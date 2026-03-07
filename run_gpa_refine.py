#!/usr/bin/env python
"""Run GPA refinement on a trained W1W2 Flow checkpoint.

Generates particles for all y values in the joint training data (matching
the pi(y) distribution) plus test y values, then refines them all together
using a conditional discriminator. Test particles get the full K iterative
updates alongside the training particles.

Usage:
    python run_gpa_refine.py --checkpoint results/circle/checkpoints/.../checkpoint_iter20000.pt \
        --problem circle --K 500 --eta 0.1 --disc-steps 3
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from problems import get_problem, PROBLEMS
from flows import W1W2Flow
from refinement.gpa import gpa_refine
from utils.evaluation import generate_posterior
from utils.integrators import euler_integrate


def parse_args():
    parser = argparse.ArgumentParser(description="GPA refinement of W1W2 Flow")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEMS.keys()))
    parser.add_argument('--K', type=int, default=500, help='Number of GPA refinement steps')
    parser.add_argument('--eta', type=float, default=0.5, help='Particle step size (lr_P)')
    parser.add_argument('--disc-steps', type=int, default=3, help='Disc updates per GPA step')
    parser.add_argument('--disc-lr', type=float, default=0.001, help='Disc learning rate')
    parser.add_argument('--L', type=float, default=1.0, help='Lipschitz constant for discriminator')
    parser.add_argument('--no-warmstart', action='store_true', help='Use fresh disc instead of warm-starting from checkpoint')
    parser.add_argument('--from-prior', action='store_true', help='Start from prior (N(0,1)) instead of flow output — sanity check')
    parser.add_argument('--prior-scale', type=float, default=1.0, help='Scale of prior N(0, scale^2) when using --from-prior')
    parser.add_argument('--n-train', type=int, default=10000, help='Training data size (also number of pi(y) particles)')
    parser.add_argument('--n-steps', type=int, default=10, help='ODE integration steps')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for disc training')
    parser.add_argument('--gp-weight', type=float, default=0.0, help='One-sided gradient penalty weight (0 = use hard projection)')
    parser.add_argument('--disc-hidden', type=int, default=32, help='Disc hidden layer width')
    parser.add_argument('--formulation', type=str, default='LT', choices=['LT', 'LT_nu', 'DV'],
                        help='KL variational formulation: LT (default), LT_nu (trainable nu), DV (Donsker-Varadhan)')
    parser.add_argument('--activation', type=str, default=None, choices=['relu', 'mollified_relu', 'silu'],
                        help='Disc activation (default: auto-select based on constraint type)')
    parser.add_argument('--n-eval', type=int, default=2000, help='Number of particles per test y for evaluation')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    # Load problem and generate joint training data (theta, y) ~ pi(theta, y)
    problem = get_problem(args.problem)
    print(f"Problem: {problem.description}")

    theta_np, y_np = problem.sample_joint(args.n_train)
    theta_data = torch.FloatTensor(theta_np).to(device)
    y_data = torch.FloatTensor(y_np).to(device)
    if y_data.dim() == 1:
        y_data = y_data.unsqueeze(1)

    # Load trained flow
    print(f"Loading checkpoint: {args.checkpoint}")
    flow = W1W2Flow.from_checkpoint(args.checkpoint, device=device)

    y_test_values = problem.default_y_test_values()

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.checkpoint).parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate particles for ALL y values in the joint data (matching pi(y))
    if args.from_prior:
        print(f"\nUsing PRIOR N(0,{args.prior_scale}^2) particles")
        train_particles = args.prior_scale * torch.randn(args.n_train, problem.theta_dim, device=device)
    else:
        print(f"\nGenerating {args.n_train} particles for y ~ pi(y)...")
        flow.vel_net.eval()
        with torch.no_grad():
            z = torch.randn(args.n_train, problem.theta_dim, device=device)
            traj = euler_integrate(flow.vel_net, z, y_data, args.n_steps)
            train_particles = traj[-1]  # (n_train, theta_dim)
    train_y = y_data.clone()
    print(f"  Generated {len(train_particles)} particles")

    # Generate evaluation particles at test y values
    print(f"\nGenerating {args.n_eval} eval particles per test y...")
    eval_particles_list = []
    eval_y_list = []
    eval_offsets = {}  # track where each test y's particles are in the concatenated array
    all_particles_before = {}

    offset = len(train_particles)  # eval particles start after train particles
    for y_val in y_test_values:
        if args.from_prior:
            samples = (args.prior_scale * torch.randn(args.n_eval, problem.theta_dim)).numpy()
        else:
            samples = generate_posterior(
                flow.vel_net, y_val, args.n_eval, args.n_steps,
                problem.theta_dim, problem.y_dim, device
            )
        all_particles_before[y_val] = samples.copy()

        eval_particles_list.append(torch.FloatTensor(samples).to(device))
        eval_y_list.append(torch.full((args.n_eval, problem.y_dim), y_val, device=device))

        eval_offsets[y_val] = (offset, offset + args.n_eval)
        offset += args.n_eval

        dist = problem.compute_distance(samples, y_val)
        print(f"  y={y_val}: mean_dist = {dist.mean():.4f}")

    # Concatenate ALL particles: train (pi(y)) + eval (test y values)
    all_particles = torch.cat([train_particles] + eval_particles_list, dim=0)
    all_y = torch.cat([train_y] + eval_y_list, dim=0)
    print(f"\nTotal particles: {len(all_particles)} ({len(train_particles)} train + {len(all_particles)-len(train_particles)} eval)")

    # Eval callback: measure distance at test y values during GPA
    def eval_callback(particles_current, step):
        with torch.no_grad():
            p_np = particles_current.cpu().numpy()
        dists = {}
        for y_val in y_test_values:
            start, end = eval_offsets[y_val]
            d = problem.compute_distance(p_np[start:end], y_val).mean()
            dists[y_val] = float(d)
        mean_d = np.mean(list(dists.values()))
        dists['mean'] = mean_d
        if step % 50 == 0:
            print(f"  [EVAL step {step}] mean_dist={mean_d:.4f}  " +
                  "  ".join(f"y={y}:{d:.4f}" for y, d in dists.items() if y != 'mean'))
        return dists

    # Run GPA refinement on ALL particles together
    # First n_train particles are coupled 1-to-1 with joint data (same y values)
    n_coupled = len(train_particles)
    result = gpa_refine(
        particles=all_particles,
        y_particles=all_y,
        theta_data=theta_data,
        y_data=y_data,
        n_coupled=n_coupled,
        disc=None if args.no_warmstart else flow.disc,
        K=args.K,
        eta=args.eta,
        disc_steps=args.disc_steps,
        disc_lr=args.disc_lr,
        L=args.L,
        batch_size=args.batch_size,
        gp_weight=args.gp_weight,
        disc_hidden=args.disc_hidden,
        formulation=args.formulation,
        activation=args.activation,
        device=device,
        verbose=True,
        eval_callback=eval_callback,
        eval_every=50,
    )

    # Extract refined eval particles per test y
    refined_all = result['particles'].cpu().numpy()
    all_particles_after = {}
    for y_val in y_test_values:
        start, end = eval_offsets[y_val]
        all_particles_after[y_val] = refined_all[start:end]

    # Plot and summarize
    n_test = len(y_test_values)
    fig, axes = plt.subplots(3, n_test, figsize=(4 * n_test, 12))

    all_before_dists = []
    all_after_dists = []

    for i, y_val in enumerate(y_test_values):
        particles_before = all_particles_before[y_val]
        particles_after = all_particles_after[y_val]

        dist_before = problem.compute_distance(particles_before, y_val)
        dist_after = problem.compute_distance(particles_after, y_val)

        all_before_dists.append(dist_before.mean())
        all_after_dists.append(dist_after.mean())

        # Row 0: Before GPA (2D scatter)
        # Row 1: After GPA (2D scatter)
        for row, (samples, label, dist) in enumerate([
            (particles_before, 'Before GPA', dist_before),
            (particles_after, 'After GPA', dist_after),
        ]):
            ax = axes[row, i]
            problem.plot_true_posterior(ax, y_val, alpha=0.7)
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5, label=label)
            ax.set_title(f'y={y_val}: dist={dist.mean():.3f} ({label})')
            ax.set_xlabel('θ₁')
            ax.set_ylabel('θ₂')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Row 2: θ₁ marginal histogram vs true PDF
        ax = axes[2, i]
        ax.hist(particles_before[:, 0], bins=50, density=True, alpha=0.4,
                color='blue', label='Before')
        ax.hist(particles_after[:, 0], bins=50, density=True, alpha=0.4,
                color='green', label='After')
        grid = np.linspace(-4, 4, 200)
        true_pdf = problem.true_posterior_pdf(grid, y_val, dim=0)
        if true_pdf is not None:
            ax.plot(grid, true_pdf, 'r-', linewidth=2, label='True p(θ₁|y)')
        ax.set_title(f'θ₁ marginal (y={y_val})')
        ax.set_xlabel('θ₁')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    warmstart_label = 'warm-start' if not args.no_warmstart else 'fresh disc'
    gp_label = f', GP={args.gp_weight}' if args.gp_weight > 0 else ', spec norm'
    fig.suptitle(
        f'GPA Refinement: {problem.name} | K={args.K}, η={args.eta}, L={args.L}, '
        f'ds={args.disc_steps}, bs={args.batch_size}{gp_label}, {warmstart_label}',
        fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    warmstart_tag = 'fresh' if args.no_warmstart else 'warm'
    if args.from_prior:
        warmstart_tag = 'prior'
    gp_tag = f'_gp{args.gp_weight}' if args.gp_weight > 0 else ''
    form_tag = f'_{args.formulation}' if args.formulation != 'LT' else ''
    act_tag = f'_{args.activation}' if args.activation else ''
    plot_path = out_dir / f'gpa_refine_K{args.K}_eta{args.eta}_L{args.L}_ds{args.disc_steps}_bs{args.batch_size}{gp_tag}{form_tag}{act_tag}_{warmstart_tag}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {plot_path}")
    plt.close()

    # Plot distance trajectory over GPA steps
    eval_history = result['history']['eval']
    if eval_history:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        steps = [s for s, _ in eval_history]
        for y_val in y_test_values:
            dists_traj = [d[y_val] for _, d in eval_history]
            ax2.plot(steps, dists_traj, 'o-', markersize=3, label=f'y={y_val}')
        mean_traj = [d['mean'] for _, d in eval_history]
        ax2.plot(steps, mean_traj, 'k-', linewidth=2, label='mean')
        ax2.set_xlabel('GPA step')
        ax2.set_ylabel('Mean distance')
        ax2.set_title(f'Distance trajectory: {problem.name} | {warmstart_tag}')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        traj_path = out_dir / f'gpa_trajectory_K{args.K}_eta{args.eta}_L{args.L}_ds{args.disc_steps}_bs{args.batch_size}{gp_tag}{form_tag}{act_tag}_{warmstart_tag}.png'
        fig2.savefig(traj_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory: {traj_path}")
        plt.close(fig2)

    # Summary
    print(f"\n{'='*50}")
    print("Summary:")
    print(f"{'='*50}")
    for y_val, db, da in zip(y_test_values, all_before_dists, all_after_dists):
        improvement = (db - da) / db * 100
        print(f"  y={y_val}: {db:.4f} -> {da:.4f} ({improvement:+.1f}%)")
    print(f"  Mean:  {np.mean(all_before_dists):.4f} -> {np.mean(all_after_dists):.4f}")


if __name__ == '__main__':
    main()
