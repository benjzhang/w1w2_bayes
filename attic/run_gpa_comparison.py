#!/usr/bin/env python
"""Head-to-head comparison: Flow+GPA vs GPA-only on the same joint data.

1. Sample joint data (theta, y) ~ pi(theta, y) ONCE
2. Path A: Generate flow particles -> refine with GPA
3. Path B: Start from prior -> refine with GPA (same joint data, same disc init seed)

Both use identical joint samples, disc architecture, and hyperparameters.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy

from problems import get_problem, PROBLEMS
from flows import W1W2Flow
from refinement.gpa import gpa_refine
from utils.evaluation import generate_posterior
from utils.integrators import euler_integrate


def parse_args():
    parser = argparse.ArgumentParser(description="GPA comparison: flow+GPA vs GPA-only")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEMS.keys()))
    parser.add_argument('--K', type=int, default=500)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--disc-steps', type=int, default=10)
    parser.add_argument('--disc-lr', type=float, default=0.001)
    parser.add_argument('--L', type=float, default=100.0)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-steps', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--gp-weight', type=float, default=0.0)
    parser.add_argument('--disc-hidden', type=int, default=32)
    parser.add_argument('--formulation', type=str, default='LT', choices=['LT', 'LT_nu', 'DV'])
    parser.add_argument('--activation', type=str, default='mollified_relu',
                        choices=['relu', 'mollified_relu', 'silu'])
    parser.add_argument('--n-eval', type=int, default=2000)
    parser.add_argument('--prior-scale', type=float, default=1.0)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def run_gpa_with_eval(particles, y_particles, theta_data, y_data, n_coupled,
                      eval_offsets, y_test_values, problem, args, device, label):
    """Run GPA and return trajectory + final particles."""

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
            print(f"  [{label} step {step}] mean_dist={mean_d:.4f}  " +
                  "  ".join(f"y={y}:{d:.4f}" for y, d in dists.items() if y != 'mean'))
        return dists

    result = gpa_refine(
        particles=particles.clone(),
        y_particles=y_particles.clone(),
        theta_data=theta_data,
        y_data=y_data,
        n_coupled=n_coupled,
        disc=None,
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
    return result


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    problem = get_problem(args.problem)
    print(f"Problem: {problem.description}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: Sample joint data ONCE ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    theta_np, y_np = problem.sample_joint(args.n_train)
    theta_data = torch.FloatTensor(theta_np).to(device)
    y_data = torch.FloatTensor(y_np).to(device)
    if y_data.dim() == 1:
        y_data = y_data.unsqueeze(1)
    print(f"Joint data: {len(theta_data)} samples")

    # Load flow
    flow = W1W2Flow.from_checkpoint(args.checkpoint, device=device)
    y_test_values = problem.default_y_test_values()

    # === Step 2: Generate particles for both paths ===
    # Use same seed for flow generation
    torch.manual_seed(args.seed + 1000)

    # Flow particles
    flow.vel_net.eval()
    with torch.no_grad():
        z_flow = torch.randn(args.n_train, problem.theta_dim, device=device)
        traj = euler_integrate(flow.vel_net, z_flow, y_data, args.n_steps)
        flow_train_particles = traj[-1]

    # Prior particles (same seed for fair comparison of disc init)
    torch.manual_seed(args.seed + 2000)
    prior_train_particles = args.prior_scale * torch.randn(args.n_train, problem.theta_dim, device=device)

    # Eval particles for both paths
    eval_particles_flow = []
    eval_particles_prior = []
    eval_y_list = []
    eval_offsets = {}
    all_particles_before_flow = {}
    all_particles_before_prior = {}

    offset = args.n_train
    for y_val in y_test_values:
        # Flow eval particles
        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        samples_flow = generate_posterior(
            flow.vel_net, y_val, args.n_eval, args.n_steps,
            problem.theta_dim, problem.y_dim, device
        )
        all_particles_before_flow[y_val] = samples_flow.copy()
        eval_particles_flow.append(torch.FloatTensor(samples_flow).to(device))

        # Prior eval particles
        torch.manual_seed(args.seed + 4000 + int(y_val * 100))
        samples_prior = (args.prior_scale * torch.randn(args.n_eval, problem.theta_dim)).numpy()
        all_particles_before_prior[y_val] = samples_prior.copy()
        eval_particles_prior.append(torch.FloatTensor(samples_prior).to(device))

        eval_y_list.append(torch.full((args.n_eval, problem.y_dim), y_val, device=device))
        eval_offsets[y_val] = (offset, offset + args.n_eval)
        offset += args.n_eval

        dist_flow = problem.compute_distance(samples_flow, y_val).mean()
        dist_prior = problem.compute_distance(samples_prior, y_val).mean()
        print(f"  y={y_val}: flow_dist={dist_flow:.4f}, prior_dist={dist_prior:.4f}")

    # Concatenate
    all_flow = torch.cat([flow_train_particles] + eval_particles_flow, dim=0)
    all_prior = torch.cat([prior_train_particles] + eval_particles_prior, dim=0)
    all_y = torch.cat([y_data] + eval_y_list, dim=0)

    # === Step 3: Run GPA for both paths ===
    print(f"\n{'='*60}")
    print("PATH A: Flow + GPA")
    print(f"{'='*60}")
    torch.manual_seed(args.seed + 5000)  # same disc init seed
    result_flow = run_gpa_with_eval(
        all_flow, all_y, theta_data, y_data, args.n_train,
        eval_offsets, y_test_values, problem, args, device, "FLOW+GPA")

    print(f"\n{'='*60}")
    print("PATH B: Prior + GPA (GPA-only)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed + 5000)  # SAME disc init seed
    result_prior = run_gpa_with_eval(
        all_prior, all_y, theta_data, y_data, args.n_train,
        eval_offsets, y_test_values, problem, args, device, "GPA-ONLY")

    # === Step 4: Plot comparison ===
    refined_flow = result_flow['particles'].cpu().numpy()
    refined_prior = result_prior['particles'].cpu().numpy()

    eval_flow = result_flow['history']['eval']
    eval_prior = result_prior['history']['eval']

    # Trajectory comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: trajectory over GPA steps
    ax = axes[0]
    steps_f = [s for s, _ in eval_flow]
    steps_p = [s for s, _ in eval_prior]
    mean_f = [d['mean'] for _, d in eval_flow]
    mean_p = [d['mean'] for _, d in eval_prior]
    ax.plot(steps_f, mean_f, 'b-o', markersize=4, linewidth=2, label='Flow + GPA')
    ax.plot(steps_p, mean_p, 'r-s', markersize=4, linewidth=2, label='GPA-only (from prior)')
    ax.set_xlabel('GPA step')
    ax.set_ylabel('Mean distance')
    ax.set_title(f'{problem.name}: Distance trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: per-y final comparison
    ax = axes[1]
    n_y = len(y_test_values)
    x_pos = np.arange(n_y)
    width = 0.2

    before_flow_vals = [problem.compute_distance(all_particles_before_flow[y], y).mean() for y in y_test_values]
    before_prior_vals = [problem.compute_distance(all_particles_before_prior[y], y).mean() for y in y_test_values]
    after_flow_vals = []
    after_prior_vals = []
    for y_val in y_test_values:
        s, e = eval_offsets[y_val]
        after_flow_vals.append(problem.compute_distance(refined_flow[s:e], y_val).mean())
        after_prior_vals.append(problem.compute_distance(refined_prior[s:e], y_val).mean())

    ax.bar(x_pos - 1.5*width, before_flow_vals, width, label='Flow (before)', color='lightblue')
    ax.bar(x_pos - 0.5*width, after_flow_vals, width, label='Flow+GPA (after)', color='blue')
    ax.bar(x_pos + 0.5*width, before_prior_vals, width, label='Prior (before)', color='lightsalmon')
    ax.bar(x_pos + 1.0*width, after_prior_vals, width, label='GPA-only (after)', color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'y={y}' for y in y_test_values])
    ax.set_ylabel('Mean distance')
    ax.set_title(f'{problem.name}: Before vs After')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Flow+GPA vs GPA-only | K={args.K}, η={args.eta}, L={args.L}, ds={args.disc_steps}', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plot_path = out_dir / f'comparison_{problem.name}_K{args.K}_eta{args.eta}_L{args.L}_ds{args.disc_steps}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {plot_path}")
    plt.close()

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'y':>6}  {'Flow before':>12} {'Flow+GPA':>12} {'Prior before':>12} {'GPA-only':>12}")
    for i, y_val in enumerate(y_test_values):
        print(f"  {y_val:>4}  {before_flow_vals[i]:>12.4f} {after_flow_vals[i]:>12.4f} "
              f"{before_prior_vals[i]:>12.4f} {after_prior_vals[i]:>12.4f}")
    print(f"  Mean  {np.mean(before_flow_vals):>12.4f} {np.mean(after_flow_vals):>12.4f} "
          f"{np.mean(before_prior_vals):>12.4f} {np.mean(after_prior_vals):>12.4f}")


if __name__ == '__main__':
    main()
