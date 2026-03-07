#!/usr/bin/env python
"""Compare baseline conditional samplers on the same problems.

Methods:
  1. W1W2 Flow (our method)
  2. W1W2 Flow + GPA refinement (our method)
  3. Conditional Flow Matching (CFM)
  4. Score-based Generative Model (SGM / diffusion)
  5. CNF with MLE + W2 regularization (CNF-MLE)

All methods use the same joint training data, same architecture (hidden, layers),
and same test y values.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from problems import get_problem, PROBLEMS
from flows import W1W2Flow
from refinement.gpa import gpa_refine
from baselines import ConditionalFlowMatching, ScoreBasedDiffusion, CNF_MLE
from utils.evaluation import generate_posterior, compute_mmd, compute_mmd_bandwidth
from utils.integrators import euler_integrate


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline comparison")
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEMS.keys()))
    parser.add_argument('--n-iters', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-eval', type=int, default=2000)
    parser.add_argument('--n-steps', type=int, default=10,
                        help='ODE integration steps for flow methods')
    parser.add_argument('--sgm-sample-steps', type=int, default=200,
                        help='Sampling steps for SGM (needs more)')
    # W1W2 specific
    parser.add_argument('--flow-lam', type=float, default=0.25)
    parser.add_argument('--flow-lip-scale', type=float, default=10.0)
    parser.add_argument('--flow-gp-lambda', type=float, default=1.0)
    parser.add_argument('--flow-disc-updates', type=int, default=5)
    # W1W2 + GPA specific
    parser.add_argument('--gpa-K', type=int, default=500)
    parser.add_argument('--gpa-eta', type=float, default=0.005)
    parser.add_argument('--gpa-L', type=float, default=1000.0)
    parser.add_argument('--gpa-disc-steps', type=int, default=10)
    parser.add_argument('--gpa-disc-hidden', type=int, default=32)
    parser.add_argument('--gpa-disc-layers', type=int, default=4)
    # CNF-MLE specific
    parser.add_argument('--cnf-lam', type=float, default=0.01)
    # Output
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    problem = get_problem(args.problem)
    print(f"Problem: {problem.description}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save hyperparams
    with open(out_dir / 'hparams.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # === Sample joint data ONCE ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    theta_np, y_np = problem.sample_joint(args.n_train)
    theta_data = torch.FloatTensor(theta_np)
    y_data = torch.FloatTensor(y_np)
    if y_data.dim() == 1:
        y_data = y_data.unsqueeze(1)
    print(f"Joint data: {len(theta_data)} samples")

    y_test_values = problem.default_y_test_values()

    # Precompute MMD bandwidths
    mmd_bandwidths = None
    if hasattr(problem, 'sample_true_posterior'):
        mmd_bandwidths = {}
        for y_val in y_test_values:
            true_samp = problem.sample_true_posterior(y_val, 5000)
            mmd_bandwidths[y_val] = compute_mmd_bandwidth(true_samp)

    # === Method 1: W1W2 Flow ===
    print(f"\n{'='*60}")
    print("Training: W1W2 Flow")
    print(f"{'='*60}")
    defaults = problem.default_hyperparams()
    use_quad = defaults.get('use_quadratic_features', False)

    w1w2 = W1W2Flow(
        theta_dim=problem.theta_dim, y_dim=problem.y_dim,
        vel_hidden=args.hidden, vel_layers=args.n_layers,
        disc_hidden=args.gpa_disc_hidden, disc_layers=args.gpa_disc_layers,
        lip_scale=args.flow_lip_scale,
        use_quadratic_features=use_quad,
        gp_lambda=args.flow_gp_lambda,
        device=device,
    )
    w1w2.train(
        theta_data, y_data,
        n_iters=args.n_iters, batch_size=args.batch_size,
        lr=args.lr, lam=args.flow_lam,
        n_steps=args.n_steps, disc_updates=args.flow_disc_updates,
        verbose=True,
    )
    torch.save({
        'vel_net': w1w2.vel_net.state_dict(),
        'disc': w1w2.disc.state_dict(),
    }, out_dir / 'w1w2_checkpoint.pt')

    # === Method 2: CFM ===
    print(f"\n{'='*60}")
    print("Training: Conditional Flow Matching")
    print(f"{'='*60}")
    cfm = ConditionalFlowMatching(
        theta_dim=problem.theta_dim, y_dim=problem.y_dim,
        hidden=args.hidden, n_layers=args.n_layers,
        device=device,
    )
    cfm.train(theta_data, y_data, n_iters=args.n_iters,
              batch_size=args.batch_size, lr=args.lr, verbose=True)
    torch.save(cfm.vel_net.state_dict(), out_dir / 'cfm_checkpoint.pt')

    # === Method 3: SGM ===
    print(f"\n{'='*60}")
    print("Training: Score-based Diffusion")
    print(f"{'='*60}")
    sgm = ScoreBasedDiffusion(
        theta_dim=problem.theta_dim, y_dim=problem.y_dim,
        hidden=args.hidden, n_layers=args.n_layers,
        device=device,
    )
    sgm.train(theta_data, y_data, n_iters=args.n_iters,
              batch_size=args.batch_size, lr=args.lr, verbose=True)
    torch.save(sgm.eps_net.state_dict(), out_dir / 'sgm_checkpoint.pt')

    # === Method 4: CNF-MLE ===
    print(f"\n{'='*60}")
    print("Training: CNF-MLE + W2")
    print(f"{'='*60}")
    cnf = CNF_MLE(
        theta_dim=problem.theta_dim, y_dim=problem.y_dim,
        hidden=args.hidden, n_layers=args.n_layers,
        device=device,
    )
    cnf.train(theta_data, y_data, n_iters=args.n_iters,
              batch_size=args.batch_size, lr=args.lr,
              lam=args.cnf_lam, n_steps=args.n_steps, verbose=True)
    torch.save(cnf.vel_net.state_dict(), out_dir / 'cnf_checkpoint.pt')

    # === Generate samples from all methods ===
    print(f"\n{'='*60}")
    print("Generating samples")
    print(f"{'='*60}")

    methods = {
        'W1W2': {},
        'W1W2+GPA': {},
        'CFM': {},
        'SGM': {},
        'CNF-MLE': {},
    }

    theta_data_dev = theta_data.to(device)
    y_data_dev = y_data.to(device)
    if y_data_dev.dim() == 1:
        y_data_dev = y_data_dev.unsqueeze(1)

    for y_val in y_test_values:
        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        methods['W1W2'][y_val] = generate_posterior(
            w1w2.vel_net, y_val, args.n_eval, args.n_steps,
            problem.theta_dim, problem.y_dim, device)

        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        methods['CFM'][y_val] = cfm.sample(y_val, args.n_eval, n_steps=50)

        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        methods['SGM'][y_val] = sgm.sample(y_val, args.n_eval,
                                           n_steps=args.sgm_sample_steps)

        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        methods['CNF-MLE'][y_val] = cnf.sample(y_val, args.n_eval, n_steps=50)

    # === W1W2 + GPA refinement ===
    print(f"\n{'='*60}")
    print("Running GPA refinement on W1W2 flow output")
    print(f"{'='*60}")

    # Build particle set for GPA: train particles + eval particles
    torch.manual_seed(args.seed + 1000)
    w1w2.vel_net.eval()
    with torch.no_grad():
        z_train = torch.randn(args.n_train, problem.theta_dim, device=device)
        traj = euler_integrate(w1w2.vel_net, z_train, y_data_dev, args.n_steps)
        train_particles = traj[-1]

    eval_particles = []
    eval_y_list = []
    eval_offsets = {}
    offset = args.n_train
    for y_val in y_test_values:
        p = torch.FloatTensor(methods['W1W2'][y_val]).to(device)
        eval_particles.append(p)
        eval_y_list.append(torch.full((args.n_eval, problem.y_dim), y_val, device=device))
        eval_offsets[y_val] = (offset, offset + args.n_eval)
        offset += args.n_eval

    all_particles = torch.cat([train_particles] + eval_particles, dim=0)
    all_y = torch.cat([y_data_dev] + eval_y_list, dim=0)

    torch.manual_seed(args.seed + 5000)
    gpa_result = gpa_refine(
        particles=all_particles.clone(),
        y_particles=all_y.clone(),
        theta_data=theta_data_dev,
        y_data=y_data_dev,
        n_coupled=args.n_train,
        disc=w1w2.disc,
        K=args.gpa_K, eta=args.gpa_eta,
        disc_steps=args.gpa_disc_steps, disc_lr=0.001,
        L=args.gpa_L, batch_size=args.batch_size,
        disc_hidden=args.gpa_disc_hidden,
        disc_layers=args.gpa_disc_layers,
        device=device, verbose=True,
    )
    refined = gpa_result['particles'].cpu().numpy()
    for y_val in y_test_values:
        s, e = eval_offsets[y_val]
        methods['W1W2+GPA'][y_val] = refined[s:e]

    # === Evaluate all methods ===
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")

    results = {}
    method_names = list(methods.keys())

    for name in method_names:
        results[name] = {'dist': {}, 'mmd': {}}
        for y_val in y_test_values:
            samples = methods[name][y_val]
            d = float(problem.compute_distance(samples, y_val).mean())
            results[name]['dist'][str(y_val)] = d
            if mmd_bandwidths is not None:
                true_samp = problem.sample_true_posterior(y_val, 5000)
                m = float(compute_mmd(samples, true_samp,
                                      bandwidth=mmd_bandwidths[y_val]))
                results[name]['mmd'][str(y_val)] = m
        results[name]['dist']['mean'] = float(np.mean(
            [results[name]['dist'][str(y)] for y in y_test_values]))
        if results[name]['mmd']:
            results[name]['mmd']['mean'] = float(np.mean(
                [results[name]['mmd'][str(y)] for y in y_test_values]))

    # Save results
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Save particles
    particles_dict = {}
    for name in method_names:
        for y_val in y_test_values:
            key = f'{name.replace("+","_").replace("-","_")}_y{y_val}'
            particles_dict[key] = methods[name][y_val]
    np.savez_compressed(out_dir / 'all_particles.npz', **particles_dict)

    # Print table
    print(f"\nDistance to manifold:")
    header = f"{'y':>6}" + "".join(f"  {n:>12}" for n in method_names)
    print(header)
    for y_val in y_test_values:
        row = f"  {y_val:>4}"
        for name in method_names:
            row += f"  {results[name]['dist'][str(y_val)]:>12.4f}"
        print(row)
    row = f"  {'Mean':>4}"
    for name in method_names:
        row += f"  {results[name]['dist']['mean']:>12.4f}"
    print(row)

    if results[method_names[0]]['mmd']:
        print(f"\nMMD² vs true posterior:")
        print(header)
        for y_val in y_test_values:
            row = f"  {y_val:>4}"
            for name in method_names:
                row += f"  {results[name]['mmd'][str(y_val)]:>12.6f}"
            print(row)
        row = f"  {'Mean':>4}"
        for name in method_names:
            row += f"  {results[name]['mmd']['mean']:>12.6f}"
        print(row)

    # === Plots ===
    # Distribution plots: one row per method, one column per y
    n_methods = len(method_names)
    n_test = len(y_test_values)
    fig, axes = plt.subplots(n_methods, n_test, figsize=(4 * n_test, 4 * n_methods),
                             squeeze=False)

    for col, y_val in enumerate(y_test_values):
        for row, name in enumerate(method_names):
            ax = axes[row, col]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            samples = methods[name][y_val]
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
            dist = results[name]['dist'][str(y_val)]
            mmd_str = ""
            if results[name]['mmd']:
                mmd_str = f", MMD²={results[name]['mmd'][str(y_val)]:.4f}"
            ax.set_title(f'{name}\ny={y_val}: dist={dist:.3f}{mmd_str}', fontsize=9)
            if row == n_methods - 1:
                ax.set_xlabel('θ₁')
            if col == 0:
                ax.set_ylabel('θ₂')
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{problem.name}: Baseline comparison ({args.n_iters} iters, '
                 f'hidden={args.hidden}, {args.n_train} samples)', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    fig.savefig(out_dir / f'baselines_{problem.name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_dir / f'baselines_{problem.name}.png'}")

    # Bar chart comparison
    fig_bar, ax_bar = plt.subplots(1, 1, figsize=(10, 5))
    x_pos = np.arange(n_test)
    width = 0.15
    bar_colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for i, name in enumerate(method_names):
        vals = [results[name]['dist'][str(y)] for y in y_test_values]
        ax_bar.bar(x_pos + i * width, vals, width, label=name, color=bar_colors[i])
    ax_bar.set_xticks(x_pos + width * (n_methods - 1) / 2)
    ax_bar.set_xticklabels([f'y={y}' for y in y_test_values])
    ax_bar.set_ylabel('Mean distance')
    ax_bar.set_title(f'{problem.name}: Distance comparison')
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3, axis='y')
    fig_bar.savefig(out_dir / f'baselines_bar_{problem.name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / f'baselines_bar_{problem.name}.png'}")


if __name__ == '__main__':
    main()
