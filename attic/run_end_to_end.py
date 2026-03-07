#!/usr/bin/env python
"""End-to-end experiment: sample data ONCE, train flow, then refine with GPA.

Both the flow training and GPA refinement use the EXACT same joint samples.
Compares Flow+GPA vs GPA-only (from prior) on identical data.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from problems import get_problem, PROBLEMS
from flows import W1W2Flow
from refinement.gpa import gpa_refine
from utils.evaluation import generate_posterior, compute_mmd, compute_mmd_bandwidth
from utils.integrators import euler_integrate


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end: train flow + GPA on same data")
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEMS.keys()))
    # Flow training
    parser.add_argument('--flow-iters', type=int, default=20000)
    parser.add_argument('--flow-lr', type=float, default=1e-3)
    parser.add_argument('--flow-lam', type=float, default=0.01)
    parser.add_argument('--flow-lip-scale', type=float, default=10.0)
    parser.add_argument('--flow-gp-lambda', type=float, default=1.0)
    parser.add_argument('--flow-n-steps', type=int, default=10)
    parser.add_argument('--flow-vel-hidden', type=int, default=128)
    parser.add_argument('--flow-vel-layers', type=int, default=3)
    parser.add_argument('--flow-disc-updates', type=int, default=5)
    # GPA refinement
    parser.add_argument('--K', type=int, default=500)
    parser.add_argument('--eta', type=float, default=0.01)
    parser.add_argument('--disc-steps', type=int, default=10)
    parser.add_argument('--disc-lr', type=float, default=0.001)
    parser.add_argument('--L', type=float, default=100.0)
    parser.add_argument('--gp-weight', type=float, default=0.0)
    parser.add_argument('--disc-hidden', type=int, default=32)
    parser.add_argument('--disc-layers', type=int, default=4)
    parser.add_argument('--formulation', type=str, default='LT', choices=['LT', 'LT_nu', 'DV'])
    parser.add_argument('--activation', type=str, default='mollified_relu',
                        choices=['relu', 'mollified_relu', 'silu'])
    # Shared
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-eval', type=int, default=2000)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def run_gpa_with_eval(particles, y_particles, theta_data, y_data, n_coupled,
                      eval_offsets, y_test_values, problem, args, device, label,
                      disc=None, mmd_bandwidths=None):
    def eval_callback(particles_current, step):
        with torch.no_grad():
            p_np = particles_current.cpu().numpy()
        dists = {}
        mmds = {}
        for y_val in y_test_values:
            start, end = eval_offsets[y_val]
            samples = p_np[start:end]
            d = problem.compute_distance(samples, y_val).mean()
            dists[y_val] = float(d)
            if hasattr(problem, 'sample_true_posterior') and mmd_bandwidths is not None:
                true_samples = problem.sample_true_posterior(y_val, len(samples))
                mmds[y_val] = compute_mmd(samples, true_samples, bandwidth=mmd_bandwidths[y_val])
        mean_d = np.mean(list(dists.values()))
        dists['mean'] = mean_d
        if mmds:
            mean_mmd = np.mean(list(mmds.values()))
            mmds['mean'] = mean_mmd
        result = {'dist': dists, 'mmd': mmds}
        if step % 50 == 0:
            mmd_str = f", mmd={mean_mmd:.6f}" if mmds else ""
            print(f"  [{label} step {step}] mean_dist={mean_d:.4f}{mmd_str}  " +
                  "  ".join(f"y={y}:{d:.4f}" for y, d in dists.items() if y != 'mean'))
        return result

    result = gpa_refine(
        particles=particles.clone(),
        y_particles=y_particles.clone(),
        theta_data=theta_data,
        y_data=y_data,
        n_coupled=n_coupled,
        disc=disc,
        K=args.K,
        eta=args.eta,
        disc_steps=args.disc_steps,
        disc_lr=args.disc_lr,
        L=args.L,
        batch_size=args.batch_size,
        gp_weight=args.gp_weight,
        disc_hidden=args.disc_hidden,
        disc_layers=args.disc_layers,
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
    print(f"Joint data: {len(theta_data)} samples (FIXED for flow + GPA)")

    # Also need CPU versions for flow training
    theta_train_cpu = torch.FloatTensor(theta_np)
    y_train_cpu = torch.FloatTensor(y_np)
    if y_train_cpu.dim() == 1:
        y_train_cpu = y_train_cpu.unsqueeze(1)

    # === Step 2: Train flow on this data ===
    print(f"\n{'='*60}")
    print(f"TRAINING FLOW ({args.flow_iters} iters)")
    print(f"{'='*60}")

    defaults = problem.default_hyperparams()
    use_quad = defaults.get('use_quadratic_features', False)

    flow = W1W2Flow(
        theta_dim=problem.theta_dim,
        y_dim=problem.y_dim,
        vel_hidden=args.flow_vel_hidden,
        vel_layers=args.flow_vel_layers,
        disc_hidden=args.disc_hidden,
        disc_layers=args.disc_layers,
        lip_scale=args.flow_lip_scale,
        use_quadratic_features=use_quad,
        gp_lambda=args.flow_gp_lambda,
        device=device
    )

    history = flow.train(
        theta_train_cpu,
        y_train_cpu,
        n_iters=args.flow_iters,
        batch_size=args.batch_size,
        lr=args.flow_lr,
        lam=args.flow_lam,
        n_steps=args.flow_n_steps,
        disc_updates=args.flow_disc_updates,
        checkpoint_dir=None,
        checkpoint_every=99999,
        verbose=True
    )

    # === Step 3: Generate particles ===
    y_test_values = problem.default_y_test_values()

    # Flow particles
    torch.manual_seed(args.seed + 1000)
    flow.vel_net.eval()
    with torch.no_grad():
        z_flow = torch.randn(args.n_train, problem.theta_dim, device=device)
        traj = euler_integrate(flow.vel_net, z_flow, y_data, args.flow_n_steps)
        flow_train_particles = traj[-1]

    # Prior particles
    torch.manual_seed(args.seed + 2000)
    prior_train_particles = torch.randn(args.n_train, problem.theta_dim, device=device)

    # Eval particles
    eval_particles_flow = []
    eval_particles_prior = []
    eval_y_list = []
    eval_offsets = {}
    all_particles_before_flow = {}
    all_particles_before_prior = {}

    offset = args.n_train
    for y_val in y_test_values:
        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        samples_flow = generate_posterior(
            flow.vel_net, y_val, args.n_eval, args.flow_n_steps,
            problem.theta_dim, problem.y_dim, device
        )
        all_particles_before_flow[y_val] = samples_flow.copy()
        eval_particles_flow.append(torch.FloatTensor(samples_flow).to(device))

        torch.manual_seed(args.seed + 4000 + int(y_val * 100))
        samples_prior = torch.randn(args.n_eval, problem.theta_dim).numpy()
        all_particles_before_prior[y_val] = samples_prior.copy()
        eval_particles_prior.append(torch.FloatTensor(samples_prior).to(device))

        eval_y_list.append(torch.full((args.n_eval, problem.y_dim), y_val, device=device))
        eval_offsets[y_val] = (offset, offset + args.n_eval)
        offset += args.n_eval

        dist_flow = problem.compute_distance(samples_flow, y_val).mean()
        dist_prior = problem.compute_distance(samples_prior, y_val).mean()
        print(f"  y={y_val}: flow_dist={dist_flow:.4f}, prior_dist={dist_prior:.4f}")

    all_flow = torch.cat([flow_train_particles] + eval_particles_flow, dim=0)
    all_prior = torch.cat([prior_train_particles] + eval_particles_prior, dim=0)
    all_y = torch.cat([y_data] + eval_y_list, dim=0)

    # Precompute MMD bandwidths from true posterior (fixed across all comparisons)
    mmd_bandwidths = None
    if hasattr(problem, 'sample_true_posterior'):
        mmd_bandwidths = {}
        for y_val in y_test_values:
            true_samp = problem.sample_true_posterior(y_val, 5000)
            mmd_bandwidths[y_val] = compute_mmd_bandwidth(true_samp)
            print(f"  MMD bandwidth for y={y_val}: {mmd_bandwidths[y_val]:.4f}")

    # === Step 4: Run GPA on SAME joint data ===
    print(f"\n{'='*60}")
    print("FLOW + GPA (warm-start disc, same joint data as flow training)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed + 5000)
    result_flow = run_gpa_with_eval(
        all_flow, all_y, theta_data, y_data, args.n_train,
        eval_offsets, y_test_values, problem, args, device, "FLOW+GPA",
        disc=flow.disc, mmd_bandwidths=mmd_bandwidths)

    print(f"\n{'='*60}")
    print("GPA-ONLY from prior (fresh disc, same joint data)")
    print(f"{'='*60}")
    torch.manual_seed(args.seed + 5000)
    result_prior = run_gpa_with_eval(
        all_prior, all_y, theta_data, y_data, args.n_train,
        eval_offsets, y_test_values, problem, args, device, "GPA-ONLY",
        disc=None, mmd_bandwidths=mmd_bandwidths)

    # === Step 5: Plot ===
    refined_flow = result_flow['particles'].cpu().numpy()
    refined_prior = result_prior['particles'].cpu().numpy()
    eval_flow = result_flow['history']['eval']
    eval_prior = result_prior['history']['eval']

    has_mmd = bool(eval_flow[0][1]['mmd'])
    n_plots = 3 if has_mmd else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))

    # Distance trajectory
    ax = axes[0]
    steps_f = [s for s, _ in eval_flow]
    steps_p = [s for s, _ in eval_prior]
    mean_f = [d['dist']['mean'] for _, d in eval_flow]
    mean_p = [d['dist']['mean'] for _, d in eval_prior]
    ax.plot(steps_f, mean_f, 'b-o', markersize=4, linewidth=2, label='Flow + GPA')
    ax.plot(steps_p, mean_p, 'r-s', markersize=4, linewidth=2, label='GPA-only (from prior)')
    ax.set_xlabel('GPA step')
    ax.set_ylabel('Mean distance')
    ax.set_title(f'{problem.name}: Distance trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # MMD trajectory
    if has_mmd:
        ax = axes[1]
        mmd_f = [max(d['mmd']['mean'], 1e-8) for _, d in eval_flow]
        mmd_p = [max(d['mmd']['mean'], 1e-8) for _, d in eval_prior]
        ax.plot(steps_f, mmd_f, 'b-o', markersize=4, linewidth=2, label='Flow + GPA')
        ax.plot(steps_p, mmd_p, 'r-s', markersize=4, linewidth=2, label='GPA-only (from prior)')
        ax.set_xlabel('GPA step')
        ax.set_ylabel('MMD²')
        ax.set_yscale('log')
        ax.set_title(f'{problem.name}: MMD² trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Bar chart
    ax = axes[-1]
    n_y = len(y_test_values)
    x_pos = np.arange(n_y)
    width = 0.2
    before_flow_vals = [problem.compute_distance(all_particles_before_flow[y], y).mean() for y in y_test_values]
    before_prior_vals = [problem.compute_distance(all_particles_before_prior[y], y).mean() for y in y_test_values]
    after_flow_vals, after_prior_vals = [], []
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

    fig.suptitle(f'End-to-end: Flow(L={args.flow_lip_scale}) + GPA(L={args.L}) vs GPA-only(L={args.L})\n'
                 f'K={args.K}, η={args.eta}, ds={args.disc_steps} | SAME {args.n_train} joint samples', fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plot_path = out_dir / f'e2e_{problem.name}_K{args.K}_eta{args.eta}_L{args.L}_ds{args.disc_steps}.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {plot_path}")
    plt.close()

    # === Distribution plots: 4 rows (flow before, flow after, prior before, prior after) x n_y cols ===
    n_test = len(y_test_values)
    fig_dist, axes_dist = plt.subplots(5, n_test, figsize=(4 * n_test, 20))

    for i, y_val in enumerate(y_test_values):
        s, e = eval_offsets[y_val]
        panels = [
            (all_particles_before_flow[y_val], 'Flow (before GPA)', 'C0'),
            (refined_flow[s:e], 'Flow + GPA', 'C1'),
            (all_particles_before_prior[y_val], 'Prior (before GPA)', 'C2'),
            (refined_prior[s:e], 'GPA-only', 'C3'),
        ]

        for row, (samples, label, color) in enumerate(panels):
            ax = axes_dist[row, i]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5, color=color, label=label, zorder=2)
            dist = problem.compute_distance(samples, y_val).mean()
            ax.set_title(f'y={y_val}: dist={dist:.3f}\n{label}', fontsize=10)
            ax.set_xlabel('θ₁')
            ax.set_ylabel('θ₂')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Row 4: θ₁ marginal histograms
        ax = axes_dist[4, i]
        ax.hist(all_particles_before_flow[y_val][:, 0], bins=50, density=True, alpha=0.3,
                color='C0', label='Flow before')
        ax.hist(refined_flow[s:e, 0], bins=50, density=True, alpha=0.3,
                color='C1', label='Flow+GPA')
        ax.hist(all_particles_before_prior[y_val][:, 0], bins=50, density=True, alpha=0.3,
                color='C2', label='Prior before')
        ax.hist(refined_prior[s:e, 0], bins=50, density=True, alpha=0.3,
                color='C3', label='GPA-only')
        grid_pts = np.linspace(-4, 4, 200)
        true_pdf = problem.true_posterior_pdf(grid_pts, y_val, dim=0)
        if true_pdf is not None:
            ax.plot(grid_pts, true_pdf, 'k-', linewidth=2, label='True p(θ₁|y)')
        ax.set_title(f'θ₁ marginal (y={y_val})', fontsize=10)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig_dist.suptitle(
        f'Distributions: Flow(L={args.flow_lip_scale}) + GPA(L={args.L}) vs GPA-only(L={args.L})\n'
        f'K={args.K}, η={args.eta}, ds={args.disc_steps} | SAME {args.n_train} joint samples', fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    dist_plot_path = out_dir / f'e2e_dist_{problem.name}_K{args.K}_eta{args.eta}_L{args.L}_ds{args.disc_steps}.png'
    fig_dist.savefig(dist_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved distributions: {dist_plot_path}")
    plt.close(fig_dist)

    # Compute final MMDs with fixed bandwidth
    mmd_before_flow, mmd_after_flow, mmd_before_prior, mmd_after_prior = {}, {}, {}, {}
    if hasattr(problem, 'sample_true_posterior') and mmd_bandwidths is not None:
        for y_val in y_test_values:
            s, e = eval_offsets[y_val]
            bw = mmd_bandwidths[y_val]
            true_samp = problem.sample_true_posterior(y_val, 5000)
            mmd_before_flow[y_val] = compute_mmd(all_particles_before_flow[y_val], true_samp, bandwidth=bw)
            mmd_after_flow[y_val] = compute_mmd(refined_flow[s:e], true_samp, bandwidth=bw)
            mmd_before_prior[y_val] = compute_mmd(all_particles_before_prior[y_val], true_samp, bandwidth=bw)
            mmd_after_prior[y_val] = compute_mmd(refined_prior[s:e], true_samp, bandwidth=bw)

    # Summary
    print(f"\n{'='*60}")
    print("END-TO-END COMPARISON (same joint data throughout)")
    print(f"{'='*60}")
    print(f"\nDistance to manifold:")
    print(f"{'y':>6}  {'Flow before':>12} {'Flow+GPA':>12} {'Prior before':>12} {'GPA-only':>12}")
    for i, y_val in enumerate(y_test_values):
        print(f"  {y_val:>4}  {before_flow_vals[i]:>12.4f} {after_flow_vals[i]:>12.4f} "
              f"{before_prior_vals[i]:>12.4f} {after_prior_vals[i]:>12.4f}")
    print(f"  Mean  {np.mean(before_flow_vals):>12.4f} {np.mean(after_flow_vals):>12.4f} "
          f"{np.mean(before_prior_vals):>12.4f} {np.mean(after_prior_vals):>12.4f}")

    if mmd_before_flow:
        print(f"\nMMD² vs true posterior:")
        print(f"{'y':>6}  {'Flow before':>12} {'Flow+GPA':>12} {'Prior before':>12} {'GPA-only':>12}")
        for y_val in y_test_values:
            print(f"  {y_val:>4}  {mmd_before_flow[y_val]:>12.6f} {mmd_after_flow[y_val]:>12.6f} "
                  f"{mmd_before_prior[y_val]:>12.6f} {mmd_after_prior[y_val]:>12.6f}")
        print(f"  Mean  {np.mean(list(mmd_before_flow.values())):>12.6f} {np.mean(list(mmd_after_flow.values())):>12.6f} "
              f"{np.mean(list(mmd_before_prior.values())):>12.6f} {np.mean(list(mmd_after_prior.values())):>12.6f}")


if __name__ == '__main__':
    main()
