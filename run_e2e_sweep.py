#!/usr/bin/env python
"""End-to-end sweep: train flow ONCE, then sweep GPA (L, eta) configs.

Saves all artifacts for reproducibility:
  - Flow checkpoint (.pt)
  - Particles before/after refinement (.npz)
  - Metrics per GPA step (JSON)
  - Plots (PNG)
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import copy

from problems import get_problem, PROBLEMS
from flows import W1W2Flow
from refinement.gpa import gpa_refine
from utils.evaluation import generate_posterior, compute_mmd, compute_mmd_bandwidth
from utils.integrators import euler_integrate


def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end sweep: train flow + sweep GPA")
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
    # GPA sweep: comma-separated L:eta pairs, e.g. "10:0.01,100:0.01,1000:0.005"
    parser.add_argument('--gpa-configs', type=str, required=True,
                        help='Comma-separated L:eta pairs, e.g. "10:0.01,100:0.01,1000:0.005"')
    parser.add_argument('--K', type=int, default=500)
    parser.add_argument('--disc-steps', type=int, default=10)
    parser.add_argument('--disc-lr', type=float, default=0.001)
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
    # Flow checkpoint
    parser.add_argument('--load-flow', type=str, default=None,
                        help='Path to saved flow checkpoint (skip training)')
    return parser.parse_args()


def parse_gpa_configs(config_str):
    """Parse 'L1:eta1,L2:eta2,...' into list of (L, eta) tuples."""
    configs = []
    for pair in config_str.split(','):
        L_str, eta_str = pair.strip().split(':')
        configs.append((float(L_str), float(eta_str)))
    return configs


def run_gpa_with_eval(particles, y_particles, theta_data, y_data, n_coupled,
                      eval_offsets, y_test_values, problem, L, eta, K,
                      disc_steps, disc_lr, batch_size, gp_weight,
                      disc_hidden, disc_layers, formulation, activation,
                      device, label, disc=None, mmd_bandwidths=None):
    """Run GPA refinement with eval callbacks. Returns result dict."""
    eval_history = []

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
                mmds[y_val] = float(compute_mmd(samples, true_samples,
                                                bandwidth=mmd_bandwidths[y_val]))
        mean_d = float(np.mean(list(dists.values())))
        dists['mean'] = mean_d
        if mmds:
            mean_mmd = float(np.mean(list(mmds.values())))
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
        K=K,
        eta=eta,
        disc_steps=disc_steps,
        disc_lr=disc_lr,
        L=L,
        batch_size=batch_size,
        gp_weight=gp_weight,
        disc_hidden=disc_hidden,
        disc_layers=disc_layers,
        formulation=formulation,
        activation=activation,
        device=device,
        verbose=True,
        eval_callback=eval_callback,
        eval_every=50,
    )
    return result


def save_metrics(out_dir, tag, eval_history, before_particles, after_particles,
                 eval_offsets, y_test_values, problem, mmd_bandwidths):
    """Save metrics JSON and particles npz."""
    # Compute final metrics
    final_metrics = {'dist': {}, 'mmd': {}}
    for y_val in y_test_values:
        s, e = eval_offsets[y_val]
        before = before_particles[y_val]
        after = after_particles[s:e]
        final_metrics['dist'][str(y_val)] = {
            'before': float(problem.compute_distance(before, y_val).mean()),
            'after': float(problem.compute_distance(after, y_val).mean()),
        }
        if hasattr(problem, 'sample_true_posterior') and mmd_bandwidths is not None:
            bw = mmd_bandwidths[y_val]
            true_samp = problem.sample_true_posterior(y_val, 5000)
            final_metrics['mmd'][str(y_val)] = {
                'before': float(compute_mmd(before, true_samp, bandwidth=bw)),
                'after': float(compute_mmd(after, true_samp, bandwidth=bw)),
            }

    # Save trajectory from eval history
    trajectory = []
    for step, data in eval_history:
        # Convert keys to strings for JSON
        entry = {'step': step}
        entry['dist'] = {str(k): v for k, v in data['dist'].items()}
        entry['mmd'] = {str(k): v for k, v in data['mmd'].items()}
        trajectory.append(entry)

    result = {
        'final': final_metrics,
        'trajectory': trajectory,
    }

    with open(out_dir / f'metrics_{tag}.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Save particles
    particles_dict = {}
    for y_val in y_test_values:
        s, e = eval_offsets[y_val]
        particles_dict[f'before_y{y_val}'] = before_particles[y_val]
        particles_dict[f'after_y{y_val}'] = after_particles[s:e]
    np.savez_compressed(out_dir / f'particles_{tag}.npz', **particles_dict)


def make_plots(out_dir, tag, problem, y_test_values, eval_offsets,
               before_flow, after_flow, eval_flow,
               before_prior, after_prior, eval_prior,
               mmd_bandwidths, title_extra=''):
    """Generate trajectory + distribution plots."""
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
    bf = [problem.compute_distance(before_flow[y], y).mean() for y in y_test_values]
    bp = [problem.compute_distance(before_prior[y], y).mean() for y in y_test_values]
    af, ap = [], []
    for y_val in y_test_values:
        s, e = eval_offsets[y_val]
        af.append(problem.compute_distance(after_flow[s:e], y_val).mean())
        ap.append(problem.compute_distance(after_prior[s:e], y_val).mean())

    ax.bar(x_pos - 1.5*width, bf, width, label='Flow (before)', color='lightblue')
    ax.bar(x_pos - 0.5*width, af, width, label='Flow+GPA (after)', color='blue')
    ax.bar(x_pos + 0.5*width, bp, width, label='Prior (before)', color='lightsalmon')
    ax.bar(x_pos + 1.0*width, ap, width, label='GPA-only (after)', color='red')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'y={y}' for y in y_test_values])
    ax.set_ylabel('Mean distance')
    ax.set_title(f'{problem.name}: Before vs After')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(title_extra, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    fig.savefig(out_dir / f'e2e_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Distribution plots
    n_test = len(y_test_values)
    fig_dist, axes_dist = plt.subplots(5, n_test, figsize=(4 * n_test, 20))

    for i, y_val in enumerate(y_test_values):
        s, e = eval_offsets[y_val]
        panels = [
            (before_flow[y_val], 'Flow (before GPA)'),
            (after_flow[s:e], 'Flow + GPA'),
            (before_prior[y_val], 'Prior (before GPA)'),
            (after_prior[s:e], 'GPA-only'),
        ]

        for row, (samples, label) in enumerate(panels):
            ax = axes_dist[row, i]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5, color='steelblue', label=label, zorder=2)
            dist = problem.compute_distance(samples, y_val).mean()
            ax.set_title(f'y={y_val}: dist={dist:.3f}\n{label}', fontsize=10)
            ax.set_xlabel('θ₁')
            ax.set_ylabel('θ₂')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Row 4: θ₁ marginal histograms
        ax = axes_dist[4, i]
        ax.hist(before_flow[y_val][:, 0], bins=50, density=True, alpha=0.3,
                color='C0', label='Flow before')
        ax.hist(after_flow[s:e, 0], bins=50, density=True, alpha=0.3,
                color='C1', label='Flow+GPA')
        ax.hist(before_prior[y_val][:, 0], bins=50, density=True, alpha=0.3,
                color='C2', label='Prior before')
        ax.hist(after_prior[s:e, 0], bins=50, density=True, alpha=0.3,
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

    fig_dist.suptitle(title_extra, fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    fig_dist.savefig(out_dir / f'e2e_dist_{tag}.png', dpi=150, bbox_inches='tight')
    plt.close(fig_dist)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Device: {device}")

    problem = get_problem(args.problem)
    print(f"Problem: {problem.description}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpa_configs = parse_gpa_configs(args.gpa_configs)
    print(f"GPA configs: {gpa_configs}")

    # Save hyperparams
    hparams = vars(args).copy()
    hparams['gpa_configs_parsed'] = [(L, eta) for L, eta in gpa_configs]
    with open(out_dir / 'hparams.json', 'w') as f:
        json.dump(hparams, f, indent=2)

    # === Step 1: Sample joint data ONCE ===
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    theta_np, y_np = problem.sample_joint(args.n_train)
    theta_data = torch.FloatTensor(theta_np).to(device)
    y_data = torch.FloatTensor(y_np).to(device)
    if y_data.dim() == 1:
        y_data = y_data.unsqueeze(1)
    print(f"Joint data: {len(theta_data)} samples")

    theta_train_cpu = torch.FloatTensor(theta_np)
    y_train_cpu = torch.FloatTensor(y_np)
    if y_train_cpu.dim() == 1:
        y_train_cpu = y_train_cpu.unsqueeze(1)

    # === Step 2: Train or load flow ===
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

    flow_ckpt_path = out_dir / 'flow_checkpoint.pt'

    if args.load_flow:
        print(f"\nLoading flow from {args.load_flow}")
        ckpt = torch.load(args.load_flow, map_location=device)
        flow.vel_net.load_state_dict(ckpt['vel_net'])
        flow.disc.load_state_dict(ckpt['disc'])
    else:
        print(f"\n{'='*60}")
        print(f"TRAINING FLOW ({args.flow_iters} iters, λ={args.flow_lam})")
        print(f"{'='*60}")

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

        # Save flow checkpoint
        torch.save({
            'vel_net': flow.vel_net.state_dict(),
            'disc': flow.disc.state_dict(),
            'hparams': flow.hparams,
            'flow_lam': args.flow_lam,
            'flow_iters': args.flow_iters,
        }, flow_ckpt_path)
        print(f"Saved flow checkpoint: {flow_ckpt_path}")

    # === Step 3: Generate particles (once, reused across GPA configs) ===
    y_test_values = problem.default_y_test_values()

    torch.manual_seed(args.seed + 1000)
    flow.vel_net.eval()
    with torch.no_grad():
        z_flow = torch.randn(args.n_train, problem.theta_dim, device=device)
        traj = euler_integrate(flow.vel_net, z_flow, y_data, args.flow_n_steps)
        flow_train_particles = traj[-1]

    torch.manual_seed(args.seed + 2000)
    prior_train_particles = torch.randn(args.n_train, problem.theta_dim, device=device)

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

    # Precompute MMD bandwidths
    mmd_bandwidths = None
    if hasattr(problem, 'sample_true_posterior'):
        mmd_bandwidths = {}
        for y_val in y_test_values:
            true_samp = problem.sample_true_posterior(y_val, 5000)
            mmd_bandwidths[y_val] = compute_mmd_bandwidth(true_samp)
            print(f"  MMD bandwidth for y={y_val}: {mmd_bandwidths[y_val]:.4f}")

    # Save flow particles (before refinement)
    flow_particles_dict = {f'y{y_val}': all_particles_before_flow[y_val] for y_val in y_test_values}
    np.savez_compressed(out_dir / 'flow_particles_before.npz', **flow_particles_dict)

    # Summary collector for all configs
    summary = {}

    # === Step 4: Sweep over GPA configs ===
    for L, eta in gpa_configs:
        tag = f"{problem.name}_L{L}_eta{eta}"
        print(f"\n{'='*60}")
        print(f"GPA CONFIG: L={L}, η={eta}")
        print(f"{'='*60}")

        # Flow + GPA (warm-start disc)
        print(f"\n--- Flow + GPA (warm-start) ---")
        torch.manual_seed(args.seed + 5000)
        result_flow = run_gpa_with_eval(
            all_flow, all_y, theta_data, y_data, args.n_train,
            eval_offsets, y_test_values, problem, L, eta, args.K,
            args.disc_steps, args.disc_lr, args.batch_size, args.gp_weight,
            args.disc_hidden, args.disc_layers, args.formulation, args.activation,
            device, f"FLOW+GPA L={L}",
            disc=flow.disc, mmd_bandwidths=mmd_bandwidths)

        # GPA-only from prior (fresh disc)
        print(f"\n--- GPA-only (fresh disc) ---")
        torch.manual_seed(args.seed + 5000)
        result_prior = run_gpa_with_eval(
            all_prior, all_y, theta_data, y_data, args.n_train,
            eval_offsets, y_test_values, problem, L, eta, args.K,
            args.disc_steps, args.disc_lr, args.batch_size, args.gp_weight,
            args.disc_hidden, args.disc_layers, args.formulation, args.activation,
            device, f"GPA-ONLY L={L}",
            disc=None, mmd_bandwidths=mmd_bandwidths)

        refined_flow = result_flow['particles'].cpu().numpy()
        refined_prior = result_prior['particles'].cpu().numpy()

        # Save metrics and particles
        save_metrics(out_dir, f'flow_gpa_{tag}',
                     result_flow['history']['eval'],
                     all_particles_before_flow, refined_flow,
                     eval_offsets, y_test_values, problem, mmd_bandwidths)
        save_metrics(out_dir, f'gpa_only_{tag}',
                     result_prior['history']['eval'],
                     all_particles_before_prior, refined_prior,
                     eval_offsets, y_test_values, problem, mmd_bandwidths)

        # Make plots
        title = (f'{problem.name}: Flow(λ={args.flow_lam}, L={args.flow_lip_scale}) '
                 f'+ GPA(L={L}, η={eta}) vs GPA-only\n'
                 f'K={args.K}, ds={args.disc_steps} | {args.n_train} joint samples')
        make_plots(out_dir, tag, problem, y_test_values, eval_offsets,
                   all_particles_before_flow, refined_flow, result_flow['history']['eval'],
                   all_particles_before_prior, refined_prior, result_prior['history']['eval'],
                   mmd_bandwidths, title_extra=title)

        # Collect summary
        bf = np.mean([problem.compute_distance(all_particles_before_flow[y], y).mean()
                      for y in y_test_values])
        af = np.mean([problem.compute_distance(refined_flow[eval_offsets[y][0]:eval_offsets[y][1]], y).mean()
                      for y in y_test_values])
        bp = np.mean([problem.compute_distance(all_particles_before_prior[y], y).mean()
                      for y in y_test_values])
        ap = np.mean([problem.compute_distance(refined_prior[eval_offsets[y][0]:eval_offsets[y][1]], y).mean()
                      for y in y_test_values])

        entry = {
            'dist_flow_before': float(bf), 'dist_flow_gpa': float(af),
            'dist_prior_before': float(bp), 'dist_gpa_only': float(ap),
        }
        if mmd_bandwidths:
            mmd_af, mmd_ap = [], []
            for y_val in y_test_values:
                s, e = eval_offsets[y_val]
                bw = mmd_bandwidths[y_val]
                true_samp = problem.sample_true_posterior(y_val, 5000)
                mmd_af.append(float(compute_mmd(refined_flow[s:e], true_samp, bandwidth=bw)))
                mmd_ap.append(float(compute_mmd(refined_prior[s:e], true_samp, bandwidth=bw)))
            entry['mmd_flow_gpa'] = float(np.mean(mmd_af))
            entry['mmd_gpa_only'] = float(np.mean(mmd_ap))

        summary[f'L={L}_eta={eta}'] = entry
        print(f"\n  Summary L={L}, η={eta}:")
        print(f"    Flow before: {bf:.4f} | Flow+GPA: {af:.4f} | GPA-only: {ap:.4f}")

    # Save overall summary
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {out_dir / 'summary.json'}")

    # Print final comparison table
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY: {problem.name}, flow λ={args.flow_lam}")
    print(f"{'='*60}")
    print(f"{'Config':>20} {'Flow before':>12} {'Flow+GPA':>12} {'GPA-only':>12}", end='')
    has_mmd = any('mmd_flow_gpa' in v for v in summary.values())
    if has_mmd:
        print(f" {'MMD F+G':>12} {'MMD GPA':>12}", end='')
    print()
    for config, vals in summary.items():
        print(f"  {config:>18} {vals['dist_flow_before']:>12.4f} {vals['dist_flow_gpa']:>12.4f} "
              f"{vals['dist_gpa_only']:>12.4f}", end='')
        if has_mmd:
            print(f" {vals.get('mmd_flow_gpa', 0):>12.6f} {vals.get('mmd_gpa_only', 0):>12.6f}", end='')
        print()


if __name__ == '__main__':
    main()
