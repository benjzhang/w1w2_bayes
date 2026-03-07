#!/usr/bin/env python
"""Visualize the evolution of Flow+GPA vs GPA-only particles over iterations.

Produces a grid: two rows per test y value (Flow+GPA, GPA-only),
columns = Prior, Flow/Prior, GPA snapshots at matched steps.
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
    parser = argparse.ArgumentParser(description="GPA evolution visualization")
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEMS.keys()))
    # Flow
    parser.add_argument('--load-flow', type=str, required=True,
                        help='Path to saved flow checkpoint')
    parser.add_argument('--flow-vel-hidden', type=int, default=128)
    parser.add_argument('--flow-vel-layers', type=int, default=3)
    parser.add_argument('--flow-n-steps', type=int, default=10)
    # GPA
    parser.add_argument('--L', type=float, default=1000.0)
    parser.add_argument('--eta', type=float, default=0.005)
    parser.add_argument('--K', type=int, default=500)
    parser.add_argument('--disc-steps', type=int, default=10)
    parser.add_argument('--disc-hidden', type=int, default=32)
    parser.add_argument('--disc-layers', type=int, default=4)
    parser.add_argument('--gp-weight', type=float, default=0.0)
    parser.add_argument('--flow-lam', type=float, default=0.25)
    parser.add_argument('--flow-lip-scale', type=float, default=10.0)
    parser.add_argument('--flow-gp-lambda', type=float, default=1.0)
    # Vis
    parser.add_argument('--snapshot-every', type=int, default=0,
                        help='Save snapshots every N steps (0 = auto pick ~8 frames)')
    parser.add_argument('--n-frames', type=int, default=8,
                        help='Number of frames to show (used when snapshot-every=0)')
    # Shared
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-eval', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def subsample_snapshots(snapshots, n_frames):
    """Pick evenly spaced subset, always including first and last."""
    if len(snapshots) <= n_frames:
        return snapshots
    indices = np.linspace(0, len(snapshots) - 1, n_frames, dtype=int)
    indices = sorted(set(indices))
    return [snapshots[i] for i in indices]


def build_gpa_particles(eval_particles_dict, train_particles, y_data,
                        y_test_values, n_eval, y_dim, device):
    """Build concatenated particle tensor and offset map."""
    n_train = len(train_particles)
    parts = [train_particles]
    y_parts = [y_data]
    offsets = {}
    offset = n_train
    for y_val in y_test_values:
        p = eval_particles_dict[y_val]
        parts.append(p)
        y_parts.append(torch.full((len(p), y_dim), y_val, device=device))
        offsets[y_val] = (offset, offset + len(p))
        offset += len(p)
    return torch.cat(parts, dim=0), torch.cat(y_parts, dim=0), offsets


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    problem = get_problem(args.problem)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = problem.default_hyperparams()
    use_quad = defaults.get('use_quadratic_features', False)

    # Load flow
    w1w2 = W1W2Flow(
        theta_dim=problem.theta_dim, y_dim=problem.y_dim,
        vel_hidden=args.flow_vel_hidden, vel_layers=args.flow_vel_layers,
        disc_hidden=args.disc_hidden, disc_layers=args.disc_layers,
        lip_scale=args.flow_lip_scale,
        use_quadratic_features=use_quad,
        gp_lambda=args.flow_gp_lambda,
        device=device,
    )
    ckpt = torch.load(args.load_flow, map_location=device)
    w1w2.vel_net.load_state_dict(ckpt['vel_net'])
    w1w2.disc.load_state_dict(ckpt['disc'])
    print(f"Loaded flow from {args.load_flow}")

    # Sample joint data
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    theta_np, y_np = problem.sample_joint(args.n_train)
    theta_data = torch.FloatTensor(theta_np).to(device)
    y_data = torch.FloatTensor(y_np).to(device)
    if y_data.dim() == 1:
        y_data = y_data.unsqueeze(1)

    y_test_values = problem.default_y_test_values()

    # Determine snapshot frequency (use denser snapshots for GIF)
    gif_snap_every = max(1, args.K // 60)  # ~60 GPA frames for GIF
    if args.snapshot_every > 0:
        snap_every = args.snapshot_every
    else:
        snap_every = gif_snap_every  # save densely, subsample for static plots

    # === Generate flow particles and initial noise ===
    torch.manual_seed(args.seed + 1000)
    w1w2.vel_net.eval()
    with torch.no_grad():
        z_train = torch.randn(args.n_train, problem.theta_dim, device=device)
        traj = euler_integrate(w1w2.vel_net, z_train, y_data, args.flow_n_steps)
        flow_train_particles = traj[-1]

    flow_eval = {}
    flow_trajectories = {}  # full ODE trajectory per y_val
    prior_eval = {}
    initial_noise = {}
    for y_val in y_test_values:
        torch.manual_seed(args.seed + 3000 + int(y_val * 100))
        z = torch.randn(args.n_eval, problem.theta_dim, device=device)
        initial_noise[y_val] = z.cpu().numpy()
        y_cond = torch.full((args.n_eval, problem.y_dim), y_val, device=device)
        with torch.no_grad():
            traj_eval = euler_integrate(w1w2.vel_net, z, y_cond, args.flow_n_steps)
        flow_eval[y_val] = traj_eval[-1]
        flow_trajectories[y_val] = [t.cpu().numpy() for t in traj_eval]
        # For GPA-only: start from the same noise
        prior_eval[y_val] = z.clone()

    # === Run Flow+GPA ===
    print(f"\n{'='*60}")
    print(f"Running Flow+GPA: K={args.K}, L={args.L}, eta={args.eta}")
    print(f"{'='*60}")
    all_p_flow, all_y_flow, offsets_flow = build_gpa_particles(
        flow_eval, flow_train_particles, y_data,
        y_test_values, args.n_eval, problem.y_dim, device)

    torch.manual_seed(args.seed + 5000)
    result_flow = gpa_refine(
        particles=all_p_flow.clone(),
        y_particles=all_y_flow.clone(),
        theta_data=theta_data, y_data=y_data,
        n_coupled=args.n_train, disc=w1w2.disc,
        K=args.K, eta=args.eta,
        disc_steps=args.disc_steps, disc_lr=0.001,
        L=args.L, batch_size=args.batch_size,
        disc_hidden=args.disc_hidden, disc_layers=args.disc_layers,
        gp_weight=args.gp_weight,
        device=device, verbose=True,
        snapshot_every=snap_every,
    )
    snaps_flow = subsample_snapshots(result_flow['snapshots'], args.n_frames)

    # === Run GPA-only (from prior) ===
    print(f"\n{'='*60}")
    print(f"Running GPA-only: K={args.K}, L={args.L}, eta={args.eta}")
    print(f"{'='*60}")

    # For GPA-only, use prior noise as initial particles for train too
    torch.manual_seed(args.seed + 1000)
    prior_train = torch.randn(args.n_train, problem.theta_dim, device=device)

    all_p_prior, all_y_prior, offsets_prior = build_gpa_particles(
        prior_eval, prior_train, y_data,
        y_test_values, args.n_eval, problem.y_dim, device)

    torch.manual_seed(args.seed + 5000)
    result_prior = gpa_refine(
        particles=all_p_prior.clone(),
        y_particles=all_y_prior.clone(),
        theta_data=theta_data, y_data=y_data,
        n_coupled=args.n_train, disc=None,
        K=args.K, eta=args.eta,
        disc_steps=args.disc_steps, disc_lr=0.001,
        L=args.L, batch_size=args.batch_size,
        disc_hidden=args.disc_hidden, disc_layers=args.disc_layers,
        gp_weight=args.gp_weight,
        device=device, verbose=True,
        snapshot_every=snap_every,
    )
    snaps_prior = subsample_snapshots(result_prior['snapshots'], args.n_frames)

    # === Get matched step indices ===
    flow_steps = [s[0] for s in snaps_flow]
    prior_steps = [s[0] for s in snaps_prior]
    print(f"Flow+GPA steps: {flow_steps}")
    print(f"GPA-only steps: {prior_steps}")

    # === Plotting ===
    n_snaps = len(snaps_flow)
    n_cols = 1 + n_snaps  # prior/noise column + snapshot columns
    n_test = len(y_test_values)
    n_rows = 2 * n_test  # 2 rows per y: Flow+GPA and GPA-only

    # Compute axis limits from flow snapshots (GPA-only may spread wider)
    all_xlim = [None, None]
    all_ylim = [None, None]
    for snapshots, offsets in [(snaps_flow, offsets_flow), (snaps_prior, offsets_prior)]:
        for _, ps in snapshots:
            for y_val in y_test_values:
                s, e = offsets[y_val]
                samp = ps.numpy()[s:e]
                xmin, xmax = samp[:, 0].min(), samp[:, 0].max()
                ymin, ymax = samp[:, 1].min(), samp[:, 1].max()
                if all_xlim[0] is None or xmin < all_xlim[0]:
                    all_xlim[0] = xmin
                if all_xlim[1] is None or xmax > all_xlim[1]:
                    all_xlim[1] = xmax
                if all_ylim[0] is None or ymin < all_ylim[0]:
                    all_ylim[0] = ymin
                if all_ylim[1] is None or ymax > all_ylim[1]:
                    all_ylim[1] = ymax
    pad_x = (all_xlim[1] - all_xlim[0]) * 0.1
    pad_y = (all_ylim[1] - all_ylim[0]) * 0.1
    all_xlim = [all_xlim[0] - pad_x, all_xlim[1] + pad_x]
    all_ylim = [all_ylim[0] - pad_y, all_ylim[1] + pad_y]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
                             squeeze=False)

    for yi, y_val in enumerate(y_test_values):
        row_flow = 2 * yi
        row_prior = 2 * yi + 1

        # Column 0: initial distribution
        for row, label, noise in [
            (row_flow, 'Flow+GPA', initial_noise[y_val]),
            (row_prior, 'GPA-only', initial_noise[y_val]),
        ]:
            ax = axes[row, 0]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            if problem.theta_dim == 2:
                ax.scatter(noise[:, 0], noise[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
            ax.set_title('Prior\nz ~ N(0,I)', fontsize=9)
            ax.set_ylabel(f'{label}\ny={y_val}', fontsize=9)
            ax.grid(True, alpha=0.3)

        # Snapshot columns
        for col_offset in range(n_snaps):
            col = col_offset + 1

            # Flow+GPA row
            step_f, ps_f = snaps_flow[col_offset]
            ax = axes[row_flow, col]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            s, e = offsets_flow[y_val]
            samples = ps_f.numpy()[s:e]
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
            dist = float(problem.compute_distance(samples, y_val).mean())
            lbl = 'Flow' if step_f == 0 else f'GPA k={step_f}'
            ax.set_title(f'{lbl}\ndist={dist:.4f}', fontsize=9)
            ax.set_xlim(all_xlim)
            ax.set_ylim(all_ylim)
            ax.grid(True, alpha=0.3)

            # GPA-only row
            step_p, ps_p = snaps_prior[col_offset]
            ax = axes[row_prior, col]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            s, e = offsets_prior[y_val]
            samples = ps_p.numpy()[s:e]
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
            dist = float(problem.compute_distance(samples, y_val).mean())
            lbl = 'Prior' if step_p == 0 else f'GPA k={step_p}'
            ax.set_title(f'{lbl}\ndist={dist:.4f}', fontsize=9)
            ax.set_xlim(all_xlim)
            ax.set_ylim(all_ylim)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f'{problem.name}: Flow+GPA vs GPA-only (L={args.L}, η={args.eta}, K={args.K})',
                 fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    save_path = out_dir / f'gpa_evolution_{problem.name}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # === Separate plots for Flow+GPA and GPA-only ===
    for run_name, snaps, offsets, init_label in [
        ('flow_gpa', snaps_flow, offsets_flow, 'Flow'),
        ('gpa_only', snaps_prior, offsets_prior, 'Prior'),
    ]:
        n_s = len(snaps)
        nc = 1 + n_s
        fig2, axes2 = plt.subplots(n_test, nc, figsize=(3.2 * nc, 3.2 * n_test),
                                   squeeze=False)
        for row, y_val in enumerate(y_test_values):
            # Column 0: prior noise
            ax = axes2[row, 0]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            z = initial_noise[y_val]
            if problem.theta_dim == 2:
                ax.scatter(z[:, 0], z[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
            ax.set_title('Prior\nz ~ N(0,I)', fontsize=9)
            ax.set_ylabel(f'y={y_val}', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Snapshot columns
            for ci, (step, ps) in enumerate(snaps):
                ax = axes2[row, ci + 1]
                problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
                s, e = offsets[y_val]
                samples = ps.numpy()[s:e]
                if problem.theta_dim == 2:
                    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5,
                               color='steelblue', zorder=2)
                dist = float(problem.compute_distance(samples, y_val).mean())
                lbl = init_label if step == 0 else f'GPA k={step}'
                ax.set_title(f'{lbl}\ndist={dist:.4f}', fontsize=9)
                ax.set_xlim(all_xlim)
                ax.set_ylim(all_ylim)
                ax.grid(True, alpha=0.3)

        title = 'Flow+GPA' if run_name == 'flow_gpa' else 'GPA-only'
        fig2.suptitle(f'{problem.name}: {title} Evolution (L={args.L}, η={args.eta}, K={args.K})',
                      fontsize=13)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        sp = out_dir / f'{run_name}_evolution_{problem.name}.png'
        fig2.savefig(sp, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {sp}")


    # === Flow evolution plot (ODE trajectory over time) ===
    n_flow_steps = len(flow_trajectories[y_test_values[0]])  # n_steps + 1
    # Pick ~8 evenly spaced time indices
    n_flow_frames = min(args.n_frames, n_flow_steps)
    flow_indices = np.linspace(0, n_flow_steps - 1, n_flow_frames, dtype=int)
    flow_indices = sorted(set(flow_indices))
    dt_flow = 1.0 / (n_flow_steps - 1)

    n_fc = len(flow_indices)
    fig3, axes3 = plt.subplots(n_test, n_fc, figsize=(3.2 * n_fc, 3.2 * n_test),
                               squeeze=False)
    for row, y_val in enumerate(y_test_values):
        traj_y = flow_trajectories[y_val]
        for ci, fi in enumerate(flow_indices):
            ax = axes3[row, ci]
            problem.plot_true_posterior(ax, y_val, alpha=0.7, zorder=1)
            samples = traj_y[fi]
            if problem.theta_dim == 2:
                ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
            t_val = fi * dt_flow
            dist = float(problem.compute_distance(samples, y_val).mean())
            ax.set_title(f't={t_val:.2f}\ndist={dist:.4f}', fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'y={y_val}', fontsize=10)
            ax.grid(True, alpha=0.3)

    fig3.suptitle(f'{problem.name}: Flow Evolution (λ={args.flow_lam}, {args.flow_n_steps} steps)',
                  fontsize=13)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    sp3 = out_dir / f'flow_evolution_{problem.name}.png'
    fig3.savefig(sp3, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sp3}")


    # === GIF: Flow + GPA evolution for one y value ===
    from matplotlib.animation import PillowWriter
    import matplotlib.animation as animation

    gif_y = y_test_values[len(y_test_values) // 2]  # pick middle y value
    print(f"\nGenerating GIF for y={gif_y}...")

    # Build frame list: flow trajectory frames + GPA snapshot frames
    gif_frames = []

    # Flow frames (all time steps)
    traj_y = flow_trajectories[gif_y]
    n_flow = len(traj_y)
    dt_f = 1.0 / (n_flow - 1)
    for fi in range(n_flow):
        t_val = fi * dt_f
        samples = traj_y[fi]
        dist = float(problem.compute_distance(samples, gif_y).mean())
        gif_frames.append(('flow', t_val, samples, dist))

    # GPA frames (all saved snapshots, not subsampled)
    all_snaps_flow = result_flow['snapshots']
    s_off, e_off = offsets_flow[gif_y]
    for step, ps in all_snaps_flow:
        if step == 0:
            continue  # already have flow t=1.0 as last flow frame
        samples = ps.numpy()[s_off:e_off]
        dist = float(problem.compute_distance(samples, gif_y).mean())
        gif_frames.append(('gpa', step, samples, dist))

    # Compute fixed axis limits across all frames
    gif_xmin, gif_xmax = np.inf, -np.inf
    gif_ymin, gif_ymax = np.inf, -np.inf
    for _, _, samples, _ in gif_frames:
        gif_xmin = min(gif_xmin, samples[:, 0].min())
        gif_xmax = max(gif_xmax, samples[:, 0].max())
        gif_ymin = min(gif_ymin, samples[:, 1].min())
        gif_ymax = max(gif_ymax, samples[:, 1].max())
    gpad_x = (gif_xmax - gif_xmin) * 0.1
    gpad_y = (gif_ymax - gif_ymin) * 0.1
    gif_xlim = [gif_xmin - gpad_x, gif_xmax + gpad_x]
    gif_ylim = [gif_ymin - gpad_y, gif_ymax + gpad_y]

    # Create animation
    fig_gif, ax_gif = plt.subplots(1, 1, figsize=(5, 5))

    def animate(i):
        ax_gif.clear()
        problem.plot_true_posterior(ax_gif, gif_y, alpha=0.7, zorder=1)
        phase, val, samples, dist = gif_frames[i]
        if problem.theta_dim == 2:
            ax_gif.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5,
                           color='steelblue', zorder=2)
        if phase == 'flow':
            ax_gif.set_title(f'Flow t={val:.2f} | dist={dist:.4f}', fontsize=11)
        else:
            ax_gif.set_title(f'GPA k={int(val)} | dist={dist:.4f}', fontsize=11)
        ax_gif.set_xlim(gif_xlim)
        ax_gif.set_ylim(gif_ylim)
        ax_gif.set_xlabel('θ₁')
        ax_gif.set_ylabel('θ₂')
        ax_gif.grid(True, alpha=0.3)
        return []

    anim = animation.FuncAnimation(fig_gif, animate, frames=len(gif_frames),
                                   interval=150, blit=True)
    gif_path = out_dir / f'flow_gpa_evolution_y{gif_y}_{problem.name}.gif'
    anim.save(str(gif_path), writer=PillowWriter(fps=8))
    plt.close()
    print(f"Saved GIF: {gif_path}")


if __name__ == '__main__':
    main()
