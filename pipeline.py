#!/usr/bin/env python
"""Unified pipeline for W1W2 flow + GPA posterior sampling.

Supports two data modes:
  --data data.npz        Load joint (theta, y) from a file
  --problem circle       Use a registered problem to generate joint data

Subcommands:
  train   Train a W1W2 flow on joint (theta, y) data
  sample  Generate conditional posterior samples for new y values
  refine  Refine samples with GPA
  plot    Pairwise corner plots for arbitrary dimensions

Examples:
  # Using a named problem
  python pipeline.py train --problem circle --n-train 5000 --output runs/circle
  python pipeline.py sample --model runs/circle/model.pt --y-values 1.0 2.0 --output runs/circle/samples.npz
  python pipeline.py refine --samples runs/circle/samples.npz --model runs/circle/model.pt --output runs/circle/refined.npz --K 500 --eta 0.005 --L 1000
  python pipeline.py plot --samples runs/circle/samples.npz --refined runs/circle/refined.npz --output runs/circle/

  # Using raw data
  python pipeline.py train --data my_data.npz --output runs/custom
  python pipeline.py sample --model runs/custom/model.pt --y-values 0.5 1.0 --output runs/custom/samples.npz
"""

import argparse
import json
import sys
import numpy as np
import torch
from pathlib import Path


def load_joint_data(args):
    """Load joint (theta, y) data from --data or --problem."""
    if args.data is not None:
        d = np.load(args.data)
        theta = d['theta']
        y = d['y']
        if y.ndim == 1:
            y = y[:, None]
        return torch.FloatTensor(theta), torch.FloatTensor(y)
    elif args.problem is not None:
        from problems import get_problem
        prob = get_problem(args.problem)
        n = getattr(args, 'n_train', 10000)
        theta_np, y_np = prob.sample_joint(n)
        y_t = torch.FloatTensor(y_np)
        if y_t.dim() == 1:
            y_t = y_t.unsqueeze(1)
        return torch.FloatTensor(theta_np), y_t
    else:
        raise ValueError("Must specify --data or --problem")


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def cmd_train(args):
    from flows import W1W2Flow

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    theta, y = load_joint_data(args)
    theta_dim, y_dim = theta.shape[1], y.shape[1]
    print(f"Data: {len(theta)} samples, theta_dim={theta_dim}, y_dim={y_dim}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    flow = W1W2Flow(
        theta_dim=theta_dim, y_dim=y_dim,
        vel_hidden=args.vel_hidden, vel_layers=args.vel_layers,
        disc_hidden=args.disc_hidden, disc_layers=args.disc_layers,
        lip_scale=args.lip_scale, gp_lambda=args.gp_lambda,
    )

    source = None
    if args.source_samples:
        src = np.load(args.source_samples)
        source = torch.from_numpy(src['theta']).float()
        print(f"Source samples: {source.shape[0]} from {args.source_samples}")

    history = flow.train(
        theta, y,
        n_iters=args.n_iters, batch_size=args.batch_size,
        lr=args.lr, lam=args.lam,
        n_steps=args.n_steps, disc_updates=args.disc_updates,
        checkpoint_dir=str(out) if args.checkpoint_every > 0 else None,
        checkpoint_every=args.checkpoint_every,
        verbose=True,
        source_samples=source,
    )

    flow.save(str(out / 'model.pt'))

    # Save hparams + training info
    info = {**flow.hparams, 'n_iters': args.n_iters, 'lr': args.lr,
            'lam': args.lam, 'n_steps': args.n_steps, 'n_train': len(theta)}
    if args.problem:
        info['problem'] = args.problem
    with open(out / 'hparams.json', 'w') as f:
        json.dump(info, f, indent=2)

    print(f"Done. Model saved to {out / 'model.pt'}")


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------

def cmd_sample(args):
    from flows import W1W2Flow

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    flow = W1W2Flow.from_checkpoint(args.model)
    y_dim = flow.y_dim
    theta_dim = flow.theta_dim
    print(f"Loaded model: theta_dim={theta_dim}, y_dim={y_dim}")

    # Parse y values
    raw = args.y_values
    if len(raw) % y_dim != 0:
        raise ValueError(f"Number of y-values ({len(raw)}) must be divisible by y_dim ({y_dim})")
    y_obs = np.array(raw).reshape(-1, y_dim)
    n_obs = len(y_obs)
    print(f"Sampling {args.n_samples} particles for {n_obs} observations")

    all_theta = []
    all_y = []
    for i in range(n_obs):
        y_t = torch.FloatTensor(y_obs[i])
        samples = flow.sample(y_t, args.n_samples, n_steps=args.n_steps)
        all_theta.append(samples.cpu().numpy())
        all_y.append(np.tile(y_obs[i], (args.n_samples, 1)))
        print(f"  y={y_obs[i]}: done")

    theta_out = np.concatenate(all_theta, axis=0)
    y_out = np.concatenate(all_y, axis=0)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, theta=theta_out, y=y_out,
             y_obs=y_obs, n_samples_per_obs=args.n_samples)
    print(f"Saved: {out} ({theta_out.shape[0]} samples)")


# ---------------------------------------------------------------------------
# Refine
# ---------------------------------------------------------------------------

def cmd_refine(args):
    from flows import W1W2Flow
    from refinement.gpa import gpa_refine
    from utils.integrators import euler_integrate

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load samples to refine
    s = np.load(args.samples)
    particles = torch.FloatTensor(s['theta']).to(device)
    y_particles = torch.FloatTensor(s['y']).to(device)
    y_obs = s['y_obs']
    n_samp = int(s['n_samples_per_obs'])
    print(f"Loaded {len(particles)} particles to refine")

    # Load training data
    theta_data, y_data = load_joint_data(args)
    theta_data = theta_data.to(device)
    y_data = y_data.to(device)
    n_data = len(theta_data)
    print(f"Loaded {n_data} training samples")

    # Generate coupled particles at training y values
    if args.model is not None:
        flow = W1W2Flow.from_checkpoint(args.model, device=device)
        flow.vel_net.eval()
        print("Generating coupled particles from flow at training y values...")
        with torch.no_grad():
            z = torch.randn(n_data, flow.theta_dim, device=device)
            from utils.integrators import euler_integrate
            traj = euler_integrate(flow.vel_net, z, y_data, args.n_steps)
            train_particles = traj[-1]
    else:
        # No model — use noisy copies of training data as coupled particles
        train_particles = theta_data + 0.1 * torch.randn_like(theta_data)

    particles = torch.cat([train_particles, particles], dim=0)
    y_particles = torch.cat([y_data, y_particles], dim=0)
    n_coupled = n_data
    print(f"  Coupled {n_coupled} training particles + {len(particles) - n_coupled} test particles")

    # Warm-start discriminator from flow if available and sizes match
    flow_disc = None
    if args.model is not None:
        fd = flow.disc
        # Check if flow disc matches requested size
        flow_h = fd.net[0].weight.shape[0]
        flow_l = sum(1 for m in fd.net if isinstance(m, torch.nn.Linear))
        if flow_h == args.disc_hidden and flow_l == args.disc_layers:
            flow_disc = fd
            print(f"  Warm-starting disc from flow ({flow_h}×{flow_l})")
        else:
            print(f"  Disc size mismatch: flow={flow_h}×{flow_l}, "
                  f"requested={args.disc_hidden}×{args.disc_layers}. Cold-starting.")

    result = gpa_refine(
        particles=particles, y_particles=y_particles,
        theta_data=theta_data, y_data=y_data,
        n_coupled=n_coupled,
        disc=flow_disc,
        K=args.K, eta=args.eta, L=args.L,
        disc_steps=args.disc_steps, disc_lr=args.disc_lr,
        batch_size=args.batch_size, gp_weight=args.gp_weight,
        disc_hidden=args.disc_hidden, disc_layers=args.disc_layers,
        formulation=args.formulation,
        normalize_grad=args.normalize_grad,
        disc_optimizer=args.disc_optimizer,
        disc_reset_every=args.disc_reset_every,
        device=device, verbose=True,
    )

    # Extract only the test particles (after the coupled training ones)
    refined = result['particles'][n_coupled:].cpu().numpy()
    y_out = y_particles[n_coupled:].cpu().numpy()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, theta=refined, y=y_out,
             y_obs=y_obs, n_samples_per_obs=n_samp)
    print(f"Saved: {out} ({refined.shape[0]} refined particles)")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_corner_plot(samples_dict, theta_dim, labels=None, max_points=2000,
                     title=None):
    """Create a corner/triangle plot for arbitrary dimensions.

    Args:
        samples_dict: dict of {label: (samples, color)} where samples is (N, theta_dim).
        theta_dim: dimension of theta.
        labels: list of axis labels, length theta_dim.
        max_points: max points per scatter.
        title: figure title.

    Returns:
        matplotlib figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = theta_dim
    if labels is None:
        labels = [f'θ_{i+1}' if d <= 10 else f'{i+1}' for i in range(d)]

    sz = max(2.2 * d, 6)
    fig, axes = plt.subplots(d, d, figsize=(sz, sz))
    if d == 1:
        axes = np.array([[axes]])

    ms = max(0.5, 4 - d * 0.2)
    alpha = max(0.05, 0.35 - d * 0.015)

    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i < j:
                # Upper triangle — hide
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal — 1D histogram
                for label, (samp, color) in samples_dict.items():
                    s = samp[:max_points] if len(samp) > max_points else samp
                    ax.hist(s[:, i], bins=40, density=True, alpha=0.5,
                            color=color, label=label)
                if i == 0:
                    ax.legend(fontsize=max(5, 8 - d * 0.3), loc='upper right')
            else:
                # Lower triangle — 2D scatter
                for label, (samp, color) in samples_dict.items():
                    idx = np.random.choice(len(samp), min(max_points, len(samp)),
                                           replace=False)
                    ax.scatter(samp[idx, j], samp[idx, i], s=ms, alpha=alpha,
                               c=color, rasterized=True)

            # Axis labels on edges only
            if i == d - 1:
                ax.set_xlabel(labels[j], fontsize=9)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i], fontsize=9)
            elif j > 0:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.94)
    return fig


def cmd_plot(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    s = np.load(args.samples)
    theta = s['theta']
    y_obs = s['y_obs']
    n_samp = int(s['n_samples_per_obs'])
    theta_dim = theta.shape[1]
    n_obs = len(y_obs)

    refined = None
    if args.refined:
        r = np.load(args.refined)
        refined = r['theta']

    true_data = None
    if args.true_data:
        t = np.load(args.true_data)
        true_data = t['theta']

    # Also try loading from --problem for true posterior
    true_from_problem = None
    if args.problem:
        from problems import get_problem
        true_from_problem = get_problem(args.problem)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    labels = args.labels if args.labels else None

    for i in range(n_obs):
        start = i * n_samp
        end = start + n_samp
        samp_i = theta[start:end]

        samples_dict = {}
        samples_dict['Flow'] = (samp_i, '#4a90d9')

        if refined is not None:
            ref_i = refined[start:end]
            samples_dict['GPA'] = (ref_i, '#d73027')

        if true_from_problem is not None:
            true_samp = true_from_problem.sample_true_posterior(
                float(y_obs[i].squeeze()), 5000)
            samples_dict['True'] = (true_samp, '#888888')
        elif true_data is not None:
            samples_dict['True'] = (true_data, '#888888')

        y_vals = y_obs[i]
        if y_vals.size <= 5:
            y_str = '_'.join(f'{v:.3g}' for v in y_vals)
            title = f'y = [{y_str}]' if y_vals.size > 1 else f'y = {y_vals.squeeze()}'
        else:
            y_str = f'obs{i}'
            title = f'y observation {i} ({y_vals.size}D)'

        fig = make_corner_plot(
            samples_dict, theta_dim,
            labels=labels, max_points=args.max_points,
            title=title,
        )
        fname = out / f'corner_{y_str}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        fig.savefig(fname.with_suffix('.pdf'), bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {fname}")

    print(f"Done. {n_obs} corner plots in {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='W1W2 Flow + GPA Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── train ──
    tp = sub.add_parser('train', help='Train a W1W2 flow')
    tp.add_argument('--seed', type=int, default=42)
    tp.add_argument('--data', type=str, default=None,
                    help='Path to .npz with keys "theta" and "y"')
    tp.add_argument('--problem', type=str, default=None,
                    help='Named problem (circle, bimodal_quadratic, etc)')
    tp.add_argument('--n-train', type=int, default=10000)
    tp.add_argument('--output', type=str, required=True)
    tp.add_argument('--n-iters', type=int, default=5000)
    tp.add_argument('--batch-size', type=int, default=256)
    tp.add_argument('--lr', type=float, default=1e-3)
    tp.add_argument('--lam', type=float, default=0.01)
    tp.add_argument('--n-steps', type=int, default=40)
    tp.add_argument('--disc-updates', type=int, default=5)
    tp.add_argument('--vel-hidden', type=int, default=256)
    tp.add_argument('--vel-layers', type=int, default=4)
    tp.add_argument('--disc-hidden', type=int, default=128)
    tp.add_argument('--disc-layers', type=int, default=3)
    tp.add_argument('--lip-scale', type=float, default=10.0)
    tp.add_argument('--gp-lambda', type=float, default=0.0)
    tp.add_argument('--checkpoint-every', type=int, default=0)
    tp.add_argument('--source-samples', type=str, default=None,
                    help='Path to .npz with "theta" key to use as source distribution '
                         'instead of N(0,I). For multi-step W1W2.')

    # ── sample ──
    sp = sub.add_parser('sample', help='Generate conditional samples')
    sp.add_argument('--seed', type=int, default=42)
    sp.add_argument('--model', type=str, required=True, help='Path to model.pt')
    sp.add_argument('--y-values', type=float, nargs='+', required=True,
                    help='Observation values (flat list, grouped by y_dim)')
    sp.add_argument('--n-samples', type=int, default=2000)
    sp.add_argument('--n-steps', type=int, default=40)
    sp.add_argument('--output', type=str, required=True)

    # ── refine ──
    rp = sub.add_parser('refine', help='Refine samples with GPA')
    rp.add_argument('--seed', type=int, default=42)
    rp.add_argument('--samples', type=str, required=True, help='Path to samples.npz')
    rp.add_argument('--data', type=str, default=None, help='Path to training data .npz')
    rp.add_argument('--problem', type=str, default=None, help='Named problem')
    rp.add_argument('--n-train', type=int, default=10000)
    rp.add_argument('--model', type=str, default=None,
                    help='Path to model.pt (for coupled flow particles)')
    rp.add_argument('--n-steps', type=int, default=40)
    rp.add_argument('--output', type=str, required=True)
    rp.add_argument('--K', type=int, default=500)
    rp.add_argument('--eta', type=float, default=0.005)
    rp.add_argument('--L', type=float, default=1000.0)
    rp.add_argument('--disc-steps', type=int, default=3)
    rp.add_argument('--disc-lr', type=float, default=0.001)
    rp.add_argument('--batch-size', type=int, default=256)
    rp.add_argument('--gp-weight', type=float, default=0.0)
    rp.add_argument('--disc-hidden', type=int, default=32)
    rp.add_argument('--disc-layers', type=int, default=4)
    rp.add_argument('--formulation', type=str, default='LT',
                    choices=['LT', 'LT_nu', 'DV'])
    rp.add_argument('--normalize-grad', action='store_true',
                    help='Normalize particle gradients to unit length')
    rp.add_argument('--disc-optimizer', type=str, default='adam',
                    choices=['adam', 'sgd'],
                    help='Optimizer for discriminator')
    rp.add_argument('--disc-reset-every', type=int, default=0,
                    help='Reset disc optimizer state every N steps (0=never)')

    # ── plot ──
    pp = sub.add_parser('plot', help='Corner plots')
    pp.add_argument('--samples', type=str, required=True)
    pp.add_argument('--refined', type=str, default=None)
    pp.add_argument('--true-data', type=str, default=None,
                    help='Path to .npz with true posterior samples')
    pp.add_argument('--problem', type=str, default=None,
                    help='Named problem (for true posterior overlay)')
    pp.add_argument('--output', type=str, required=True)
    pp.add_argument('--labels', type=str, nargs='+', default=None)
    pp.add_argument('--max-points', type=int, default=2000)

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'sample':
        cmd_sample(args)
    elif args.command == 'refine':
        cmd_refine(args)
    elif args.command == 'plot':
        cmd_plot(args)


if __name__ == '__main__':
    main()
