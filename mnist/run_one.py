#!/usr/bin/env python
"""Run a single flow training + eval, or GPA refinement.

Used by SLURM array jobs to parallelize the sweep.

Usage:
    # Train a flow
    python -m mnist.run_one flow --name lam0.1_lip10 --lam 0.1 --lip-scale 10 \
        --arch mlp --hidden 512 --n-layers 4 --n-iters 15000

    # Run GPA on a trained flow
    python -m mnist.run_one gpa --checkpoint results/mnist/sweep/best/model.pt \
        --name gpa_eta0.01_L10 --eta 0.01 --L 10 --K 300
"""

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .data import MNISTInpainting
from .networks import build_networks, VelocityMLP, VelocityCNN, DiscriminatorMLP
from .evaluate import compute_metrics, plot_completions, plot_flow_vs_gpa


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def euler_integrate(vel_net, z0, y, n_steps=10):
    dt = 1.0 / n_steps
    theta = z0
    traj = [theta]
    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        theta = theta + dt * v
        traj.append(theta)
    return traj


def _spectral_norm_projection(layer, target_norm):
    with torch.no_grad():
        W = layer.weight
        if W.dim() > 2:
            W_2d = W.reshape(W.shape[0], -1)
        else:
            W_2d = W
        sigma = torch.linalg.norm(W_2d, ord=2)
        if sigma > 1e-6:
            W.mul_(target_norm / sigma)


def _gradient_penalty(disc, real, fake, y, L):
    """Gradient penalty matching reference: lambda * E[max(0, ||grad||^2 - L^2)]."""
    alpha = torch.rand(real.shape[0], 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = disc(interp, y)
    grads = torch.autograd.grad(
        d_interp.sum(), interp, create_graph=True
    )[0]
    grad_sq = (grads ** 2).sum(dim=1)
    penalty = torch.nn.functional.relu(grad_sq - L ** 2).mean()
    return penalty


def _project_disc_weights(disc, L):
    layers = []
    if hasattr(disc, 'convs'):
        layers += [m for m in disc.convs if isinstance(m, (nn.Linear, nn.Conv2d))]
    if hasattr(disc, 'net'):
        layers += [m for m in disc.net if isinstance(m, (nn.Linear, nn.Conv2d))]
    D = len(layers)
    if D == 0:
        return
    per_layer_norm = L ** (1.0 / D)
    for layer in layers:
        _spectral_norm_projection(layer, per_layer_norm)


def load_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    hp = state['hparams']
    arch = hp.get('arch_type', 'mlp')
    if arch == 'cnn':
        vel_net = VelocityCNN(
            ch=hp.get('cnn_ch', 8),
            fc_hidden=hp.get('hidden', 512),
        )
    else:
        vel_net = VelocityMLP(
            theta_dim=hp['theta_dim'], y_dim=hp['y_dim'],
            hidden=hp.get('hidden', 512), n_layers=hp.get('n_layers', 4),
        )
    vel_net.load_state_dict(state['vel_net_state'])
    vel_net = vel_net.to(device)
    vel_net.eval()
    return vel_net, hp


# ---------------------------------------------------------------------------
# Flow training
# ---------------------------------------------------------------------------

def run_flow(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading MNIST data...")
    train_data = MNISTInpainting(train=True, data_root='./data')
    test_data = MNISTInpainting(train=False, data_root='./data')
    theta_train, y_train = train_data.get_tensors()
    theta_train = theta_train.to(device)
    y_train = y_train.to(device)
    y_test, theta_true = test_data.get_test_images(n=args.n_test)
    theta_dim = theta_train.shape[1]
    y_dim = y_train.shape[1]
    n_data = len(theta_train)

    print(f"Training: {args.name}")
    print(f"  arch={args.arch}, hidden={args.hidden}, "
          f"layers={args.n_layers}, cnn_ch={args.cnn_channels}")
    lr_disc = args.lr_disc if args.lr_disc is not None else args.lr
    print(f"  lam={args.lam}, lip={args.lip_scale}, lr_gen={args.lr}, "
          f"lr_disc={lr_disc}, iters={args.n_iters}, steps={args.n_steps}")

    vel_net, disc = build_networks(
        args.arch, theta_dim, y_dim,
        hidden=args.hidden, n_layers=args.n_layers,
        cnn_channels=args.cnn_channels,
    )
    vel_net = vel_net.to(device)
    disc = disc.to(device)

    n_vel = sum(p.numel() for p in vel_net.parameters())
    n_disc = sum(p.numel() for p in disc.parameters())
    print(f"  Params: vel={n_vel:,}, disc={n_disc:,}")

    opt_vel = optim.Adam(vel_net.parameters(), lr=args.lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr_disc)
    use_gp = args.lip_mode == 'gp'
    if not use_gp:
        _project_disc_weights(disc, args.lip_scale)

    def compute_L_dual(phi_gen, phi_real):
        """Compute the dual objective depending on f-star choice."""
        if args.fstar == 'reverse_kl':
            # Reference: D_loss_1 = mean(fake), D_loss_2 = log(mean(exp(real)))
            return phi_gen.mean() - torch.logsumexp(phi_real.squeeze(), dim=0) + np.log(phi_real.shape[0])
        else:
            # KL conjugate: f*(t) = exp(t-1)
            return phi_gen.mean() - torch.exp(phi_real - 1).mean()

    history = {'L_dual': [], 'KE': [], 'iters': []}
    t_start = time.time()

    for it in range(1, args.n_iters + 1):
        idx = torch.randint(0, n_data, (args.batch_size,))
        theta_batch = theta_train[idx]
        y_batch = y_train[idx]
        z = torch.rand(args.batch_size, theta_dim, device=device)  # Uniform[0,1] like reference

        for _ in range(args.disc_updates):
            traj = euler_integrate(vel_net, z, y_batch, args.n_steps)
            theta_gen = traj[-1].detach()
            phi_gen = disc(theta_gen, y_batch)
            phi_real = disc(theta_batch, y_batch)
            L_dual = compute_L_dual(phi_gen, phi_real)
            if use_gp:
                gp = _gradient_penalty(disc, theta_batch, theta_gen, y_batch, args.lip_scale)
                disc_loss = -L_dual + args.gp_lambda * gp
            else:
                disc_loss = -L_dual
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()
            if not use_gp:
                _project_disc_weights(disc, args.lip_scale)

        traj = euler_integrate(vel_net, z, y_batch, args.n_steps)
        theta_gen = traj[-1]
        phi_gen = disc(theta_gen, y_batch)
        phi_real = disc(theta_batch, y_batch)
        L_dual = compute_L_dual(phi_gen, phi_real)

        KE = 0.0
        dt = 1.0 / args.n_steps
        for i, theta_t in enumerate(traj[:-1]):
            t = torch.full((theta_t.shape[0], 1), i * dt, device=device)
            v = vel_net(t, theta_t, y_batch)
            KE += 0.5 * (v ** 2).sum(dim=1).mean() * dt

        loss = L_dual + args.lam * KE
        opt_vel.zero_grad()
        loss.backward()
        opt_vel.step()

        history['L_dual'].append(L_dual.item())
        history['KE'].append(KE.item() if isinstance(KE, torch.Tensor) else KE)
        history['iters'].append(it)

        if it % 500 == 0:
            elapsed = time.time() - t_start
            recent_dual = np.mean(history['L_dual'][-200:])
            recent_ke = np.mean(history['KE'][-200:])
            print(f"  [{it}/{args.n_iters}] L_dual={recent_dual:.4f}, "
                  f"KE={recent_ke:.4f} ({elapsed:.0f}s)")

    # Save model
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pt"
    state = {
        'hparams': {
            'theta_dim': theta_dim, 'y_dim': y_dim,
            'arch_type': args.arch,
            'hidden': args.hidden, 'n_layers': args.n_layers,
            'cnn_channels': args.cnn_channels,
            'lip_scale': args.lip_scale,
            'lr_gen': args.lr, 'lr_disc': lr_disc,
        },
        'vel_net_state': vel_net.state_dict(),
        'disc_state': disc.state_dict(),
        'history': history,
        'iter': args.n_iters,
    }
    torch.save(state, model_path)
    print(f"Model saved: {model_path}")

    # Evaluate
    print("Evaluating...")
    vel_net.eval()
    all_metrics = []
    all_samples = []
    with torch.no_grad():
        for i in range(args.n_test):
            y_i = y_test[i:i+1].expand(args.n_eval_samples, -1).to(device)
            z = torch.rand(args.n_eval_samples, theta_dim, device=device)
            traj = euler_integrate(vel_net, z, y_i, args.n_steps)
            samples = traj[-1].cpu()
            all_samples.append(samples)
            metrics = compute_metrics(samples, theta_true[i])
            all_metrics.append(metrics)

    mean_mse = np.mean([m['mse'] for m in all_metrics])
    mean_div = np.mean([m['diversity'] for m in all_metrics])
    print(f"Result: MSE={mean_mse:.4f}, diversity={mean_div:.4f}")

    # Save completions plot
    plot_completions(
        y_test.numpy(), [s.numpy() for s in all_samples],
        theta_true.numpy(),
        save_path=str(output_dir / "completions.png"),
        n_show=5, title=f"{args.name} (MSE={mean_mse:.4f})"
    )

    # Save metrics JSON
    result = {
        'name': args.name, 'type': 'flow',
        'arch_type': args.arch,
        'hidden': args.hidden, 'n_layers': args.n_layers,
        'lam': args.lam, 'lip_scale': args.lip_scale,
        'lr_gen': args.lr, 'lr_disc': lr_disc,
        'mean_mse': mean_mse, 'mean_diversity': mean_div,
        'model_path': str(model_path),
        'elapsed_s': time.time() - t_start,
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Done: {output_dir}")


# ---------------------------------------------------------------------------
# GPA refinement
# ---------------------------------------------------------------------------

def run_gpa(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print(f"Loading checkpoint: {args.checkpoint}")
    vel_net, hp = load_checkpoint(args.checkpoint, device)
    theta_dim = hp['theta_dim']
    y_dim = hp['y_dim']

    print("Loading MNIST data...")
    train_data = MNISTInpainting(train=True, data_root='./data')
    test_data = MNISTInpainting(train=False, data_root='./data')
    theta_train, y_train = train_data.get_tensors()
    theta_train = theta_train.to(device)
    y_train = y_train.to(device)
    y_test, theta_true = test_data.get_test_images(n=args.n_test)

    n_total = args.n_test * args.n_samples
    y_particles = y_test.repeat_interleave(args.n_samples, dim=0).to(device)

    print(f"Generating {args.n_samples} initial particles per test image...")
    with torch.no_grad():
        z = torch.rand(n_total, theta_dim, device=device)
        traj = euler_integrate(vel_net, z, y_particles, args.n_steps)
        particles = traj[-1].clone()

    # Save flow-only samples for comparison plot
    flow_samples = []
    for i in range(args.n_test):
        start = i * args.n_samples
        end = start + args.n_samples
        flow_samples.append(particles[start:end].detach().cpu())

    # Build conditional real-data pools: for each test y, find nearest
    # training images and sample real theta from those (same y for fake/real).
    n_train = len(theta_train)
    print("Building conditional real-data pools (nearest neighbors in y)...")
    n_neighbors = 200  # pool size per test image
    # y_test: (n_test, y_dim), y_train: (n_train, y_dim)
    # For each test y, find closest training y's
    y_test_dev = y_test.to(device)
    # Compute distances in batches to avoid OOM
    nn_indices = []  # (n_test, n_neighbors) indices into training set
    for i in range(args.n_test):
        dists = ((y_train - y_test_dev[i:i+1]) ** 2).sum(dim=1)  # (n_train,)
        _, top_idx = dists.topk(n_neighbors, largest=False)
        nn_indices.append(top_idx)
    nn_indices = torch.stack(nn_indices)  # (n_test, n_neighbors)
    # Expand to per-particle: each particle inherits its test image's pool
    # particle i belongs to test image i // n_samples
    particle_nn = nn_indices.repeat_interleave(args.n_samples, dim=0)  # (n_total, n_neighbors)
    print(f"  Built pools: {n_neighbors} neighbors per test image")

    use_gp = args.lip_mode == 'gp'
    print(f"GPA: {args.name} (K={args.K}, eta={args.eta}, L={args.L}, fstar={args.fstar}, lip={args.lip_mode})")
    disc = DiscriminatorMLP(theta_dim=theta_dim, y_dim=y_dim, hidden=512, n_layers=4).to(device)
    if not use_gp:
        _project_disc_weights(disc, args.L)
    opt_disc = optim.Adam(disc.parameters(), lr=args.disc_lr)

    def compute_L_dual(phi_gen, phi_real):
        if args.fstar == 'reverse_kl':
            return phi_gen.mean() - torch.logsumexp(phi_real.squeeze(), dim=0) + np.log(phi_real.shape[0])
        else:
            return phi_gen.mean() - torch.exp(phi_real - 1).mean()

    t_start = time.time()
    for k in range(args.K):
        for _ in range(args.disc_steps):
            idx = torch.randint(0, n_total, (args.batch_size,))
            p_batch = particles[idx].detach()
            y_batch = y_particles[idx]
            # Sample real theta from conditional pool (same y neighborhood)
            pool = particle_nn[idx]  # (batch_size, n_neighbors)
            pick = torch.randint(0, n_neighbors, (args.batch_size,))
            real_idx = pool[torch.arange(args.batch_size), pick]
            t_real = theta_train[real_idx]
            # Use same y for fake and real (conditional formulation)
            phi_fake = disc(p_batch, y_batch)
            phi_real = disc(t_real, y_batch)
            L_dual = compute_L_dual(phi_fake, phi_real)
            if use_gp:
                gp = _gradient_penalty(disc, t_real, p_batch, y_batch, args.L)
                disc_loss = -L_dual + args.gp_lambda * gp
            else:
                disc_loss = -L_dual
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()
            if not use_gp:
                _project_disc_weights(disc, args.L)

        particles.requires_grad_(True)
        phi = disc(particles, y_particles)
        grad = torch.autograd.grad(phi.sum(), particles, create_graph=False)[0]
        with torch.no_grad():
            particles = (particles - args.eta * grad).detach()

        if (k + 1) % 50 == 0:
            print(f"  [GPA {k+1}/{args.K}] L_dual={L_dual.item():.4f}")

    # Evaluate
    all_metrics = []
    all_samples = []
    for i in range(args.n_test):
        start = i * args.n_samples
        end = start + args.n_samples
        samples_i = particles[start:end].detach().cpu()
        all_samples.append(samples_i)
        metrics = compute_metrics(samples_i, theta_true[i])
        all_metrics.append(metrics)

    mean_mse = np.mean([m['mse'] for m in all_metrics])
    mean_div = np.mean([m['diversity'] for m in all_metrics])
    print(f"Result: MSE={mean_mse:.4f}, diversity={mean_div:.4f}")

    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPA-only completions
    plot_completions(
        y_test.numpy(), [s.numpy() for s in all_samples],
        theta_true.numpy(),
        save_path=str(output_dir / "completions.png"),
        n_show=5, title=f"GPA: {args.name} (MSE={mean_mse:.4f})"
    )

    # Flow-only completions
    flow_mse = np.mean([compute_metrics(fs, theta_true[i])['mse']
                        for i, fs in enumerate(flow_samples)])
    plot_completions(
        y_test.numpy(), [s.numpy() for s in flow_samples],
        theta_true.numpy(),
        save_path=str(output_dir / "flow_completions.png"),
        n_show=5, title=f"Flow only (MSE={flow_mse:.4f})"
    )

    # Combined: flow -> GPA comparison
    plot_flow_vs_gpa(
        y_test.numpy(), theta_true.numpy(),
        [s.numpy() for s in flow_samples],
        [s.numpy() for s in all_samples],
        save_path=str(output_dir / "flow_vs_gpa.png"),
        flow_mse=flow_mse, gpa_mse=mean_mse,
        title=args.name,
    )

    result = {
        'name': args.name, 'type': 'flow+gpa',
        'checkpoint': args.checkpoint,
        'K': args.K, 'eta': args.eta, 'L': args.L,
        'fstar': args.fstar, 'lip_mode': args.lip_mode,
        'gp_lambda': args.gp_lambda,
        'mean_mse': mean_mse, 'mean_diversity': mean_div,
        'elapsed_s': time.time() - t_start,
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Done: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)

    # Flow subcommand
    fp = sub.add_parser('flow')
    fp.add_argument('--name', required=True)
    fp.add_argument('--arch', default='mlp', choices=['mlp', 'cnn'])
    fp.add_argument('--hidden', type=int, default=512)
    fp.add_argument('--n-layers', type=int, default=4)
    fp.add_argument('--cnn-channels', type=int, default=32)
    fp.add_argument('--n-iters', type=int, default=15000)
    fp.add_argument('--batch-size', type=int, default=256)
    fp.add_argument('--lr', type=float, default=1e-4,
                    help='Generator (velocity) learning rate')
    fp.add_argument('--lr-disc', type=float, default=None,
                    help='Discriminator learning rate (default: same as --lr)')
    fp.add_argument('--lam', type=float, default=0.1)
    fp.add_argument('--lip-scale', type=float, default=10.0)
    fp.add_argument('--fstar', default='kl', choices=['kl', 'reverse_kl'],
                    help='f-star: kl=exp(t-1), reverse_kl=log(mean(exp(t)))')
    fp.add_argument('--lip-mode', default='project', choices=['project', 'gp'],
                    help='Lipschitz enforcement: hard projection or gradient penalty')
    fp.add_argument('--gp-lambda', type=float, default=0.1,
                    help='Gradient penalty coefficient (only used with --lip-mode gp)')
    fp.add_argument('--n-steps', type=int, default=10)
    fp.add_argument('--disc-updates', type=int, default=5)
    fp.add_argument('--n-test', type=int, default=10)
    fp.add_argument('--n-eval-samples', type=int, default=64)
    fp.add_argument('--output-dir', default='results/mnist/sweep')

    # GPA subcommand
    gp = sub.add_parser('gpa')
    gp.add_argument('--name', required=True)
    gp.add_argument('--checkpoint', required=True)
    gp.add_argument('--K', type=int, default=300)
    gp.add_argument('--eta', type=float, default=0.01)
    gp.add_argument('--L', type=float, default=10.0)
    gp.add_argument('--n-samples', type=int, default=64)
    gp.add_argument('--n-test', type=int, default=10)
    gp.add_argument('--n-steps', type=int, default=10)
    gp.add_argument('--disc-steps', type=int, default=3)
    gp.add_argument('--disc-lr', type=float, default=1e-3)
    gp.add_argument('--batch-size', type=int, default=256)
    gp.add_argument('--fstar', default='kl', choices=['kl', 'reverse_kl'],
                    help='f-star for GPA disc (default: kl)')
    gp.add_argument('--lip-mode', default='project', choices=['project', 'gp'],
                    help='Lipschitz enforcement for GPA disc')
    gp.add_argument('--gp-lambda', type=float, default=0.1,
                    help='Gradient penalty coefficient for GPA disc')
    gp.add_argument('--output-dir', default='results/mnist/sweep')

    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)

    if args.mode == 'flow':
        run_flow(args)
    elif args.mode == 'gpa':
        run_gpa(args)


if __name__ == '__main__':
    main()
