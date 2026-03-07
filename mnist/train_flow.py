"""W1W2 Flow training for MNIST bottom-half inpainting.

Self-contained: does not import from parent package.

Usage:
    python -m mnist.train_flow --n-iters 20000 --output-dir results/mnist
"""

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .data import MNISTInpainting
from .networks import VelocityMLP, DiscriminatorMLP


# ---------------------------------------------------------------------------
# Utilities (self-contained, copied from parent codebase)
# ---------------------------------------------------------------------------

def euler_integrate(vel_net, z0, y, n_steps=20):
    """Euler-integrate the velocity field from t=0 to t=1.

    Args:
        vel_net: Velocity network with forward(t, theta, y).
        z0: Initial noise, shape (batch, theta_dim).
        y: Conditioning, shape (batch, y_dim).
        n_steps: Number of integration steps.

    Returns:
        List of tensors [z0, z1, ..., z_{n_steps}], each (batch, theta_dim).
    """
    dt = 1.0 / n_steps
    theta = z0
    traj = [z0]
    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        theta = theta + dt * v
        traj.append(theta)
    return traj


def _spectral_norm_projection(layer, target_norm):
    """Hard spectral norm projection: W <- target_norm * W / ||W||_2."""
    with torch.no_grad():
        W = layer.weight
        sigma = torch.linalg.norm(W, ord=2)
        if sigma > 1e-6:
            W.mul_(target_norm / sigma)


def _project_disc_weights(disc, L):
    """Project all linear layers so the network has Lipschitz constant L.

    Each layer gets spectral norm L^(1/D) where D is the number of layers.
    """
    linear_layers = [m for m in disc.net if isinstance(m, nn.Linear)]
    D = len(linear_layers)
    per_layer_norm = L ** (1.0 / D)
    for layer in linear_layers:
        _spectral_norm_projection(layer, per_layer_norm)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    print("Loading MNIST training data...")
    data = MNISTInpainting(train=True, data_root='./data')
    theta_data, y_data = data.get_tensors()
    theta_data = theta_data.to(device)
    y_data = y_data.to(device)
    n_data = theta_data.shape[0]
    print(f"  {n_data} training images, theta_dim={theta_data.shape[1]}, y_dim={y_data.shape[1]}")

    theta_dim = theta_data.shape[1]  # 392
    y_dim = y_data.shape[1]          # 392

    # Networks
    vel_net = VelocityMLP(
        theta_dim=theta_dim, y_dim=y_dim,
        hidden=args.hidden, n_layers=args.n_layers
    ).to(device)
    disc = DiscriminatorMLP(
        theta_dim=theta_dim, y_dim=y_dim,
        hidden=args.hidden, n_layers=args.n_layers
    ).to(device)

    n_vel_params = sum(p.numel() for p in vel_net.parameters())
    n_disc_params = sum(p.numel() for p in disc.parameters())
    print(f"  VelocityMLP: {n_vel_params:,} params")
    print(f"  DiscriminatorMLP: {n_disc_params:,} params")

    opt_vel = optim.Adam(vel_net.parameters(), lr=args.lr)
    opt_disc = optim.Adam(disc.parameters(), lr=args.lr)

    # Initial projection
    _project_disc_weights(disc, args.lip_scale)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {'L_dual': [], 'KE': [], 'iters': []}

    print(f"\nTraining W1W2 Flow: {args.n_iters} iters, lam={args.lam}, "
          f"disc_updates={args.disc_updates}, Lip={args.lip_scale}, "
          f"n_steps={args.n_steps}, lr={args.lr}, bs={args.batch_size}")

    for it in range(1, args.n_iters + 1):
        # Sample batch
        idx = torch.randint(0, n_data, (args.batch_size,))
        theta_batch = theta_data[idx]
        y_batch = y_data[idx]
        z = torch.randn(args.batch_size, theta_dim, device=device)

        # --- Discriminator updates ---
        for _ in range(args.disc_updates):
            traj = euler_integrate(vel_net, z, y_batch, args.n_steps)
            theta_gen = traj[-1].detach()

            phi_gen = disc(theta_gen, y_batch)
            phi_real = disc(theta_batch, y_batch)
            L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

            disc_loss = -L_dual

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            # Hard spectral norm projection
            _project_disc_weights(disc, args.lip_scale)

        # --- Velocity update ---
        traj = euler_integrate(vel_net, z, y_batch, args.n_steps)
        theta_gen = traj[-1]

        phi_gen = disc(theta_gen, y_batch)
        phi_real = disc(theta_batch, y_batch)
        L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

        # Kinetic energy
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

        # Record
        history['L_dual'].append(L_dual.item())
        history['KE'].append(KE.item() if isinstance(KE, torch.Tensor) else KE)
        history['iters'].append(it)

        # Progress
        if it % 500 == 0:
            recent_dual = np.mean(history['L_dual'][-100:])
            recent_ke = np.mean(history['KE'][-100:])
            print(f"[{it}/{args.n_iters}] L_dual={recent_dual:.4f}, KE={recent_ke:.4f}")

        # Checkpoint
        if it % args.checkpoint_every == 0:
            ckpt_path = output_dir / f"checkpoint_iter{it}.pt"
            state = {
                'hparams': {
                    'theta_dim': theta_dim,
                    'y_dim': y_dim,
                    'hidden': args.hidden,
                    'n_layers': args.n_layers,
                    'lip_scale': args.lip_scale,
                },
                'vel_net_state': vel_net.state_dict(),
                'disc_state': disc.state_dict(),
                'history': history,
                'iter': it,
                'args': vars(args),
            }
            torch.save(state, ckpt_path)
            print(f"  [Checkpoint saved: {ckpt_path}]")

    # Save final model
    model_path = output_dir / "model.pt"
    state = {
        'hparams': {
            'theta_dim': theta_dim,
            'y_dim': y_dim,
            'hidden': args.hidden,
            'n_layers': args.n_layers,
            'lip_scale': args.lip_scale,
        },
        'vel_net_state': vel_net.state_dict(),
        'disc_state': disc.state_dict(),
        'history': history,
        'iter': args.n_iters,
        'args': vars(args),
    }
    torch.save(state, model_path)
    print(f"Model saved: {model_path}")

    return vel_net, disc, history


def main():
    parser = argparse.ArgumentParser(description='Train W1W2 flow for MNIST inpainting')
    parser.add_argument('--n-iters', type=int, default=20000, help='Number of training iterations')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lam', type=float, default=0.1, help='Kinetic energy weight')
    parser.add_argument('--n-steps', type=int, default=20, help='ODE integration steps')
    parser.add_argument('--disc-updates', type=int, default=5, help='Disc updates per velocity update')
    parser.add_argument('--lip-scale', type=float, default=10.0, help='Lipschitz constant for discriminator')
    parser.add_argument('--hidden', type=int, default=512, help='Hidden layer width')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--output-dir', type=str, default='results/mnist', help='Output directory')
    parser.add_argument('--checkpoint-every', type=int, default=2000, help='Checkpoint frequency')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
