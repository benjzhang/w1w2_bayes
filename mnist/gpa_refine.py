"""GPA refinement for MNIST bottom-half inpainting.

Self-contained: does not import from parent package.

Load a trained flow checkpoint, generate initial particles via flow,
then refine them using the conditional Generative Particle Algorithm.

Usage:
    python -m mnist.gpa_refine --checkpoint results/mnist/model.pt --output-dir results/mnist
"""

import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .data import MNISTInpainting
from .networks import VelocityMLP, DiscriminatorMLP


# ---------------------------------------------------------------------------
# Utilities (self-contained)
# ---------------------------------------------------------------------------

def euler_integrate(vel_net, z0, y, n_steps=20):
    """Euler-integrate the velocity field from t=0 to t=1."""
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
    """Hard spectral norm projection."""
    with torch.no_grad():
        W = layer.weight
        sigma = torch.linalg.norm(W, ord=2)
        if sigma > 1e-6:
            W.mul_(target_norm / sigma)


def _project_disc_weights(disc, L):
    """Project all linear layers for Lipschitz constant L."""
    linear_layers = [m for m in disc.net if isinstance(m, nn.Linear)]
    D = len(linear_layers)
    per_layer_norm = L ** (1.0 / D)
    for layer in linear_layers:
        _spectral_norm_projection(layer, per_layer_norm)


def load_checkpoint(checkpoint_path, device):
    """Load flow checkpoint and return velocity net."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = state['hparams']

    vel_net = VelocityMLP(
        theta_dim=hparams['theta_dim'],
        y_dim=hparams['y_dim'],
        hidden=hparams.get('hidden', 512),
        n_layers=hparams.get('n_layers', 4),
    ).to(device)
    vel_net.load_state_dict(state['vel_net_state'])
    vel_net.eval()

    return vel_net, hparams


# ---------------------------------------------------------------------------
# GPA refinement
# ---------------------------------------------------------------------------

def gpa_refine(
    particles,       # (N, theta_dim) initial particles
    y_particles,     # (N, y_dim) conditioning for each particle
    theta_data,      # (n_data, theta_dim) real joint samples
    y_data,          # (n_data, y_dim) real joint y
    K=500,
    eta=0.01,
    disc_steps=3,
    disc_lr=1e-3,
    L=10.0,
    batch_size=256,
    disc_hidden=512,
    disc_layers=4,
    device=None,
    verbose=True,
):
    """Refine particles using conditional Lipschitz-regularized GPA.

    Args:
        particles: Initial particles, shape (N, theta_dim).
        y_particles: Conditioning y for each particle, shape (N, y_dim).
        theta_data: Joint samples theta, shape (n_data, theta_dim).
        y_data: Joint samples y, shape (n_data, y_dim).
        K: Number of GPA outer steps.
        eta: Particle step size.
        disc_steps: Discriminator updates per outer step.
        disc_lr: Discriminator learning rate.
        L: Lipschitz constant.
        batch_size: Batch size for disc training.
        disc_hidden: Discriminator hidden width.
        disc_layers: Number of discriminator layers.
        device: Device.
        verbose: Print progress.

    Returns:
        Dict with 'particles', 'history'.
    """
    if device is None:
        device = particles.device

    particles = particles.clone().detach().to(device)
    y_particles = y_particles.to(device)
    theta_data = theta_data.to(device)
    y_data = y_data.to(device)

    n_particles = len(particles)
    n_data = len(theta_data)
    n_coupled = min(n_particles, n_data)
    theta_dim = particles.shape[1]
    y_dim = y_particles.shape[1]

    # Initialize discriminator
    disc = DiscriminatorMLP(
        theta_dim=theta_dim, y_dim=y_dim,
        hidden=disc_hidden, n_layers=disc_layers
    ).to(device)
    _project_disc_weights(disc, L)

    opt_disc = optim.Adam(disc.parameters(), lr=disc_lr)

    if verbose:
        print(f"  GPA: {n_particles} particles, {n_data} joint samples, "
              f"K={K}, eta={eta}, L={L}")

    history = {'L_dual': [], 'grad_norm': [], 'step': []}

    for k in range(K):
        # --- Train discriminator with coupled samples ---
        for _ in range(disc_steps):
            idx = torch.randint(0, n_coupled, (batch_size,))

            particles_batch = particles[idx].detach()
            y_batch = y_particles[idx]
            theta_real = theta_data[idx]

            phi_fake = disc(particles_batch, y_batch)
            phi_real = disc(theta_real, y_batch)

            # LT formulation
            L_dual = phi_fake.mean() - torch.exp(phi_real - 1).mean()
            disc_loss = -L_dual

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            _project_disc_weights(disc, L)

        # --- Update particles along -grad phi ---
        particles.requires_grad_(True)
        phi = disc(particles, y_particles)
        grad = torch.autograd.grad(
            outputs=phi.sum(), inputs=particles,
            create_graph=False
        )[0]

        with torch.no_grad():
            grad_norm = grad.norm(dim=1).mean().item()
            particles = particles - eta * grad
            particles = particles.detach()

        history['L_dual'].append(L_dual.item())
        history['grad_norm'].append(grad_norm)
        history['step'].append(k)

        if verbose and (k + 1) % 50 == 0:
            print(f"  [GPA {k+1}/{K}] L_dual={L_dual.item():.4f}, "
                  f"grad_norm={grad_norm:.4f}")

    return {
        'particles': particles,
        'history': history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='GPA refinement for MNIST inpainting')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to flow checkpoint')
    parser.add_argument('--K', type=int, default=500, help='Number of GPA steps')
    parser.add_argument('--eta', type=float, default=0.01, help='Particle step size')
    parser.add_argument('--L', type=float, default=10.0, help='Lipschitz constant')
    parser.add_argument('--n-samples', type=int, default=64, help='Particles per test image')
    parser.add_argument('--n-test', type=int, default=10, help='Number of test images')
    parser.add_argument('--n-steps', type=int, default=20, help='ODE integration steps')
    parser.add_argument('--disc-steps', type=int, default=3, help='Disc updates per GPA step')
    parser.add_argument('--disc-lr', type=float, default=1e-3, help='Discriminator learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for disc training')
    parser.add_argument('--output-dir', type=str, default='results/mnist', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load flow
    print(f"Loading checkpoint: {args.checkpoint}")
    vel_net, hparams = load_checkpoint(args.checkpoint, device)
    theta_dim = hparams['theta_dim']
    y_dim = hparams['y_dim']

    # Load data
    print("Loading MNIST data...")
    train_data = MNISTInpainting(train=True, data_root='./data')
    test_data = MNISTInpainting(train=False, data_root='./data')

    theta_train, y_train = train_data.get_tensors()
    y_test, theta_test_true = test_data.get_test_images(n=args.n_test)

    # Generate initial particles via flow
    print(f"Generating {args.n_samples} particles per test image via flow...")
    vel_net.eval()

    # For GPA, we need particles for each test y, plus joint (theta, y) from training
    # Replicate each test y n_samples times
    n_total = args.n_test * args.n_samples
    y_particles = y_test.repeat_interleave(args.n_samples, dim=0).to(device)  # (n_total, y_dim)

    with torch.no_grad():
        z = torch.randn(n_total, theta_dim, device=device)
        traj = euler_integrate(vel_net, z, y_particles, args.n_steps)
        particles_init = traj[-1]  # (n_total, theta_dim)

    print(f"  Initial particles shape: {particles_init.shape}")

    # For GPA, the "real" data is joint samples from training set
    # We need to match the particle count: use the same y values
    # Sample from training data matching the test y values
    # The coupled sampling means same indices for fake and real share same y.
    # We construct a real dataset that matches: for each test image's y,
    # find training samples with the same y (but MNIST y are unique, so we
    # just use random training samples paired with the particle y values).
    #
    # Strategy: replicate training data to create coupled (theta_real, y_particle) pairs.
    # For each particle at y_i, we pair it with a randomly chosen theta from training
    # data that has the same y_i. Since exact y matches are rare, we use the
    # particle's y and pair with random training theta (approximate).
    #
    # Actually, the standard GPA approach: fake = particles at their y,
    # real = joint samples (theta, y) from training data. For coupled sampling,
    # we store both with matched indices. So we just use training data directly.

    # Use a subset of training data for GPA (matched to n_total particles)
    # Repeat training data indices to match particle count
    n_train = len(theta_train)
    # Create a "real" buffer: randomly sample n_total training pairs
    perm = torch.randperm(n_train)[:n_total]
    if n_total > n_train:
        # Need to repeat
        reps = (n_total // n_train) + 1
        perm = torch.cat([torch.randperm(n_train) for _ in range(reps)])[:n_total]

    theta_real_buf = theta_train[perm]
    y_real_buf = y_train[perm]

    # For coupled sampling, we need fake and real to share y.
    # Set y_real = y_particles (the test y values), pair with random training theta.
    # This way idx samples from both will share the same y.
    theta_real_buf = theta_train[perm]
    y_real_buf = y_particles.cpu()  # Use particle y values as the shared y

    print(f"Running GPA refinement (K={args.K}, eta={args.eta}, L={args.L})...")
    result = gpa_refine(
        particles=particles_init,
        y_particles=y_particles,
        theta_data=theta_real_buf,
        y_data=y_real_buf,
        K=args.K,
        eta=args.eta,
        disc_steps=args.disc_steps,
        disc_lr=args.disc_lr,
        L=args.L,
        batch_size=args.batch_size,
        disc_hidden=hparams.get('hidden', 512),
        disc_layers=hparams.get('n_layers', 4),
        device=device,
        verbose=True,
    )

    refined_particles = result['particles']  # (n_total, theta_dim)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = output_dir / "gpa_particles.pt"
    torch.save({
        'particles_init': particles_init.cpu(),
        'particles_refined': refined_particles.cpu(),
        'y_test': y_test,
        'theta_test_true': theta_test_true,
        'n_samples': args.n_samples,
        'n_test': args.n_test,
        'history': result['history'],
        'args': vars(args),
    }, save_path)
    print(f"GPA results saved: {save_path}")


if __name__ == '__main__':
    main()
