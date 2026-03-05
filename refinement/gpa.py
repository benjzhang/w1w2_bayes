"""Conditional Generative Particle Algorithm (GPA) refinement.

Given approximate posterior particles from a trained flow, refine them
by alternating discriminator updates and particle gradient steps:

    x_i^{k+1} = x_i^{k} - eta * grad_x phi_k^*(x_i^{k}, y_i)

The discriminator phi(theta, y) is trained on:
  - Fake: stored particles (theta_i, y_i) from the approximate posterior
  - Real: joint samples (theta, y) ~ pi(theta, y) from the simulator

This learns the conditional structure across all y values simultaneously.

Based on: Gu et al., "Lipschitz-regularized generative particles algorithm"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Optional, Dict


def _spectral_norm_projection(layer: nn.Linear, target_norm: float):
    """Hard spectral norm projection: W <- target_norm * W / ||W||_2."""
    with torch.no_grad():
        W = layer.weight
        sigma = torch.linalg.norm(W, ord=2)
        if sigma > 1e-6:
            W.mul_(target_norm / sigma)


def _project_disc_weights(disc: nn.Module, L: float):
    """Project all linear layers so the network has Lipschitz constant L.

    Each layer gets spectral norm L^(1/D) where D is the number of layers.
    """
    linear_layers = [m for m in disc.net if isinstance(m, nn.Linear)]
    D = len(linear_layers)
    per_layer_norm = L ** (1.0 / D)
    for layer in linear_layers:
        _spectral_norm_projection(layer, per_layer_norm)


class GPADiscriminator(nn.Module):
    """Discriminator for conditional GPA with hard spectral norm projection.

    Architecture follows GPA reference: SiLU activations, 4 layers.
    """

    def __init__(self, theta_dim: int, y_dim: int, hidden: int = 32, n_layers: int = 4):
        super().__init__()
        input_dim = theta_dim + y_dim
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([theta, y], dim=1)
        return self.net(inp)


def gpa_refine(
    particles: torch.Tensor,
    y_particles: torch.Tensor,
    theta_data: torch.Tensor,
    y_data: torch.Tensor,
    n_coupled: Optional[int] = None,
    disc: Optional[nn.Module] = None,
    K: int = 500,
    eta: float = 0.5,
    disc_steps: int = 3,
    disc_lr: float = 0.001,
    L: float = 1.0,
    batch_size: int = 256,
    gp_weight: float = 0.0,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """Refine particles using conditional Lipschitz-regularized GPA.

    Disc training uses coupled samples at the same y:
      sup_phi (1/N) sum_i [phi(particle_i, y_i) - exp(phi(theta_i, y_i) - 1)]

    where (theta_i, y_i) ~ pi(theta, y) and particle_i ~ rho_T(theta|y_i).
    The first n_coupled particles correspond 1-to-1 with joint samples
    (same y values), ensuring proper conditioning.

    Then updates ALL particles: theta_i -= eta * grad_theta phi(theta_i, y_i)

    Args:
        particles: Initial particles, shape (N, theta_dim). First n_coupled
            entries correspond to joint samples (same y values).
        y_particles: Conditioning y for each particle, shape (N, y_dim).
        theta_data: Joint samples, shape (n_data, theta_dim).
        y_data: Joint samples, shape (n_data, y_dim).
        n_coupled: Number of particles coupled with joint data (default: n_data).
        disc: Pre-trained discriminator to warm-start from.
        K: Number of GPA outer steps.
        eta: Particle step size.
        disc_steps: Discriminator updates per outer step.
        disc_lr: Discriminator learning rate.
        L: Lipschitz constant.
        batch_size: Batch size for disc training.
        device: Device.
        verbose: Print progress.

    Returns:
        Dict with 'particles', 'history', 'disc'.
    """
    if device is None:
        device = particles.device

    particles = particles.clone().detach().to(device)
    y_particles = y_particles.to(device)
    theta_data = theta_data.to(device)
    y_data = y_data.to(device)

    n_particles = len(particles)
    n_data = len(theta_data)
    if n_coupled is None:
        n_coupled = n_data
    theta_dim = particles.shape[1]
    y_dim = y_particles.shape[1]

    # Initialize discriminator
    if disc is not None:
        disc = copy.deepcopy(disc).to(device)
        disc.lip_scale = 1.0
    else:
        disc = GPADiscriminator(
            theta_dim=theta_dim, y_dim=y_dim,
            hidden=32, n_layers=4
        ).to(device)

    if gp_weight == 0:
        _project_disc_weights(disc, L)

    if verbose:
        print(f"  GPA: {n_particles} particles ({n_coupled} coupled), "
              f"{n_data} joint samples, L={L}")

    history = {'L_dual': [], 'grad_norm': [], 'step': []}

    for k in range(K):
        # --- Train discriminator with coupled samples ---
        opt_disc = optim.Adam(disc.parameters(), lr=disc_lr)

        for _ in range(disc_steps):
            # Sample SAME indices for both fake and real (coupled at same y)
            idx = torch.randint(0, n_coupled, (batch_size,))

            # Fake: particles at y_i
            particles_batch = particles[idx].detach()
            y_batch = y_particles[idx]

            # Real: joint samples at SAME y_i
            theta_real = theta_data[idx]

            phi_fake = disc(particles_batch, y_batch)
            phi_real = disc(theta_real, y_batch)

            # KL variational: E_fake[phi] - E_real[exp(phi - 1)]
            L_dual = phi_fake.mean() - torch.exp(phi_real - 1).mean()
            disc_loss = -L_dual

            # One-sided gradient penalty on interpolates
            if gp_weight > 0:
                alpha = torch.rand(len(idx), 1, device=device)
                interp = (alpha * particles_batch + (1 - alpha) * theta_real).requires_grad_(True)
                phi_interp = disc(interp, y_batch)
                grad_interp = torch.autograd.grad(
                    phi_interp.sum(), interp, create_graph=True
                )[0]
                grad_norms = grad_interp.norm(2, dim=1)
                gp = torch.clamp(grad_norms - L, min=0).pow(2).mean()
                disc_loss = disc_loss + gp_weight * gp

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            if gp_weight == 0:
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

        if verbose and (k + 1) % 10 == 0:
            print(f"  [GPA {k+1}/{K}] L_dual={L_dual.item():.4f}, "
                  f"grad_norm={grad_norm:.4f}")

    return {
        'particles': particles,
        'history': history,
        'disc': disc,
    }
