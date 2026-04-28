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

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from typing import Optional, Dict


class MollifiedReLU(nn.Module):
    """Smooth C³ ReLU approximation with Lipschitz constant 1.

    From Gu et al.: ReLU_s^eps(x) =
        0                                          if x <= 0
        x²/(4ε) + ε(cos(πx/ε) - 1)/(2π²)         if 0 < x < 2ε
        x - ε                                      if x >= 2ε
    """

    def __init__(self, eps: float = 0.5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        mask_neg = x <= 0
        mask_mid = (x > 0) & (x < 2 * eps)
        mask_pos = x >= 2 * eps
        mid_val = x.pow(2) / (4 * eps) + eps * (torch.cos(math.pi * x / eps) - 1) / (2 * math.pi ** 2)
        return torch.where(mask_neg, torch.zeros_like(x),
               torch.where(mask_mid, mid_val, x - eps))


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
    """Discriminator for conditional GPA.

    Uses ReLU (Lip-1 compatible) for hard spectral norm projection,
    or SiLU for gradient penalty mode.
    """

    def __init__(self, theta_dim: int, y_dim: int, hidden: int = 32,
                 n_layers: int = 4, activation: str = 'relu'):
        super().__init__()
        input_dim = theta_dim + y_dim
        if activation == 'relu':
            act_cls = nn.ReLU
        elif activation == 'mollified_relu':
            act_cls = lambda: MollifiedReLU(eps=0.5)
        else:
            act_cls = nn.SiLU
        layers = [nn.Linear(input_dim, hidden), act_cls()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), act_cls()])
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
    disc_hidden: int = 32,
    disc_layers: int = 4,
    formulation: str = 'LT',
    activation: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    eval_callback=None,
    eval_every: int = 50,
    snapshot_every: int = 0,
    normalize_grad: bool = False,
    disc_optimizer: str = 'adam',
    disc_reset_every: int = 0,
) -> Dict:
    """Refine particles using conditional Lipschitz-regularized GPA.

    Supports three KL variational formulations:
      LT:    E_P[phi] - E_Q[exp(phi - 1)]                    (nu = 0 fixed)
      LT_nu: E_P[phi] - E_Q[exp(phi - nu - 1)] - nu          (nu optimized)
      DV:    E_P[phi] - log E_Q[exp(phi)]                     (Donsker-Varadhan)

    Args:
        particles: Initial particles, shape (N, theta_dim).
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
        formulation: 'LT', 'LT_nu', or 'DV'.
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
        # Strip spectral norm hooks if using gradient penalty
        if gp_weight > 0:
            for name, module in list(disc.named_modules()):
                if isinstance(module, nn.Linear):
                    try:
                        nn.utils.remove_spectral_norm(module)
                    except ValueError:
                        pass  # no spectral norm on this layer
    else:
        # Auto-select activation if not specified
        if activation is not None:
            act = activation
        else:
            act = 'relu' if gp_weight == 0 else 'silu'
        disc = GPADiscriminator(
            theta_dim=theta_dim, y_dim=y_dim,
            hidden=disc_hidden, n_layers=disc_layers, activation=act
        ).to(device)

    if gp_weight == 0:
        _project_disc_weights(disc, L)

    # Trainable nu for LT_nu formulation
    nu = torch.tensor(0.0, device=device, requires_grad=(formulation == 'LT_nu'))

    if verbose:
        print(f"  GPA: {n_particles} particles ({n_coupled} coupled), "
              f"{n_data} joint samples, L={L}, formulation={formulation}")

    history = {'L_dual': [], 'grad_norm': [], 'step': [], 'eval': []}
    snapshots = []
    if snapshot_every > 0:
        snapshots.append((0, particles.detach().cpu().clone()))

    # Persistent optimizer across GPA steps
    disc_params = list(disc.parameters())
    if formulation == 'LT_nu':
        disc_params.append(nu)
    if disc_optimizer == 'sgd':
        opt_disc = optim.SGD(disc_params, lr=disc_lr)
    else:
        opt_disc = optim.Adam(disc_params, lr=disc_lr)

    # Eval at step 0 (before any updates)
    if eval_callback is not None:
        eval_result = eval_callback(particles, 0)
        history['eval'].append((0, eval_result))

    for k in range(K):
        # Reset optimizer state periodically (keep disc weights)
        if disc_reset_every > 0 and k > 0 and k % disc_reset_every == 0:
            if disc_optimizer == 'sgd':
                opt_disc = optim.SGD(disc_params, lr=disc_lr)
            else:
                opt_disc = optim.Adam(disc_params, lr=disc_lr)

        # --- Train discriminator with coupled samples ---

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

            # KL variational dual
            if formulation == 'DV':
                # Donsker-Varadhan: E_P[phi] - log E_Q[exp(phi)]
                L_dual = phi_fake.mean() - torch.logsumexp(phi_real, dim=0).squeeze() + torch.log(torch.tensor(float(len(phi_real)), device=device))
            elif formulation == 'LT_nu':
                # LT with trainable nu: E_P[phi] - E_Q[exp(phi - nu - 1)] - nu
                L_dual = phi_fake.mean() - torch.exp(phi_real - nu - 1).mean() - nu
            else:
                # LT (default): E_P[phi] - E_Q[exp(phi - 1)]
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

            if gp_weight == 0 and L > 0 and not np.isinf(L):
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
            if normalize_grad:
                grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-8)
            particles = particles - eta * grad
            particles = particles.detach()

        history['L_dual'].append(L_dual.item())
        history['grad_norm'].append(grad_norm)
        history['step'].append(k)

        if snapshot_every > 0 and (k + 1) % snapshot_every == 0:
            snapshots.append((k + 1, particles.detach().cpu().clone()))

        if eval_callback is not None and (k + 1) % eval_every == 0:
            eval_result = eval_callback(particles, k + 1)
            history['eval'].append((k + 1, eval_result))

        if verbose and (k + 1) % 10 == 0:
            nu_str = f", nu={nu.item():.4f}" if formulation == 'LT_nu' else ""
            print(f"  [GPA {k+1}/{K}] L_dual={L_dual.item():.4f}, "
                  f"grad_norm={grad_norm:.4f}{nu_str}")

    result = {
        'particles': particles,
        'history': history,
        'disc': disc,
    }
    if snapshot_every > 0:
        result['snapshots'] = snapshots
    return result
