"""Continuous Normalizing Flow with Maximum Likelihood + W2 regularization.

Learns a velocity field v(t, θ; y) via the change-of-variables formula:

  log p₁(θ₁|y) = log p₀(θ₀) - ∫₀¹ tr(∂v/∂θ) dt

where θ₁ = θ₀ + ∫₀¹ v(t, θ_t; y) dt.

Training:
  - Given (θ₁, y) from joint data
  - Integrate backward to get θ₀ and the log-determinant
  - Loss = -log p₁(θ₁|y) + λ * KE
  - Uses Hutchinson's trace estimator for ∂v/∂θ

Sampling:
  - θ₀ ~ N(0,I)
  - Integrate forward: dθ/dt = v(t, θ; y) from t=0 to t=1
"""

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional

from nn import VelocityNet
from utils.integrators import euler_integrate


class CNF_MLE:
    """CNF trained with MLE + kinetic energy (W2) regularization."""

    def __init__(
        self,
        theta_dim: int,
        y_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.theta_dim = theta_dim
        self.y_dim = y_dim

        self.vel_net = VelocityNet(
            theta_dim=theta_dim,
            y_dim=y_dim,
            hidden=hidden,
            n_layers=n_layers,
            activation='silu'
        ).to(device)

    def _integrate_backward_with_logdet(self, theta_1, y, n_steps=10):
        """Integrate from t=1 to t=0, computing log-determinant via Hutchinson.

        Returns θ₀ and Δlog p = -∫₀¹ tr(∂v/∂θ) dt (computed backward).
        """
        dt = 1.0 / n_steps
        theta = theta_1.clone().requires_grad_(True)
        delta_logp = torch.zeros(theta.shape[0], device=self.device)

        for i in range(n_steps, 0, -1):
            t_val = i * dt
            t = torch.full((theta.shape[0], 1), t_val, device=self.device)

            # Hutchinson's trace estimator: tr(∂v/∂θ) ≈ εᵀ (∂v/∂θ) ε
            eps = torch.randn_like(theta)

            with torch.enable_grad():
                theta_in = theta.detach().requires_grad_(True)
                v = self.vel_net(t, theta_in, y)

                # Compute εᵀ (∂v/∂θ) ε via vector-Jacobian product
                vjp = torch.autograd.grad(
                    outputs=v, inputs=theta_in,
                    grad_outputs=eps,
                    create_graph=True, retain_graph=True
                )[0]
                trace_est = (vjp * eps).sum(dim=1)

            # Backward Euler: θ_{t-dt} = θ_t - v * dt
            with torch.no_grad():
                v_detached = self.vel_net(t, theta, y)
                theta = theta - dt * v_detached

            # Accumulate log-determinant
            delta_logp = delta_logp - dt * trace_est

        return theta, delta_logp

    def train(
        self,
        theta_data: torch.Tensor,
        y_data: torch.Tensor,
        n_iters: int = 20000,
        batch_size: int = 256,
        lr: float = 1e-3,
        lam: float = 0.01,
        n_steps: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        theta_data = theta_data.to(self.device)
        y_data = y_data.to(self.device)

        dataset = TensorDataset(theta_data, y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = optim.Adam(self.vel_net.parameters(), lr=lr)
        history = {'nll': [], 'KE': [], 'iters': []}

        if verbose:
            print(f"Training CNF-MLE: {n_iters} iters, λ={lam}, n_steps={n_steps}")

        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch

        data_iter = infinite_loader()

        for it in range(1, n_iters + 1):
            theta_1, y = next(data_iter)

            # Integrate backward to get θ₀ and log-determinant
            theta_0, delta_logp = self._integrate_backward_with_logdet(
                theta_1, y, n_steps=n_steps)

            # log p₁(θ₁|y) = log p₀(θ₀) + delta_logp
            # log p₀(θ₀) = -½ ||θ₀||² - d/2 log(2π)
            log_p0 = -0.5 * (theta_0 ** 2).sum(dim=1) - 0.5 * self.theta_dim * math.log(2 * math.pi)
            log_p1 = log_p0 + delta_logp

            nll = -log_p1.mean()

            # Kinetic energy regularization (forward direction)
            dt = 1.0 / n_steps
            z = torch.randn_like(theta_1)
            theta_t = z.clone()
            ke = 0.0
            for i in range(n_steps):
                t = torch.full((theta_t.shape[0], 1), i * dt, device=self.device)
                v = self.vel_net(t, theta_t, y)
                ke += 0.5 * (v ** 2).sum(dim=1).mean() * dt
                theta_t = theta_t + dt * v.detach()

            loss = nll + lam * ke

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.vel_net.parameters(), 10.0)
            optimizer.step()

            if it % 500 == 0:
                history['nll'].append(nll.item())
                history['KE'].append(ke.item() if isinstance(ke, torch.Tensor) else ke)
                history['iters'].append(it)
                if verbose:
                    print(f"  [{it}/{n_iters}] nll={nll.item():.4f}, KE={ke.item():.4f}")

        return history

    def sample(self, y_val, n_samples: int, n_steps: int = 50) -> torch.Tensor:
        """Sample θ ~ p(θ|y) by integrating forward."""
        self.vel_net.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.theta_dim, device=self.device)
            if isinstance(y_val, (int, float)):
                y = torch.full((n_samples, self.y_dim), y_val, device=self.device)
            else:
                y = torch.FloatTensor(y_val).to(self.device).unsqueeze(0).expand(n_samples, -1)
            traj = euler_integrate(self.vel_net, z, y, n_steps)
        return traj[-1].cpu().numpy()
