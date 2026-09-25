"""Score-based Generative Model (SGM) / Conditional Diffusion.

VP-SDE formulation for conditional posterior sampling.

Forward SDE:
  dθ = -½ β(t) θ dt + √β(t) dW,  t ∈ [0, T]
  where β(t) = β_min + (β_max - β_min)*t  (linear schedule)

This gives: θ_t | θ_0 ~ N(√ᾱ_t θ_0, (1 - ᾱ_t) I)
  where ᾱ_t = exp(-½ ∫₀ᵗ β(s) ds)

Training (denoising score matching):
  - Sample (θ₀, y) from joint, t ~ U(0,T), ε ~ N(0,I)
  - θ_t = √ᾱ_t θ₀ + √(1 - ᾱ_t) ε
  - Loss = ||s(θ_t, t; y) + ε / √(1 - ᾱ_t)||²
    equivalently: train ε-prediction network ε̂(θ_t, t; y)
    Loss = ||ε̂(θ_t, t; y) - ε||²

Sampling (probability flow ODE):
  dθ/dt = -½ β(t) [θ + s(θ, t; y)]  from t=T to t=0
  or equivalently using the ε-prediction:
  dθ/dt = -½ β(t) [θ + ε̂(θ, t; y) / √(1 - ᾱ_t) * (-1)]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Dict, Any, Optional


class ScoreNet(nn.Module):
    """Noise prediction network ε̂(θ, t; y).

    Same architecture as VelocityNet: takes (t, θ, y), outputs θ-dim vector.
    """

    def __init__(self, theta_dim: int, y_dim: int, hidden: int = 128,
                 n_layers: int = 3):
        super().__init__()
        input_dim = 1 + theta_dim + y_dim
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, theta_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, theta, y):
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)


class ScoreBasedDiffusion:
    """VP-SDE conditional diffusion for posterior sampling."""

    def __init__(
        self,
        theta_dim: int,
        y_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

        self.eps_net = ScoreNet(
            theta_dim=theta_dim, y_dim=y_dim,
            hidden=hidden, n_layers=n_layers
        ).to(device)

    def _beta(self, t):
        """Linear beta schedule."""
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def _alpha_bar(self, t):
        """ᾱ_t = exp(-½ ∫₀ᵗ β(s) ds) = exp(-½ (β_min*t + (β_max-β_min)*t²/2))"""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def train(
        self,
        theta_data: torch.Tensor,
        y_data: torch.Tensor,
        n_iters: int = 20000,
        batch_size: int = 256,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        theta_data = theta_data.to(self.device)
        y_data = y_data.to(self.device)

        dataset = TensorDataset(theta_data, y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        optimizer = optim.Adam(self.eps_net.parameters(), lr=lr)
        history = {'loss': [], 'iters': []}

        if verbose:
            print(f"Training SGM: {n_iters} iters, β=[{self.beta_min}, {self.beta_max}]")

        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch

        data_iter = infinite_loader()

        for it in range(1, n_iters + 1):
            theta_0, y = next(data_iter)
            bs = theta_0.shape[0]

            # Sample t ~ U(eps, T), ε ~ N(0,I)
            t = torch.rand(bs, 1, device=self.device) * (self.T - 1e-5) + 1e-5
            eps = torch.randn_like(theta_0)

            # Forward diffusion: θ_t = √ᾱ_t θ₀ + √(1 - ᾱ_t) ε
            alpha_bar_t = self._alpha_bar(t)
            theta_t = torch.sqrt(alpha_bar_t) * theta_0 + torch.sqrt(1 - alpha_bar_t) * eps

            # Predict noise
            eps_pred = self.eps_net(t, theta_t, y)
            loss = ((eps_pred - eps) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 500 == 0:
                history['loss'].append(loss.item())
                history['iters'].append(it)
                if verbose:
                    print(f"  [{it}/{n_iters}] loss={loss.item():.4f}")

        return history

    def sample(self, y_val, n_samples: int, n_steps: int = 200) -> torch.Tensor:
        """Sample using the probability flow ODE (reverse-time)."""
        self.eps_net.eval()
        with torch.no_grad():
            # Start from pure noise at t=T
            theta = torch.randn(n_samples, self.theta_dim, device=self.device)
            if isinstance(y_val, (int, float)):
                y = torch.full((n_samples, self.y_dim), y_val, device=self.device)
            else:
                y = torch.FloatTensor(y_val).to(self.device).unsqueeze(0).expand(n_samples, -1)

            dt = self.T / n_steps
            for i in range(n_steps):
                t_val = self.T - i * dt
                t = torch.full((n_samples, 1), t_val, device=self.device)

                alpha_bar_t = self._alpha_bar(t)
                beta_t = self._beta(t)

                eps_pred = self.eps_net(t, theta, y)

                # Probability flow ODE:
                # dθ/dt = -½ β(t) [θ + score]
                # score = -ε̂ / √(1 - ᾱ_t)
                score = -eps_pred / torch.sqrt(1 - alpha_bar_t)
                drift = -0.5 * beta_t * (theta + score)

                # Reverse time: θ_{t-dt} = θ_t - drift * dt
                theta = theta - drift * dt

        return theta.cpu().numpy()
