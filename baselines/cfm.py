"""Conditional Flow Matching with mini-batch OT coupling (Tong et al. 2023).

Learns a conditional velocity field v(t, θ; y) by regressing onto
the OT-coupled path between prior N(0,I) and joint samples.

Training:
  - Sample (θ₁, y) from joint, θ₀ ~ N(0,I)
  - Compute mini-batch OT coupling: match θ₀ to θ₁ via linear assignment
  - θ_t = (1-t)θ₀[π] + tθ₁  (OT-coupled interpolation)
  - u_t = θ₁ - θ₀[π]         (conditional velocity)
  - Loss = ||v(t, θ_t; y) - u_t||²

Sampling:
  - θ₀ ~ N(0,I)
  - Integrate dθ/dt = v(t, θ; y) from t=0 to t=1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import linear_sum_assignment
from typing import Dict, Any, Optional, Callable

from nn import VelocityNet
from utils.integrators import euler_integrate


def _ot_coupling(theta_0, theta_1):
    """Compute mini-batch OT coupling via linear assignment.

    Returns permutation indices for theta_0 that minimize total squared distance.
    """
    cost = torch.cdist(theta_0, theta_1, p=2).pow(2)
    _, col_idx = linear_sum_assignment(cost.cpu().numpy())
    return torch.tensor(col_idx, device=theta_0.device)


class ConditionalFlowMatching:
    """OT-CFM for conditional posterior sampling."""

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

        optimizer = optim.Adam(self.vel_net.parameters(), lr=lr)
        history = {'loss': [], 'iters': []}

        if verbose:
            print(f"Training OT-CFM: {n_iters} iters, lr={lr}, batch_size={batch_size}")

        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch

        data_iter = infinite_loader()

        for it in range(1, n_iters + 1):
            theta_1, y = next(data_iter)
            bs = theta_1.shape[0]

            # Sample θ₀ ~ N(0,I)
            theta_0 = torch.randn(bs, self.theta_dim, device=self.device)

            # Mini-batch OT coupling: permute θ₀ to match θ₁
            perm = _ot_coupling(theta_0, theta_1)
            theta_0 = theta_0[perm]

            # Sample t ~ U(0,1)
            t = torch.rand(bs, 1, device=self.device)

            # OT-coupled path: θ_t = (1-t)θ₀ + tθ₁
            theta_t = (1 - t) * theta_0 + t * theta_1

            # Target velocity: u_t = θ₁ - θ₀
            u_t = theta_1 - theta_0

            # Predict and compute loss
            v_pred = self.vel_net(t, theta_t, y)
            loss = ((v_pred - u_t) ** 2).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % 500 == 0:
                history['loss'].append(loss.item())
                history['iters'].append(it)
                if verbose:
                    print(f"  [{it}/{n_iters}] loss={loss.item():.4f}")

        return history

    def sample(self, y_val: float, n_samples: int, n_steps: int = 50) -> torch.Tensor:
        """Sample θ ~ p(θ|y) by integrating the learned velocity."""
        self.vel_net.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.theta_dim, device=self.device)
            y = torch.full((n_samples, self.y_dim), y_val, device=self.device)
            traj = euler_integrate(self.vel_net, z, y, n_steps)
        return traj[-1].cpu().numpy()
