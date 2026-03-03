"""Velocity network for conditional flow."""

import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    """Velocity field v(t, θ; y) : [0,1] × ℝⁿ × ℝᵐ → ℝⁿ.

    Dimension-agnostic: works for any θ ∈ ℝⁿ, y ∈ ℝᵐ.

    Args:
        theta_dim: Dimension of θ (n)
        y_dim: Dimension of conditioning variable y (m)
        hidden: Hidden layer width
        n_layers: Number of layers (minimum 2)
        activation: Activation function ('silu', 'tanh', 'relu')
    """

    def __init__(
        self,
        theta_dim: int,
        y_dim: int,
        hidden: int = 256,
        n_layers: int = 4,
        activation: str = 'silu'
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim

        # Input: [t, θ, y] = 1 + theta_dim + y_dim
        input_dim = 1 + theta_dim + y_dim

        # Activation
        if activation == 'silu':
            act_fn = nn.SiLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = [nn.Linear(input_dim, hidden), act_fn()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), act_fn()])
        layers.append(nn.Linear(hidden, theta_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute velocity v(t, θ; y).

        Args:
            t: Time, shape (batch, 1)
            theta: Position, shape (batch, theta_dim)
            y: Conditioning, shape (batch, y_dim)

        Returns:
            Velocity, shape (batch, theta_dim)
        """
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)
