"""Discriminator (critic) network for Wasserstein duality."""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Discriminator φ(θ; y) : ℝⁿ × ℝᵐ → ℝ with spectral normalization.

    Used for Wasserstein-1 duality:
        W₁(p, q) = sup_{‖φ‖_Lip ≤ 1} E_p[φ] - E_q[φ]

    Spectral normalization ensures Lipschitz constraint.

    Args:
        theta_dim: Dimension of θ (n)
        y_dim: Dimension of conditioning variable y (m)
        hidden: Hidden layer width
        n_layers: Number of layers
        lip_scale: Lipschitz constant scale (multiplies output)
        use_quadratic_features: Add θ² and cross-product features
            (useful for nonlinear constraints like parabolas)
        use_spectral_norm: Apply spectral normalization to layers.
            Set False when using gradient penalty instead.
        activation: Activation function
    """

    def __init__(
        self,
        theta_dim: int,
        y_dim: int,
        hidden: int = 128,
        n_layers: int = 3,
        lip_scale: float = 1.0,
        use_quadratic_features: bool = False,
        use_spectral_norm: bool = True,
        activation: str = 'silu'
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim
        self.lip_scale = lip_scale
        self.use_quadratic_features = use_quadratic_features

        # Input dimension
        # Base: theta_dim + y_dim
        # With quadratic: theta_dim + theta_dim*(theta_dim+1)/2 + y_dim
        if use_quadratic_features:
            n_quad = theta_dim * (theta_dim + 1) // 2  # θᵢ² and θᵢθⱼ terms
            input_dim = theta_dim + n_quad + y_dim
        else:
            input_dim = theta_dim + y_dim

        # Activation
        if activation == 'silu':
            act_fn = nn.SiLU
        elif activation == 'tanh':
            act_fn = nn.Tanh
        elif activation == 'relu':
            act_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Optionally wrap layers with spectral normalization
        def maybe_sn(layer):
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # Build network
        layers = [
            maybe_sn(nn.Linear(input_dim, hidden)),
            act_fn()
        ]
        for _ in range(n_layers - 2):
            layers.extend([
                maybe_sn(nn.Linear(hidden, hidden)),
                act_fn()
            ])
        layers.append(maybe_sn(nn.Linear(hidden, 1)))

        self.net = nn.Sequential(*layers)

    def _compute_quadratic_features(self, theta: torch.Tensor) -> torch.Tensor:
        """Compute quadratic features: θᵢ² and θᵢθⱼ for all i ≤ j."""
        features = []
        d = theta.shape[1]

        # θᵢ² terms
        for i in range(d):
            features.append(theta[:, i:i+1] ** 2)

        # θᵢθⱼ terms (i < j)
        for i in range(d):
            for j in range(i + 1, d):
                features.append(theta[:, i:i+1] * theta[:, j:j+1])

        return torch.cat(features, dim=1)

    def forward(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute discriminator value φ(θ; y).

        Args:
            theta: Position, shape (batch, theta_dim)
            y: Conditioning, shape (batch, y_dim)

        Returns:
            Discriminator output, shape (batch, 1)
        """
        if self.use_quadratic_features:
            quad_features = self._compute_quadratic_features(theta)
            inp = torch.cat([theta, quad_features, y], dim=1)
        else:
            inp = torch.cat([theta, y], dim=1)

        return self.lip_scale * self.net(inp)
