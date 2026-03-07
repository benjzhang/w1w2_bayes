"""Bimodal quadratic constraint: mixture of two parabolas.

With probability p:     θ₁² + θ₂ = y + offset  (shifted branch)
With probability 1-p:   θ₁² + θ₂ = y           (original branch)

For a given y, the posterior has two branches:
  θ₂ = y + offset - θ₁²   and   θ₂ = y - θ₁²
"""

import numpy as np
from typing import Tuple, List, Optional
from .base import BaseProblem


class BimodalQuadraticProblem(BaseProblem):
    """Degenerate inverse problem with bimodal quadratic constraint.

    Prior: θ ~ N(0, I) in ℝ²
    Forward: mixture of two quadratics
        y = θ₁² + θ₂ - offset   with probability mix_prob
        y = θ₁² + θ₂            with probability 1 - mix_prob
    Posterior: supported on TWO parabolas
        θ₂ = y + offset - θ₁²   and   θ₂ = y - θ₁²

    Args:
        offset: Vertical separation between the two parabola branches.
        mix_prob: Probability of the shifted branch.
    """

    def __init__(self, offset: float = 2.0, mix_prob: float = 0.5):
        self.offset = offset
        self.mix_prob = mix_prob

    @property
    def name(self) -> str:
        return "bimodal_quadratic"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def y_dim(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return (
            f"Bimodal Quadratic: two parabola branches "
            f"(offset={self.offset}, mix_prob={self.mix_prob})"
        )

    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.randn(n, 2)
        # Choose which branch each sample comes from
        use_shifted = np.random.rand(n) < self.mix_prob
        y = theta[:, 0]**2 + theta[:, 1]
        # Shifted branch: θ₁² + θ₂ = y + offset → y = θ₁² + θ₂ - offset
        y[use_shifted] -= self.offset
        return theta, y

    def sample_true_posterior(self, y_obs: float, n_samples: int) -> np.ndarray:
        """Sample from true posterior: mixture of two parabolas.

        p(θ|y) = mix_prob * N(θ₁;0,1)δ(θ₂ - (y+offset-θ₁²))
               + (1-mix_prob) * N(θ₁;0,1)δ(θ₂ - (y-θ₁²))
        """
        theta1 = np.random.randn(n_samples)
        use_shifted = np.random.rand(n_samples) < self.mix_prob
        theta2 = np.where(use_shifted,
                          y_obs + self.offset - theta1**2,
                          y_obs - theta1**2)
        return np.stack([theta1, theta2], axis=1)

    def compute_distance(self, theta: np.ndarray, y_obs: float) -> np.ndarray:
        """Minimum vertical distance to either parabola branch."""
        # Branch 1 (shifted): θ₂ = y + offset - θ₁²
        dist1 = np.abs(theta[:, 1] - (y_obs + self.offset - theta[:, 0]**2))
        # Branch 2 (original): θ₂ = y - θ₁²
        dist2 = np.abs(theta[:, 1] - (y_obs - theta[:, 0]**2))
        return np.minimum(dist1, dist2)

    def true_posterior_pdf(self, grid: np.ndarray, y_obs: float, dim: int = 0) -> Optional[np.ndarray]:
        """True marginal: mixture of two quartic distributions.

        p(θ₁|y) ∝ mix_prob * N(θ₁;0,1) * N(y + offset - θ₁²; 0,1)
                  + (1 - mix_prob) * N(θ₁;0,1) * N(y - θ₁²; 0,1)
        """
        if dim != 0:
            return None

        _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

        # Shifted branch: p(θ₁) * p(θ₂ = y + offset - θ₁²)
        #   = exp(-θ₁²/2) * exp(-(y + offset - θ₁²)²/2)
        log_p1 = -grid**2 / 2 - (y_obs + self.offset - grid**2)**2 / 2

        # Original branch: p(θ₁) * p(θ₂ = y - θ₁²)
        log_p2 = -grid**2 / 2 - (y_obs - grid**2)**2 / 2

        # Numerically stable mixture
        log_max = np.maximum(log_p1, log_p2).max()
        p1 = np.exp(log_p1 - log_max)
        p2 = np.exp(log_p2 - log_max)
        p = self.mix_prob * p1 + (1 - self.mix_prob) * p2

        Z = _trapz(p, grid)
        return p / Z if Z > 0 else p

    def plot_true_posterior(self, ax, y_obs: float, **kwargs):
        """Plot both constraint parabolas."""
        t1 = np.linspace(-3, 3, 100)
        # Shifted branch
        t2_shifted = y_obs + self.offset - t1**2
        ax.plot(t1, t2_shifted, 'r-', linewidth=2,
                label=f'θ₂=y+{self.offset}-θ₁²', **kwargs)
        # Original branch
        t2_orig = y_obs - t1**2
        ax.plot(t1, t2_orig, 'b-', linewidth=2,
                label='θ₂=y-θ₁²', **kwargs)

    def default_y_test_values(self) -> List[float]:
        return [-1.0, 0.0, 1.0, 2.0]

    def default_hyperparams(self) -> dict:
        return {
            'n_epochs': 300,
            'batch_size': 256,
            'lr': 1e-3,
            'lam': 0.01,
            'n_steps': 40,
            'disc_updates': 5,
            'lip_scale': 10.0,
            'vel_hidden': 256,
            'vel_layers': 4,
            'use_quadratic_features': True,
        }
