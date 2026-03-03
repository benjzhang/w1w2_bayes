"""Linear constraint problem: y = θ₁ + θ₂."""

import numpy as np
from typing import Tuple, List
from .base import BaseProblem


class LinearProblem(BaseProblem):
    """Degenerate inverse problem with linear constraint.

    Prior: θ ~ N(0, I) in ℝ²
    Forward: y = θ₁ + θ₂ (exact, no noise)
    Posterior: supported on LINE θ₁ + θ₂ = y

    True posterior: p(θ₁|y) = N(y/2, 1/2), θ₂ = y - θ₁
    Correlation: ρ = -1 (perfect anti-correlation)
    """

    @property
    def name(self) -> str:
        return "linear"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def y_dim(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Linear: y = θ₁ + θ₂, posterior on line"

    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.randn(n, 2)
        y = theta[:, 0] + theta[:, 1]
        return theta, y

    def compute_distance(self, theta: np.ndarray, y_obs: float) -> np.ndarray:
        """Distance to line θ₁ + θ₂ = y."""
        return np.abs(theta[:, 0] + theta[:, 1] - y_obs) / np.sqrt(2)

    def true_posterior_pdf(self, grid: np.ndarray, y_obs: float, dim: int = 0) -> np.ndarray:
        """True marginal: θ₁|y ~ N(y/2, 1/2)."""
        mean = y_obs / 2
        std = np.sqrt(0.5)
        return np.exp(-0.5 * ((grid - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    def plot_true_posterior(self, ax, y_obs: float, **kwargs):
        """Plot the constraint line θ₁ + θ₂ = y."""
        t = np.linspace(-4, 4, 100)
        ax.plot(t, y_obs - t, 'r-', linewidth=2, label=f'θ₁+θ₂={y_obs}', **kwargs)

    def default_y_test_values(self) -> List[float]:
        return [-2.0, -1.0, 0.0, 1.0, 2.0]

    def default_hyperparams(self) -> dict:
        return {
            'n_epochs': 300,
            'batch_size': 256,
            'lr': 1e-3,
            'lam': 0.01,
            'n_steps': 20,
            'disc_updates': 5,
            'lip_scale': 10.0,
            'vel_hidden': 128,
            'vel_layers': 3,
        }
