"""Circle constraint problem: y = θ₁² + θ₂²."""

import numpy as np
from typing import Tuple, List, Optional
from .base import BaseProblem


class CircleProblem(BaseProblem):
    """Degenerate inverse problem with circular constraint.

    Prior: θ ~ N(0, I) in ℝ²
    Forward: y = θ₁² + θ₂² (radius squared)
    Posterior: UNIFORM on circle of radius √y

    Note: y ~ χ²(2), so y > 0 always.
    """

    @property
    def name(self) -> str:
        return "circle"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def y_dim(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Circle: y = θ₁² + θ₂², posterior uniform on circle"

    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.randn(n, 2)
        y = theta[:, 0]**2 + theta[:, 1]**2
        return theta, y

    def compute_distance(self, theta: np.ndarray, y_obs: float) -> np.ndarray:
        """Distance to circle of radius √y."""
        r_true = np.sqrt(y_obs)
        r_points = np.sqrt(theta[:, 0]**2 + theta[:, 1]**2)
        return np.abs(r_points - r_true)

    def sample_true_posterior(self, y_obs: float, n_samples: int) -> np.ndarray:
        """Sample from true posterior: uniform on circle of radius √y."""
        r = np.sqrt(y_obs)
        angles = np.random.uniform(0, 2*np.pi, n_samples)
        theta1 = r * np.cos(angles)
        theta2 = r * np.sin(angles)
        return np.stack([theta1, theta2], axis=1)

    def true_posterior_pdf(self, grid: np.ndarray, y_obs: float, dim: int = 0) -> Optional[np.ndarray]:
        """Marginal p(θ₁|y) for circle: arcsine-like distribution."""
        r = np.sqrt(y_obs)
        # p(θ₁|y) ∝ 1/√(r² - θ₁²) for |θ₁| < r
        pdf = np.zeros_like(grid)
        valid = np.abs(grid) < r
        pdf[valid] = 1.0 / np.sqrt(r**2 - grid[valid]**2)
        # Normalize
        _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        Z = _trapz(pdf, grid)
        return pdf / Z if Z > 0 else pdf

    def plot_true_posterior(self, ax, y_obs: float, **kwargs):
        """Plot the constraint circle."""
        r = np.sqrt(y_obs)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta), r * np.sin(theta), 'r-',
                linewidth=2, label=f'r={r:.2f}', **kwargs)

    def default_y_test_values(self) -> List[float]:
        return [0.25, 0.5, 1.0, 2.0, 4.0]

    def default_hyperparams(self) -> dict:
        return {
            'n_epochs': 300,
            'batch_size': 256,
            'lr': 1e-3,
            'lam': 0.05,
            'n_steps': 20,
            'disc_updates': 5,
            'lip_scale': 10.0,
            'vel_hidden': 128,
            'vel_layers': 3,
            'use_quadratic_features': False,
        }
