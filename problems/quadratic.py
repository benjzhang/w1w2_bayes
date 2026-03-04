"""Quadratic constraint problem: y = θ₁² + θ₂."""

import numpy as np
from typing import Tuple, List, Optional
from .base import BaseProblem


class QuadraticProblem(BaseProblem):
    """Degenerate inverse problem with quadratic (parabola) constraint.

    Prior: θ ~ N(0, I) in ℝ²
    Forward: y = θ₁² + θ₂ (exact, no noise)
    Posterior: supported on PARABOLA θ₂ = y - θ₁²

    True posterior marginal:
        p(θ₁|y) ∝ exp(-θ₁²/2) exp(-(y - θ₁²)²/2)
                = exp(-θ₁⁴/2 + (y - 0.5)θ₁² - y²/2)
    """

    @property
    def name(self) -> str:
        return "quadratic"

    @property
    def theta_dim(self) -> int:
        return 2

    @property
    def y_dim(self) -> int:
        return 1

    @property
    def description(self) -> str:
        return "Quadratic: y = θ₁² + θ₂, posterior on parabola"

    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.random.randn(n, 2)
        y = theta[:, 0]**2 + theta[:, 1]
        return theta, y

    def compute_distance(self, theta: np.ndarray, y_obs: float) -> np.ndarray:
        """Vertical distance to parabola θ₂ = y - θ₁²."""
        return np.abs(theta[:, 1] - (y_obs - theta[:, 0]**2))

    def true_posterior_pdf(self, grid: np.ndarray, y_obs: float, dim: int = 0) -> Optional[np.ndarray]:
        """True marginal: p(θ₁|y) ∝ exp(-θ₁⁴/2 + (y-0.5)θ₁² - y²/2)."""
        if dim != 0:
            return None
        log_p = -grid**4/2 + (y_obs - 0.5)*grid**2 - y_obs**2/2
        _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
        p = np.exp(log_p - log_p.max())
        Z = _trapz(p, grid)
        return p / Z if Z > 0 else p

    def plot_true_posterior(self, ax, y_obs: float, **kwargs):
        """Plot the constraint parabola θ₂ = y - θ₁²."""
        t1 = np.linspace(-3, 3, 100)
        t2 = y_obs - t1**2
        ax.plot(t1, t2, 'r-', linewidth=2, label=f'θ₂=y-θ₁²', **kwargs)

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
