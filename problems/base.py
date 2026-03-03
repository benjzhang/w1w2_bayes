"""Base class for inverse problems."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy as np


class BaseProblem(ABC):
    """Base class for Bayesian inverse problems.

    Defines:
    - Prior: p(θ)
    - Forward model: y = f(θ) + noise
    - Posterior: p(θ|y)

    For degenerate problems, the posterior is supported on a manifold.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name for the problem."""
        pass

    @property
    @abstractmethod
    def theta_dim(self) -> int:
        """Dimension of the parameter θ."""
        pass

    @property
    @abstractmethod
    def y_dim(self) -> int:
        """Dimension of the observation y."""
        pass

    @property
    def description(self) -> str:
        """Human-readable description of the problem."""
        return f"{self.name}: θ ∈ ℝ^{self.theta_dim}, y ∈ ℝ^{self.y_dim}"

    @abstractmethod
    def sample_joint(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from joint distribution p(θ, y).

        Returns:
            theta: (n, theta_dim) samples from prior
            y: (n, y_dim) corresponding observations
        """
        pass

    @abstractmethod
    def compute_distance(self, theta: np.ndarray, y_obs: float) -> np.ndarray:
        """Compute distance from samples to true posterior manifold.

        Args:
            theta: (n, theta_dim) samples
            y_obs: observation value (scalar for y_dim=1)

        Returns:
            distances: (n,) distance to constraint manifold
        """
        pass

    def true_posterior_pdf(self, grid: np.ndarray, y_obs: float, dim: int = 0) -> Optional[np.ndarray]:
        """Compute true marginal posterior PDF (if available analytically).

        Args:
            grid: points at which to evaluate PDF
            y_obs: observation value
            dim: which dimension's marginal to compute

        Returns:
            pdf values at grid points, or None if not available
        """
        return None

    def plot_true_posterior(self, ax, y_obs: float, **kwargs):
        """Plot the true posterior manifold on given axes.

        Should be overridden by subclasses for problem-specific visualization.
        """
        pass

    def default_y_test_values(self) -> List[float]:
        """Default y values to use for evaluation."""
        return [0.0, 1.0, 2.0]

    def default_hyperparams(self) -> dict:
        """Default hyperparameters for training on this problem."""
        return {
            'n_epochs': 300,
            'batch_size': 256,
            'lr': 1e-3,
            'lam': 0.01,
            'n_steps': 40,
            'disc_updates': 5,
            'lip_scale': 10.0,
        }
