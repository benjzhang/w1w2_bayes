"""Base class for conditional flow models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import torch


class BaseFlow(ABC):
    """Abstract base class for conditional flow models.

    All flow implementations should inherit from this class.
    """

    @abstractmethod
    def train(
        self,
        theta_data: torch.Tensor,
        y_data: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the flow model.

        Args:
            theta_data: Training samples, shape (n, theta_dim)
            y_data: Conditioning values, shape (n, y_dim)
            **kwargs: Additional training arguments

        Returns:
            Training history dictionary
        """
        pass

    @abstractmethod
    def sample(
        self,
        y: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
        """Sample from the learned conditional distribution.

        Args:
            y: Conditioning value, shape (y_dim,) or (batch, y_dim)
            n_samples: Number of samples to generate

        Returns:
            Samples, shape (n_samples, theta_dim)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk.

        Args:
            path: Path to save directory or file
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load the model from disk.

        Args:
            path: Path to saved model
        """
        pass

    @property
    @abstractmethod
    def theta_dim(self) -> int:
        """Dimension of θ."""
        pass

    @property
    @abstractmethod
    def y_dim(self) -> int:
        """Dimension of y."""
        pass
