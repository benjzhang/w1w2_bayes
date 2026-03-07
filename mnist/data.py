"""MNIST bottom-half inpainting data module.

y = top half of 28x28 image (rows 0-13, flattened to 392 dims)
theta = bottom half (rows 14-27, flattened to 392 dims)
Pixel values normalized to [0, 1].
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


class MNISTInpainting:
    """MNIST dataset split into top-half (y) and bottom-half (theta).

    Args:
        train: If True, use training set; otherwise test set.
        data_root: Directory for downloading/caching MNIST.
    """

    def __init__(self, train: bool = True, data_root: str = './data'):
        from torchvision import datasets, transforms

        dataset = datasets.MNIST(
            root=data_root, train=train, download=True,
            transform=transforms.ToTensor()  # gives [0, 1] range, shape (1, 28, 28)
        )

        # Stack all images into a single tensor: (N, 1, 28, 28)
        all_images = torch.stack([img for img, _ in dataset])  # (N, 1, 28, 28)
        all_images = all_images.squeeze(1)  # (N, 28, 28)

        # Split into top half (y) and bottom half (theta)
        self.y_data = all_images[:, :14, :].reshape(-1, 392)   # (N, 392)
        self.theta_data = all_images[:, 14:, :].reshape(-1, 392)  # (N, 392)
        self.n_samples = self.y_data.shape[0]

        # Store labels for potential later use
        self.labels = torch.tensor([label for _, label in dataset])

    def sample_joint(self, n: int):
        """Sample n joint (theta, y) pairs.

        Args:
            n: Number of samples.

        Returns:
            Tuple of numpy arrays (theta, y), each shape (n, 392).
        """
        idx = np.random.randint(0, self.n_samples, size=n)
        theta = self.theta_data[idx].numpy()
        y = self.y_data[idx].numpy()
        return theta, y

    def get_test_images(self, n: int = 10):
        """Get n test images with top (y) and true bottom (theta).

        Args:
            n: Number of test images to return.

        Returns:
            Tuple of tensors (y, theta_true), each shape (n, 392).
        """
        idx = np.arange(min(n, self.n_samples))
        return self.y_data[idx], self.theta_data[idx]

    def get_tensors(self):
        """Return full dataset as tensors.

        Returns:
            Tuple (theta_data, y_data), each shape (N, 392).
        """
        return self.theta_data, self.y_data

    def get_dataloader(self, batch_size: int = 256, shuffle: bool = True):
        """Create a DataLoader over (theta, y) pairs.

        Args:
            batch_size: Batch size.
            shuffle: Whether to shuffle.

        Returns:
            DataLoader yielding (theta, y) batches.
        """
        dataset = TensorDataset(self.theta_data, self.y_data)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
