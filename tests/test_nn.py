"""Tests for neural network components."""

import pytest
import torch
from nn import VelocityNet, Discriminator


class TestVelocityNet:
    def test_output_shape(self):
        net = VelocityNet(theta_dim=2, y_dim=1, hidden=64, n_layers=3)
        batch_size = 32

        t = torch.rand(batch_size, 1)
        theta = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)

        v = net(t, theta, y)
        assert v.shape == (batch_size, 2)

    def test_higher_dimensions(self):
        net = VelocityNet(theta_dim=5, y_dim=3, hidden=64, n_layers=3)
        batch_size = 16

        t = torch.rand(batch_size, 1)
        theta = torch.randn(batch_size, 5)
        y = torch.randn(batch_size, 3)

        v = net(t, theta, y)
        assert v.shape == (batch_size, 5)

    def test_different_activations(self):
        for act in ['silu', 'tanh', 'relu']:
            net = VelocityNet(theta_dim=2, y_dim=1, hidden=32, n_layers=2, activation=act)
            t = torch.rand(4, 1)
            theta = torch.randn(4, 2)
            y = torch.randn(4, 1)
            v = net(t, theta, y)
            assert v.shape == (4, 2)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            VelocityNet(theta_dim=2, y_dim=1, activation='invalid')

    def test_gradient_flow(self):
        net = VelocityNet(theta_dim=2, y_dim=1, hidden=32, n_layers=2)
        t = torch.rand(4, 1, requires_grad=True)
        theta = torch.randn(4, 2, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)

        v = net(t, theta, y)
        loss = v.sum()
        loss.backward()

        assert t.grad is not None
        assert theta.grad is not None
        assert y.grad is not None


class TestDiscriminator:
    def test_output_shape(self):
        disc = Discriminator(theta_dim=2, y_dim=1, hidden=64)
        batch_size = 32

        theta = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)

        phi = disc(theta, y)
        assert phi.shape == (batch_size, 1)

    def test_lip_scale(self):
        disc1 = Discriminator(theta_dim=2, y_dim=1, hidden=32, lip_scale=1.0)
        disc10 = Discriminator(theta_dim=2, y_dim=1, hidden=32, lip_scale=10.0)

        # Copy weights
        disc10.load_state_dict(disc1.state_dict())

        theta = torch.randn(4, 2)
        y = torch.randn(4, 1)

        phi1 = disc1(theta, y)
        phi10 = disc10(theta, y)

        # Output should scale by lip_scale ratio
        torch.testing.assert_close(phi10, phi1 * 10.0)

    def test_quadratic_features(self):
        disc = Discriminator(theta_dim=2, y_dim=1, hidden=32, use_quadratic_features=True)
        batch_size = 8

        theta = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)

        phi = disc(theta, y)
        assert phi.shape == (batch_size, 1)

    def test_quadratic_features_higher_dim(self):
        # For theta_dim=3, should have 3 + 6 = 9 quad features (3 squares + 3 cross products)
        disc = Discriminator(theta_dim=3, y_dim=2, hidden=32, use_quadratic_features=True)
        theta = torch.randn(4, 3)
        y = torch.randn(4, 2)

        phi = disc(theta, y)
        assert phi.shape == (4, 1)

    def test_spectral_norm_applied(self):
        disc = Discriminator(theta_dim=2, y_dim=1, hidden=32)
        # Check that spectral norm hooks are present
        has_spectral_norm = False
        for module in disc.modules():
            if hasattr(module, 'weight_orig'):
                has_spectral_norm = True
                break
        assert has_spectral_norm
