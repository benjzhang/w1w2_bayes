"""Tests for ODE integrators."""

import pytest
import torch
from nn import VelocityNet
from utils.integrators import euler_integrate, compute_kinetic_energy, rk4_integrate


class TestEulerIntegrate:
    @pytest.fixture
    def vel_net(self):
        return VelocityNet(theta_dim=2, y_dim=1, hidden=32, n_layers=2)

    def test_trajectory_length(self, vel_net):
        z0 = torch.randn(16, 2)
        y = torch.randn(16, 1)
        n_steps = 20

        traj = euler_integrate(vel_net, z0, y, n_steps=n_steps)
        assert len(traj) == n_steps + 1  # includes initial point

    def test_trajectory_shapes(self, vel_net):
        batch_size = 16
        z0 = torch.randn(batch_size, 2)
        y = torch.randn(batch_size, 1)

        traj = euler_integrate(vel_net, z0, y, n_steps=10)
        for state in traj:
            assert state.shape == (batch_size, 2)

    def test_initial_point_preserved(self, vel_net):
        z0 = torch.randn(8, 2)
        y = torch.randn(8, 1)

        traj = euler_integrate(vel_net, z0, y, n_steps=5)
        torch.testing.assert_close(traj[0], z0)

    def test_zero_velocity_unchanged(self):
        """With zero velocity, trajectory should stay constant."""
        class ZeroVelocity(torch.nn.Module):
            def forward(self, t, theta, y):
                return torch.zeros_like(theta)

        z0 = torch.randn(4, 2)
        y = torch.randn(4, 1)

        traj = euler_integrate(ZeroVelocity(), z0, y, n_steps=10)
        for state in traj:
            torch.testing.assert_close(state, z0)


class TestComputeKineticEnergy:
    def test_zero_velocity_zero_ke(self):
        """Zero velocity should give zero KE."""
        class ZeroVelocity(torch.nn.Module):
            def forward(self, t, theta, y):
                return torch.zeros_like(theta)

        z0 = torch.randn(4, 2)
        y = torch.randn(4, 1)

        ke = compute_kinetic_energy(ZeroVelocity(), z0, y, n_steps=10)
        assert ke.item() == pytest.approx(0.0, abs=1e-6)

    def test_constant_velocity_ke(self):
        """Constant unit velocity: KE = 0.5 * ||v||^2 * T = 0.5 * 2 * 1 = 1."""
        class ConstantVelocity(torch.nn.Module):
            def forward(self, t, theta, y):
                return torch.ones_like(theta)

        z0 = torch.zeros(1, 2)
        y = torch.zeros(1, 1)

        ke = compute_kinetic_energy(ConstantVelocity(), z0, y, n_steps=100)
        # KE = 0.5 * (1^2 + 1^2) * 1 = 1.0
        assert ke.item() == pytest.approx(1.0, rel=0.05)

    def test_ke_is_nonnegative(self):
        vel_net = VelocityNet(theta_dim=2, y_dim=1, hidden=32, n_layers=2)
        z0 = torch.randn(8, 2)
        y = torch.randn(8, 1)

        ke = compute_kinetic_energy(vel_net, z0, y, n_steps=10)
        assert ke.item() >= 0


class TestRK4Integrate:
    def test_trajectory_length(self):
        vel_net = VelocityNet(theta_dim=2, y_dim=1, hidden=32, n_layers=2)
        z0 = torch.randn(8, 2)
        y = torch.randn(8, 1)
        n_steps = 20

        traj = rk4_integrate(vel_net, z0, y, n_steps=n_steps)
        assert len(traj) == n_steps + 1

    def test_more_accurate_than_euler(self):
        """RK4 should be more accurate for smooth problems."""
        # Simple ODE: dz/dt = -z, solution z(t) = z0 * exp(-t)
        class ExpDecay(torch.nn.Module):
            def forward(self, t, theta, y):
                return -theta

        z0 = torch.tensor([[1.0, 1.0]])
        y = torch.zeros(1, 1)
        n_steps = 10

        euler_traj = euler_integrate(ExpDecay(), z0.clone(), y, n_steps)
        rk4_traj = rk4_integrate(ExpDecay(), z0.clone(), y, n_steps)

        # True solution at t=1
        true_final = z0 * torch.exp(torch.tensor(-1.0))

        euler_error = (euler_traj[-1] - true_final).abs().mean()
        rk4_error = (rk4_traj[-1] - true_final).abs().mean()

        # RK4 should have smaller error
        assert rk4_error < euler_error
