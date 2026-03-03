"""ODE integrators for flow models."""

import torch
from typing import List, Tuple


def euler_integrate(
    vel_net,
    z0: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 20
) -> List[torch.Tensor]:
    """Integrate dθ/dt = v(t, θ; y) from t=0 to t=1 using Euler method.

    Args:
        vel_net: Velocity network with forward(t, theta, y)
        z0: Initial positions, shape (batch, theta_dim)
        y: Conditioning values, shape (batch, y_dim)
        n_steps: Number of integration steps

    Returns:
        List of tensors [z0, z1, ..., z_n_steps], each shape (batch, theta_dim)
    """
    dt = 1.0 / n_steps
    traj = [z0]
    theta = z0

    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        theta = theta + dt * v
        traj.append(theta)

    return traj


def compute_kinetic_energy(
    vel_net,
    z0: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 20
) -> torch.Tensor:
    """Compute kinetic energy ∫₀¹ ½‖v(t,θ(t);y)‖² dt along trajectory.

    This is the regularization term in the Benamou-Brenier formulation.

    Args:
        vel_net: Velocity network with forward(t, theta, y)
        z0: Initial positions, shape (batch, theta_dim)
        y: Conditioning values, shape (batch, y_dim)
        n_steps: Number of integration steps

    Returns:
        Scalar kinetic energy (averaged over batch)
    """
    dt = 1.0 / n_steps
    theta = z0
    ke = 0.0

    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        ke += 0.5 * (v ** 2).sum(dim=1).mean() * dt
        theta = theta + dt * v.detach()

    return ke


def rk4_integrate(
    vel_net,
    z0: torch.Tensor,
    y: torch.Tensor,
    n_steps: int = 20
) -> List[torch.Tensor]:
    """Integrate using 4th-order Runge-Kutta (more accurate than Euler).

    Args:
        vel_net: Velocity network with forward(t, theta, y)
        z0: Initial positions, shape (batch, theta_dim)
        y: Conditioning values, shape (batch, y_dim)
        n_steps: Number of integration steps

    Returns:
        List of tensors [z0, z1, ..., z_n_steps]
    """
    dt = 1.0 / n_steps
    traj = [z0]
    theta = z0

    for i in range(n_steps):
        t_i = i * dt
        t = torch.full((theta.shape[0], 1), t_i, device=theta.device)
        t_mid = torch.full((theta.shape[0], 1), t_i + dt/2, device=theta.device)
        t_end = torch.full((theta.shape[0], 1), t_i + dt, device=theta.device)

        k1 = vel_net(t, theta, y)
        k2 = vel_net(t_mid, theta + dt/2 * k1, y)
        k3 = vel_net(t_mid, theta + dt/2 * k2, y)
        k4 = vel_net(t_end, theta + dt * k3, y)

        theta = theta + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        traj.append(theta)

    return traj
