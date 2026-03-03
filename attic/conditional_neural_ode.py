"""
Conditional Neural ODE (Continuous Normalizing Flow) for Degenerate Posterior

Uses ODE-based flow trained with MLE (not Wasserstein like OT-Flow).
Computes log-det-Jacobian via trace estimator (Hutchinson).

Problem:
- Prior: θ ~ N(0, I) in ℝ²
- Forward: y = θ₁ + θ₂ (exact, no noise)
- Posterior: supported on LINE θ₁ + θ₂ = y

Key difference from RealNVP: continuous transformation, no coupling structure.
Key difference from OT-Flow: MLE objective, not Wasserstein.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Target Distribution
# ============================================================================

def sample_joint(n):
    """Sample from joint: θ ~ N(0, I), y = θ₁ + θ₂."""
    theta = np.random.randn(n, 2)
    y = theta[:, 0] + theta[:, 1]
    return theta, y


def compute_line_distance(theta, y_obs):
    """Compute distance from points to line θ₁ + θ₂ = y."""
    return np.abs(theta[:, 0] + theta[:, 1] - y_obs) / np.sqrt(2)


# ============================================================================
# Neural ODE Components
# ============================================================================

class ODEFunc(nn.Module):
    """
    Velocity field v(t, x; y) for the ODE dx/dt = v(t, x; y).
    Conditioned on y.
    """
    def __init__(self, hidden=128):
        super().__init__()
        # Input: [t, x1, x2, y] = 4 dims
        # Output: [dx1/dt, dx2/dt] = 2 dims
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t, x, y):
        """
        Args:
            t: scalar or (batch,) tensor
            x: (batch, 2) positions
            y: (batch, 1) conditioning
        Returns:
            dx/dt: (batch, 2)
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1:
            t = t.unsqueeze(1)

        inp = torch.cat([t, x, y], dim=1)
        return self.net(inp)


def euler_solve(odefunc, x0, y, n_steps=20):
    """Simple Euler solver for ODE."""
    dt = 1.0 / n_steps
    x = x0
    traj = [x]
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        dx = odefunc(t, x, y)
        x = x + dt * dx
        traj.append(x)
    return x, traj


def euler_solve_with_trace(odefunc, x0, y, n_steps=20):
    """
    Euler solver that also computes log|det J| via Hutchinson trace estimator.

    For CNF: d(log|det J|)/dt = tr(∂v/∂x)
    Use Hutchinson: tr(A) ≈ E[ε^T A ε] where ε ~ N(0, I)
    """
    dt = 1.0 / n_steps
    x = x0.clone().requires_grad_(True)
    log_det = torch.zeros(x.shape[0], device=x.device)

    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)

        # Compute velocity
        v = odefunc(t, x, y)

        # Hutchinson trace estimator for divergence
        # tr(∂v/∂x) = E[ε^T (∂v/∂x) ε]
        eps = torch.randn_like(x)

        # Compute ε^T (∂v/∂x) via vector-Jacobian product
        vjp = torch.autograd.grad(
            outputs=v,
            inputs=x,
            grad_outputs=eps,
            create_graph=True,
            retain_graph=True
        )[0]

        # tr(∂v/∂x) ≈ ε^T vjp = sum(eps * vjp)
        trace_est = (eps * vjp).sum(dim=1)

        # Update log determinant
        log_det = log_det + dt * trace_est

        # Euler step (detach to avoid backprop through trajectory)
        x = x + dt * v.detach()
        x = x.detach().requires_grad_(True)

    return x, log_det


# ============================================================================
# Conditional Neural ODE Flow
# ============================================================================

class ConditionalCNF(nn.Module):
    """Conditional Continuous Normalizing Flow."""

    def __init__(self, hidden=128, n_steps=20):
        super().__init__()
        self.odefunc = ODEFunc(hidden=hidden)
        self.n_steps = n_steps

    def forward(self, z, y):
        """Transform z → x (generation direction)."""
        x, _ = euler_solve(self.odefunc, z, y, self.n_steps)
        return x

    def inverse_with_logdet(self, x, y):
        """
        Transform x → z and compute log|det J|.
        For training: we need p(x|y) = p(z) * |det dz/dx|
        """
        # Run ODE backwards: integrate from t=1 to t=0
        # Equivalently: use negative velocity
        return self._backward_solve_with_trace(x, y)

    def _backward_solve_with_trace(self, x1, y):
        """Backward ODE solve with trace for log-det."""
        dt = 1.0 / self.n_steps
        x = x1.clone().requires_grad_(True)
        log_det = torch.zeros(x.shape[0], device=x.device)

        for i in range(self.n_steps):
            # t goes from 1 to 0
            t_val = 1.0 - i * dt
            t = torch.full((x.shape[0],), t_val, device=x.device)

            v = self.odefunc(t, x, y)

            # Hutchinson trace estimator
            eps = torch.randn_like(x)
            vjp = torch.autograd.grad(
                outputs=v,
                inputs=x,
                grad_outputs=eps,
                create_graph=True,
                retain_graph=True
            )[0]
            trace_est = (eps * vjp).sum(dim=1)

            # Negative dt because going backward
            log_det = log_det - dt * trace_est

            # Backward Euler step (negative velocity)
            x = x - dt * v.detach()
            x = x.detach().requires_grad_(True)

        return x, log_det

    def log_prob(self, x, y):
        """Compute log p(x|y)."""
        z, log_det_inv = self.inverse_with_logdet(x, y)

        # Base distribution: N(0, I)
        log_pz = -0.5 * (z**2).sum(dim=1) - np.log(2 * np.pi)

        # log p(x) = log p(z) + log|det dz/dx|
        return log_pz + log_det_inv


# ============================================================================
# Training
# ============================================================================

def compute_kinetic_energy(odefunc, z0, y, n_steps=20):
    """Compute kinetic energy along trajectory."""
    dt = 1.0 / n_steps
    x = z0
    ke = 0.0

    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        v = odefunc(t, x, y)
        ke += 0.5 * (v**2).sum(dim=1).mean() * dt
        x = x + dt * v.detach()

    return ke


def train(n_epochs=300, batch_size=256, lr=1e-3, hidden=128, n_steps=20, lam=0.1):
    """Train conditional CNF with maximum likelihood + KE regularization."""

    n_train = 10000
    theta_np, y_np = sample_joint(n_train)

    theta_train = torch.FloatTensor(theta_np).to(device)
    y_train = torch.FloatTensor(y_np).unsqueeze(1).to(device)

    dataset = TensorDataset(theta_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    flow = ConditionalCNF(hidden=hidden, n_steps=n_steps).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=lr)

    history = {'nll': [], 'ke': []}

    print(f"Training Conditional CNF (Neural ODE): {n_epochs} epochs")
    print(f"Architecture: hidden={hidden}, n_steps={n_steps}, λ_KE={lam}")
    print("Target: DEGENERATE posterior on line θ₁ + θ₂ = y")

    for epoch in range(n_epochs):
        epoch_nll = []
        epoch_ke = []

        for theta_data, y_data in loader:
            # Sample z from base distribution
            z = torch.randn_like(theta_data)

            # Compute kinetic energy along forward trajectory
            ke = compute_kinetic_energy(flow.odefunc, z, y_data, n_steps)

            # Compute log probability
            log_prob = flow.log_prob(theta_data, y_data)
            nll = -log_prob.mean()

            # Total loss: NLL + λ * KE
            loss = nll + lam * ke

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=10.0)
            optimizer.step()

            epoch_nll.append(nll.item())
            epoch_ke.append(ke.item())

        history['nll'].append(np.mean(epoch_nll))
        history['ke'].append(np.mean(epoch_ke))

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1}/{n_epochs}] NLL={history['nll'][-1]:.4f}, KE={history['ke'][-1]:.4f}")

    return flow, history


# ============================================================================
# Generation & Evaluation
# ============================================================================

def generate_posterior(flow, y_obs, n_samples=1000):
    flow.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y = torch.full((n_samples, 1), y_obs, device=device)
        theta = flow.forward(z, y)
        return theta.cpu().numpy()


def evaluate(flow, history):
    """Compare generated posteriors to true degenerate posteriors."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    y_test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    n_samples = 3000

    # Top row: posterior samples
    for i, y_val in enumerate(y_test_values[:4]):
        ax = axes[0, i]

        # True posterior: line θ₁ + θ₂ = y
        t_line = np.linspace(-3, 3, 100)
        ax.plot(t_line, y_val - t_line, 'r-', linewidth=2, label='True (line)', alpha=0.7)

        # Generated
        theta_gen = generate_posterior(flow, y_val, n_samples)
        ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='CNF')

        # Metrics
        dist = compute_line_distance(theta_gen, y_val)
        corr = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(f'y={y_val}: dist={dist.mean():.3f}, ρ={corr:.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Bottom left: y=2 posterior
    ax = axes[1, 0]
    y_val = 2.0
    t_line = np.linspace(-3, 3, 100)
    ax.plot(t_line, y_val - t_line, 'r-', linewidth=2, label='True', alpha=0.7)
    theta_gen = generate_posterior(flow, y_val, n_samples)
    ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='CNF')
    dist = compute_line_distance(theta_gen, y_val)
    corr = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title(f'y={y_val}: dist={dist.mean():.3f}, ρ={corr:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Training curve
    ax = axes[1, 1]
    ax.plot(history['nll'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NLL')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)

    # Distance histogram for y=0
    ax = axes[1, 2]
    theta_gen = generate_posterior(flow, 0.0, 5000)
    dist = compute_line_distance(theta_gen, 0.0)
    ax.hist(dist, bins=50, density=True, alpha=0.7)
    ax.axvline(dist.mean(), color='r', linestyle='--', label=f'mean={dist.mean():.3f}')
    ax.set_xlabel('Distance to line')
    ax.set_ylabel('Density')
    ax.set_title('Dist to θ₁+θ₂=0 (y=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics summary
    ax = axes[1, 3]
    text = "Conditional CNF (Neural ODE)\n"
    text += "=" * 30 + "\n\n"
    text += "Problem: θ₁ + θ₂ = y (degenerate)\n"
    text += "Target: ρ = -1.0, dist = 0\n\n"

    for y_val in [0.0, 1.0]:
        theta_gen = generate_posterior(flow, y_val, 5000)
        dist = compute_line_distance(theta_gen, y_val)
        corr = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]
        text += f"y={y_val}:\n"
        text += f"  ρ = {corr:.4f}\n"
        text += f"  dist = {dist.mean():.4f}\n\n"

    text += "ODE-based, no coupling structure"

    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Results Summary')

    plt.tight_layout()
    plt.savefig('results_cnf_ke.png', dpi=150)
    print("Saved: results_cnf_ke.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Conditional Neural ODE (CNF): DEGENERATE Inverse Problem")
    print("Prior: θ ~ N(0, I)")
    print("Forward: y = θ₁ + θ₂  (exact, no noise!)")
    print("Posterior: supported on LINE θ₁ + θ₂ = y")
    print("")
    print("CNF = Continuous Normalizing Flow (ODE-based)")
    print("Trained with MLE (like NF), but continuous transformation")
    print("=" * 60)

    import sys
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1

    print(f"KE regularization: λ = {lam}")

    flow, history = train(
        n_epochs=300,
        batch_size=256,
        lr=1e-3,
        hidden=128,
        n_steps=20,
        lam=lam
    )

    print("\nGenerating evaluation plots...")
    evaluate(flow, history)

    # Final validation
    print("\n" + "=" * 60)
    print("Final validation:")
    for y_test in [0.0, 1.0]:
        theta_gen = generate_posterior(flow, y_test, 5000)
        dist = compute_line_distance(theta_gen, y_test)
        corr = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]
        print(f"  y={y_test}: mean dist to line = {np.mean(dist):.4f}, corr = {corr:.4f}")

    print("\nCompare with:")
    print("  RealNVP: dist ≈ 0.005, corr ≈ -1.000")
    print("  OT-Flow (Lip=10, λ=0.01): dist ≈ 0.008, corr ≈ -0.9999")
