"""
Conditional OT-Flow for Circular Constraint

Degenerate inverse problem where posterior lies on a CIRCLE:
- Prior: θ ~ N(0, I) in ℝ²
- Forward: y = θ₁² + θ₂² (radius squared)
- Posterior: UNIFORM on circle of radius √y

Uses Wasserstein-1 dual + kinetic energy regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import sys

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Target Distribution
# ============================================================================

def sample_joint(n):
    """
    Sample from joint: θ ~ N(0, I), y = θ₁² + θ₂².
    Note: y ~ χ²(2), so y > 0 always.
    """
    theta = np.random.randn(n, 2)
    y = theta[:, 0]**2 + theta[:, 1]**2
    return theta, y


def sample_true_posterior(y_obs, n_samples):
    """Sample from true posterior: uniform on circle of radius √y."""
    r = np.sqrt(y_obs)
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    theta1 = r * np.cos(angles)
    theta2 = r * np.sin(angles)
    return np.stack([theta1, theta2], axis=1)


def compute_circle_distance(theta, y_obs):
    """Compute distance from points to circle θ₁² + θ₂² = y."""
    r_true = np.sqrt(y_obs)
    r_points = np.sqrt(theta[:, 0]**2 + theta[:, 1]**2)
    return np.abs(r_points - r_true)


# ============================================================================
# Networks
# ============================================================================

class VelocityNet(nn.Module):
    """v(t, θ; y) : [0,1] × ℝ² × ℝ → ℝ²"""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden),  # [t, θ₁, θ₂, y]
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)  # velocity in ℝ²
        )

    def forward(self, t, theta, y):
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)


class Discriminator(nn.Module):
    """φ(θ; y) : ℝ² × ℝ → ℝ with spectral norm."""
    def __init__(self, hidden=128, lip_scale=1.0):
        super().__init__()
        self.lip_scale = lip_scale
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(3, hidden)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, theta, y):
        inp = torch.cat([theta, y], dim=1)
        return self.lip_scale * self.net(inp)


# ============================================================================
# ODE Integration
# ============================================================================

def euler_integrate(vel_net, z0, y, n_steps=20):
    """Integrate dθ/dt = v(t, θ; y) from t=0 to t=1."""
    dt = 1.0 / n_steps
    traj = [z0]
    theta = z0
    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        theta = theta + dt * v
        traj.append(theta)
    return traj


# ============================================================================
# Training
# ============================================================================

def train(n_epochs=300, batch_size=256, lr=1e-3, lam=0.05, n_steps=20, disc_updates=5, lip_scale=1.0):
    """Train conditional OT-flow for the circular constraint."""

    n_train = 10000
    theta_np, y_np = sample_joint(n_train)

    theta_train = torch.FloatTensor(theta_np).to(device)
    y_train = torch.FloatTensor(y_np).unsqueeze(1).to(device)

    dataset = TensorDataset(theta_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vel_net = VelocityNet(hidden=128).to(device)
    disc = Discriminator(hidden=128, lip_scale=lip_scale).to(device)

    opt_vel = optim.Adam(vel_net.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)

    history = {'L_dual': [], 'KE': []}

    print(f"Training: {n_epochs} epochs, λ={lam}, disc_updates={disc_updates}, Lip={lip_scale}")
    print("Target: DEGENERATE posterior on CIRCLE θ₁² + θ₂² = y")

    for epoch in range(n_epochs):
        epoch_dual, epoch_ke = [], []

        for theta_data, y_data in loader:
            bs = theta_data.shape[0]
            z = torch.randn(bs, 2, device=device)

            # --- Discriminator updates ---
            for _ in range(disc_updates):
                traj = euler_integrate(vel_net, z, y_data, n_steps)
                theta_gen = traj[-1].detach()

                phi_gen = disc(theta_gen, y_data)
                phi_real = disc(theta_data, y_data)
                L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

                opt_disc.zero_grad()
                (-L_dual).backward()
                opt_disc.step()

            # --- Velocity update ---
            traj = euler_integrate(vel_net, z, y_data, n_steps)
            theta_gen = traj[-1]

            phi_gen = disc(theta_gen, y_data)
            phi_real = disc(theta_data, y_data)
            L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

            # Kinetic energy
            KE = 0.0
            dt = 1.0 / n_steps
            for i, theta_t in enumerate(traj[:-1]):
                t = torch.full((theta_t.shape[0], 1), i * dt, device=device)
                v = vel_net(t, theta_t, y_data)
                KE += 0.5 * (v**2).sum(dim=1).mean() * dt

            loss = L_dual + lam * KE
            opt_vel.zero_grad()
            loss.backward()
            opt_vel.step()

            epoch_dual.append(L_dual.item())
            epoch_ke.append(KE.item())

        history['L_dual'].append(np.mean(epoch_dual))
        history['KE'].append(np.mean(epoch_ke))

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1}/{n_epochs}] L_dual={history['L_dual'][-1]:.4f}, KE={history['KE'][-1]:.4f}")

    return vel_net, disc, history


# ============================================================================
# Generation & Evaluation
# ============================================================================

def generate_posterior(vel_net, y_obs, n_samples=1000, n_steps=20):
    """Sample from learned p(θ|y)."""
    vel_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y = torch.full((n_samples, 1), y_obs, device=device)
        traj = euler_integrate(vel_net, z, y, n_steps)
        return traj[-1].cpu().numpy()


def evaluate(vel_net, history, suffix=""):
    """Compare generated posteriors to true circular posteriors."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    y_test_values = [0.5, 1.0, 2.0, 4.0]
    n_samples = 3000

    # Top row: posterior samples
    for i, y_val in enumerate(y_test_values):
        ax = axes[0, i]

        # True posterior: circle of radius √y
        r = np.sqrt(y_val)
        theta_circle = np.linspace(0, 2*np.pi, 100)
        ax.plot(r * np.cos(theta_circle), r * np.sin(theta_circle),
                'r-', linewidth=2, label=f'True (r={r:.2f})', alpha=0.7)

        # Generated
        theta_gen = generate_posterior(vel_net, y_val, n_samples)
        ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='OT-Flow')

        # Metrics
        dist = compute_circle_distance(theta_gen, y_val)

        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(f'y={y_val}: dist={dist.mean():.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    # Bottom left: Angular distribution for y=1
    ax = axes[1, 0]
    theta_gen = generate_posterior(vel_net, 1.0, 5000)
    angles = np.arctan2(theta_gen[:, 1], theta_gen[:, 0])
    ax.hist(angles, bins=50, density=True, alpha=0.7, label='OT-Flow')
    ax.axhline(1/(2*np.pi), color='r', linestyle='--', label='Uniform')
    ax.set_xlabel('Angle (rad)')
    ax.set_ylabel('Density')
    ax.set_title('Angular Distribution (y=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training curves
    ax = axes[1, 1]
    ax.plot(history['L_dual'], label='L_dual')
    ax.plot(history['KE'], label='KE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distance histogram for y=1
    ax = axes[1, 2]
    theta_gen = generate_posterior(vel_net, 1.0, 5000)
    dist = compute_circle_distance(theta_gen, 1.0)
    ax.hist(dist, bins=50, density=True, alpha=0.7)
    ax.axvline(dist.mean(), color='r', linestyle='--', label=f'mean={dist.mean():.3f}')
    ax.set_xlabel('Distance to circle')
    ax.set_ylabel('Density')
    ax.set_title('Dist to circle (y=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics summary
    ax = axes[1, 3]
    text = "OT-Flow (Wasserstein + KE)\n"
    text += "=" * 30 + "\n\n"
    text += "Problem: θ₁² + θ₂² = y (circle)\n"
    text += "Target: uniform on circle\n\n"

    for y_val in [1.0, 2.0, 4.0]:
        theta_gen = generate_posterior(vel_net, y_val, 5000)
        dist = compute_circle_distance(theta_gen, y_val)
        r_gen = np.sqrt(theta_gen[:, 0]**2 + theta_gen[:, 1]**2)
        text += f"y={y_val} (r={np.sqrt(y_val):.1f}):\n"
        text += f"  dist = {dist.mean():.4f}\n"
        text += f"  r_mean = {r_gen.mean():.3f}\n\n"

    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Results Summary')

    plt.tight_layout()
    fname = f'results_circle{suffix}.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    lip_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    print("=" * 60)
    print("Conditional OT-Flow: CIRCULAR Constraint")
    print("Prior: θ ~ N(0, I)")
    print("Forward: y = θ₁² + θ₂²")
    print("Posterior: UNIFORM on circle of radius √y")
    print(f"Lipschitz constant: {lip_scale}, λ (KE penalty): {lam}")
    print("=" * 60)

    vel_net, disc, history = train(
        n_epochs=300,
        batch_size=256,
        lr=1e-3,
        lam=lam,
        n_steps=20,
        disc_updates=5,
        lip_scale=lip_scale
    )

    print("\nGenerating evaluation plots...")
    suffix = f"_lip{int(lip_scale)}_lam{lam}"
    evaluate(vel_net, history, suffix=suffix)

    # Final validation
    print("\n" + "=" * 60)
    print("Final validation:")
    for y_test in [1.0, 2.0, 4.0]:
        theta_gen = generate_posterior(vel_net, y_test, 5000)
        dist = compute_circle_distance(theta_gen, y_test)
        r_gen = np.sqrt(theta_gen[:, 0]**2 + theta_gen[:, 1]**2)
        print(f"  y={y_test} (r={np.sqrt(y_test):.2f}): dist = {np.mean(dist):.4f}, r_mean = {r_gen.mean():.3f}")
