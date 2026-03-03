"""
Conditional OT-Flow for Degenerate Bayesian Inverse Problem

Setup:
- Prior: θ = (θ₁, θ₂) ~ N(0, I)  [2D standard normal]
- Forward model: y = θ₁ + θ₂  (NO noise — exact observation)
- Posterior: p(θ|y) is DEGENERATE — supported exactly on line θ₁ + θ₂ = y

This is the hard case: the posterior is a 1D distribution embedded in 2D.
The posterior has correlation = -1 (perfect anti-correlation).

Analytical posterior (degenerate case):
    Given y, the posterior is the prior restricted to the line θ₁ + θ₂ = y.
    Parameterize line as θ = (s, y-s). Under prior:
        p(s|y) ∝ exp(-½(s² + (y-s)²)) = exp(-(s - y/2)²) × const
    So: s|y ~ N(y/2, 1/2)

    This means:
        θ₁|y ~ N(y/2, 1/2)
        θ₂|y = y - θ₁|y ~ N(y/2, 1/2)
        Corr(θ₁, θ₂|y) = -1  (exactly!)
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
# Problem Setup — Degenerate Case
# ============================================================================

def sample_prior(n):
    """Sample θ ~ N(0, I) from prior."""
    return np.random.randn(n, 2)


def forward_model_exact(theta):
    """y = θ₁ + θ₂ (no noise!)"""
    return theta[:, 0] + theta[:, 1]


def sample_posterior_degenerate(y_obs, n_samples):
    """
    Sample from degenerate posterior p(θ|y) on line θ₁ + θ₂ = y.

    The posterior is the prior restricted to the constraint line.
    Parameterize as θ = (s, y-s) where s|y ~ N(y/2, 1/2).
    """
    # s | y ~ N(y/2, 1/2)
    s = np.random.randn(n_samples) * np.sqrt(0.5) + y_obs / 2
    theta1 = s
    theta2 = y_obs - s
    return np.stack([theta1, theta2], axis=1)


def sample_joint(n):
    """
    Sample (θ, y) from joint distribution.
    θ ~ prior, y = θ₁ + θ₂ exactly.
    """
    theta = sample_prior(n)
    y = forward_model_exact(theta)
    return theta, y


# ============================================================================
# Networks — need more capacity for degenerate case
# ============================================================================

class VelocityNet(nn.Module):
    """v(t, θ; y) : [0,1] × ℝ² × ℝ → ℝ²"""
    def __init__(self, hidden=128):
        super().__init__()
        # Input: [t, θ₁, θ₂, y] = 4 dims
        self.net = nn.Sequential(
            nn.Linear(4, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t, theta, y):
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)


class Discriminator(nn.Module):
    """φ(θ; y) : ℝ² × ℝ → ℝ with spectral norm.

    lip_scale: Lipschitz constant upper bound. Default 1.0 gives standard
               spectral normalization. Larger values allow sharper gradients
               which can help concentrate mass on low-dimensional manifolds.
    """
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
    """Train conditional OT-flow for the degenerate inverse problem.

    lip_scale: Lipschitz constant for discriminator. Larger values allow
               sharper gradients to push samples onto constraint manifold.
    """

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
    print("Target: DEGENERATE posterior on line θ₁ + θ₂ = y")

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

def generate_posterior(vel_net, y_obs, n_samples, n_steps=20):
    """Generate samples from p(θ|y)."""
    vel_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y = torch.full((n_samples, 1), y_obs, device=device)
        traj = euler_integrate(vel_net, z, y, n_steps)
        theta_gen = traj[-1].cpu().numpy()
    return theta_gen


def compute_line_distance(theta, y_obs):
    """Compute distance from points to line θ₁ + θ₂ = y."""
    # Line: θ₁ + θ₂ - y = 0, normal = [1, 1]/√2
    return np.abs(theta[:, 0] + theta[:, 1] - y_obs) / np.sqrt(2)


def evaluate(vel_net, suffix=""):
    """Compare generated posteriors to analytical degenerate posteriors."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    y_test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    n_samples = 3000

    # Top row: posterior samples for different y values
    for i, y_obs in enumerate(y_test_values[:4]):
        ax = axes[0, i]

        # Analytical posterior samples (on the line!)
        theta_true = sample_posterior_degenerate(y_obs, n_samples)

        # Generated posterior samples
        theta_gen = generate_posterior(vel_net, y_obs, n_samples)

        ax.scatter(theta_true[:, 0], theta_true[:, 1], alpha=0.4, s=8, label='Analytical')
        ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.4, s=8, label='Generated')

        # Draw the constraint line
        xlim = [-3, 3]
        ax.plot(xlim, [y_obs - xlim[0], y_obs - xlim[1]], 'k-', lw=2, alpha=0.7)

        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(f'p(θ|y={y_obs})')
        ax.legend(fontsize=8)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Bottom left: y=2.0
    ax = axes[1, 0]
    y_obs = 2.0
    theta_true = sample_posterior_degenerate(y_obs, n_samples)
    theta_gen = generate_posterior(vel_net, y_obs, n_samples)
    ax.scatter(theta_true[:, 0], theta_true[:, 1], alpha=0.4, s=8, label='Analytical')
    ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.4, s=8, label='Generated')
    xlim = [-3, 3]
    ax.plot(xlim, [y_obs - xlim[0], y_obs - xlim[1]], 'k-', lw=2, alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_title(f'p(θ|y={y_obs})')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Bottom middle: distance to line histogram
    ax = axes[1, 1]
    y_obs = 0.0
    theta_gen = generate_posterior(vel_net, y_obs, 5000)
    distances = compute_line_distance(theta_gen, y_obs)
    ax.hist(distances, bins=50, alpha=0.7, density=True, edgecolor='black')
    ax.axvline(np.mean(distances), color='r', linestyle='--', label=f'Mean={np.mean(distances):.3f}')
    ax.axvline(np.median(distances), color='g', linestyle='--', label=f'Median={np.median(distances):.3f}')
    ax.set_xlabel('Distance to line θ₁+θ₂=y')
    ax.set_ylabel('Density')
    ax.set_title('How close to constraint? (y=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right-middle: correlation check
    ax = axes[1, 2]
    correlations = []
    y_range = np.linspace(-2, 2, 20)
    for y_val in y_range:
        theta_gen = generate_posterior(vel_net, y_val, 1000)
        if theta_gen.std(axis=0).min() > 1e-6:  # avoid div by zero
            corr = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]
            correlations.append(corr)
        else:
            correlations.append(-1.0)

    ax.plot(y_range, correlations, 'bo-', markersize=4)
    ax.axhline(-1.0, color='r', linestyle='--', label='Target (ρ=-1)')
    ax.set_xlabel('y')
    ax.set_ylabel('Correlation(θ₁, θ₂|y)')
    ax.set_title('Posterior Correlation vs y')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-1.1, 0])

    # Bottom far right: statistics
    ax = axes[1, 3]
    y_obs = 0.0
    theta_gen = generate_posterior(vel_net, y_obs, 5000)
    mean_gen = theta_gen.mean(axis=0)
    std_gen = theta_gen.std(axis=0)
    corr_gen = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]
    dist_mean = np.mean(compute_line_distance(theta_gen, y_obs))
    dist_std = np.std(compute_line_distance(theta_gen, y_obs))

    # Analytical values
    # θ₁|y ~ N(y/2, 1/2), so for y=0: mean=0, std=√0.5≈0.707

    text = f"y = {y_obs}\n\n"
    text += "Analytical (degenerate):\n"
    text += f"  mean = [0.000, 0.000]\n"
    text += f"  std  = [0.707, 0.707]\n"
    text += f"  corr = -1.000\n"
    text += f"  dist to line = 0.000\n\n"
    text += "Generated:\n"
    text += f"  mean = [{mean_gen[0]:.3f}, {mean_gen[1]:.3f}]\n"
    text += f"  std  = [{std_gen[0]:.3f}, {std_gen[1]:.3f}]\n"
    text += f"  corr = {corr_gen:.3f}\n"
    text += f"  dist to line = {dist_mean:.3f} ± {dist_std:.3f}"

    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Posterior Statistics')

    plt.tight_layout()
    fname = f'results_degenerate{suffix}.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys

    # Allow lip_scale and lam to be passed as command line arguments
    # Usage: python script.py [lip_scale] [lam]
    lip_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    print("=" * 60)
    print("Conditional OT-Flow: DEGENERATE Bayesian Inverse Problem")
    print("Prior: θ ~ N(0, I)")
    print("Forward: y = θ₁ + θ₂  (exact, no noise!)")
    print("Posterior: supported on LINE θ₁ + θ₂ = y")
    print("Target correlation: -1.0 (perfect anti-correlation)")
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
    # Include both lip and lam in filename
    suffix = f"_lip{int(lip_scale)}" if lam == 0.05 else f"_lip{int(lip_scale)}_lam{lam}"
    evaluate(vel_net, suffix=suffix)

    # Print validation
    print("\n" + "=" * 60)
    print("Final validation:")
    for y_test in [0.0, 1.0]:
        theta_gen = generate_posterior(vel_net, y_test, 5000)
        dist = compute_line_distance(theta_gen, y_test)
        corr = np.corrcoef(theta_gen[:, 0], theta_gen[:, 1])[0, 1]
        print(f"  y={y_test}: mean dist to line = {np.mean(dist):.4f}, corr = {corr:.4f}")
