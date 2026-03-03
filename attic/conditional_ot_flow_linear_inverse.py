"""
Conditional OT-Flow for Linear Bayesian Inverse Problem

Setup:
- Prior: θ = (θ₁, θ₂) ~ N(0, I)  [2D standard normal]
- Forward model: y = θ₁ + θ₂ + ε, where ε ~ N(0, σ²)
- Posterior: p(θ|y) is concentrated along the line θ₁ + θ₂ ≈ y

This is a classic example where the posterior has strong correlation structure.
The likelihood constrains θ to lie near a 1D manifold (a line) in 2D space.

Analytical posterior (for reference):
    p(θ|y) = N(μ_post, Σ_post)
    Σ_post = (I + A^T A / σ²)^{-1}  where A = [1, 1]
    μ_post = Σ_post A^T y / σ²
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
# Problem Setup
# ============================================================================

SIGMA_OBS = 0.3  # Observation noise std


def compute_analytical_posterior(y, sigma=SIGMA_OBS):
    """
    Compute analytical posterior parameters for the linear inverse problem.

    Prior: θ ~ N(0, I)
    Likelihood: y | θ ~ N(θ₁ + θ₂, σ²)

    Returns: (mean, covariance) of posterior
    """
    sigma2 = sigma ** 2
    # A = [1, 1], so A^T A = [[1,1],[1,1]]
    # Σ_post = (I + A^T A / σ²)^{-1}
    precision = np.array([[1 + 1/sigma2, 1/sigma2],
                          [1/sigma2, 1 + 1/sigma2]])
    cov_post = np.linalg.inv(precision)

    # μ_post = Σ_post @ A^T @ y / σ²
    A = np.array([1, 1])
    mean_post = cov_post @ A * y / sigma2

    return mean_post, cov_post


def sample_prior(n):
    """Sample θ ~ N(0, I) from prior."""
    return np.random.randn(n, 2)


def forward_model(theta, sigma=SIGMA_OBS):
    """y = θ₁ + θ₂ + ε, ε ~ N(0, σ²)"""
    y = theta[:, 0] + theta[:, 1] + np.random.randn(len(theta)) * sigma
    return y


def sample_posterior(y_obs, n_samples, sigma=SIGMA_OBS):
    """Sample from analytical posterior p(θ|y)."""
    mean, cov = compute_analytical_posterior(y_obs, sigma)
    return np.random.multivariate_normal(mean, cov, n_samples)


def sample_joint(n):
    """
    Sample (θ, y) from joint distribution.
    θ ~ prior, y ~ forward_model(θ)
    """
    theta = sample_prior(n)
    y = forward_model(theta)
    return theta, y


# ============================================================================
# Networks
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
            nn.Linear(hidden, 2)  # Output: velocity in θ-space
        )

    def forward(self, t, theta, y):
        # t: (batch, 1), theta: (batch, 2), y: (batch, 1)
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)


class Discriminator(nn.Module):
    """φ(θ; y) : ℝ² × ℝ → ℝ with spectral norm."""
    def __init__(self, hidden=128):
        super().__init__()
        # Input: [θ₁, θ₂, y] = 3 dims
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(3, hidden)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, theta, y):
        inp = torch.cat([theta, y], dim=1)
        return self.net(inp)


# ============================================================================
# ODE Integration
# ============================================================================

def euler_integrate(vel_net, z0, y, n_steps=20):
    """
    Integrate dθ/dt = v(t, θ; y) from t=0 to t=1.

    Args:
        z0: initial positions (batch, 2), sampled from N(0, I)
        y: conditioning values (batch, 1), the observed data

    Returns:
        trajectory: list of (batch, 2) tensors
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


# ============================================================================
# Training
# ============================================================================

def train(n_epochs=300, batch_size=256, lr=1e-3, lam=0.1, n_steps=20, disc_updates=3):
    """Train conditional OT-flow for the inverse problem."""

    # Generate training data: (θ, y) pairs from joint
    n_train = 10000
    theta_np, y_np = sample_joint(n_train)

    theta_train = torch.FloatTensor(theta_np).to(device)
    y_train = torch.FloatTensor(y_np).unsqueeze(1).to(device)

    dataset = TensorDataset(theta_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vel_net = VelocityNet(hidden=128).to(device)
    disc = Discriminator(hidden=128).to(device)

    opt_vel = optim.Adam(vel_net.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)

    history = {'L_dual': [], 'KE': []}

    print(f"Training: {n_epochs} epochs, λ={lam}, disc_updates={disc_updates}")
    print(f"Observation noise σ={SIGMA_OBS}")

    for epoch in range(n_epochs):
        epoch_dual, epoch_ke = [], []

        for theta_data, y_data in loader:
            bs = theta_data.shape[0]

            # Reference: z ~ N(0, I) for θ
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
    """
    Generate samples from p(θ|y) by flowing from reference.

    Args:
        y_obs: scalar observation value
        n_samples: number of posterior samples
    """
    vel_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y = torch.full((n_samples, 1), y_obs, device=device)
        traj = euler_integrate(vel_net, z, y, n_steps)
        theta_gen = traj[-1].cpu().numpy()
    return theta_gen


def evaluate(vel_net):
    """Compare generated posteriors to analytical posteriors."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    y_test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    n_samples = 3000

    # Top row: posterior samples for different y values
    for i, y_obs in enumerate(y_test_values[:4]):
        ax = axes[0, i]

        # Analytical posterior samples
        theta_true = sample_posterior(y_obs, n_samples)

        # Generated posterior samples
        theta_gen = generate_posterior(vel_net, y_obs, n_samples)

        ax.scatter(theta_true[:, 0], theta_true[:, 1], alpha=0.3, s=5, label='Analytical')
        ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='Generated')

        # Draw the constraint line θ₁ + θ₂ = y
        xlim = [-3, 3]
        ax.plot(xlim, [y_obs - xlim[0], y_obs - xlim[1]], 'k--', alpha=0.5, label=f'θ₁+θ₂={y_obs}')

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
    theta_true = sample_posterior(y_obs, n_samples)
    theta_gen = generate_posterior(vel_net, y_obs, n_samples)
    ax.scatter(theta_true[:, 0], theta_true[:, 1], alpha=0.3, s=5, label='Analytical')
    ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='Generated')
    xlim = [-3, 3]
    ax.plot(xlim, [y_obs - xlim[0], y_obs - xlim[1]], 'k--', alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_title(f'p(θ|y={y_obs})')
    ax.legend(fontsize=8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Bottom middle: marginal comparison for y=0
    ax = axes[1, 1]
    y_obs = 0.0
    theta_true = sample_posterior(y_obs, n_samples)
    theta_gen = generate_posterior(vel_net, y_obs, n_samples)
    ax.hist(theta_true[:, 0], bins=50, alpha=0.5, density=True, label='Analytical θ₁')
    ax.hist(theta_gen[:, 0], bins=50, alpha=0.5, density=True, label='Generated θ₁')
    ax.set_xlabel('θ₁')
    ax.set_title(f'Marginal p(θ₁|y={y_obs})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: joint samples from training distribution
    ax = axes[1, 2]
    theta_joint, y_joint = sample_joint(2000)
    ax.scatter(theta_joint[:, 0], theta_joint[:, 1], c=y_joint, cmap='coolwarm',
               alpha=0.5, s=10)
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.set_title('Joint samples (color=y)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('y')

    # Bottom far right: covariance check
    ax = axes[1, 3]
    y_obs = 0.0
    mean_true, cov_true = compute_analytical_posterior(y_obs)
    theta_gen = generate_posterior(vel_net, y_obs, 5000)
    mean_gen = theta_gen.mean(axis=0)
    cov_gen = np.cov(theta_gen.T)

    # Display comparison
    text = f"y = {y_obs}\n\n"
    text += "Analytical:\n"
    text += f"  mean = [{mean_true[0]:.3f}, {mean_true[1]:.3f}]\n"
    text += f"  cov = [[{cov_true[0,0]:.3f}, {cov_true[0,1]:.3f}],\n"
    text += f"         [{cov_true[1,0]:.3f}, {cov_true[1,1]:.3f}]]\n\n"
    text += "Generated:\n"
    text += f"  mean = [{mean_gen[0]:.3f}, {mean_gen[1]:.3f}]\n"
    text += f"  cov = [[{cov_gen[0,0]:.3f}, {cov_gen[0,1]:.3f}],\n"
    text += f"         [{cov_gen[1,0]:.3f}, {cov_gen[1,1]:.3f}]]"

    ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Posterior Statistics')

    plt.tight_layout()
    plt.savefig('results_linear_inverse.png', dpi=150)
    print("Saved: results_linear_inverse.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Conditional OT-Flow: Linear Bayesian Inverse Problem")
    print("Prior: θ ~ N(0, I)")
    print(f"Forward: y = θ₁ + θ₂ + ε, ε ~ N(0, {SIGMA_OBS}²)")
    print("Posterior concentrates along line θ₁ + θ₂ ≈ y")
    print("=" * 60)

    vel_net, disc, history = train(
        n_epochs=300,
        batch_size=256,
        lr=1e-3,
        lam=0.1,
        n_steps=20,
        disc_updates=3
    )

    print("\nGenerating evaluation plots...")
    evaluate(vel_net)

    # Print final statistics for a test case
    print("\n" + "=" * 60)
    print("Final validation (y=1.0):")
    y_test = 1.0
    mean_true, cov_true = compute_analytical_posterior(y_test)
    theta_gen = generate_posterior(vel_net, y_test, 5000)
    mean_gen = theta_gen.mean(axis=0)
    cov_gen = np.cov(theta_gen.T)

    print(f"  Analytical mean: [{mean_true[0]:.4f}, {mean_true[1]:.4f}]")
    print(f"  Generated mean:  [{mean_gen[0]:.4f}, {mean_gen[1]:.4f}]")
    print(f"  Analytical cov diagonal: [{cov_true[0,0]:.4f}, {cov_true[1,1]:.4f}]")
    print(f"  Generated cov diagonal:  [{cov_gen[0,0]:.4f}, {cov_gen[1,1]:.4f}]")
    print(f"  Analytical correlation:  {cov_true[0,1]/np.sqrt(cov_true[0,0]*cov_true[1,1]):.4f}")
    print(f"  Generated correlation:   {cov_gen[0,1]/np.sqrt(cov_gen[0,0]*cov_gen[1,1]):.4f}")
