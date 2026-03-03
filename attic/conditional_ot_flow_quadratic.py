"""
Conditional OT-Flow for Quadratic (Parabola) Constraint

Degenerate inverse problem where posterior lies on a PARABOLA:
- Prior: θ ~ N(0, I) in ℝ²
- Forward: y = θ₁² + θ₂
- Posterior: on parabola θ₂ = y - θ₁²

This is nonlinear but still an open curve (unlike circle).
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
    Sample from joint: θ ~ N(0, I), y = θ₁² + θ₂.
    """
    theta = np.random.randn(n, 2)
    y = theta[:, 0]**2 + theta[:, 1]
    return theta, y


def compute_parabola_distance(theta, y_obs):
    """Compute distance from points to parabola θ₁² + θ₂ = y."""
    # Constraint: θ₂ = y - θ₁²
    # For each point, distance to nearest point on parabola
    # Approximate: vertical distance |θ₂ - (y - θ₁²)|
    return np.abs(theta[:, 1] - (y_obs - theta[:, 0]**2))


def true_theta1_marginal(t1_grid, y_obs):
    """
    Compute true posterior marginal p(θ₁|y) for parabola constraint.

    Prior: θ ~ N(0, I)
    Constraint: y = θ₁² + θ₂

    p(θ₁|y) ∝ p(θ₁) p(θ₂ = y - θ₁²)
            = exp(-θ₁²/2) exp(-(y - θ₁²)²/2)
            = exp(-θ₁⁴/2 + (y - 0.5)θ₁² - y²/2)
    """
    log_p = -t1_grid**4/2 + (y_obs - 0.5)*t1_grid**2 - y_obs**2/2
    p = np.exp(log_p - log_p.max())  # numerical stability
    # Normalize via trapezoidal rule
    Z = np.trapz(p, t1_grid)
    return p / Z


# ============================================================================
# Networks
# ============================================================================

class VelocityNet(nn.Module):
    """v(t, θ; y) : [0,1] × ℝ² × ℝ → ℝ²"""
    def __init__(self, hidden=256, n_layers=4):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t, theta, y):
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)


class Discriminator(nn.Module):
    """φ(θ; y) : ℝ² × ℝ → ℝ with spectral norm.

    Uses quadratic features to help with nonlinear constraints.
    """
    def __init__(self, hidden=128, lip_scale=1.0, use_quadratic_features=False):
        super().__init__()
        self.lip_scale = lip_scale
        self.use_quadratic_features = use_quadratic_features

        # Input: [θ₁, θ₂, y] or [θ₁, θ₂, θ₁², θ₂², θ₁θ₂, y]
        input_dim = 6 if use_quadratic_features else 3

        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, hidden)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.SiLU(),
            nn.utils.spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, theta, y):
        if self.use_quadratic_features:
            # Add quadratic features: θ₁², θ₂², θ₁θ₂
            t1, t2 = theta[:, 0:1], theta[:, 1:2]
            quad_features = torch.cat([t1**2, t2**2, t1*t2], dim=1)
            inp = torch.cat([theta, quad_features, y], dim=1)
        else:
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

def train(n_epochs=300, batch_size=256, lr=1e-3, lam=0.01, n_steps=40, disc_updates=5, lip_scale=10.0, use_quad_features=False):
    """Train conditional OT-flow for the quadratic constraint."""

    n_train = 10000
    theta_np, y_np = sample_joint(n_train)

    theta_train = torch.FloatTensor(theta_np).to(device)
    y_train = torch.FloatTensor(y_np).unsqueeze(1).to(device)

    dataset = TensorDataset(theta_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vel_net = VelocityNet(hidden=256, n_layers=4).to(device)
    disc = Discriminator(hidden=128, lip_scale=lip_scale, use_quadratic_features=use_quad_features).to(device)

    opt_vel = optim.Adam(vel_net.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)

    history = {'L_dual': [], 'KE': []}

    print(f"Training: {n_epochs} epochs, λ={lam}, disc_updates={disc_updates}, Lip={lip_scale}, quad_features={use_quad_features}")
    print("Target: DEGENERATE posterior on PARABOLA θ₁² + θ₂ = y")

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

def generate_posterior(vel_net, y_obs, n_samples=1000, n_steps=40):
    """Sample from learned p(θ|y)."""
    vel_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y = torch.full((n_samples, 1), y_obs, device=device)
        traj = euler_integrate(vel_net, z, y, n_steps)
        return traj[-1].cpu().numpy()


def evaluate(vel_net, history, suffix=""):
    """Compare generated posteriors to true parabola posteriors."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    y_test_values = [-1.0, 0.0, 1.0, 2.0]
    n_samples = 3000

    # Top row: posterior samples
    for i, y_val in enumerate(y_test_values):
        ax = axes[0, i]

        # True posterior: parabola θ₂ = y - θ₁²
        t1 = np.linspace(-3, 3, 100)
        t2 = y_val - t1**2
        ax.plot(t1, t2, 'r-', linewidth=2, label=f'True (parabola)', alpha=0.7)

        # Generated
        theta_gen = generate_posterior(vel_net, y_val, n_samples)
        ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='OT-Flow')

        # Metrics
        dist = compute_parabola_distance(theta_gen, y_val)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-10, 5)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(f'y={y_val}: dist={dist.mean():.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom left: θ₁ distribution for y=0
    ax = axes[1, 0]
    y_test = 0.0
    theta_gen = generate_posterior(vel_net, y_test, 5000)
    ax.hist(theta_gen[:, 0], bins=50, density=True, alpha=0.7, label='OT-Flow')
    # True marginal: p(θ₁|y) ∝ exp(-θ₁⁴/2 + (y-0.5)θ₁²)
    t1_grid = np.linspace(-3, 3, 200)
    true_pdf = true_theta1_marginal(t1_grid, y_test)
    ax.plot(t1_grid, true_pdf, 'r-', linewidth=2, label='True p(θ₁|y)')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('Density')
    ax.set_title(f'θ₁ Distribution (y={y_test})')
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

    # Distance histogram for y=0
    ax = axes[1, 2]
    theta_gen = generate_posterior(vel_net, 0.0, 5000)
    dist = compute_parabola_distance(theta_gen, 0.0)
    ax.hist(dist, bins=50, density=True, alpha=0.7)
    ax.axvline(dist.mean(), color='r', linestyle='--', label=f'mean={dist.mean():.3f}')
    ax.set_xlabel('Distance to parabola')
    ax.set_ylabel('Density')
    ax.set_title('Dist to parabola (y=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics summary
    ax = axes[1, 3]
    text = "OT-Flow (Wasserstein + KE)\n"
    text += "=" * 30 + "\n\n"
    text += "Problem: θ₁² + θ₂ = y (parabola)\n"
    text += "p(θ₁|y) ∝ exp(-θ₁⁴/2 + (y-½)θ₁²)\n\n"

    for y_val in [-1.0, 0.0, 1.0, 2.0]:
        theta_gen = generate_posterior(vel_net, y_val, 5000)
        dist = compute_parabola_distance(theta_gen, y_val)
        # Compute std of true marginal
        t1_grid = np.linspace(-4, 4, 500)
        true_pdf = true_theta1_marginal(t1_grid, y_val)
        true_std = np.sqrt(np.trapz(t1_grid**2 * true_pdf, t1_grid))
        text += f"y={y_val}: dist={dist.mean():.3f}, σ_true={true_std:.2f}\n"

    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Results Summary')

    plt.tight_layout()
    fname = f'results_quadratic{suffix}.png'
    plt.savefig(fname, dpi=150)
    print(f"Saved: {fname}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    lip_scale = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    use_quad = len(sys.argv) > 3 and sys.argv[3].lower() == 'quad'

    print("=" * 60)
    print("Conditional OT-Flow: QUADRATIC (Parabola) Constraint")
    print("Prior: θ ~ N(0, I)")
    print("Forward: y = θ₁² + θ₂")
    print("Posterior: on parabola θ₂ = y - θ₁²")
    print(f"Lipschitz constant: {lip_scale}, λ (KE penalty): {lam}")
    print(f"Quadratic features: {use_quad}")
    print("=" * 60)

    vel_net, disc, history = train(
        n_epochs=300,
        batch_size=256,
        lr=1e-3,
        lam=lam,
        n_steps=40,
        disc_updates=5,
        lip_scale=lip_scale,
        use_quad_features=use_quad
    )

    print("\nGenerating evaluation plots...")
    suffix = f"_lip{int(lip_scale)}_lam{lam}" + ("_quad" if use_quad else "")
    evaluate(vel_net, history, suffix=suffix)

    # Final validation
    print("\n" + "=" * 60)
    print("Final validation:")
    for y_test in [0.0, 1.0, 2.0]:
        theta_gen = generate_posterior(vel_net, y_test, 5000)
        dist = compute_parabola_distance(theta_gen, y_test)
        print(f"  y={y_test}: mean dist to parabola = {np.mean(dist):.4f}")
