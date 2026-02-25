"""
Conditional OT-Flow with Benamou-Brenier Formulation
Clean implementation where conditioning variable is NOT flowed.

Setup:
- Conditioning variable: y ~ p(y) [known marginal, sampled directly]
- Target: x | y ~ p(x|y) [learned via flow]
- Reference: z ~ N(0, 1)
- Flow: z → x conditioned on y

The flow only operates on x. Variable y is sampled from its known marginal
and passed as conditioning — no transport needed in that direction.
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
# Target Distribution: 1D Conditional
# ============================================================================
# y ~ Uniform(-3, 3)
# x | y ~ Gamma(2, 0.3) + tanh(y)

def sample_y_marginal(n):
    """Sample from marginal p(y) = Uniform(-3, 3)."""
    return np.random.uniform(-3, 3, n)

def sample_x_given_y(y):
    """Sample x | y ~ Gamma(2, 0.3) + tanh(y)."""
    n = len(y)
    gamma = np.random.gamma(2.0, 0.3, n)
    return gamma + np.tanh(y)

def sample_joint(n):
    """Sample (x, y) from joint."""
    y = sample_y_marginal(n)
    x = sample_x_given_y(y)
    return x, y


# ============================================================================
# Networks
# ============================================================================

class VelocityNet(nn.Module):
    """v(t, x; y) : [0,1] × ℝ × ℝ → ℝ"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),  # [t, x, y]
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, t, x, y):
        # t, x, y: (batch, 1)
        inp = torch.cat([t, x, y], dim=1)
        return self.net(inp)


class Discriminator(nn.Module):
    """φ(x; y) : ℝ × ℝ → ℝ with spectral norm."""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2, hidden)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)


# ============================================================================
# ODE Integration
# ============================================================================

def euler_integrate(vel_net, z0, y, n_steps=10):
    """
    Integrate dx/dt = v(t, x; y) from t=0 to t=1.

    Args:
        z0: initial positions (batch, 1), sampled from N(0,1)
        y: conditioning values (batch, 1), sampled from p(y)

    Returns:
        trajectory: list of (batch, 1) tensors, length n_steps+1
    """
    dt = 1.0 / n_steps
    traj = [z0]
    x = z0
    for i in range(n_steps):
        t = torch.full_like(x, i * dt)
        v = vel_net(t, x, y)
        x = x + dt * v
        traj.append(x)
    return traj


# ============================================================================
# Training
# ============================================================================

def train(n_epochs=500, batch_size=256, lr=1e-3, lam=0.5, n_steps=10, disc_updates=5):
    """
    Train conditional OT-flow.

    Key: y comes from data marginal. At test time, sample y ~ p(y) directly.
    """
    # Generate training data
    n_train = 10000
    x_np, y_np = sample_joint(n_train)
    x_train = torch.FloatTensor(x_np).unsqueeze(1).to(device)
    y_train = torch.FloatTensor(y_np).unsqueeze(1).to(device)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vel_net = VelocityNet(hidden=64).to(device)
    disc = Discriminator(hidden=64).to(device)

    opt_vel = optim.Adam(vel_net.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)

    history = {'L_dual': [], 'KE': []}

    print(f"Training: {n_epochs} epochs, λ={lam}, disc_updates={disc_updates}")

    for epoch in range(n_epochs):
        epoch_dual, epoch_ke = [], []

        for x_data, y_data in loader:
            bs = x_data.shape[0]

            # Reference: z ~ N(0,1) for x, y from data
            z = torch.randn(bs, 1, device=device)

            # --- Discriminator updates ---
            for _ in range(disc_updates):
                traj = euler_integrate(vel_net, z, y_data, n_steps)
                x_gen = traj[-1].detach()

                phi_gen = disc(x_gen, y_data)
                phi_real = disc(x_data, y_data)
                L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

                opt_disc.zero_grad()
                (-L_dual).backward()
                opt_disc.step()

            # --- Velocity update ---
            traj = euler_integrate(vel_net, z, y_data, n_steps)
            x_gen = traj[-1]

            phi_gen = disc(x_gen, y_data)
            phi_real = disc(x_data, y_data)
            L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

            # Kinetic energy
            KE = 0.0
            dt = 1.0 / n_steps
            for i, xt in enumerate(traj[:-1]):
                t = torch.full_like(xt, i * dt)
                v = vel_net(t, xt, y_data)
                KE += 0.5 * (v**2).mean() * dt

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

def generate(vel_net, y_values, n_steps=10):
    """
    Generate x | y by:
    1. Sample z ~ N(0,1)
    2. Integrate flow conditioned on y

    Args:
        y_values: numpy array of conditioning values
    Returns:
        x_gen: numpy array of generated x values
    """
    vel_net.eval()
    n = len(y_values)
    with torch.no_grad():
        z = torch.randn(n, 1, device=device)
        y = torch.FloatTensor(y_values).unsqueeze(1).to(device)
        traj = euler_integrate(vel_net, z, y, n_steps)
        x_gen = traj[-1].cpu().numpy().flatten()
    return x_gen


def evaluate(vel_net):
    """Generate plots comparing true vs generated conditionals."""

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    y_test = [-2, -1, 0, 1, 2]
    n_samples = 2000

    # Top row: scatter of joint
    ax = axes[0, 0]
    y_true = sample_y_marginal(n_samples)
    x_true = sample_x_given_y(y_true)
    ax.scatter(y_true, x_true, alpha=0.3, s=10, label='True')

    y_gen = sample_y_marginal(n_samples)  # Sample y from marginal directly!
    x_gen = generate(vel_net, y_gen)
    ax.scatter(y_gen, x_gen, alpha=0.3, s=10, label='Generated')
    ax.set_xlabel('y')
    ax.set_ylabel('x')
    ax.set_title('Joint Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Remaining plots: conditional histograms
    for i, y_val in enumerate(y_test[:3]):
        ax = axes[0, i+1]

        # True conditional
        y_cond = np.full(n_samples, y_val)
        x_true_cond = sample_x_given_y(y_cond)

        # Generated conditional
        x_gen_cond = generate(vel_net, y_cond)

        ax.hist(x_true_cond, bins=40, alpha=0.5, density=True, label='True')
        ax.hist(x_gen_cond, bins=40, alpha=0.5, density=True, label='Gen')
        ax.set_title(f'p(x|y={y_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for i, y_val in enumerate(y_test[3:]):
        ax = axes[1, i]
        y_cond = np.full(n_samples, y_val)
        x_true_cond = sample_x_given_y(y_cond)
        x_gen_cond = generate(vel_net, y_cond)

        ax.hist(x_true_cond, bins=40, alpha=0.5, density=True, label='True')
        ax.hist(x_gen_cond, bins=40, alpha=0.5, density=True, label='Gen')
        ax.set_title(f'p(x|y={y_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print("Saved: results.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Conditional OT-Flow: 1D Gamma Example")
    print("y ~ Uniform(-3, 3), x|y ~ Gamma(2, 0.3) + tanh(y)")
    print("="*60)

    vel_net, disc, history = train(
        n_epochs=500,
        batch_size=256,
        lr=1e-3,
        lam=0.5,
        n_steps=10,
        disc_updates=5
    )

    print("\nGenerating evaluation plots...")
    evaluate(vel_net)

    print("\nKey point: at generation time, y is sampled from its marginal")
    print("Uniform(-3, 3) directly — no flow needed in that direction!")
