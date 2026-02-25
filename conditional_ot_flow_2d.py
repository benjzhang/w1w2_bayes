"""
Conditional OT-Flow: 2D Hierarchical Funnel
Clean implementation where conditioning variable v is NOT flowed.

Setup:
- Conditioning: v ~ N(0, 2) [sampled directly from known marginal]
- Target: (x₁, x₂) | v ~ hierarchical funnel
    x₁ | v ~ N(0, exp(v))
    x₂ | v, x₁ ~ N(0.8·x₁, exp(v/2))
- Reference: z ~ N(0, I₂)
- Flow: z → (x₁, x₂) conditioned on v

The flow only operates on x = (x₁, x₂). Variable v is sampled from N(0,2)
and passed as conditioning — no transport needed in that direction.

Key: we augment the conditioning with scale features [v, exp(v/2), exp(v/4)]
so the network doesn't have to learn the exponential mapping.
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
# Target Distribution: 2D Hierarchical Funnel
# ============================================================================

def sample_v_marginal(n):
    """Sample from marginal p(v) = N(0, 2)."""
    return np.random.normal(0, 2, n)

def sample_x_given_v(v):
    """
    Sample (x₁, x₂) | v from hierarchical funnel.
    x₁ | v ~ N(0, exp(v))
    x₂ | v, x₁ ~ N(0.8·x₁, exp(v/2))
    """
    n = len(v)
    std1 = np.exp(v / 2)
    x1 = np.random.normal(0, std1)

    std2 = np.exp(v / 4)
    x2 = np.random.normal(0.8 * x1, std2)

    return np.stack([x1, x2], axis=1)  # (n, 2)

def sample_joint(n):
    """Sample (x, v) from joint."""
    v = sample_v_marginal(n)
    x = sample_x_given_v(v)
    return x, v


# ============================================================================
# Networks with Scale-Augmented Conditioning
# ============================================================================

def encode_v(v):
    """
    Augment v with explicit scale features.
    Input: v (batch, 1)
    Output: [v, exp(v/2), exp(v/4)] (batch, 3)
    """
    return torch.cat([v, torch.exp(v / 2), torch.exp(v / 4)], dim=1)


class VelocityNet2D(nn.Module):
    """v(t, x; v) : [0,1] × ℝ² × ℝ → ℝ²"""
    def __init__(self, hidden=256):
        super().__init__()
        # Input: [t, x1, x2, v, exp(v/2), exp(v/4)] = 6D
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2)
        )

    def forward(self, t, x, v):
        # t: (batch, 1), x: (batch, 2), v: (batch, 1)
        v_enc = encode_v(v)  # (batch, 3)
        inp = torch.cat([t, x, v_enc], dim=1)  # (batch, 6)
        return self.net(inp)


class Discriminator2D(nn.Module):
    """φ(x; v) : ℝ² × ℝ → ℝ with spectral norm."""
    def __init__(self, hidden=256):
        super().__init__()
        # Input: [x1, x2, v, exp(v/2), exp(v/4)] = 5D
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(5, hidden)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden, 1))
        )

    def forward(self, x, v):
        v_enc = encode_v(v)
        inp = torch.cat([x, v_enc], dim=1)
        return self.net(inp)


# ============================================================================
# ODE Integration
# ============================================================================

def euler_integrate(vel_net, z0, v, n_steps=10):
    """Integrate dx/dt = v(t, x; v) from t=0 to t=1."""
    dt = 1.0 / n_steps
    traj = [z0]
    x = z0
    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), i * dt, device=x.device)
        vel = vel_net(t, x, v)
        x = x + dt * vel
        traj.append(x)
    return traj


# ============================================================================
# Training
# ============================================================================

def train(n_epochs=500, batch_size=256, lr=1e-3, lam=0.1, n_steps=10, disc_updates=5):
    """Train conditional OT-flow for 2D funnel."""

    # Generate training data
    n_train = 12000
    x_np, v_np = sample_joint(n_train)
    x_train = torch.FloatTensor(x_np).to(device)      # (n, 2)
    v_train = torch.FloatTensor(v_np).unsqueeze(1).to(device)  # (n, 1)

    dataset = TensorDataset(x_train, v_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    vel_net = VelocityNet2D(hidden=256).to(device)
    disc = Discriminator2D(hidden=256).to(device)

    opt_vel = optim.Adam(vel_net.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)

    history = {'L_dual': [], 'KE': []}

    print(f"Training 2D funnel: {n_epochs} epochs, λ={lam}, disc_updates={disc_updates}")

    for epoch in range(n_epochs):
        epoch_dual, epoch_ke = [], []

        for x_data, v_data in loader:
            bs = x_data.shape[0]

            # Reference: z ~ N(0, I₂), v from data
            z = torch.randn(bs, 2, device=device)

            # --- Discriminator updates ---
            for _ in range(disc_updates):
                traj = euler_integrate(vel_net, z, v_data, n_steps)
                x_gen = traj[-1].detach()

                phi_gen = disc(x_gen, v_data)
                phi_real = disc(x_data, v_data)
                L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

                opt_disc.zero_grad()
                (-L_dual).backward()
                opt_disc.step()

            # --- Velocity update ---
            traj = euler_integrate(vel_net, z, v_data, n_steps)
            x_gen = traj[-1]

            phi_gen = disc(x_gen, v_data)
            phi_real = disc(x_data, v_data)
            L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

            # Kinetic energy (sum over 2D)
            KE = 0.0
            dt = 1.0 / n_steps
            for i, xt in enumerate(traj[:-1]):
                t = torch.full((bs, 1), i * dt, device=device)
                vel = vel_net(t, xt, v_data)
                KE += 0.5 * (vel**2).sum(dim=1).mean() * dt

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

def generate(vel_net, v_values, n_steps=10):
    """
    Generate (x₁, x₂) | v by integrating flow.

    Args:
        v_values: numpy array of conditioning values
    Returns:
        x_gen: (n, 2) numpy array
    """
    vel_net.eval()
    n = len(v_values)
    with torch.no_grad():
        z = torch.randn(n, 2, device=device)
        v = torch.FloatTensor(v_values).unsqueeze(1).to(device)
        traj = euler_integrate(vel_net, z, v, n_steps)
        x_gen = traj[-1].cpu().numpy()
    return x_gen


def evaluate(vel_net):
    """Generate comparison plots."""

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    v_test = [-3, -1, 1, 3]
    n_samples = 2000

    # Top row: conditional scatter plots
    for i, v_val in enumerate(v_test):
        ax = axes[0, i]

        # True conditional
        v_cond = np.full(n_samples, v_val)
        x_true = sample_x_given_v(v_cond)

        # Generated conditional
        x_gen = generate(vel_net, v_cond)

        ax.scatter(x_true[:, 0], x_true[:, 1], alpha=0.3, s=10, label='True')
        ax.scatter(x_gen[:, 0], x_gen[:, 1], alpha=0.3, s=10, label='Gen')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_title(f'p(x|v={v_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    # Bottom row: marginal histograms for x₁
    for i, v_val in enumerate(v_test):
        ax = axes[1, i]

        v_cond = np.full(n_samples, v_val)
        x_true = sample_x_given_v(v_cond)
        x_gen = generate(vel_net, v_cond)

        ax.hist(x_true[:, 0], bins=40, alpha=0.5, density=True, label='True x₁')
        ax.hist(x_gen[:, 0], bins=40, alpha=0.5, density=True, label='Gen x₁')
        ax.set_xlabel('x₁')
        ax.set_title(f'Marginal p(x₁|v={v_val})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results_2d.png', dpi=150)
    print("Saved: results_2d.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Conditional OT-Flow: 2D Hierarchical Funnel")
    print("v ~ N(0, 2)")
    print("x₁|v ~ N(0, exp(v)),  x₂|v,x₁ ~ N(0.8·x₁, exp(v/2))")
    print("="*60)

    vel_net, disc, history = train(
        n_epochs=500,
        batch_size=256,
        lr=1e-3,
        lam=0.1,
        n_steps=10,
        disc_updates=5
    )

    print("\nGenerating evaluation plots...")
    evaluate(vel_net)

    print("\nKey point: at generation time, v is sampled from N(0, 2)")
    print("directly — no flow needed in that direction!")
