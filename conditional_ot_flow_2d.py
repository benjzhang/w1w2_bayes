"""
Conditional Optimal Transport Flow - 2D Degenerate Distribution
Based on a Hierarchical Bayesian Model (Funnel-like Distribution)

This implements conditional generation for a challenging 2D distribution
that is highly degenerate and inspired by hierarchical Bayesian models.

Target Distribution (Hierarchical Model):
- v ~ N(0, 2)  [hyperparameter, like log-precision]
- x₁|v ~ N(0, exp(v))  [parameter 1]
- x₂|v ~ N(0.8*x₁, exp(v/2))  [parameter 2, correlated with x₁]

This creates a degenerate 3D joint distribution where:
- When v is negative: x₁, x₂ are tightly concentrated (high precision)
- When v is positive: x₁, x₂ are spread out (low precision)
- x₂ is correlated with x₁, creating additional degeneracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Data Generation
# ============================================================================

def sample_hierarchical_funnel(n_samples):
    """
    Sample from hierarchical funnel distribution:
    - v ~ N(0, 2)
    - x₁|v ~ N(0, exp(v))
    - x₂|v ~ N(0.8*x₁, exp(v/2))

    This creates a degenerate distribution where the spread of (x₁, x₂)
    depends exponentially on v, mimicking hierarchical Bayesian models.

    Returns:
        x1, x2, v: Arrays of shape (n_samples,)
    """
    # Sample hyperparameter v (log-precision)
    v = np.random.normal(0, 2, size=n_samples)

    # Sample x₁ conditioned on v
    std1 = np.exp(v / 2)  # Standard deviation depends on v
    x1 = np.random.normal(0, std1)

    # Sample x₂ conditioned on v and x₁ (correlated)
    std2 = np.exp(v / 4)  # Smaller variance than x₁
    x2 = np.random.normal(0.8 * x1, std2)

    return x1, x2, v


def generate_data(n_total=15000, train_split=0.8):
    """Generate and split data into training and validation sets."""
    x1, x2, v = sample_hierarchical_funnel(n_total)

    # Convert to torch tensors
    x1_tensor = torch.FloatTensor(x1).unsqueeze(1)  # (n, 1)
    x2_tensor = torch.FloatTensor(x2).unsqueeze(1)  # (n, 1)
    v_tensor = torch.FloatTensor(v).unsqueeze(1)    # (n, 1)

    # Concatenate x = [x1, x2]
    x_tensor = torch.cat([x1_tensor, x2_tensor], dim=1)  # (n, 2)

    # Split
    n_train = int(n_total * train_split)
    x_train, x_val = x_tensor[:n_train], x_tensor[n_train:]
    v_train, v_val = v_tensor[:n_train], v_tensor[n_train:]

    print(f"Generated {n_train} training samples and {n_total - n_train} validation samples")
    print(f"x shape: {x_train.shape}, v shape: {v_train.shape}")

    return (x_train, v_train), (x_val, v_val)


# ============================================================================
# Neural Networks
# ============================================================================

def encode_v(v):
    """
    Augment conditioning variable with explicit scale features.

    The funnel distribution has std(x₁) = exp(v/2), so passing these
    scale terms directly saves the network from having to discover the
    exponential relationship on its own.

    Args:
        v: Conditioning values, shape (batch_size, 1)
    Returns:
        Augmented features [v, exp(v/2), exp(v/4)], shape (batch_size, 3)
    """
    return torch.cat([v, torch.exp(v / 2), torch.exp(v / 4)], dim=1)


class VelocityNetwork2D(nn.Module):
    """
    Velocity field v_θ(t, x; v): [0,1] × ℝ² × ℝ → ℝ²

    Maps (t, x₁, x₂, v, exp(v/2), exp(v/4)) → (v_x₁, v_x₂)
    """
    def __init__(self, hidden_size=256):
        super(VelocityNetwork2D, self).__init__()

        # Input: [t, x1, x2, v, exp(v/2), exp(v/4)] = 6D
        # Output: [v_x1, v_x2] = 2D
        self.network = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)  # 2D velocity
        )

    def forward(self, t, x, v):
        """
        Args:
            t: Time, shape (batch_size, 1)
            x: Position in 2D, shape (batch_size, 2)
            v: Conditioning variable, shape (batch_size, 1)

        Returns:
            Velocity, shape (batch_size, 2)
        """
        v_enc = encode_v(v)  # (batch_size, 3)
        inputs = torch.cat([t, x, v_enc], dim=1)  # (batch_size, 6)
        return self.network(inputs)


class Discriminator2D(nn.Module):
    """
    Discriminator φ_ψ(x; v): ℝ² × ℝ → ℝ

    Maps (x₁, x₂, v, exp(v/2), exp(v/4)) → scalar
    """
    def __init__(self, hidden_size=256):
        super(Discriminator2D, self).__init__()

        # Input: [x1, x2, v, exp(v/2), exp(v/4)] = 5D
        # Output: scalar
        self.network = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(5, hidden_size)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden_size, 1))
        )

    def forward(self, x, v):
        """
        Args:
            x: Data in 2D, shape (batch_size, 2)
            v: Conditioning variable, shape (batch_size, 1)

        Returns:
            Discriminator output, shape (batch_size, 1)
        """
        v_enc = encode_v(v)  # (batch_size, 3)
        inputs = torch.cat([x, v_enc], dim=1)  # (batch_size, 5)
        return self.network(inputs)


# ============================================================================
# ODE Integration
# ============================================================================

def integrate_ode_euler(velocity_net, z0, v, n_steps=10, T=1.0):
    """
    Integrate ODE: dx/dt = v_θ(t, x(t); v)

    Args:
        velocity_net: Velocity network
        z0: Initial positions, shape (batch_size, 2)
        v: Conditioning values, shape (batch_size, 1)
        n_steps: Number of Euler steps
        T: Final time

    Returns:
        trajectory: List of positions, each shape (batch_size, 2)
    """
    dt = T / n_steps
    batch_size = z0.shape[0]

    trajectory = [z0]
    x_t = z0

    for step in range(n_steps):
        t = torch.full((batch_size, 1), step * dt, device=z0.device)
        v_t = velocity_net(t, x_t, v)
        x_t = x_t + dt * v_t
        trajectory.append(x_t)

    return trajectory


# ============================================================================
# Training
# ============================================================================

def train_conditional_ot_2d(
    train_data,
    val_data,
    n_epochs=500,
    batch_size=256,
    lr=1e-3,
    lambda_param=1.0,
    n_steps=10,
    disc_updates=5,
    device='cpu'
):
    """Train 2D conditional OT-Flow model."""
    x_train, v_train = train_data
    x_val, v_val = val_data

    # Data loader
    train_dataset = TensorDataset(x_train, v_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Networks (larger for 2D)
    velocity_net = VelocityNetwork2D(hidden_size=128).to(device)
    discriminator = Discriminator2D(hidden_size=128).to(device)

    # Optimizers
    optimizer_vel = optim.Adam(velocity_net.parameters(), lr=lr)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr)

    # History
    history = {
        'L_dual': [],
        'kinetic': [],
        'loss_disc': [],
        'loss_vel': []
    }

    print(f"\nTraining 2D model for {n_epochs} epochs...")
    print(f"Batch size: {batch_size}, λ: {lambda_param}, ODE steps: {n_steps}")
    print(f"Discriminator updates: {disc_updates}:1")

    for epoch in range(n_epochs):
        epoch_metrics = {
            'L_dual': [],
            'kinetic': [],
            'loss_disc': [],
            'loss_vel': []
        }

        for batch_idx, (x_data, v_data) in enumerate(train_loader):
            x_data = x_data.to(device)
            v_data = v_data.to(device)
            current_batch_size = x_data.shape[0]

            # Sample from reference (2D Gaussian)
            z = torch.randn(current_batch_size, 2, device=device)

            # Update Discriminator
            for _ in range(disc_updates):
                trajectory = integrate_ode_euler(velocity_net, z, v_data, n_steps=n_steps)
                x_T = trajectory[-1]

                phi_generated = discriminator(x_T.detach(), v_data)
                phi_real = discriminator(x_data, v_data)
                L_dual = phi_generated.mean() - torch.exp(phi_real - 1).mean()

                loss_disc = -L_dual
                optimizer_disc.zero_grad()
                loss_disc.backward()
                optimizer_disc.step()

            # Update Velocity
            trajectory = integrate_ode_euler(velocity_net, z, v_data, n_steps=n_steps)
            x_T = trajectory[-1]

            phi_generated = discriminator(x_T, v_data)
            phi_real = discriminator(x_data, v_data)
            L_dual = phi_generated.mean() - torch.exp(phi_real - 1).mean()

            # Kinetic energy
            kinetic = 0.0
            dt = 1.0 / n_steps
            for step, x_t in enumerate(trajectory[:-1]):
                t = torch.full((current_batch_size, 1), step * dt, device=device)
                v_t = velocity_net(t, x_t, v_data)
                kinetic += 0.5 * (v_t ** 2).sum(dim=1).mean() * dt  # Sum over 2D

            loss_vel = L_dual + lambda_param * kinetic
            optimizer_vel.zero_grad()
            loss_vel.backward()
            optimizer_vel.step()

            # Record
            epoch_metrics['L_dual'].append(L_dual.item())
            epoch_metrics['kinetic'].append(kinetic.item())
            epoch_metrics['loss_disc'].append(loss_disc.item())
            epoch_metrics['loss_vel'].append(loss_vel.item())

        # Average
        for key in epoch_metrics:
            history[key].append(np.mean(epoch_metrics[key]))

        # Print
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"L_dual: {history['L_dual'][-1]:.4f}, "
                  f"Kinetic: {history['kinetic'][-1]:.4f}, "
                  f"Loss_vel: {history['loss_vel'][-1]:.4f}")

    print("\nTraining completed!")
    return velocity_net, discriminator, history


# ============================================================================
# Visualization
# ============================================================================

def generate_samples_2d(velocity_net, n_samples, v_values, n_steps=10, device='cpu'):
    """Generate 2D samples."""
    velocity_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        v = torch.FloatTensor(v_values).to(device)
        if len(v.shape) == 1:
            v = v.unsqueeze(1)

        trajectory = integrate_ode_euler(velocity_net, z, v, n_steps=n_steps)
        x_generated = trajectory[-1]

    return x_generated.cpu().numpy()


def plot_3d_joint_distribution(velocity_net, x_val, v_val, n_steps=10, device='cpu'):
    """Plot 3D scatter: true vs generated joint distribution."""
    fig = plt.figure(figsize=(16, 6))

    # True distribution
    ax1 = fig.add_subplot(131, projection='3d')
    x1_true = x_val[:, 0].cpu().numpy()
    x2_true = x_val[:, 1].cpu().numpy()
    v_true = v_val.cpu().numpy().flatten()

    ax1.scatter(v_true, x1_true, x2_true, alpha=0.3, s=5, c=v_true, cmap='viridis')
    ax1.set_xlabel('v (condition)', fontsize=10)
    ax1.set_ylabel('x₁', fontsize=10)
    ax1.set_zlabel('x₂', fontsize=10)
    ax1.set_title('True Joint Distribution', fontsize=12)

    # Generated distribution
    ax2 = fig.add_subplot(132, projection='3d')
    n_gen = len(x_val)
    v_gen = v_val.cpu().numpy().flatten()
    x_gen = generate_samples_2d(velocity_net, n_gen, v_gen, n_steps=n_steps, device=device)

    ax2.scatter(v_gen, x_gen[:, 0], x_gen[:, 1], alpha=0.3, s=5, c=v_gen, cmap='viridis')
    ax2.set_xlabel('v (condition)', fontsize=10)
    ax2.set_ylabel('x₁', fontsize=10)
    ax2.set_zlabel('x₂', fontsize=10)
    ax2.set_title('Generated Joint Distribution', fontsize=12)

    # Side by side 2D projections
    ax3 = fig.add_subplot(133)
    ax3.scatter(x1_true, x2_true, alpha=0.3, s=10, label='True', c='blue')
    ax3.scatter(x_gen[:, 0], x_gen[:, 1], alpha=0.3, s=10, label='Generated', c='orange')
    ax3.set_xlabel('x₁', fontsize=11)
    ax3.set_ylabel('x₂', fontsize=11)
    ax3.set_title('2D Projection (x₁, x₂)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_conditional_slices_2d(velocity_net, v_test_values=[-3, -1, 1, 3], n_samples=2000, n_steps=10, device='cpu'):
    """Plot 2D conditional distributions for fixed v values."""
    n_conditions = len(v_test_values)
    fig, axes = plt.subplots(2, n_conditions, figsize=(5*n_conditions, 10))

    for idx, v_val in enumerate(v_test_values):
        # Generate true samples
        v_samples = np.full(n_samples, v_val)
        std1 = np.exp(v_val / 2)
        x1_true = np.random.normal(0, std1, n_samples)
        std2 = np.exp(v_val / 4)
        x2_true = np.random.normal(0.8 * x1_true, std2)

        # Generate from model
        x_gen = generate_samples_2d(velocity_net, n_samples, v_samples, n_steps=n_steps, device=device)

        # Scatter plot
        axes[0, idx].scatter(x1_true, x2_true, alpha=0.3, s=5, label='True', c='blue')
        axes[0, idx].scatter(x_gen[:, 0], x_gen[:, 1], alpha=0.3, s=5, label='Generated', c='orange')
        axes[0, idx].set_xlabel('x₁', fontsize=11)
        axes[0, idx].set_ylabel('x₂', fontsize=11)
        axes[0, idx].set_title(f'p(x₁, x₂|v={v_val})', fontsize=12)
        axes[0, idx].legend(markerscale=3)
        axes[0, idx].grid(True, alpha=0.3)
        axes[0, idx].set_aspect('equal', adjustable='box')

        # Marginal histograms for x₁
        axes[1, idx].hist(x1_true, bins=40, alpha=0.5, density=True, label='True x₁', color='blue')
        axes[1, idx].hist(x_gen[:, 0], bins=40, alpha=0.5, density=True, label='Gen x₁', color='orange')
        axes[1, idx].set_xlabel('x₁', fontsize=11)
        axes[1, idx].set_ylabel('Density', fontsize=11)
        axes[1, idx].set_title(f'Marginal p(x₁|v={v_val})', fontsize=12)
        axes[1, idx].legend()
        axes[1, idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_curves_2d(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    epochs = range(1, len(history['L_dual']) + 1)

    axes[0].plot(epochs, history['L_dual'], linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('L_dual', fontsize=12)
    axes[0].set_title('Dual Loss', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['kinetic'], color='red', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Kinetic Energy', fontsize=12)
    axes[1].set_title('Kinetic Energy', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_flow_trajectories_2d(velocity_net, v_val=0.0, n_trajectories=15, n_steps=50, device='cpu'):
    """Visualize 2D flow trajectories."""
    velocity_net.eval()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    with torch.no_grad():
        z0 = torch.randn(n_trajectories, 2, device=device)
        v = torch.full((n_trajectories, 1), v_val, device=device)

        trajectory = integrate_ode_euler(velocity_net, z0, v, n_steps=n_steps)
        trajectory_np = [x.cpu().numpy() for x in trajectory]

        # Plot trajectories in x₁-x₂ space
        for i in range(n_trajectories):
            x1_traj = [trajectory_np[t][i, 0] for t in range(len(trajectory_np))]
            x2_traj = [trajectory_np[t][i, 1] for t in range(len(trajectory_np))]
            axes[0].plot(x1_traj, x2_traj, alpha=0.6, linewidth=2)
            axes[0].scatter(x1_traj[0], x2_traj[0], c='green', s=50, marker='o', zorder=5)
            axes[0].scatter(x1_traj[-1], x2_traj[-1], c='red', s=50, marker='x', zorder=5)

        axes[0].set_xlabel('x₁', fontsize=12)
        axes[0].set_ylabel('x₂', fontsize=12)
        axes[0].set_title(f'Flow Trajectories in (x₁, x₂) Space (v={v_val})', fontsize=13)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(['Start', 'End'], loc='best')

        # Plot x₁ over time
        times = np.linspace(0, 1, n_steps + 1)
        for i in range(n_trajectories):
            x1_traj = [trajectory_np[t][i, 0] for t in range(len(trajectory_np))]
            axes[1].plot(times, x1_traj, alpha=0.6, linewidth=1.5)

        axes[1].set_xlabel('Time t', fontsize=12)
        axes[1].set_ylabel('x₁(t)', fontsize=12)
        axes[1].set_title(f'x₁ Trajectories Over Time (v={v_val})', fontsize=13)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Conditional OT-Flow: 2D Hierarchical Funnel Distribution")
    print("=" * 70)

    # Generate data
    print("\n[1/4] Generating training data...")
    train_data, val_data = generate_data(n_total=15000, train_split=0.8)
    x_train, v_train = train_data
    x_val, v_val = val_data

    # Train
    print("\n[2/4] Training model...")
    velocity_net, discriminator, history = train_conditional_ot_2d(
        train_data=train_data,
        val_data=val_data,
        n_epochs=500,
        batch_size=256,
        lr=1e-3,
        lambda_param=0.1,
        n_steps=10,
        disc_updates=5,
        device=device
    )

    # Visualize
    print("\n[3/4] Generating visualizations...")

    print("  - Plotting 3D joint distribution...")
    fig1 = plot_3d_joint_distribution(velocity_net, x_val, v_val, n_steps=10, device=device)
    plt.savefig('joint_distribution_2d.png', dpi=150, bbox_inches='tight')
    print("    Saved: joint_distribution_2d.png")

    print("  - Plotting conditional slices...")
    fig2 = plot_conditional_slices_2d(velocity_net, v_test_values=[-3, -1, 1, 3], n_samples=2000, n_steps=10, device=device)
    plt.savefig('conditional_slices_2d.png', dpi=150, bbox_inches='tight')
    print("    Saved: conditional_slices_2d.png")

    print("  - Plotting training curves...")
    fig3 = plot_training_curves_2d(history)
    plt.savefig('training_curves_2d.png', dpi=150, bbox_inches='tight')
    print("    Saved: training_curves_2d.png")

    print("  - Plotting flow trajectories...")
    fig4 = plot_flow_trajectories_2d(velocity_net, v_val=0.0, n_trajectories=15, n_steps=50, device=device)
    plt.savefig('flow_trajectories_2d.png', dpi=150, bbox_inches='tight')
    print("    Saved: flow_trajectories_2d.png")

    # Summary
    print("\n[4/4] Evaluation Summary")
    print("=" * 70)
    print("\nThis model learns to generate from a highly degenerate 2D distribution")
    print("inspired by hierarchical Bayesian models (funnel distribution).")
    print("\nKey Challenge:")
    print("  - When v is negative: high precision → narrow conditional p(x|v)")
    print("  - When v is positive: low precision → wide conditional p(x|v)")
    print("  - Variance changes exponentially with v (10⁶× range!)")
    print("\nCheck visualizations to assess if the model handles this degeneracy!")
    print("=" * 70)

    plt.close('all')


if __name__ == "__main__":
    main()
