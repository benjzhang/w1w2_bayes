"""
Conditional Optimal Transport Flow using Benamou-Brenier Formulation
with Lipschitz-Regularized f-Divergence

This implements a conditional generative model that learns to transport
a reference distribution (Gaussian) to a target conditional distribution
using continuous normalizing flows with kinetic energy regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import seaborn as sns
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# ============================================================================
# Data Generation
# ============================================================================

def sample_target_distribution(n_samples):
    """
    Sample from the target distribution π(x, y):
    - y ~ Uniform(-3, 3)
    - x|y ~ Gamma(shape=2, scale=0.3) + tanh(y)

    Args:
        n_samples: Number of samples to generate

    Returns:
        x, y: Numpy arrays of shape (n_samples,)
    """
    # Sample y from uniform distribution
    y = np.random.uniform(-3, 3, size=n_samples)

    # Sample x|y from shifted Gamma distribution
    gamma_samples = np.random.gamma(shape=2.0, scale=0.3, size=n_samples)
    x = gamma_samples + np.tanh(y)

    return x, y


def generate_data(n_total=10000, train_split=0.8):
    """Generate and split data into training and validation sets."""
    x, y = sample_target_distribution(n_total)

    # Convert to torch tensors
    x_tensor = torch.FloatTensor(x).unsqueeze(1)  # (n, 1)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)  # (n, 1)

    # Split into train and validation
    n_train = int(n_total * train_split)

    x_train, x_val = x_tensor[:n_train], x_tensor[n_train:]
    y_train, y_val = y_tensor[:n_train], y_tensor[n_train:]

    print(f"Generated {n_train} training samples and {n_total - n_train} validation samples")

    return (x_train, y_train), (x_val, y_val)


# ============================================================================
# Neural Network Architectures
# ============================================================================

class VelocityNetwork(nn.Module):
    """
    Time-dependent velocity field v_θ(t, x; y): [0,1] × ℝ × ℝ → ℝ

    Architecture: 3-layer MLP with hidden size 64
    """
    def __init__(self, hidden_size=64):
        super(VelocityNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(3, hidden_size),  # Input: [t, x, y]
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # Output: velocity (scalar)
        )

    def forward(self, t, x, y):
        """
        Args:
            t: Time values, shape (batch_size, 1)
            x: Position values, shape (batch_size, 1)
            y: Conditioning values, shape (batch_size, 1)

        Returns:
            Velocity field values, shape (batch_size, 1)
        """
        inputs = torch.cat([t, x, y], dim=1)
        return self.network(inputs)


class Discriminator(nn.Module):
    """
    Lipschitz-constrained discriminator φ_ψ(x; y): ℝ × ℝ → ℝ

    Uses spectral normalization on all linear layers to enforce Lipschitz constraint.
    Architecture: 3-layer MLP with hidden size 64
    """
    def __init__(self, hidden_size=64):
        super(Discriminator, self).__init__()

        # Apply spectral normalization to all linear layers
        self.network = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(2, hidden_size)),  # Input: [x, y]
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            nn.utils.spectral_norm(nn.Linear(hidden_size, 1))  # Output: scalar
        )

    def forward(self, x, y):
        """
        Args:
            x: Data values, shape (batch_size, 1)
            y: Conditioning values, shape (batch_size, 1)

        Returns:
            Discriminator outputs, shape (batch_size, 1)
        """
        inputs = torch.cat([x, y], dim=1)
        return self.network(inputs)


# ============================================================================
# ODE Integration
# ============================================================================

def integrate_ode_euler(velocity_net, z0, y, n_steps=10, T=1.0):
    """
    Integrate the ODE dx/dt = v_θ(t, x(t); y) using Euler method.

    Args:
        velocity_net: The velocity network v_θ
        z0: Initial positions, shape (batch_size, 1)
        y: Conditioning values, shape (batch_size, 1)
        n_steps: Number of Euler steps
        T: Final time (default 1.0)

    Returns:
        trajectory: List of positions at each time step, length n_steps + 1
    """
    dt = T / n_steps
    batch_size = z0.shape[0]

    trajectory = [z0]
    x_t = z0

    for step in range(n_steps):
        t = torch.full((batch_size, 1), step * dt, device=z0.device)
        v_t = velocity_net(t, x_t, y)
        x_t = x_t + dt * v_t
        trajectory.append(x_t)

    return trajectory


# ============================================================================
# Training
# ============================================================================

def train_conditional_ot(
    train_data,
    val_data,
    n_epochs=300,
    batch_size=256,
    lr=1e-3,
    lambda_param=0.5,
    n_steps=10,
    disc_updates=1,
    device='cpu'
):
    """
    Train the conditional OT-Flow model.

    Args:
        train_data: Tuple of (x_train, y_train)
        val_data: Tuple of (x_val, y_val)
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        lambda_param: Kinetic energy regularization weight
        n_steps: Number of ODE integration steps
        disc_updates: Number of discriminator updates per velocity update
        device: Device to train on

    Returns:
        velocity_net, discriminator, history
    """
    x_train, y_train = train_data
    x_val, y_val = val_data

    # Create data loader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize networks
    velocity_net = VelocityNetwork(hidden_size=64).to(device)
    discriminator = Discriminator(hidden_size=64).to(device)

    # Optimizers
    optimizer_vel = optim.Adam(velocity_net.parameters(), lr=lr)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr)

    # Training history
    history = {
        'L_dual': [],
        'kinetic': [],
        'loss_disc': [],
        'loss_vel': []
    }

    print(f"\nStarting training for {n_epochs} epochs...")
    print(f"Batch size: {batch_size}, λ: {lambda_param}, ODE steps: {n_steps}")
    print(f"Discriminator updates per velocity update: {disc_updates}")

    for epoch in range(n_epochs):
        epoch_metrics = {
            'L_dual': [],
            'kinetic': [],
            'loss_disc': [],
            'loss_vel': []
        }

        for batch_idx, (x_data, y_data) in enumerate(train_loader):
            x_data = x_data.to(device)
            y_data = y_data.to(device)
            current_batch_size = x_data.shape[0]

            # Sample from reference distribution
            z = torch.randn(current_batch_size, 1, device=device)

            # ================================================================
            # Update Discriminator (maximize L_dual)
            # ================================================================
            for _ in range(disc_updates):
                # Forward integrate ODE
                trajectory = integrate_ode_euler(velocity_net, z, y_data, n_steps=n_steps)
                x_T = trajectory[-1]

                # Compute adversarial loss
                phi_generated = discriminator(x_T.detach(), y_data)  # Detach to avoid grad flow
                phi_real = discriminator(x_data, y_data)

                # L_dual = E[φ(x_generated)] - E[exp(φ(x_real) - 1)]
                L_dual = phi_generated.mean() - torch.exp(phi_real - 1).mean()

                # Discriminator wants to maximize L_dual, so minimize -L_dual
                loss_disc = -L_dual

                optimizer_disc.zero_grad()
                loss_disc.backward()
                optimizer_disc.step()

            # ================================================================
            # Update Velocity Network (minimize L_dual + λ * kinetic)
            # ================================================================

            # Forward integrate ODE (need fresh trajectory with gradients)
            trajectory = integrate_ode_euler(velocity_net, z, y_data, n_steps=n_steps)
            x_T = trajectory[-1]

            # Compute adversarial loss
            phi_generated = discriminator(x_T, y_data)
            phi_real = discriminator(x_data, y_data)
            L_dual = phi_generated.mean() - torch.exp(phi_real - 1).mean()

            # Compute kinetic energy: ∫₀¹ 0.5 * ||v_θ(t, x(t); y)||² dt
            kinetic = 0.0
            dt = 1.0 / n_steps
            for step, x_t in enumerate(trajectory[:-1]):  # Don't include final point
                t = torch.full((current_batch_size, 1), step * dt, device=device)
                v_t = velocity_net(t, x_t, y_data)
                kinetic += 0.5 * (v_t ** 2).mean() * dt

            # Total velocity loss
            loss_vel = L_dual + lambda_param * kinetic

            optimizer_vel.zero_grad()
            loss_vel.backward()
            optimizer_vel.step()

            # Record metrics
            epoch_metrics['L_dual'].append(L_dual.item())
            epoch_metrics['kinetic'].append(kinetic.item())
            epoch_metrics['loss_disc'].append(loss_disc.item())
            epoch_metrics['loss_vel'].append(loss_vel.item())

        # Average metrics over epoch
        for key in epoch_metrics:
            history[key].append(np.mean(epoch_metrics[key]))

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"L_dual: {history['L_dual'][-1]:.4f}, "
                  f"Kinetic: {history['kinetic'][-1]:.4f}, "
                  f"Loss_disc: {history['loss_disc'][-1]:.4f}, "
                  f"Loss_vel: {history['loss_vel'][-1]:.4f}")

    print("\nTraining completed!")
    return velocity_net, discriminator, history


# ============================================================================
# Visualization
# ============================================================================

def generate_samples(velocity_net, n_samples, y_values, n_steps=10, device='cpu'):
    """
    Generate samples from the learned conditional distribution.

    Args:
        velocity_net: Trained velocity network
        n_samples: Number of samples to generate
        y_values: Conditioning values, shape (n_samples, 1)
        n_steps: Number of ODE integration steps
        device: Device to use

    Returns:
        Generated samples, shape (n_samples, 1)
    """
    velocity_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 1, device=device)
        y = torch.FloatTensor(y_values).to(device)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        trajectory = integrate_ode_euler(velocity_net, z, y, n_steps=n_steps)
        x_generated = trajectory[-1]

    return x_generated.cpu().numpy()


def plot_joint_distribution(velocity_net, x_true, y_true, n_steps=10, device='cpu'):
    """Plot joint distribution: true vs generated."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # True distribution
    axes[0].scatter(y_true.cpu().numpy(), x_true.cpu().numpy(), alpha=0.3, s=10)
    axes[0].set_xlabel('y', fontsize=12)
    axes[0].set_ylabel('x', fontsize=12)
    axes[0].set_title('True Joint Distribution π(x, y)', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    # Generated distribution
    n_gen = len(x_true)
    y_gen = y_true.cpu().numpy()
    x_gen = generate_samples(velocity_net, n_gen, y_gen, n_steps=n_steps, device=device)

    axes[1].scatter(y_gen, x_gen, alpha=0.3, s=10, color='orange')
    axes[1].set_xlabel('y', fontsize=12)
    axes[1].set_ylabel('x', fontsize=12)
    axes[1].set_title('Generated Joint Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_conditional_slices(velocity_net, y_test_values=[-2, 0, 1, 2], n_samples=2000, n_steps=10, device='cpu'):
    """Plot conditional distributions for fixed y values."""
    n_conditions = len(y_test_values)
    fig, axes = plt.subplots(1, n_conditions, figsize=(5*n_conditions, 4))

    if n_conditions == 1:
        axes = [axes]

    for idx, y_val in enumerate(y_test_values):
        # Generate true samples
        y_samples = np.full(n_samples, y_val)
        gamma_samples = np.random.gamma(shape=2.0, scale=0.3, size=n_samples)
        x_true = gamma_samples + np.tanh(y_val)

        # Generate samples from learned distribution
        x_gen = generate_samples(velocity_net, n_samples, y_samples, n_steps=n_steps, device=device)

        # Plot histograms
        axes[idx].hist(x_true, bins=40, alpha=0.5, density=True, label='True', color='blue')
        axes[idx].hist(x_gen.flatten(), bins=40, alpha=0.5, density=True, label='Generated', color='orange')
        axes[idx].set_xlabel('x', fontsize=12)
        axes[idx].set_ylabel('Density', fontsize=12)
        axes[idx].set_title(f'Conditional p(x|y={y_val})', fontsize=14)
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_curves(history):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    epochs = range(1, len(history['L_dual']) + 1)

    # L_dual
    axes[0].plot(epochs, history['L_dual'], label='L_dual', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('L_dual', fontsize=12)
    axes[0].set_title('Dual Loss Over Training', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Kinetic energy
    axes[1].plot(epochs, history['kinetic'], label='Kinetic Energy', color='red', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Kinetic Energy', fontsize=12)
    axes[1].set_title('Kinetic Energy Over Training', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_flow_trajectories(velocity_net, y_val=0.0, n_trajectories=20, n_steps=50, device='cpu'):
    """
    Visualize flow trajectories from t=0 to t=1 for a fixed y value.

    Args:
        velocity_net: Trained velocity network
        y_val: Fixed conditioning value
        n_trajectories: Number of trajectories to plot
        n_steps: Number of time steps for visualization (finer than training)
        device: Device to use
    """
    velocity_net.eval()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    with torch.no_grad():
        # Sample initial positions
        z0 = torch.randn(n_trajectories, 1, device=device)
        y = torch.full((n_trajectories, 1), y_val, device=device)

        # Integrate ODE
        trajectory = integrate_ode_euler(velocity_net, z0, y, n_steps=n_steps)

        # Convert to numpy
        trajectory_np = [x.cpu().numpy() for x in trajectory]
        times = np.linspace(0, 1, n_steps + 1)

        # Plot each trajectory
        for i in range(n_trajectories):
            positions = [trajectory_np[t][i, 0] for t in range(len(trajectory_np))]
            ax.plot(times, positions, alpha=0.6, linewidth=1.5)

        ax.set_xlabel('Time t', fontsize=12)
        ax.set_ylabel('Position x(t)', fontsize=12)
        ax.set_title(f'Flow Trajectories for y = {y_val}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=np.tanh(y_val), color='red', linestyle='--',
                   label=f'Target mean ≈ {np.tanh(y_val):.2f}', linewidth=2)
        ax.legend()

    plt.tight_layout()
    return fig


def plot_velocity_magnitude(velocity_net, y_val=0.0, n_samples=100, n_steps=50, device='cpu'):
    """
    Plot velocity field magnitude over time along trajectories.
    For optimal transport, this should be roughly constant.
    """
    velocity_net.eval()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    with torch.no_grad():
        z0 = torch.randn(n_samples, 1, device=device)
        y = torch.full((n_samples, 1), y_val, device=device)

        dt = 1.0 / n_steps
        trajectory = integrate_ode_euler(velocity_net, z0, y, n_steps=n_steps)

        # Compute velocity magnitude at each time step
        times = []
        velocities = []

        for step in range(n_steps):
            t = torch.full((n_samples, 1), step * dt, device=device)
            x_t = trajectory[step]
            v_t = velocity_net(t, x_t, y)

            times.append(step * dt)
            velocities.append(v_t.abs().mean().cpu().numpy())

        ax.plot(times, velocities, linewidth=2, label='Mean |v(t, x; y)|')
        ax.set_xlabel('Time t', fontsize=12)
        ax.set_ylabel('Mean Velocity Magnitude', fontsize=12)
        ax.set_title(f'Velocity Magnitude Over Time (y = {y_val})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Conditional Optimal Transport Flow - Benamou-Brenier Formulation")
    print("=" * 70)

    # Generate data
    print("\n[1/4] Generating training data...")
    train_data, val_data = generate_data(n_total=10000, train_split=0.8)
    x_train, y_train = train_data
    x_val, y_val = val_data

    # Train model
    print("\n[2/4] Training model...")
    velocity_net, discriminator, history = train_conditional_ot(
        train_data=train_data,
        val_data=val_data,
        n_epochs=500,
        batch_size=256,
        lr=1e-3,
        lambda_param=0.5,
        n_steps=10,
        disc_updates=5,
        device=device
    )

    # Generate visualizations
    print("\n[3/4] Generating visualizations...")

    # 1. Joint distribution
    print("  - Plotting joint distribution...")
    fig1 = plot_joint_distribution(velocity_net, x_val, y_val, n_steps=10, device=device)
    plt.savefig('joint_distribution.png', dpi=150, bbox_inches='tight')
    print("    Saved: joint_distribution.png")

    # 2. Conditional slices
    print("  - Plotting conditional slices...")
    fig2 = plot_conditional_slices(velocity_net, y_test_values=[-2, 0, 1, 2], n_samples=2000, n_steps=10, device=device)
    plt.savefig('conditional_slices.png', dpi=150, bbox_inches='tight')
    print("    Saved: conditional_slices.png")

    # 3. Training curves
    print("  - Plotting training curves...")
    fig3 = plot_training_curves(history)
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("    Saved: training_curves.png")

    # 4. Flow trajectories
    print("  - Plotting flow trajectories...")
    fig4 = plot_flow_trajectories(velocity_net, y_val=0.0, n_trajectories=20, n_steps=50, device=device)
    plt.savefig('flow_trajectories.png', dpi=150, bbox_inches='tight')
    print("    Saved: flow_trajectories.png")

    # 5. Velocity magnitude
    print("  - Plotting velocity magnitude...")
    fig5 = plot_velocity_magnitude(velocity_net, y_val=0.0, n_samples=100, n_steps=50, device=device)
    plt.savefig('velocity_magnitude.png', dpi=150, bbox_inches='tight')
    print("    Saved: velocity_magnitude.png")

    # Summary
    print("\n[4/4] Evaluation Summary")
    print("=" * 70)
    print("\nQualitative Assessment:")
    print("  - Check 'conditional_slices.png' for histogram matching")
    print("  - Generated samples should closely match true conditional distributions")
    print("  - Training curves should show convergence without collapse")
    print("  - Flow trajectories should show smooth transport from Gaussian to target")
    print("  - Velocity magnitude should be relatively stable (optimal transport property)")

    print("\n" + "=" * 70)
    print("All visualizations saved! Review the images to assess model performance.")
    print("=" * 70)

    plt.close('all')


if __name__ == "__main__":
    main()
