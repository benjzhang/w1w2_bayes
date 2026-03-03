"""
Conditional Normalizing Flow Baseline for Degenerate Posterior

Standard NF trained with MLE on the same degenerate inverse problem:
- Prior: θ ~ N(0, I) in ℝ²
- Forward: y = θ₁ + θ₂ (exact, no noise)
- Posterior: supported on LINE θ₁ + θ₂ = y

This should struggle because:
1. Bijection ℝ² → ℝ² cannot map to 1D manifold
2. Log-likelihood → -∞ on measure-zero sets
3. MLE training tries to minimize variance → numerical instability
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
# Target Distribution (same as OT-Flow version)
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
# Conditional Affine Coupling Flow (RealNVP-style)
# ============================================================================

class ConditionalAffineCoupling(nn.Module):
    """
    Affine coupling layer: given input [x1, x2] and condition y,
    - Keep x1 fixed
    - Transform x2 → s(x1, y) * x2 + t(x1, y)
    """
    def __init__(self, hidden=64, mask_first=True):
        super().__init__()
        self.mask_first = mask_first

        # Scale and translate networks
        self.scale_net = nn.Sequential(
            nn.Linear(2, hidden),  # [x_masked, y]
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x, y):
        """Forward: z → x, returns (x, log_det)."""
        if self.mask_first:
            x1, x2 = x[:, 0:1], x[:, 1:2]
        else:
            x1, x2 = x[:, 1:2], x[:, 0:1]

        cond = torch.cat([x1, y], dim=1)
        s = self.scale_net(cond)
        t = self.translate_net(cond)

        # Affine transform
        x2_new = x2 * torch.exp(s) + t
        log_det = s.sum(dim=1)

        if self.mask_first:
            x_out = torch.cat([x1, x2_new], dim=1)
        else:
            x_out = torch.cat([x2_new, x1], dim=1)

        return x_out, log_det

    def inverse(self, x, y):
        """Inverse: x → z."""
        if self.mask_first:
            x1, x2 = x[:, 0:1], x[:, 1:2]
        else:
            x1, x2 = x[:, 1:2], x[:, 0:1]

        cond = torch.cat([x1, y], dim=1)
        s = self.scale_net(cond)
        t = self.translate_net(cond)

        x2_new = (x2 - t) * torch.exp(-s)

        if self.mask_first:
            return torch.cat([x1, x2_new], dim=1)
        else:
            return torch.cat([x2_new, x1], dim=1)


class ConditionalRealNVP(nn.Module):
    """Stack of conditional affine coupling layers."""

    def __init__(self, n_layers=6, hidden=64):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(hidden, mask_first=(i % 2 == 0))
            for i in range(n_layers)
        ])

    def forward(self, z, y):
        """z → x with log determinant."""
        log_det_total = 0
        x = z
        for layer in self.layers:
            x, log_det = layer(x, y)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, x, y):
        """x → z."""
        z = x
        for layer in reversed(self.layers):
            z = layer.inverse(z, y)
        return z

    def log_prob(self, x, y):
        """Compute log p(x|y) using change of variables."""
        z = self.inverse(x, y)

        # Base distribution: N(0, I)
        log_pz = -0.5 * (z**2).sum(dim=1) - np.log(2 * np.pi)

        # Forward pass to get log det
        _, log_det = self.forward(z, y)

        # log p(x|y) = log p(z) - log |det J| = log p(z) + log |det J^{-1}|
        # But we computed log det of forward (z→x), so subtract
        return log_pz - log_det


# ============================================================================
# Training
# ============================================================================

def train(n_epochs=300, batch_size=256, lr=1e-3, n_layers=6, hidden=64):
    """Train conditional NF with maximum likelihood."""

    n_train = 10000
    theta_np, y_np = sample_joint(n_train)

    theta_train = torch.FloatTensor(theta_np).to(device)
    y_train = torch.FloatTensor(y_np).unsqueeze(1).to(device)

    dataset = TensorDataset(theta_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    flow = ConditionalRealNVP(n_layers=n_layers, hidden=hidden).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=lr)

    history = {'nll': []}

    print(f"Training Conditional NF: {n_epochs} epochs, {n_layers} layers")
    print("Target: DEGENERATE posterior on line θ₁ + θ₂ = y")
    print("(MLE training should struggle with this!)")

    for epoch in range(n_epochs):
        epoch_nll = []

        for theta_data, y_data in loader:
            log_prob = flow.log_prob(theta_data, y_data)
            nll = -log_prob.mean()

            optimizer.zero_grad()
            nll.backward()

            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=10.0)

            optimizer.step()
            epoch_nll.append(nll.item())

        history['nll'].append(np.mean(epoch_nll))

        if (epoch + 1) % 50 == 0:
            print(f"[{epoch+1}/{n_epochs}] NLL={history['nll'][-1]:.4f}")

    return flow, history


# ============================================================================
# Generation & Evaluation
# ============================================================================

def generate_posterior(flow, y_obs, n_samples=1000):
    """Sample from learned p(θ|y)."""
    flow.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, 2, device=device)
        y = torch.full((n_samples, 1), y_obs, device=device)
        theta, _ = flow.forward(z, y)
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
        theta1_line = t_line
        theta2_line = y_val - t_line
        ax.plot(theta1_line, theta2_line, 'r-', linewidth=2, label='True (line)', alpha=0.7)

        # Generated
        theta_gen = generate_posterior(flow, y_val, n_samples)
        ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='NF')

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
    ax.scatter(theta_gen[:, 0], theta_gen[:, 1], alpha=0.3, s=5, label='NF')
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
    text = "Conditional NF (RealNVP)\n"
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

    text += "\nNF cannot learn degenerate\n"
    text += "distributions (measure zero)!"

    ax.text(0.05, 0.5, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', family='monospace')
    ax.axis('off')
    ax.set_title('Results Summary')

    plt.tight_layout()
    plt.savefig('results_nf_baseline.png', dpi=150)
    print("Saved: results_nf_baseline.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Conditional NF Baseline: DEGENERATE Inverse Problem")
    print("Prior: θ ~ N(0, I)")
    print("Forward: y = θ₁ + θ₂  (exact, no noise!)")
    print("Posterior: supported on LINE θ₁ + θ₂ = y")
    print("")
    print("EXPECTED: NF will fail to concentrate on the line")
    print("because bijection ℝ² → ℝ² cannot map to 1D manifold!")
    print("=" * 60)

    flow, history = train(
        n_epochs=300,
        batch_size=256,
        lr=1e-3,
        n_layers=8,
        hidden=128
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

    print("\nCompare with OT-Flow (Lip=10, λ=0.01):")
    print("  y=0.0: dist ≈ 0.008, corr ≈ -0.9999")
    print("  y=1.0: dist ≈ 0.010, corr ≈ -0.9998")
