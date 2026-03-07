"""Evaluation and visualization for MNIST inpainting.

Self-contained: does not import from parent package.

Usage:
    python -m mnist.evaluate --checkpoint results/mnist/model.pt --output-dir results/mnist
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from .data import MNISTInpainting
from .networks import VelocityMLP


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def euler_integrate(vel_net, z0, y, n_steps=20):
    """Euler-integrate the velocity field from t=0 to t=1."""
    dt = 1.0 / n_steps
    theta = z0
    traj = [z0]
    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        theta = theta + dt * v
        traj.append(theta)
    return traj


def load_checkpoint(checkpoint_path, device):
    """Load flow checkpoint and return velocity net and hparams."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = state['hparams']

    vel_net = VelocityMLP(
        theta_dim=hparams['theta_dim'],
        y_dim=hparams['y_dim'],
        hidden=hparams.get('hidden', 512),
        n_layers=hparams.get('n_layers', 4),
    ).to(device)
    vel_net.load_state_dict(state['vel_net_state'])
    vel_net.eval()

    return vel_net, hparams, state


def reconstruct_image(y, theta):
    """Reconstruct a 28x28 image from top half (y) and bottom half (theta).

    Args:
        y: Top half, shape (392,) numpy array or tensor.
        theta: Bottom half, shape (392,) numpy array or tensor.

    Returns:
        28x28 numpy array with pixel values in [0, 1].
    """
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(theta, torch.Tensor):
        theta = theta.cpu().numpy()

    top = y.reshape(14, 28)
    bottom = theta.reshape(14, 28)
    img = np.vstack([top, bottom])
    return np.clip(img, 0.0, 1.0)


def compute_metrics(theta_samples, theta_true):
    """Compute evaluation metrics.

    Args:
        theta_samples: Generated bottom halves, shape (n_samples, 392).
        theta_true: True bottom half, shape (392,).

    Returns:
        Dict with 'mse' and 'diversity'.
    """
    if isinstance(theta_samples, torch.Tensor):
        theta_samples = theta_samples.cpu().numpy()
    if isinstance(theta_true, torch.Tensor):
        theta_true = theta_true.cpu().numpy()

    # MSE to ground truth (averaged over samples)
    mse = np.mean((theta_samples - theta_true[None, :]) ** 2)

    # Diversity: std of completions per pixel, averaged
    diversity = np.mean(np.std(theta_samples, axis=0))

    return {'mse': float(mse), 'diversity': float(diversity)}


def plot_completions(y_test, theta_samples_list, theta_true, save_path,
                     n_show=5, title=None):
    """Plot a grid of inpainting completions.

    Each row is one test image. Columns show:
      [Original] [Top only] [Completion 1] ... [Completion n_show]

    Args:
        y_test: Top halves, shape (n_test, 392).
        theta_samples_list: List of arrays, each shape (n_samples, 392),
            one per test image.
        theta_true: True bottom halves, shape (n_test, 392).
        save_path: Path to save the figure.
        n_show: Number of sample completions to show per image.
        title: Optional figure title.
    """
    n_test = len(y_test)
    n_cols = 2 + n_show  # original, top-only, completions
    fig, axes = plt.subplots(n_test, n_cols, figsize=(2 * n_cols, 2 * n_test))

    if n_test == 1:
        axes = axes[None, :]  # ensure 2D

    for i in range(n_test):
        y_i = y_test[i]
        theta_true_i = theta_true[i]
        samples_i = theta_samples_list[i]

        # Original full image
        ax = axes[i, 0]
        img_full = reconstruct_image(y_i, theta_true_i)
        ax.imshow(img_full, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            ax.set_title('Original', fontsize=9)
        ax.axis('off')

        # Top half only (bottom half blank)
        ax = axes[i, 1]
        img_top = reconstruct_image(y_i, np.zeros(392))
        ax.imshow(img_top, cmap='gray', vmin=0, vmax=1)
        if i == 0:
            ax.set_title('Observed (top)', fontsize=9)
        ax.axis('off')

        # Sample completions
        for j in range(n_show):
            ax = axes[i, 2 + j]
            if j < len(samples_i):
                theta_j = samples_i[j]
                img_j = reconstruct_image(y_i, theta_j)
                ax.imshow(img_j, cmap='gray', vmin=0, vmax=1)
            else:
                ax.axis('off')
            if i == 0:
                ax.set_title(f'Sample {j+1}', fontsize=9)
            ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.93)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_metrics_summary(all_metrics, save_path, title=None):
    """Plot a bar chart of MSE and diversity per test image.

    Args:
        all_metrics: List of metric dicts (one per test image).
        save_path: Path to save figure.
        title: Optional title.
    """
    n = len(all_metrics)
    mses = [m['mse'] for m in all_metrics]
    divs = [m['diversity'] for m in all_metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(range(n), mses, color='steelblue')
    ax1.set_xlabel('Test image')
    ax1.set_ylabel('MSE')
    ax1.set_title('Pixel MSE to ground truth')
    ax1.axhline(np.mean(mses), color='red', linestyle='--',
                label=f'mean={np.mean(mses):.4f}')
    ax1.legend()

    ax2.bar(range(n), divs, color='coral')
    ax2.set_xlabel('Test image')
    ax2.set_ylabel('Diversity (mean std)')
    ax2.set_title('Sample diversity')
    ax2.axhline(np.mean(divs), color='red', linestyle='--',
                label=f'mean={np.mean(divs):.4f}')
    ax2.legend()

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate MNIST inpainting')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to flow checkpoint')
    parser.add_argument('--gpa-particles', type=str, default=None,
                        help='Path to GPA particles file (optional)')
    parser.add_argument('--n-test', type=int, default=10, help='Number of test images')
    parser.add_argument('--n-samples', type=int, default=8, help='Samples per test image for flow')
    parser.add_argument('--n-show', type=int, default=5, help='Completions to show per image')
    parser.add_argument('--n-steps', type=int, default=20, help='ODE integration steps')
    parser.add_argument('--output-dir', type=str, default='results/mnist', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    vel_net, hparams, state = load_checkpoint(args.checkpoint, device)
    theta_dim = hparams['theta_dim']
    y_dim = hparams['y_dim']

    # Load test data
    test_data = MNISTInpainting(train=False, data_root='./data')
    y_test, theta_true = test_data.get_test_images(n=args.n_test)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Flow samples ---
    print(f"Generating {args.n_samples} flow samples per test image...")
    flow_samples_list = []
    flow_metrics = []

    for i in range(args.n_test):
        y_i = y_test[i:i+1].expand(args.n_samples, -1).to(device)
        z = torch.randn(args.n_samples, theta_dim, device=device)

        with torch.no_grad():
            traj = euler_integrate(vel_net, z, y_i, args.n_steps)
            samples = traj[-1].cpu()

        flow_samples_list.append(samples.numpy())
        metrics = compute_metrics(samples, theta_true[i])
        flow_metrics.append(metrics)
        print(f"  Image {i}: MSE={metrics['mse']:.4f}, diversity={metrics['diversity']:.4f}")

    # Plot flow completions
    plot_completions(
        y_test.numpy(), flow_samples_list, theta_true.numpy(),
        save_path=str(output_dir / "flow_completions.png"),
        n_show=args.n_show,
        title="W1W2 Flow Completions"
    )

    # Plot metrics
    plot_metrics_summary(
        flow_metrics,
        save_path=str(output_dir / "flow_metrics.png"),
        title="W1W2 Flow Metrics"
    )

    # Print summary
    mean_mse = np.mean([m['mse'] for m in flow_metrics])
    mean_div = np.mean([m['diversity'] for m in flow_metrics])
    print(f"\nFlow results: mean MSE={mean_mse:.4f}, mean diversity={mean_div:.4f}")

    # --- GPA-refined samples (if available) ---
    gpa_path = args.gpa_particles
    if gpa_path is None:
        # Try default path
        default_gpa = output_dir / "gpa_particles.pt"
        if default_gpa.exists():
            gpa_path = str(default_gpa)

    if gpa_path and Path(gpa_path).exists():
        print(f"\nLoading GPA particles: {gpa_path}")
        gpa_data = torch.load(gpa_path, map_location='cpu', weights_only=False)
        gpa_particles = gpa_data['particles_refined']
        gpa_n_samples = gpa_data['n_samples']
        gpa_n_test = gpa_data['n_test']
        gpa_y_test = gpa_data['y_test']
        gpa_theta_true = gpa_data['theta_test_true']

        n_eval = min(args.n_test, gpa_n_test)

        gpa_samples_list = []
        gpa_metrics = []
        for i in range(n_eval):
            start = i * gpa_n_samples
            end = start + gpa_n_samples
            samples_i = gpa_particles[start:end].numpy()
            gpa_samples_list.append(samples_i)

            metrics = compute_metrics(
                gpa_particles[start:end], gpa_theta_true[i]
            )
            gpa_metrics.append(metrics)
            print(f"  Image {i}: MSE={metrics['mse']:.4f}, "
                  f"diversity={metrics['diversity']:.4f}")

        # Plot GPA completions
        plot_completions(
            gpa_y_test[:n_eval].numpy(), gpa_samples_list,
            gpa_theta_true[:n_eval].numpy(),
            save_path=str(output_dir / "gpa_completions.png"),
            n_show=args.n_show,
            title="GPA-Refined Completions"
        )

        plot_metrics_summary(
            gpa_metrics,
            save_path=str(output_dir / "gpa_metrics.png"),
            title="GPA-Refined Metrics"
        )

        mean_mse_gpa = np.mean([m['mse'] for m in gpa_metrics])
        mean_div_gpa = np.mean([m['diversity'] for m in gpa_metrics])
        print(f"\nGPA results: mean MSE={mean_mse_gpa:.4f}, "
              f"mean diversity={mean_div_gpa:.4f}")

        # --- Side-by-side comparison ---
        print("\n--- Comparison ---")
        print(f"{'':>10s} {'MSE':>10s} {'Diversity':>10s}")
        print(f"{'Flow':>10s} {mean_mse:.4f} {mean_div:>10.4f}")
        print(f"{'GPA':>10s} {mean_mse_gpa:.4f} {mean_div_gpa:>10.4f}")
    else:
        print("\nNo GPA particles found. Run gpa_refine first to compare.")

    # --- Plot training curves if available ---
    if 'history' in state:
        history = state['history']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if 'L_dual' in history:
            ax1.plot(history['L_dual'], alpha=0.3, color='blue')
            # Smoothed
            window = min(100, len(history['L_dual']))
            if window > 1:
                smoothed = np.convolve(history['L_dual'],
                                       np.ones(window)/window, mode='valid')
                ax1.plot(smoothed, color='blue', linewidth=2)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('L_dual')
            ax1.set_title('Dual Loss')
            ax1.grid(True, alpha=0.3)

        if 'KE' in history:
            ax2.plot(history['KE'], alpha=0.3, color='orange')
            window = min(100, len(history['KE']))
            if window > 1:
                smoothed = np.convolve(history['KE'],
                                       np.ones(window)/window, mode='valid')
                ax2.plot(smoothed, color='orange', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('KE')
            ax2.set_title('Kinetic Energy')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        curves_path = output_dir / "training_curves.png"
        plt.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {curves_path}")


if __name__ == '__main__':
    main()
