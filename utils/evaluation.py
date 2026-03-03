"""Evaluation and plotting utilities for conditional flows."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from pathlib import Path

from .integrators import euler_integrate


def generate_posterior(
    vel_net,
    y_obs: float,
    n_samples: int = 1000,
    n_steps: int = 40,
    theta_dim: int = 2,
    y_dim: int = 1,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """Sample from learned posterior p(θ|y).

    Args:
        vel_net: Trained velocity network
        y_obs: Observation value
        n_samples: Number of samples to generate
        n_steps: Number of integration steps
        theta_dim: Dimension of θ
        y_dim: Dimension of y
        device: Device to use

    Returns:
        Samples, shape (n_samples, theta_dim)
    """
    if device is None:
        device = next(vel_net.parameters()).device

    vel_net.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, theta_dim, device=device)
        if y_dim == 1:
            y = torch.full((n_samples, 1), y_obs, device=device)
        else:
            y = torch.tensor(y_obs, device=device).expand(n_samples, y_dim)
        traj = euler_integrate(vel_net, z, y, n_steps)
        return traj[-1].cpu().numpy()


def evaluate_flow(
    vel_net,
    problem,
    y_test_values: Optional[List[float]] = None,
    n_samples: int = 5000,
    n_steps: int = 40,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Evaluate a trained flow on a problem.

    Args:
        vel_net: Trained velocity network
        problem: Problem instance with compute_distance method
        y_test_values: Test y values (defaults to problem's defaults)
        n_samples: Number of samples per y value
        n_steps: Integration steps
        device: Device to use

    Returns:
        Dictionary with evaluation metrics
    """
    if y_test_values is None:
        y_test_values = problem.default_y_test_values()

    if device is None:
        device = next(vel_net.parameters()).device

    results = {
        'y_values': y_test_values,
        'mean_distances': [],
        'std_distances': [],
        'samples': {},
    }

    for y_val in y_test_values:
        samples = generate_posterior(
            vel_net, y_val, n_samples, n_steps,
            problem.theta_dim, problem.y_dim, device
        )
        dist = problem.compute_distance(samples, y_val)

        results['mean_distances'].append(float(np.mean(dist)))
        results['std_distances'].append(float(np.std(dist)))
        results['samples'][y_val] = samples

    return results


def plot_results(
    vel_net,
    problem,
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    y_test_values: Optional[List[float]] = None,
    n_samples: int = 3000,
    n_steps: int = 40,
    device: Optional[torch.device] = None,
    run_config: Optional[Dict[str, Any]] = None
) -> plt.Figure:
    """Generate evaluation plots for a trained flow.

    Args:
        vel_net: Trained velocity network
        problem: Problem instance
        history: Training history with 'L_dual' and 'KE' keys
        save_path: Path to save figure (optional)
        y_test_values: Test y values
        n_samples: Samples per y value for plotting
        n_steps: Integration steps
        device: Device to use
        run_config: Dict with run configuration for labeling
            Keys: 'epochs', 'lam', 'lip_scale', 'vel_layers', 'vel_hidden',
                  'n_steps', 'quad_features', 'run_id'

    Returns:
        Matplotlib figure
    """
    if y_test_values is None:
        y_test_values = problem.default_y_test_values()

    if device is None:
        device = next(vel_net.parameters()).device

    if run_config is None:
        run_config = {}

    n_test = len(y_test_values)
    fig, axes = plt.subplots(2, max(4, n_test), figsize=(4 * max(4, n_test), 8))

    # Create descriptive title
    run_id = run_config.get('run_id', '')
    if run_id:
        fig.suptitle(f"W1W2 Flow: {problem.name} | {run_id}", fontsize=14, fontweight='bold')
    else:
        fig.suptitle(f"W1W2 Flow: {problem.name}", fontsize=14, fontweight='bold')

    # Top row: posterior samples for each y value
    for i, y_val in enumerate(y_test_values[:4]):
        ax = axes[0, i]

        # Generate samples
        samples = generate_posterior(
            vel_net, y_val, n_samples, n_steps,
            problem.theta_dim, problem.y_dim, device
        )

        # Plot true posterior if available
        problem.plot_true_posterior(ax, y_val, alpha=0.7)

        # Plot generated samples
        if problem.theta_dim == 2:
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=5, label='W1W2 Flow')

        # Compute distance
        dist = problem.compute_distance(samples, y_val)

        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_title(f'y={y_val}: dist={dist.mean():.3f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Bottom left: marginal distribution
    ax = axes[1, 0]
    y_test = y_test_values[0] if y_test_values else 0.0
    samples = generate_posterior(
        vel_net, y_test, 5000, n_steps,
        problem.theta_dim, problem.y_dim, device
    )
    ax.hist(samples[:, 0], bins=50, density=True, alpha=0.7, label='W1W2 Flow')

    # True marginal if available
    grid = np.linspace(-4, 4, 200)
    true_pdf = problem.true_posterior_pdf(grid, y_test, dim=0)
    if true_pdf is not None:
        ax.plot(grid, true_pdf, 'r-', linewidth=2, label='True p(θ₁|y)')

    ax.set_xlabel('θ₁')
    ax.set_ylabel('Density')
    ax.set_title(f'θ₁ Marginal (y={y_test})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Training curves
    ax = axes[1, 1]
    if 'L_dual' in history:
        ax.plot(history['L_dual'], label='L_dual')
    if 'KE' in history:
        ax.plot(history['KE'], label='KE')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Distance histogram
    ax = axes[1, 2]
    samples = generate_posterior(
        vel_net, y_test, 5000, n_steps,
        problem.theta_dim, problem.y_dim, device
    )
    dist = problem.compute_distance(samples, y_test)
    ax.hist(dist, bins=50, density=True, alpha=0.7)
    ax.axvline(dist.mean(), color='r', linestyle='--', label=f'mean={dist.mean():.3f}')
    ax.set_xlabel('Distance to constraint')
    ax.set_ylabel('Density')
    ax.set_title(f'Distance (y={y_test})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Statistics summary with hyperparameters
    ax = axes[1, 3]

    # Build config summary
    text = "W1W2 Flow Results\n"
    text += "=" * 32 + "\n\n"
    text += f"Problem: {problem.name}\n"
    text += f"{problem.description}\n\n"

    # Hyperparameters section
    text += "Hyperparameters:\n"
    text += "-" * 32 + "\n"
    if 'epochs' in run_config:
        text += f"  epochs: {run_config['epochs']}\n"
    if 'lam' in run_config:
        text += f"  λ (KE): {run_config['lam']}\n"
    if 'lip_scale' in run_config:
        text += f"  Lip scale: {run_config['lip_scale']}\n"
    if 'vel_layers' in run_config and 'vel_hidden' in run_config:
        text += f"  Velocity: {run_config['vel_layers']}×{run_config['vel_hidden']}\n"
    if 'n_steps' in run_config:
        text += f"  ODE steps: {run_config['n_steps']}\n"
    if run_config.get('quad_features'):
        text += f"  Quad features: Yes\n"

    text += "\nResults:\n"
    text += "-" * 32 + "\n"

    for y_val in y_test_values:
        samples = generate_posterior(
            vel_net, y_val, 5000, n_steps,
            problem.theta_dim, problem.y_dim, device
        )
        dist = problem.compute_distance(samples, y_val)
        text += f"  y={y_val:5.1f}: dist={dist.mean():.4f}\n"

    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    ax.set_title('Configuration & Results')

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)  # Make room for suptitle

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig
