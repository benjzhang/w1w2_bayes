"""Replot baseline comparison for bimodal quadratic with consistent axes.

Loads saved particles from 14_sweep/baselines and replots with same
axis ranges as bimodal_intro.py.
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from problems.bimodal_quadratic import BimodalQuadraticProblem

np.random.seed(42)

prob = BimodalQuadraticProblem(offset=2.0, mix_prob=0.5)
base_dir = Path('results/bimodal_quadratic/14_sweep/baselines')

# Load saved particles and results
particles = np.load(base_dir / 'all_particles.npz')
with open(base_dir / 'results.json') as f:
    results = json.load(f)

y_test_values = [-1.0, 0.0, 1.0, 2.0]
method_names = ['W1W2', 'W1W2+GPA', 'CFM', 'SGM']  # skip CNF-MLE (NaN)
method_keys = ['W1W2', 'W1W2_GPA', 'CFM', 'SGM']
method_display = ['W1W2 Flow', 'W1W2 + GPA', 'CFM', 'SGM']
method_colors = ['#4a90d9', '#d73027', '#1a9850', '#e69f00']

xlim = (-3.5, 3.5)
ylim = (-5, 5)

# ── Main comparison figure: one row per method, one col per y ──
n_methods = len(method_names)
n_test = len(y_test_values)

fig, axes = plt.subplots(n_methods, n_test,
                         figsize=(3.5 * n_test, 3.5 * n_methods))

for col, y_val in enumerate(y_test_values):
    # True posterior samples for reference
    true_samples = prob.sample_true_posterior(y_val, 5000)
    t1 = np.linspace(*xlim, 300)
    branch1 = y_val - t1**2
    branch2 = y_val + prob.offset - t1**2

    for row, (name, key, display, color) in enumerate(
            zip(method_names, method_keys, method_display, method_colors)):
        ax = axes[row, col]

        # Draw true posterior curves
        ax.plot(t1, branch1, '-', color='#aaaaaa', lw=1.5, zorder=1)
        ax.plot(t1, branch2, '-', color='#cccccc', lw=1.5, zorder=1)

        # Scatter generated samples
        pkey = f'{key}_y{y_val}'
        samples = particles[pkey]
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.3,
                   c=color, zorder=2, rasterized=True)


        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.15)

        if row == 0:
            ax.set_title(f'$y = {y_val:.0f}$', fontsize=13, fontweight='bold')
        if col == 0:
            ax.set_ylabel(f'{display}\n' + r'$\theta_2$', fontsize=11)
        else:
            ax.set_ylabel(r'$\theta_2$', fontsize=10)
        if row == n_methods - 1:
            ax.set_xlabel(r'$\theta_1$', fontsize=11)

plt.tight_layout()
out = Path(__file__).resolve().parent / 'baselines_bimodal.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")


# ── Single-method figures ──
def plot_single_method(name, key, display, color, filename):
    fig, axes = plt.subplots(1, n_test, figsize=(3.5 * n_test, 3.5),
                             sharey=True)
    t1 = np.linspace(*xlim, 300)
    for col, y_val in enumerate(y_test_values):
        ax = axes[col]
        branch1 = y_val - t1**2
        branch2 = y_val + prob.offset - t1**2
        ax.plot(t1, branch1, '-', color='#aaaaaa', lw=1.5, zorder=1)
        ax.plot(t1, branch2, '-', color='#cccccc', lw=1.5, zorder=1)
        pkey = f'{key}_y{y_val}'
        samples = particles[pkey]
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.3,
                   c=color, zorder=2, rasterized=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.15)
        ax.set_title(f'$y = {y_val:.0f}$', fontsize=13, fontweight='bold')
        ax.set_xlabel(r'$\theta_1$', fontsize=11)
    axes[0].set_ylabel(r'$\theta_2$', fontsize=11)
    fig.suptitle(display, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    out_path = Path(__file__).resolve().parent / filename
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.pdf')}")

plot_single_method('W1W2', 'W1W2', 'W1W2 Flow', '#4a90d9', 'w1w2_bimodal.png')
plot_single_method('W1W2+GPA', 'W1W2_GPA', 'W1W2 Flow + GPA', '#d73027', 'w1w2_gpa_bimodal.png')
