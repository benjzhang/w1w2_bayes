"""Three-row comparison: True posterior / W1W2 Flow / W1W2 + GPA.

Columns = y values, rows = truth, flow, flow+gpa.
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

particles = np.load(base_dir / 'all_particles.npz')

y_test_values = [-1.0, 0.0, 1.0, 2.0]
xlim = (-3.5, 3.5)
ylim = (-5, 5)

n_cols = len(y_test_values)
n_rows = 3
row_labels = [r'True posterior $p(\theta \mid y)$', 'W1W2 Flow', 'W1W2 Flow + GPA']
row_colors = ['#d73027', '#4a90d9', '#d73027']
row_keys = [None, 'W1W2', 'W1W2_GPA']

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
                         sharex=True, sharey=True)

t1 = np.linspace(*xlim, 300)

for col, y_val in enumerate(y_test_values):
    branch1 = y_val - t1**2
    branch2 = y_val + prob.offset - t1**2

    for row in range(n_rows):
        ax = axes[row, col]
        # Reference curves
        ax.plot(t1, branch1, '-', color='#aaaaaa', lw=1.5, zorder=1)
        ax.plot(t1, branch2, '-', color='#cccccc', lw=1.5, zorder=1)

        if row == 0:
            # True posterior samples
            post = prob.sample_true_posterior(y_val, 5000)
            ax.scatter(post[:, 0], post[:, 1], s=2, alpha=0.2,
                       c=row_colors[row], zorder=2, rasterized=True)
        else:
            pkey = f'{row_keys[row]}_y{y_val}'
            samples = particles[pkey]
            ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.3,
                       c=row_colors[row], zorder=2, rasterized=True)

        ax.grid(True, alpha=0.15)

    axes[0, col].set_title(f'$y = {y_val:.0f}$', fontsize=13, fontweight='bold')

axes[0, 0].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)

for row in range(n_rows):
    axes[row, 0].set_ylabel(row_labels[row] + '\n' + r'$\theta_2$', fontsize=11)
for col in range(n_cols):
    axes[n_rows - 1, col].set_xlabel(r'$\theta_1$', fontsize=11)

plt.tight_layout()
out = Path(__file__).resolve().parent / 'truth_flow_gpa.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
