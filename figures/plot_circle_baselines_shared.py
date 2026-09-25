"""Circle baselines comparison with shared axes across all panels."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path('results/circle/14_sweep/baselines')
particles = np.load(base_dir / 'all_particles.npz')

y_test_values = [0.25, 0.5, 1.0, 2.0, 4.0]
# Drop CNF-MLE (all NaN)
method_names = ['W1W2', 'W1W2_GPA', 'CFM', 'SGM']
method_labels = ['W1W2 Flow', 'W1W2 + GPA', 'CFM', 'SGM']
method_colors = ['#4a90d9', '#d73027', '#2ca02c', '#ff7f0e']

lim = 2.5
n_rows = len(method_names)
n_cols = len(y_test_values)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
                         sharex=True, sharey=True)

for col, y_val in enumerate(y_test_values):
    # True circle
    r = np.sqrt(y_val)
    ang = np.linspace(0, 2 * np.pi, 200)
    cx, cy = r * np.cos(ang), r * np.sin(ang)

    for row, (mkey, mlabel, mcol) in enumerate(zip(method_names, method_labels, method_colors)):
        ax = axes[row, col]

        # Draw true circle
        ax.plot(cx, cy, 'r-', lw=1.8, alpha=0.6, zorder=1)

        # Plot samples
        key = f'{mkey}_y{y_val}'
        samples = particles[key]
        ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.25,
                   c=mcol, zorder=2, rasterized=True)

        ax.set_aspect('equal')
        ax.grid(True, alpha=0.15)

    axes[0, col].set_title(f'$y = {y_val}$\n$r = {r:.2f}$', fontsize=11, fontweight='bold')

axes[0, 0].set_xlim(-lim, lim)
axes[0, 0].set_ylim(-lim, lim)

for row, mlabel in enumerate(method_labels):
    axes[row, 0].set_ylabel(mlabel + '\n' + r'$\theta_2$', fontsize=10)
for col in range(n_cols):
    axes[n_rows - 1, col].set_xlabel(r'$\theta_1$', fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent / 'circle_baselines_shared.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
