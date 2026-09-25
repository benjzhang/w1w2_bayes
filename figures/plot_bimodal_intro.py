"""Presentation figure: introduce the bimodal quadratic inverse problem.

Columns = y values, Rows = Prior / Likelihood / Posterior.
Matches the layout of the baselines figures for easy comparison.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from problems.bimodal_quadratic import BimodalQuadraticProblem
from scipy.stats import multivariate_normal

np.random.seed(42)

prob = BimodalQuadraticProblem(offset=2.0, mix_prob=0.5)
y_values = [-1.0, 0.0, 1.0, 2.0]

xlim = (-3.5, 3.5)
ylim = (-5, 5)

# Prior density on grid
xg = np.linspace(*xlim, 300)
yg = np.linspace(*ylim, 300)
X, Y = np.meshgrid(xg, yg)
pos = np.stack([X.ravel(), Y.ravel()], axis=1)
rv = multivariate_normal(mean=[0, 0], cov=np.eye(2))
Z = rv.pdf(pos).reshape(X.shape)

n_cols = len(y_values)
n_rows = 3  # prior, likelihood, posterior
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows),
                         sharex=True, sharey=True)

row_labels = [r'Prior $p(\theta)$', 'Likelihood', r'Posterior $p(\theta \mid y)$']

for col, y_obs in enumerate(y_values):
    post_samples = prob.sample_true_posterior(y_obs, 5000)
    t1 = np.linspace(*xlim, 300)

    # ─── Row 0: Prior ───
    ax = axes[0, col]
    ax.contourf(X, Y, Z, levels=30, cmap='Blues', alpha=0.8)
    ax.contour(X, Y, Z, levels=6, colors='#2166ac', linewidths=0.5, alpha=0.5)
    ax.set_title(f'$y = {y_obs:.0f}$', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.15)

    # ─── Row 1: Likelihood level sets ───
    ax = axes[1, col]
    ax.plot(t1, y_obs - t1**2, '-', color='#2166ac', lw=2.5,
            label=r'$\theta_2 = y - \theta_1^2$')
    ax.plot(t1, y_obs + prob.offset - t1**2, '-', color='#1a9850', lw=2.5,
            label=r'$\theta_2 = y + \Delta - \theta_1^2$')
    if col == 0:
        ax.legend(fontsize=8, loc='lower center', framealpha=0.9)
    ax.grid(True, alpha=0.15)

    # ─── Row 2: Posterior samples ───
    ax = axes[2, col]
    ax.scatter(post_samples[:, 0], post_samples[:, 1], s=2, alpha=0.2,
               c='#d73027', rasterized=True)
    ax.grid(True, alpha=0.15)

# Set shared limits
axes[0, 0].set_xlim(xlim)
axes[0, 0].set_ylim(ylim)

# Only label edges
for row in range(n_rows):
    axes[row, 0].set_ylabel(row_labels[row] + '\n' + r'$\theta_2$', fontsize=11)
for col in range(n_cols):
    axes[n_rows - 1, col].set_xlabel(r'$\theta_1$', fontsize=11)

plt.tight_layout()
out = Path(__file__).resolve().parent / 'bimodal_intro.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
