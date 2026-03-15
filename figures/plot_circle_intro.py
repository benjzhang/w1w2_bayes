"""Presentation figure: introduce the circle inverse problem.

Columns = y values, Rows = Prior / Likelihood / Posterior.
Matches the layout of the bimodal_intro figure.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from problems.circle import CircleProblem
from scipy.stats import multivariate_normal

np.random.seed(42)

prob = CircleProblem()
y_values = [0.25, 0.5, 1.0, 2.0, 4.0]

lim = 2.5

# Prior density on grid
xg = np.linspace(-lim, lim, 300)
yg = np.linspace(-lim, lim, 300)
X, Y = np.meshgrid(xg, yg)
pos = np.stack([X.ravel(), Y.ravel()], axis=1)
rv = multivariate_normal(mean=[0, 0], cov=np.eye(2))
Z = rv.pdf(pos).reshape(X.shape)

n_cols = len(y_values)
n_rows = 3
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows),
                         sharex=True, sharey=True)

row_labels = [r'Prior $p(\theta)$', r'Likelihood $p(y \mid \theta)$', r'Posterior $p(\theta \mid y)$']

for col, y_obs in enumerate(y_values):
    r = np.sqrt(y_obs)
    ang = np.linspace(0, 2 * np.pi, 500)

    # ─── Row 0: Prior ───
    ax = axes[0, col]
    ax.contourf(X, Y, Z, levels=30, cmap='Blues', alpha=0.8)
    ax.contour(X, Y, Z, levels=6, colors='#2166ac', linewidths=0.5, alpha=0.5)
    ax.set_title(f'$y = {y_obs}$\n$r = {r:.2f}$', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    # ─── Row 1: Likelihood — the constraint circle ───
    ax = axes[1, col]
    # Shade interior to indicate where y = θ₁² + θ₂² = y_obs is satisfied
    theta_fill = np.linspace(0, 2 * np.pi, 500)
    ax.fill(r * np.cos(theta_fill), r * np.sin(theta_fill),
            color='#2166ac', alpha=0.08)
    ax.plot(r * np.cos(ang), r * np.sin(ang), '-', color='#2166ac',
            lw=2.5, label=r'$\|\theta\|^2 = y$')
    if col == 0:
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    # ─── Row 2: Posterior samples (uniform on circle) ───
    ax = axes[2, col]
    post_samples = prob.sample_true_posterior(y_obs, 3000)
    ax.scatter(post_samples[:, 0], post_samples[:, 1], s=3, alpha=0.3,
               c='#d73027', rasterized=True)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

axes[0, 0].set_xlim(-lim, lim)
axes[0, 0].set_ylim(-lim, lim)

for row in range(n_rows):
    axes[row, 0].set_ylabel(row_labels[row] + '\n' + r'$\theta_2$', fontsize=10)
for col in range(n_cols):
    axes[n_rows - 1, col].set_xlabel(r'$\theta_1$', fontsize=10)

plt.tight_layout()
out = Path(__file__).resolve().parent / 'circle_intro.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
plt.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved: {out}")
print(f"Saved: {out.with_suffix('.pdf')}")
