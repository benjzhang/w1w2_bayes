#!/usr/bin/env python
"""Plot the true posterior for the bimodal quadratic problem.

For each y value, the posterior p(θ|y) is a mixture of two distributions,
each supported on a parabola:
    Branch 1 (shifted): θ₂ = y + offset - θ₁²
    Branch 2 (original): θ₂ = y - θ₁²

The density along each branch is determined by the N(0,I) prior:
    p(θ₁ | y, branch k) ∝ exp(-θ₁²/2) · exp(-(y + c_k - θ₁²)²/2)

where c_k is the offset for that branch.
"""

import numpy as np
import matplotlib.pyplot as plt
from problems import get_problem

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def sample_true_posterior(problem, y_obs, n_samples=5000):
    """Sample from the true posterior p(θ|y) for the bimodal quadratic problem.

    Strategy: sample θ₁ from the marginal p(θ₁|y) via rejection sampling,
    then assign to a branch with the correct conditional probability,
    then set θ₂ deterministically from the parabola equation.
    """
    offset = problem.offset
    mix_prob = problem.mix_prob

    # Unnormalized log-densities for each branch as a function of θ₁
    def log_density_branch(t1, c):
        return -t1**2 / 2 - (y_obs + c - t1**2)**2 / 2

    # Fine grid for building the proposal
    t1_grid = np.linspace(-4, 4, 10000)

    log_p1 = log_density_branch(t1_grid, offset)   # shifted branch
    log_p2 = log_density_branch(t1_grid, 0.0)      # original branch

    # Unnormalized mixture density
    log_max = max(log_p1.max(), log_p2.max())
    p1 = np.exp(log_p1 - log_max)
    p2 = np.exp(log_p2 - log_max)
    p_mix = mix_prob * p1 + (1 - mix_prob) * p2

    # Normalize to get a proper PDF on the grid
    Z = _trapz(p_mix, t1_grid)
    pdf = p_mix / Z

    # CDF for inverse transform sampling
    cdf = np.cumsum(pdf) * (t1_grid[1] - t1_grid[0])
    cdf = cdf / cdf[-1]  # ensure it ends at 1

    # Sample θ₁ via inverse CDF
    u = np.random.rand(n_samples)
    t1_samples = np.interp(u, cdf, t1_grid)

    # For each sample, decide which branch using the conditional probability
    # p(branch 1 | θ₁, y) = mix_prob * p1(θ₁) / (mix_prob * p1(θ₁) + (1-mix_prob) * p2(θ₁))
    lp1 = log_density_branch(t1_samples, offset)
    lp2 = log_density_branch(t1_samples, 0.0)
    lp_max = np.maximum(lp1, lp2)
    w1 = mix_prob * np.exp(lp1 - lp_max)
    w2 = (1 - mix_prob) * np.exp(lp2 - lp_max)
    prob_branch1 = w1 / (w1 + w2)

    use_branch1 = np.random.rand(n_samples) < prob_branch1

    # Set θ₂ from the parabola
    t2_samples = np.where(use_branch1,
                          y_obs + offset - t1_samples**2,
                          y_obs - t1_samples**2)

    return np.column_stack([t1_samples, t2_samples]), use_branch1


def main():
    problem = get_problem('bimodal_quadratic')
    y_values = problem.default_y_test_values()
    n_samples = 5000
    out_dir = 'results/bimodal_quadratic'

    # --- Figure 1: Parabola curves ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    axes1 = axes1.flatten()

    for i, y_obs in enumerate(y_values):
        ax = axes1[i]
        t1 = np.linspace(-3.5, 3.5, 200)
        ax.plot(t1, y_obs + problem.offset - t1**2, 'r-', linewidth=2,
                label=f'θ₂ = y + {problem.offset} − θ₁²')
        ax.plot(t1, y_obs - t1**2, 'b-', linewidth=2,
                label='θ₂ = y − θ₁²')

        # Plot marginal density
        t1_grid = np.linspace(-4, 4, 500)
        pdf = problem.true_posterior_pdf(t1_grid, y_obs, dim=0)
        if pdf is not None:
            ylim_lo = y_obs - 3.5**2
            pdf_scaled = ylim_lo - 0.5 + pdf / pdf.max() * 1.5
            ax.plot(t1_grid, pdf_scaled, 'k-', linewidth=1, alpha=0.5,
                    label='p(θ₁|y) marginal')

        ax.set_title(f'y = {y_obs}', fontsize=14)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_xlim(-3.5, 3.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

    fig1.suptitle('Constraint Curves — Bimodal Quadratic\n'
                  'θ₁² + θ₂ = y (blue) and θ₁² + θ₂ = y + offset (red)',
                  fontsize=13)
    plt.tight_layout()
    path1 = f'{out_dir}/true_posterior_curves.png'
    fig1.savefig(path1, dpi=150, bbox_inches='tight')
    print(f'Saved: {path1}')
    plt.close(fig1)

    # --- Figure 2: Scatter plots ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    axes2 = axes2.flatten()

    for i, y_obs in enumerate(y_values):
        ax = axes2[i]
        samples, branch_mask = sample_true_posterior(problem, y_obs, n_samples)

        ax.scatter(samples[branch_mask, 0], samples[branch_mask, 1],
                   s=3, alpha=0.3, c='red', label='Branch 1 (shifted)')
        ax.scatter(samples[~branch_mask, 0], samples[~branch_mask, 1],
                   s=3, alpha=0.3, c='blue', label='Branch 2 (original)')

        ax.set_title(f'y = {y_obs}', fontsize=14)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.set_xlim(-3.5, 3.5)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.2)

        n1 = branch_mask.sum()
        n2 = (~branch_mask).sum()
        ax.text(0.02, 0.02, f'Branch 1: {n1} ({100*n1/n_samples:.0f}%)\n'
                             f'Branch 2: {n2} ({100*n2/n_samples:.0f}%)',
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig2.suptitle('True Posterior Samples p(θ|y) — Bimodal Quadratic\n'
                  'Prior: θ ~ N(0, I₂)',
                  fontsize=13)
    plt.tight_layout()
    path2 = f'{out_dir}/true_posterior_samples.png'
    fig2.savefig(path2, dpi=150, bbox_inches='tight')
    print(f'Saved: {path2}')
    plt.close(fig2)


if __name__ == '__main__':
    main()
