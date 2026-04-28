"""Langevin dynamics animation on a donut posterior.

Likelihood: p(y|θ) ∝ exp(-(y - ||θ||²)² / 2σ²)
Prior: p(θ) = N(0, I)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path

np.random.seed(42)

y_obs = 2.0
r_true = np.sqrt(y_obs)
sigma_like = 0.15

n_particles = 2000
n_steps = 150
dt = 0.001
save_every = 1

particles = np.random.randn(n_particles, 2)

frames = []
for step in range(n_steps + 1):
    if step % save_every == 0:
        frames.append((step, particles.copy()))
    if step < n_steps:
        r2 = particles[:, 0]**2 + particles[:, 1]**2
        grad = -particles - (r2 - y_obs)[:, None] * 2 * particles / sigma_like**2
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad = np.where(grad_norm > 100, grad * 100 / grad_norm, grad)
        particles = particles + dt * grad + np.sqrt(2 * dt) * np.random.randn(n_particles, 2)

lim = 2.5

# Donut density on polar grid
r_ring = np.linspace(0, lim, 200)
theta_grid = np.linspace(0, 2 * np.pi, 200)
R, TH = np.meshgrid(r_ring, theta_grid)
XG = R * np.cos(TH)
YG = R * np.sin(TH)
log_post = -0.5 * R**2 - (R**2 - y_obs)**2 / (2 * sigma_like**2)
post_density = np.exp(log_post - log_post.max())

fig, ax = plt.subplots(1, 1, figsize=(5, 5))

def animate(i):
    ax.clear()
    ax.contourf(XG, YG, post_density, levels=30, cmap='Reds', alpha=0.4, zorder=0)
    ax.contour(XG, YG, post_density, levels=6, colors='#d73027', linewidths=0.5, alpha=0.4, zorder=0)
    step, pts = frames[i]
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.3, s=5, color='steelblue', zorder=2)
    ax.set_title(f'Langevin step {step}', fontsize=11)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('θ₁')
    ax.set_ylabel('θ₂')
    ax.grid(True, alpha=0.3)
    return []

anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                interval=100, blit=True)
out = Path(__file__).resolve().parent / 'langevin_circle.gif'
anim.save(str(out), writer=PillowWriter(fps=10))
plt.close()
print(f"Saved: {out} ({len(frames)} frames)")
