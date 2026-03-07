#!/usr/bin/env python
"""Self-driving adaptive MNIST inpainting sweep.

Three phases:
  Phase 1: Quick probe — train at each architecture tier for 5k iters.
           If MSE is good enough, stop. Otherwise escalate (MLP small →
           MLP large → CNN).
  Phase 2: Full sweep — train multiple hyperparameter configs at the chosen
           architecture. Pick the best by MSE.
  Phase 3: GPA refinement — run GPA on the best flow with several configs.

Submit and walk away:
    sbatch mnist/sweep.sh

Or run directly on a GPU node:
    python -m mnist.sweep
"""

import argparse
import csv
import json
import time
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from .data import MNISTInpainting
from .networks import (
    VelocityMLP, DiscriminatorMLP,
    VelocityCNN, DiscriminatorCNN,
    build_networks,
)
from .evaluate import (
    reconstruct_image, compute_metrics, plot_completions, plot_metrics_summary,
)


# ---------------------------------------------------------------------------
# Utilities (self-contained)
# ---------------------------------------------------------------------------

def euler_integrate(vel_net, z0, y, n_steps=20):
    dt = 1.0 / n_steps
    theta = z0
    traj = [z0]
    for i in range(n_steps):
        t = torch.full((theta.shape[0], 1), i * dt, device=theta.device)
        v = vel_net(t, theta, y)
        theta = theta + dt * v
        traj.append(theta)
    return traj


def _spectral_norm_projection(layer, target_norm):
    with torch.no_grad():
        W = layer.weight
        # For Conv2d (4D), reshape to 2D: (out_ch, in_ch*kH*kW)
        if W.dim() > 2:
            W_2d = W.reshape(W.shape[0], -1)
        else:
            W_2d = W
        sigma = torch.linalg.norm(W_2d, ord=2)
        if sigma > 1e-6:
            W.mul_(target_norm / sigma)


def _project_disc_weights(disc, L):
    """Project discriminator weights for Lipschitz constant L.

    Works for both MLP (all layers in disc.net) and CNN (conv layers in
    disc.convs + linear layers in disc.net).
    """
    # Collect all weight-bearing layers
    layers = []
    if hasattr(disc, 'convs'):
        layers += [m for m in disc.convs if isinstance(m, (nn.Linear, nn.Conv2d))]
    if hasattr(disc, 'net'):
        layers += [m for m in disc.net if isinstance(m, (nn.Linear, nn.Conv2d))]

    D = len(layers)
    if D == 0:
        return
    per_layer_norm = L ** (1.0 / D)
    for layer in layers:
        _spectral_norm_projection(layer, per_layer_norm)


def write_csv(path, results):
    if not results:
        return
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def print_summary(results, best_flow_name=None):
    print(f"\n{'='*70}")
    print("SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Name':<45s} {'Type':<10s} {'MSE':>8s} {'Diversity':>10s}")
    print("-" * 75)
    for r in results:
        mse_val = r.get('mean_mse', float('nan'))
        div_val = r.get('mean_diversity', float('nan'))
        mse_str = f"{mse_val:.4f}" if not np.isnan(mse_val) else "FAILED"
        div_str = f"{div_val:.4f}" if not np.isnan(div_val) else "FAILED"
        marker = " *best" if r.get('name') == best_flow_name else ""
        print(f"{r['name']:<45s} {r['type']:<10s} {mse_str:>8s} {div_str:>10s}{marker}")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class FlowConfig:
    name: str
    arch_type: str = 'mlp'       # 'mlp' or 'cnn'
    n_iters: int = 15000
    batch_size: int = 256
    lr: float = 1e-4
    lam: float = 0.1
    n_steps: int = 10
    disc_updates: int = 5
    lip_scale: float = 10.0
    hidden: int = 512            # MLP hidden width
    n_layers: int = 4            # MLP layers
    cnn_channels: int = 32       # CNN base channels


@dataclass
class GPAConfig:
    name: str
    K: int = 300
    eta: float = 0.01
    L: float = 10.0
    disc_steps: int = 3
    disc_lr: float = 1e-3
    batch_size: int = 256
    n_samples: int = 64


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_flow(cfg: FlowConfig, theta_data, y_data, output_dir: Path, device):
    """Train one flow config. Returns (vel_net, disc, history, model_path)."""
    print(f"\n{'='*60}")
    print(f"Training: {cfg.name}")
    print(f"  arch={cfg.arch_type}, lam={cfg.lam}, lip={cfg.lip_scale}, "
          f"lr={cfg.lr}, iters={cfg.n_iters}")
    if cfg.arch_type == 'mlp':
        print(f"  hidden={cfg.hidden}, layers={cfg.n_layers}")
    else:
        print(f"  cnn_channels={cfg.cnn_channels}")
    print(f"{'='*60}")

    theta_dim = theta_data.shape[1]
    y_dim = y_data.shape[1]
    n_data = len(theta_data)

    vel_net, disc = build_networks(
        cfg.arch_type,
        theta_dim=theta_dim, y_dim=y_dim,
        hidden=cfg.hidden, n_layers=cfg.n_layers,
        cnn_channels=cfg.cnn_channels,
    )
    vel_net = vel_net.to(device)
    disc = disc.to(device)

    n_vel = sum(p.numel() for p in vel_net.parameters())
    n_disc = sum(p.numel() for p in disc.parameters())
    print(f"  Params: vel={n_vel:,}, disc={n_disc:,}")

    opt_vel = optim.Adam(vel_net.parameters(), lr=cfg.lr)
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.lr)
    _project_disc_weights(disc, cfg.lip_scale)

    history = {'L_dual': [], 'KE': [], 'iters': []}
    t_start = time.time()

    for it in range(1, cfg.n_iters + 1):
        idx = torch.randint(0, n_data, (cfg.batch_size,))
        theta_batch = theta_data[idx]
        y_batch = y_data[idx]
        z = torch.randn(cfg.batch_size, theta_dim, device=device)

        # Discriminator updates
        for _ in range(cfg.disc_updates):
            traj = euler_integrate(vel_net, z, y_batch, cfg.n_steps)
            theta_gen = traj[-1].detach()
            phi_gen = disc(theta_gen, y_batch)
            phi_real = disc(theta_batch, y_batch)
            L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()
            disc_loss = -L_dual
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()
            _project_disc_weights(disc, cfg.lip_scale)

        # Velocity update
        traj = euler_integrate(vel_net, z, y_batch, cfg.n_steps)
        theta_gen = traj[-1]
        phi_gen = disc(theta_gen, y_batch)
        phi_real = disc(theta_batch, y_batch)
        L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

        KE = 0.0
        dt = 1.0 / cfg.n_steps
        for i, theta_t in enumerate(traj[:-1]):
            t = torch.full((theta_t.shape[0], 1), i * dt, device=device)
            v = vel_net(t, theta_t, y_batch)
            KE += 0.5 * (v ** 2).sum(dim=1).mean() * dt

        loss = L_dual + cfg.lam * KE
        opt_vel.zero_grad()
        loss.backward()
        opt_vel.step()

        history['L_dual'].append(L_dual.item())
        history['KE'].append(KE.item() if isinstance(KE, torch.Tensor) else KE)
        history['iters'].append(it)

        if it % 1000 == 0:
            elapsed = time.time() - t_start
            recent_dual = np.mean(history['L_dual'][-200:])
            recent_ke = np.mean(history['KE'][-200:])
            print(f"  [{it}/{cfg.n_iters}] L_dual={recent_dual:.4f}, "
                  f"KE={recent_ke:.4f} ({elapsed:.0f}s)")

    # Save model
    run_dir = output_dir / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "model.pt"
    state = {
        'hparams': {
            'theta_dim': theta_dim, 'y_dim': y_dim,
            'arch_type': cfg.arch_type,
            'hidden': cfg.hidden, 'n_layers': cfg.n_layers,
            'cnn_channels': cfg.cnn_channels,
            'lip_scale': cfg.lip_scale,
        },
        'vel_net_state': vel_net.state_dict(),
        'disc_state': disc.state_dict(),
        'history': history,
        'iter': cfg.n_iters,
        'config': asdict(cfg),
    }
    torch.save(state, model_path)
    elapsed = time.time() - t_start
    print(f"  Saved: {model_path} ({elapsed:.0f}s)")

    return vel_net, disc, history, model_path


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_flow(vel_net, y_test, theta_true, theta_dim, n_samples=64,
                  n_steps=20, device='cpu'):
    vel_net.eval()
    n_test = len(y_test)
    all_metrics = []
    all_samples = []

    with torch.no_grad():
        for i in range(n_test):
            y_i = y_test[i:i+1].expand(n_samples, -1).to(device)
            z = torch.randn(n_samples, theta_dim, device=device)
            traj = euler_integrate(vel_net, z, y_i, n_steps)
            samples = traj[-1].cpu()
            all_samples.append(samples)
            metrics = compute_metrics(samples, theta_true[i])
            all_metrics.append(metrics)

    mean_mse = np.mean([m['mse'] for m in all_metrics])
    mean_div = np.mean([m['diversity'] for m in all_metrics])
    return all_metrics, all_samples, mean_mse, mean_div


def compute_naive_baseline(y_test, theta_true, theta_train):
    mean_bottom = theta_train.mean(dim=0).cpu()
    mse = ((theta_true - mean_bottom.unsqueeze(0)) ** 2).mean().item()
    return mse


# ---------------------------------------------------------------------------
# GPA refinement
# ---------------------------------------------------------------------------

def run_gpa(cfg: GPAConfig, vel_net, theta_data, y_data,
            y_test, theta_true, theta_dim, y_dim, n_steps, device):
    print(f"\n  GPA: {cfg.name} (K={cfg.K}, eta={cfg.eta}, L={cfg.L})")

    vel_net.eval()
    n_test = len(y_test)
    n_total = n_test * cfg.n_samples

    y_particles = y_test.repeat_interleave(cfg.n_samples, dim=0).to(device)
    with torch.no_grad():
        z = torch.randn(n_total, theta_dim, device=device)
        traj = euler_integrate(vel_net, z, y_particles, n_steps)
        particles = traj[-1].clone()

    n_train = len(theta_data)
    perm = torch.cat([torch.randperm(n_train) for _ in range((n_total // n_train) + 1)])[:n_total]
    theta_real_buf = theta_data[perm].to(device)

    # GPA always uses MLP discriminator (no spatial structure needed for
    # gradient-based particle updates in flat theta space)
    disc = DiscriminatorMLP(
        theta_dim=theta_dim, y_dim=y_dim,
        hidden=512, n_layers=4
    ).to(device)
    _project_disc_weights(disc, cfg.L)
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.disc_lr)

    n_coupled = n_total

    for k in range(cfg.K):
        for _ in range(cfg.disc_steps):
            idx = torch.randint(0, n_coupled, (cfg.batch_size,))
            p_batch = particles[idx].detach()
            y_batch = y_particles[idx]
            t_real = theta_real_buf[idx]

            phi_fake = disc(p_batch, y_batch)
            phi_real = disc(t_real, y_batch)
            L_dual = phi_fake.mean() - torch.exp(phi_real - 1).mean()

            opt_disc.zero_grad()
            (-L_dual).backward()
            opt_disc.step()
            _project_disc_weights(disc, cfg.L)

        particles.requires_grad_(True)
        phi = disc(particles, y_particles)
        grad = torch.autograd.grad(phi.sum(), particles, create_graph=False)[0]
        with torch.no_grad():
            particles = (particles - cfg.eta * grad).detach()

        if (k + 1) % 100 == 0:
            print(f"    [GPA {k+1}/{cfg.K}] L_dual={L_dual.item():.4f}")

    all_metrics = []
    all_samples = []
    for i in range(n_test):
        start = i * cfg.n_samples
        end = start + cfg.n_samples
        samples_i = particles[start:end].detach().cpu()
        all_samples.append(samples_i)
        metrics = compute_metrics(samples_i, theta_true[i])
        all_metrics.append(metrics)

    mean_mse = np.mean([m['mse'] for m in all_metrics])
    mean_div = np.mean([m['diversity'] for m in all_metrics])
    return all_metrics, all_samples, mean_mse, mean_div


# ---------------------------------------------------------------------------
# Architecture tiers — probed in order, escalating if needed
# ---------------------------------------------------------------------------

ARCH_TIERS = [
    {
        'name': 'mlp_small',
        'arch_type': 'mlp',
        'hidden': 512, 'n_layers': 4, 'cnn_channels': 32,
        'probe_iters': 5000, 'full_iters': 15000,
    },
    {
        'name': 'mlp_large',
        'arch_type': 'mlp',
        'hidden': 1024, 'n_layers': 6, 'cnn_channels': 32,
        'probe_iters': 5000, 'full_iters': 20000,
    },
    {
        'name': 'cnn_small',
        'arch_type': 'cnn',
        'hidden': 512, 'n_layers': 4, 'cnn_channels': 32,
        'probe_iters': 5000, 'full_iters': 15000,
    },
    {
        'name': 'cnn_large',
        'arch_type': 'cnn',
        'hidden': 512, 'n_layers': 4, 'cnn_channels': 64,
        'probe_iters': 5000, 'full_iters': 20000,
    },
]

# Escalation thresholds
ESCALATE_RATIO = 0.85   # must beat naive MSE by at least 15%
GOOD_ENOUGH_MSE = 0.04  # absolute threshold — clearly working


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Adaptive MNIST inpainting sweep')
    parser.add_argument('--output-dir', type=str, default='results/mnist/sweep')
    parser.add_argument('--n-test', type=int, default=10)
    parser.add_argument('--n-eval-samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-tier', type=int, default=len(ARCH_TIERS) - 1,
                        help='Max architecture tier to try (0-indexed)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    print("Loading MNIST data...")
    train_data = MNISTInpainting(train=True, data_root='./data')
    test_data = MNISTInpainting(train=False, data_root='./data')

    theta_train, y_train = train_data.get_tensors()
    theta_train = theta_train.to(device)
    y_train = y_train.to(device)

    y_test, theta_true = test_data.get_test_images(n=args.n_test)
    theta_dim = theta_train.shape[1]
    y_dim = y_train.shape[1]
    print(f"  Train: {len(theta_train)}, theta_dim={theta_dim}, y_dim={y_dim}")

    naive_mse = compute_naive_baseline(y_test, theta_true, theta_train)
    print(f"  Naive baseline MSE (predict training mean): {naive_mse:.4f}")

    all_results = []
    csv_path = output_dir / "sweep_results.csv"

    # ==================================================================
    # Phase 1: Probe — find the right architecture tier
    # ==================================================================
    print(f"\n{'#'*60}")
    print("PHASE 1: Architecture probe")
    print(f"{'#'*60}")

    chosen_tier = None

    for tier_idx, tier in enumerate(ARCH_TIERS):
        if tier_idx > args.max_tier:
            break

        print(f"\n--- Probing tier {tier_idx}: {tier['name']} "
              f"({tier['arch_type']}) ---")

        probe_cfg = FlowConfig(
            name=f"probe_{tier['name']}",
            arch_type=tier['arch_type'],
            hidden=tier['hidden'],
            n_layers=tier['n_layers'],
            cnn_channels=tier['cnn_channels'],
            n_iters=tier['probe_iters'],
            lam=0.1,
            lip_scale=10.0,
        )

        try:
            vel_net, disc, history, model_path = train_flow(
                probe_cfg, theta_train, y_train, output_dir, device
            )
            _, samples_list, probe_mse, probe_div = evaluate_flow(
                vel_net, y_test, theta_true, theta_dim,
                n_samples=args.n_eval_samples, n_steps=probe_cfg.n_steps,
                device=device
            )

            ratio = probe_mse / naive_mse if naive_mse > 0 else float('inf')
            print(f"  Probe result: MSE={probe_mse:.4f}, "
                  f"diversity={probe_div:.4f}")
            print(f"  vs naive: {naive_mse:.4f} (ratio={ratio:.2f})")

            all_results.append({
                'name': probe_cfg.name, 'type': 'probe',
                'arch_type': tier['arch_type'],
                'hidden': tier['hidden'], 'n_layers': tier['n_layers'],
                'cnn_channels': tier['cnn_channels'],
                'mean_mse': probe_mse, 'mean_diversity': probe_div,
                'model_path': str(model_path),
            })

            run_dir = output_dir / probe_cfg.name
            plot_completions(
                y_test.numpy(), [s.numpy() for s in samples_list],
                theta_true.numpy(),
                save_path=str(run_dir / "completions.png"),
                n_show=5,
                title=f"Probe: {tier['name']} (MSE={probe_mse:.4f})"
            )

            # Decision
            if probe_mse < GOOD_ENOUGH_MSE:
                print(f"  >> MSE < {GOOD_ENOUGH_MSE} — using {tier['name']}")
                chosen_tier = tier_idx
                break
            elif probe_mse < ESCALATE_RATIO * naive_mse:
                improvement = (1 - ratio) * 100
                print(f"  >> Beating naive by {improvement:.1f}% — "
                      f"using {tier['name']}")
                chosen_tier = tier_idx
                break
            else:
                print(f"  >> ratio={ratio:.2f} > {ESCALATE_RATIO}, "
                      f"escalating...")

        except Exception as e:
            print(f"  !! Probe FAILED: {e}")
            traceback.print_exc()
            all_results.append({
                'name': probe_cfg.name if 'probe_cfg' in dir() else f"probe_tier{tier_idx}",
                'type': 'probe',
                'mean_mse': float('nan'), 'mean_diversity': float('nan'),
                'error': str(e),
            })
            continue

    if chosen_tier is None:
        chosen_tier = min(args.max_tier, len(ARCH_TIERS) - 1)
        print(f"\n  No tier clearly beat naive — using last: "
              f"{ARCH_TIERS[chosen_tier]['name']}")

    tier = ARCH_TIERS[chosen_tier]
    print(f"\n>> Selected: {tier['name']} ({tier['arch_type']})")

    # Incremental save after probes
    write_csv(csv_path, all_results)

    # ==================================================================
    # Phase 2: Full hyperparameter sweep at chosen tier
    # ==================================================================
    print(f"\n{'#'*60}")
    print(f"PHASE 2: Full sweep — {tier['name']}")
    print(f"{'#'*60}")

    sweep_configs = [
        FlowConfig(
            name=f"{tier['name']}_lam{lam}_lip{lip}",
            arch_type=tier['arch_type'],
            hidden=tier['hidden'], n_layers=tier['n_layers'],
            cnn_channels=tier['cnn_channels'],
            n_iters=tier['full_iters'],
            lam=lam, lip_scale=lip,
        )
        for lam in [0.01, 0.1, 0.5]
        for lip in [5.0, 10.0]
    ]

    best_flow_mse = float('inf')
    best_flow_name = None
    best_vel_net = None
    best_flow_cfg = None

    for cfg in sweep_configs:
        try:
            vel_net, disc, history, model_path = train_flow(
                cfg, theta_train, y_train, output_dir, device
            )
            metrics_list, samples_list, mean_mse, mean_div = evaluate_flow(
                vel_net, y_test, theta_true, theta_dim,
                n_samples=args.n_eval_samples, n_steps=cfg.n_steps,
                device=device
            )

            result = {
                'name': cfg.name, 'type': 'flow',
                'arch_type': cfg.arch_type,
                'hidden': cfg.hidden, 'n_layers': cfg.n_layers,
                'cnn_channels': cfg.cnn_channels,
                'lam': cfg.lam, 'lip_scale': cfg.lip_scale,
                'mean_mse': mean_mse, 'mean_diversity': mean_div,
                'model_path': str(model_path),
            }
            all_results.append(result)
            print(f"  >> {cfg.name}: MSE={mean_mse:.4f}, diversity={mean_div:.4f}")

            run_dir = output_dir / cfg.name
            plot_completions(
                y_test.numpy(), [s.numpy() for s in samples_list],
                theta_true.numpy(),
                save_path=str(run_dir / "completions.png"),
                n_show=5, title=f"Flow: {cfg.name}"
            )

            if mean_mse < best_flow_mse:
                best_flow_mse = mean_mse
                best_flow_name = cfg.name
                best_vel_net = vel_net
                best_flow_cfg = cfg

        except Exception as e:
            print(f"  !! FAILED: {cfg.name}: {e}")
            traceback.print_exc()
            all_results.append({
                'name': cfg.name, 'type': 'flow',
                'arch_type': cfg.arch_type,
                'lam': cfg.lam, 'lip_scale': cfg.lip_scale,
                'mean_mse': float('nan'), 'mean_diversity': float('nan'),
                'error': str(e),
            })

    write_csv(csv_path, all_results)

    print(f"\n{'='*60}")
    print(f"Phase 2 done. Best: {best_flow_name} (MSE={best_flow_mse:.4f})")
    print(f"  Naive baseline: {naive_mse:.4f}")
    print(f"{'='*60}")

    # ==================================================================
    # Phase 3: GPA refinement on best flow
    # ==================================================================
    print(f"\n{'#'*60}")
    print("PHASE 3: GPA refinement")
    print(f"{'#'*60}")

    if best_vel_net is not None:
        gpa_configs = [
            GPAConfig(name="gpa_eta0.005_L10", eta=0.005, L=10.0, K=300),
            GPAConfig(name="gpa_eta0.01_L10",  eta=0.01,  L=10.0, K=300),
            GPAConfig(name="gpa_eta0.005_L5",  eta=0.005, L=5.0,  K=300),
            GPAConfig(name="gpa_eta0.01_L5",   eta=0.01,  L=5.0,  K=300),
        ]

        for gpa_cfg in gpa_configs:
            try:
                metrics_list, samples_list, mean_mse, mean_div = run_gpa(
                    gpa_cfg, best_vel_net, theta_train, y_train,
                    y_test, theta_true, theta_dim, y_dim,
                    n_steps=best_flow_cfg.n_steps, device=device
                )

                result = {
                    'name': f"{best_flow_name}+{gpa_cfg.name}",
                    'type': 'flow+gpa',
                    'flow': best_flow_name,
                    'gpa_eta': gpa_cfg.eta, 'gpa_L': gpa_cfg.L,
                    'gpa_K': gpa_cfg.K,
                    'mean_mse': mean_mse, 'mean_diversity': mean_div,
                }
                all_results.append(result)
                print(f"  >> {gpa_cfg.name}: MSE={mean_mse:.4f}, "
                      f"diversity={mean_div:.4f}")

                gpa_dir = output_dir / best_flow_name / gpa_cfg.name
                gpa_dir.mkdir(parents=True, exist_ok=True)
                plot_completions(
                    y_test.numpy(), [s.numpy() for s in samples_list],
                    theta_true.numpy(),
                    save_path=str(gpa_dir / "completions.png"),
                    n_show=5,
                    title=f"GPA: {gpa_cfg.name} (on {best_flow_name})"
                )

            except Exception as e:
                print(f"  !! GPA FAILED: {gpa_cfg.name}: {e}")
                traceback.print_exc()
                all_results.append({
                    'name': f"{best_flow_name}+{gpa_cfg.name}",
                    'type': 'flow+gpa', 'flow': best_flow_name,
                    'mean_mse': float('nan'), 'mean_diversity': float('nan'),
                    'error': str(e),
                })
    else:
        print("  No successful flow — skipping GPA.")

    # ==================================================================
    # Final summary
    # ==================================================================
    write_csv(csv_path, all_results)

    json_path = output_dir / "sweep_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'naive_mse': naive_mse,
            'chosen_tier': chosen_tier,
            'chosen_arch': tier['name'],
            'best_flow': best_flow_name,
            'best_flow_mse': best_flow_mse,
            'results': all_results,
        }, f, indent=2, default=str)

    print_summary(all_results, best_flow_name)
    print(f"\nNaive baseline MSE: {naive_mse:.4f}")
    print(f"Chosen architecture: tier {chosen_tier} ({tier['name']})")
    print(f"Results: {output_dir}")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == '__main__':
    main()
