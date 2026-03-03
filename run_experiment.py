#!/usr/bin/env python
"""
Run conditional flow experiments on inverse problems.

Usage:
    python run_experiment.py --problem linear
    python run_experiment.py --problem quadratic --epochs 500 --lip-scale 10
    python run_experiment.py --problem circle --lam 0.05

    # Resume from checkpoint
    python run_experiment.py --problem quadratic --resume checkpoints/quadratic/checkpoint_epoch100.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from problems import get_problem, PROBLEMS
from flows import W1W2Flow
from utils.evaluation import evaluate_flow, plot_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train conditional flows on inverse problems"
    )

    # Problem selection
    parser.add_argument(
        '--problem', type=str, default='linear',
        choices=list(PROBLEMS.keys()),
        help='Problem to solve'
    )

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (default: problem default)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--lam', type=float, default=None, help='KE regularization weight (default: problem default)')
    parser.add_argument('--n-steps', type=int, default=None, help='ODE integration steps (default: problem default)')
    parser.add_argument('--disc-updates', type=int, default=5, help='Discriminator updates per generator update')

    # Network architecture
    parser.add_argument('--vel-hidden', type=int, default=None, help='Velocity network hidden width')
    parser.add_argument('--vel-layers', type=int, default=None, help='Velocity network layers')
    parser.add_argument('--disc-hidden', type=int, default=128, help='Discriminator hidden width')
    parser.add_argument('--lip-scale', type=float, default=None, help='Lipschitz scale')
    parser.add_argument('--quad-features', action='store_true', help='Use quadratic features in discriminator')

    # Data
    parser.add_argument('--n-train', type=int, default=10000, help='Number of training samples')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory for checkpoints')
    parser.add_argument('--checkpoint-every', type=int, default=50, help='Checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')

    # Output
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--run-name', type=str, default=None, help='Custom run name (auto-generated if not provided)')
    parser.add_argument('--save-model', action='store_true', default=True, help='Save final model')
    parser.add_argument('--no-save-model', action='store_false', dest='save_model')

    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    return parser.parse_args()


def make_run_id(problem_name: str, epochs: int, lam: float, lip_scale: float,
                vel_layers: int, vel_hidden: int, n_steps: int,
                quad_features: bool) -> str:
    """Create a descriptive run ID from hyperparameters."""
    parts = [
        f"ep{epochs}",
        f"lam{lam}",
        f"lip{lip_scale}",
        f"v{vel_layers}x{vel_hidden}",
        f"st{n_steps}",
    ]
    if quad_features:
        parts.append("quad")
    return "_".join(parts)


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load problem
    problem = get_problem(args.problem)
    print(f"\nProblem: {problem.description}")
    print(f"  θ ∈ ℝ^{problem.theta_dim}, y ∈ ℝ^{problem.y_dim}")

    # Get default hyperparameters
    defaults = problem.default_hyperparams()

    # Merge with CLI args
    n_epochs = args.epochs or defaults.get('n_epochs', 300)
    lam = args.lam if args.lam is not None else defaults.get('lam', 0.01)
    n_steps = args.n_steps or defaults.get('n_steps', 40)
    vel_hidden = args.vel_hidden or defaults.get('vel_hidden', 256)
    vel_layers = args.vel_layers or defaults.get('vel_layers', 4)
    lip_scale = args.lip_scale if args.lip_scale is not None else defaults.get('lip_scale', 10.0)
    use_quad_features = args.quad_features or defaults.get('use_quadratic_features', False)

    # Create run ID and config
    if args.run_name:
        run_id = args.run_name
    else:
        run_id = make_run_id(
            problem.name, n_epochs, lam, lip_scale,
            vel_layers, vel_hidden, n_steps, use_quad_features
        )

    run_config = {
        'run_id': run_id,
        'problem': problem.name,
        'epochs': n_epochs,
        'lam': lam,
        'lip_scale': lip_scale,
        'vel_layers': vel_layers,
        'vel_hidden': vel_hidden,
        'n_steps': n_steps,
        'quad_features': use_quad_features,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'disc_updates': args.disc_updates,
        'seed': args.seed,
    }

    print(f"\nRun ID: {run_id}")

    # Setup directories
    output_dir = Path(args.output_dir) / problem.name
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = output_dir / 'checkpoints' / run_id

    # Generate training data
    print(f"\nGenerating {args.n_train} training samples...")
    theta_np, y_np = problem.sample_joint(args.n_train)
    theta_train = torch.FloatTensor(theta_np)
    y_train = torch.FloatTensor(y_np)
    if y_train.dim() == 1:
        y_train = y_train.unsqueeze(1)

    # Create or load flow
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        flow = W1W2Flow.from_checkpoint(args.resume, device=device)
    else:
        print(f"\nCreating W1W2 Flow:")
        print(f"  VelocityNet: {vel_layers} layers × {vel_hidden} hidden")
        print(f"  Discriminator: Lip={lip_scale}, quad_features={use_quad_features}")
        print(f"  λ={lam}, n_steps={n_steps}")

        flow = W1W2Flow(
            theta_dim=problem.theta_dim,
            y_dim=problem.y_dim,
            vel_hidden=vel_hidden,
            vel_layers=vel_layers,
            disc_hidden=args.disc_hidden,
            lip_scale=lip_scale,
            use_quadratic_features=use_quad_features,
            device=device
        )

    # Training callback for intermediate evaluation
    def eval_callback(epoch, metrics):
        if epoch % 100 == 0:
            results = evaluate_flow(flow.vel_net, problem, n_samples=1000, n_steps=n_steps, device=device)
            mean_dist = np.mean(results['mean_distances'])
            print(f"  [Eval @ epoch {epoch}] mean_dist={mean_dist:.4f}")

    # Train
    print(f"\nTraining for {n_epochs} epochs...")
    history = flow.train(
        theta_train,
        y_train,
        n_epochs=n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lam=lam,
        n_steps=n_steps,
        disc_updates=args.disc_updates,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=args.checkpoint_every,
        eval_callback=eval_callback,
        verbose=not args.quiet
    )

    # Save final model with descriptive name
    if args.save_model:
        model_path = output_dir / f"model_{run_id}.pt"
        flow.save(str(model_path))

    # Final evaluation
    print("\n" + "=" * 60)
    print(f"Final Evaluation ({run_id}):")
    results = evaluate_flow(flow.vel_net, problem, n_steps=n_steps, device=device)
    for y_val, dist in zip(results['y_values'], results['mean_distances']):
        print(f"  y={y_val}: mean_dist = {dist:.4f}")

    # Generate plots with descriptive name
    plot_path = output_dir / f"results_{run_id}.png"
    plot_results(
        flow.vel_net,
        problem,
        history,
        save_path=str(plot_path),
        n_steps=n_steps,
        device=device,
        run_config=run_config
    )

    print(f"\nResults saved to: {output_dir}")
    print(f"  Model: model_{run_id}.pt")
    print(f"  Plot:  results_{run_id}.png")


if __name__ == "__main__":
    main()
