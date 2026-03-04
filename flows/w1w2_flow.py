"""W1W2 Flow: Wasserstein-1 dual + Kinetic Energy regularization.

Based on the Benamou-Brenier formulation of optimal transport:
    W₂²(p, q) = inf_{v} ∫₀¹ E_ρ[½‖v(t,x)‖²] dt
    s.t. ∂ρ/∂t + ∇·(ρv) = 0, ρ(0)=p, ρ(1)=q

We use W₁ dual for the terminal constraint plus KE regularization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import json

from .base import BaseFlow
from nn import VelocityNet, Discriminator
from utils.integrators import euler_integrate, compute_kinetic_energy


class W1W2Flow(BaseFlow):
    """W1W2 Flow for conditional posterior sampling.

    Combines:
    - Wasserstein-1 dual objective for distribution matching
    - Kinetic energy regularization for smooth trajectories

    Supports:
    - Arbitrary θ and y dimensions
    - Model checkpointing during training
    - Final model saving
    - Intermediate evaluation callbacks
    """

    def __init__(
        self,
        theta_dim: int,
        y_dim: int,
        vel_hidden: int = 256,
        vel_layers: int = 4,
        disc_hidden: int = 128,
        disc_layers: int = 3,
        lip_scale: float = 10.0,
        use_quadratic_features: bool = False,
        gp_lambda: float = 0.0,
        device: Optional[torch.device] = None
    ):
        """Initialize W1W2 Flow.

        Args:
            theta_dim: Dimension of θ
            y_dim: Dimension of conditioning variable y
            vel_hidden: Hidden layer width for velocity network
            vel_layers: Number of layers in velocity network
            disc_hidden: Hidden layer width for discriminator
            disc_layers: Number of layers in discriminator
            lip_scale: Lipschitz constant scale for discriminator
            use_quadratic_features: Add quadratic features to discriminator
            gp_lambda: One-sided gradient penalty coefficient.
                When > 0, replaces spectral norm with penalty
                λ * E[max(0, ||∇φ||² - 1)].
            device: Device to use (defaults to CUDA if available)
        """
        self._theta_dim = theta_dim
        self._y_dim = y_dim
        self.lip_scale = lip_scale
        self.use_quadratic_features = use_quadratic_features
        self.gp_lambda = gp_lambda

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # When using gradient penalty, disable spectral norm
        use_spectral_norm = (gp_lambda == 0.0)

        # Networks
        self.vel_net = VelocityNet(
            theta_dim=theta_dim,
            y_dim=y_dim,
            hidden=vel_hidden,
            n_layers=vel_layers,
            activation='silu'
        ).to(device)

        self.disc = Discriminator(
            theta_dim=theta_dim,
            y_dim=y_dim,
            hidden=disc_hidden,
            n_layers=disc_layers,
            lip_scale=lip_scale,
            use_quadratic_features=use_quadratic_features,
            use_spectral_norm=use_spectral_norm,
            activation='silu'
        ).to(device)

        # Store hyperparameters for saving
        self.hparams = {
            'theta_dim': theta_dim,
            'y_dim': y_dim,
            'vel_hidden': vel_hidden,
            'vel_layers': vel_layers,
            'disc_hidden': disc_hidden,
            'disc_layers': disc_layers,
            'lip_scale': lip_scale,
            'use_quadratic_features': use_quadratic_features,
            'gp_lambda': gp_lambda,
        }

    @property
    def theta_dim(self) -> int:
        return self._theta_dim

    @property
    def y_dim(self) -> int:
        return self._y_dim

    def _gradient_penalty(
        self,
        theta_real: torch.Tensor,
        theta_gen: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """One-sided gradient penalty: E[max(0, ||∇_θ φ(θ̂, y)||² - 1)].

        Evaluates on interpolated points θ̂ = ε·θ_real + (1-ε)·θ_gen.
        """
        bs = theta_real.shape[0]
        eps = torch.rand(bs, 1, device=self.device)
        theta_hat = (eps * theta_real + (1 - eps) * theta_gen).requires_grad_(True)

        phi_hat = self.disc(theta_hat, y)

        grad = torch.autograd.grad(
            outputs=phi_hat,
            inputs=theta_hat,
            grad_outputs=torch.ones_like(phi_hat),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_norm_sq = (grad ** 2).sum(dim=1)
        penalty = torch.clamp(grad_norm_sq - 1.0, min=0.0).mean()
        return penalty

    def train(
        self,
        theta_data: torch.Tensor,
        y_data: torch.Tensor,
        n_iters: int = 5000,
        batch_size: int = 256,
        lr: float = 1e-3,
        lam: float = 0.01,
        n_steps: int = 40,
        disc_updates: int = 5,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every: int = 500,
        eval_callback: Optional[Callable[[int, Dict], None]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Train the W1W2 flow.

        Args:
            theta_data: Training samples, shape (n, theta_dim)
            y_data: Conditioning values, shape (n, y_dim)
            n_iters: Number of training iterations (velocity updates)
            batch_size: Batch size
            lr: Learning rate
            lam: Kinetic energy regularization weight
            n_steps: Number of ODE integration steps
            disc_updates: Number of discriminator updates per generator update
            checkpoint_dir: Directory to save checkpoints (None = no checkpoints)
            checkpoint_every: Save checkpoint every N iterations
            eval_callback: Optional callback(iter, metrics) for intermediate evaluation
            verbose: Print progress

        Returns:
            Training history dictionary with 'L_dual', 'KE', 'iters' lists
        """
        # Move data to device
        theta_data = theta_data.to(self.device)
        y_data = y_data.to(self.device)

        dataset = TensorDataset(theta_data, y_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        opt_vel = optim.Adam(self.vel_net.parameters(), lr=lr)
        opt_disc = optim.Adam(self.disc.parameters(), lr=lr)

        history = {'L_dual': [], 'KE': [], 'iters': []}

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        if verbose:
            constraint = f"GP λ_gp={self.gp_lambda}" if self.gp_lambda > 0 else f"Lip={self.lip_scale}"
            print(f"Training W1W2 Flow: {n_iters} iters, λ={lam}, "
                  f"disc_updates={disc_updates}, {constraint}")

        # Infinite data iterator
        def infinite_loader():
            while True:
                for batch in loader:
                    yield batch

        data_iter = infinite_loader()

        for it in range(1, n_iters + 1):
            theta_batch, y_batch = next(data_iter)
            bs = theta_batch.shape[0]
            z = torch.randn(bs, self._theta_dim, device=self.device)

            # --- Discriminator updates ---
            for _ in range(disc_updates):
                traj = euler_integrate(self.vel_net, z, y_batch, n_steps)
                theta_gen = traj[-1].detach()

                phi_gen = self.disc(theta_gen, y_batch)
                phi_real = self.disc(theta_batch, y_batch)
                L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

                disc_loss = -L_dual
                if self.gp_lambda > 0:
                    gp = self._gradient_penalty(theta_batch, theta_gen, y_batch)
                    disc_loss = disc_loss + self.gp_lambda * gp

                opt_disc.zero_grad()
                disc_loss.backward()
                opt_disc.step()

            # --- Velocity update ---
            traj = euler_integrate(self.vel_net, z, y_batch, n_steps)
            theta_gen = traj[-1]

            phi_gen = self.disc(theta_gen, y_batch)
            phi_real = self.disc(theta_batch, y_batch)
            L_dual = phi_gen.mean() - torch.exp(phi_real - 1).mean()

            # Kinetic energy
            KE = 0.0
            dt = 1.0 / n_steps
            for i, theta_t in enumerate(traj[:-1]):
                t = torch.full((theta_t.shape[0], 1), i * dt, device=self.device)
                v = self.vel_net(t, theta_t, y_batch)
                KE += 0.5 * (v ** 2).sum(dim=1).mean() * dt

            loss = L_dual + lam * KE
            opt_vel.zero_grad()
            loss.backward()
            opt_vel.step()

            # Record history
            history['L_dual'].append(L_dual.item())
            history['KE'].append(KE.item())
            history['iters'].append(it)

            # Checkpoint
            if checkpoint_dir and it % checkpoint_every == 0:
                self._save_checkpoint(
                    Path(checkpoint_dir) / f"checkpoint_iter{it}.pt",
                    it,
                    history
                )
                if verbose:
                    print(f"  [Checkpoint saved: iter {it}]")

            # Evaluation callback
            if eval_callback:
                eval_callback(it, {
                    'iter': it,
                    'L_dual': L_dual.item(),
                    'KE': KE.item(),
                })

            # Progress
            if verbose and it % 500 == 0:
                recent_dual = np.mean(history['L_dual'][-100:])
                recent_ke = np.mean(history['KE'][-100:])
                print(f"[{it}/{n_iters}] L_dual={recent_dual:.4f}, "
                      f"KE={recent_ke:.4f}")

        return history

    def sample(
        self,
        y: torch.Tensor,
        n_samples: int,
        n_steps: int = 40
    ) -> torch.Tensor:
        """Sample from p(θ|y).

        Args:
            y: Conditioning value, shape (y_dim,) or (batch, y_dim)
            n_samples: Number of samples
            n_steps: Integration steps

        Returns:
            Samples, shape (n_samples, theta_dim)
        """
        self.vel_net.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self._theta_dim, device=self.device)

            # Handle scalar y
            if y.dim() == 0 or (y.dim() == 1 and y.shape[0] == self._y_dim):
                y = y.view(1, -1).expand(n_samples, -1)
            elif y.dim() == 1:
                y = y.unsqueeze(1).expand(n_samples, -1)

            y = y.to(self.device)
            traj = euler_integrate(self.vel_net, z, y, n_steps)
            return traj[-1]

    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save file (will save as .pt file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'hparams': self.hparams,
            'vel_net_state': self.vel_net.state_dict(),
            'disc_state': self.disc.state_dict(),
        }
        torch.save(state, path)
        print(f"Model saved: {path}")

    def load(self, path: str) -> None:
        """Load model from disk.

        Args:
            path: Path to saved model
        """
        state = torch.load(path, map_location=self.device)

        # Verify dimensions match
        if state['hparams']['theta_dim'] != self._theta_dim:
            raise ValueError(f"theta_dim mismatch: {state['hparams']['theta_dim']} vs {self._theta_dim}")
        if state['hparams']['y_dim'] != self._y_dim:
            raise ValueError(f"y_dim mismatch: {state['hparams']['y_dim']} vs {self._y_dim}")

        self.vel_net.load_state_dict(state['vel_net_state'])
        self.disc.load_state_dict(state['disc_state'])
        print(f"Model loaded: {path}")

    def _save_checkpoint(
        self,
        path: Path,
        epoch: int,
        history: Dict[str, List]
    ) -> None:
        """Save a training checkpoint.

        Args:
            path: Path to checkpoint file
            epoch: Current epoch
            history: Training history
        """
        state = {
            'epoch': epoch,
            'hparams': self.hparams,
            'vel_net_state': self.vel_net.state_dict(),
            'disc_state': self.disc.state_dict(),
            'history': history,
        }
        torch.save(state, path)

    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[torch.device] = None) -> 'W1W2Flow':
        """Load a flow from a checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to use

        Returns:
            W1W2Flow instance with loaded weights
        """
        state = torch.load(path, map_location=device or 'cpu')
        hparams = state['hparams']

        flow = cls(
            theta_dim=hparams['theta_dim'],
            y_dim=hparams['y_dim'],
            vel_hidden=hparams.get('vel_hidden', 256),
            vel_layers=hparams.get('vel_layers', 4),
            disc_hidden=hparams.get('disc_hidden', 128),
            disc_layers=hparams.get('disc_layers', 3),
            lip_scale=hparams.get('lip_scale', 10.0),
            use_quadratic_features=hparams.get('use_quadratic_features', False),
            gp_lambda=hparams.get('gp_lambda', 0.0),
            device=device
        )

        flow.vel_net.load_state_dict(state['vel_net_state'])
        flow.disc.load_state_dict(state['disc_state'])

        return flow
