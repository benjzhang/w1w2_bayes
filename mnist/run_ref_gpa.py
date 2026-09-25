"""Load reference W1W2 flow model (TF weights) into PyTorch, generate samples, apply GPA.

The reference model uses a potential-based velocity field:
  v = -grad_U / alpha1
where U is a CNN+FNN network that outputs a scalar potential.

Architecture (from saved weights):
  Conv: (7,7,1,8) -> maxpool -> (7,7,8,8) -> maxpool -> flatten (392 dims)
  FNN: [392+1, 512, 512, 512, 1]  (the +1 is for time)

Usage:
  python -m mnist.run_ref_gpa --K 5000 --eta 0.5
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .data import MNISTInpainting
from .run_one import (
    DiscriminatorMLP, _project_disc_weights, _gradient_penalty,
    compute_metrics, plot_completions,
)


class RefPotentialCNN(nn.Module):
    """Reproduce the reference TF potential network in PyTorch.

    TF architecture:
      conv1: (7,7,1,8) + relu + maxpool(2,2)   -> (14,14,8)
      conv2: (7,7,8,8) + relu + maxpool(2,2)   -> (7,7,8)
      flatten -> 392
      concat time -> 393
      fc1: 393 -> 512 (tanh)
      fc2: 512 -> 512 (tanh)
      fc3: 512 -> 512 (tanh)
      fc4: 512 -> 1 (linear)
    """

    def __init__(self, conv_weights, fc_weights, fc_biases):
        super().__init__()
        # Conv layers (TF format: [H, W, C_in, C_out] -> PyTorch: [C_out, C_in, H, W])
        self.conv1 = nn.Conv2d(1, 8, 7, padding=3, bias=False)
        self.conv2 = nn.Conv2d(8, 8, 7, padding=3, bias=False)

        # Load conv weights (transpose from TF to PyTorch format)
        self.conv1.weight.data = torch.from_numpy(
            conv_weights[0].transpose(3, 2, 0, 1).copy()
        )
        self.conv2.weight.data = torch.from_numpy(
            conv_weights[1].transpose(3, 2, 0, 1).copy()
        )

        # FC layers
        self.fc_layers = nn.ModuleList()
        for i in range(len(fc_weights)):
            in_dim, out_dim = fc_weights[i].shape
            fc = nn.Linear(in_dim, out_dim)
            fc.weight.data = torch.from_numpy(fc_weights[i].T.copy())
            fc.bias.data = torch.from_numpy(fc_biases[i].flatten().copy())
            self.fc_layers.append(fc)

    def forward(self, x_img, t):
        """
        Args:
            x_img: (B, 1, 28, 28) image tensor
            t: (B, 1) time tensor
        Returns:
            U: (B, 1) potential value
        """
        # Conv + relu + maxpool
        h = F.max_pool2d(F.relu(self.conv1(x_img)), 2)  # (B, 8, 14, 14)
        h = F.max_pool2d(F.relu(self.conv2(h)), 2)       # (B, 8, 7, 7)

        # Flatten — match TF's transpose ordering
        # TF: (B,H,W,C) -> transpose(3,2,1,0) -> (C,W,H,B) -> reshape(-1,B) -> transpose -> (B, C*W*H)
        # PyTorch: (B,C,H,W) -> permute(0,1,3,2) -> (B,C,W,H) -> reshape(B,-1)
        B = h.shape[0]
        h = h.permute(0, 1, 3, 2).reshape(B, -1)  # (B, 392)

        # Concat time
        h_t = torch.cat([h, t], dim=1)  # (B, 393)

        # FC layers with tanh (except last)
        for i, fc in enumerate(self.fc_layers):
            h_t = fc(h_t)
            if i < len(self.fc_layers) - 1:
                h_t = torch.tanh(h_t)

        return h_t  # (B, 1)


def load_ref_model(pickle_path, device):
    """Load reference model weights from pickle file."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    G_W, G_b, G_conv_W = data[0], data[1], data[2]
    D_W, D_b, D_conv_W = data[3], data[4], data[5]
    T, dt = data[6], data[7]
    alpha1 = data[13]

    model = RefPotentialCNN(G_conv_W, G_W, G_b).to(device)
    model.eval()

    return model, T, dt, alpha1


def generate_ref_samples(model, n_samples, T, dt, alpha1, device):
    """Generate samples using the reference potential flow.

    v = -grad_U / alpha1, forward Euler integration.
    """
    steps = int(T / dt)
    # Start from Uniform[0, 1] like the reference
    x = torch.rand(n_samples, 1, 28, 28, device=device)
    x.requires_grad_(True)

    for i in range(steps):
        t_val = i * dt
        t = torch.full((n_samples, 1), t_val, device=device)
        U = model(x, t)
        grad_U = torch.autograd.grad(U.sum(), x, create_graph=False)[0]
        with torch.no_grad():
            x = x - dt * grad_U / alpha1
        x.requires_grad_(True)

    return x.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle', type=str,
                        default='/work/pi_markos_umass_edu/bjzhang_umass_edu/w1w2_reference/assets/MNIST/D1P2/Flow-GAN-Rep3/result.pickle')
    parser.add_argument('--name', type=str, default='ref_gpa')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of particles to generate and refine')
    parser.add_argument('--K', type=int, default=5000, help='GPA steps')
    parser.add_argument('--eta', type=float, default=0.5, help='GPA step size')
    parser.add_argument('--L', type=float, default=1.0, help='Lipschitz constant')
    parser.add_argument('--disc-steps', type=int, default=5)
    parser.add_argument('--disc-lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--fstar', default='kl', choices=['kl', 'reverse_kl'])
    parser.add_argument('--lip-mode', default='project', choices=['project', 'gp'])
    parser.add_argument('--gp-lambda', type=float, default=0.1)
    parser.add_argument('--output-dir', default='results/mnist/sweep')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load reference model
    print(f"Loading reference model: {args.pickle}")
    model, T, dt, alpha1 = load_ref_model(args.pickle, device)
    print(f"  T={T}, dt={dt}, steps={int(T/dt)}, alpha1={alpha1}")

    # Generate initial particles
    print(f"Generating {args.n_samples} particles from reference flow...")
    with torch.no_grad():
        # Actually need grad for potential
        pass
    particles = generate_ref_samples(model, args.n_samples, T, dt, alpha1, device)
    particles_flat = particles.reshape(args.n_samples, 784)
    print(f"  Particles shape: {particles.shape}, range: [{particles.min():.3f}, {particles.max():.3f}]")

    # Visualize pre-GPA samples
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(particles[i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
    # Show 10 more
    for i in range(10):
        axes[1, i].imshow(particles[10+i, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    plt.suptitle('Reference flow output (pre-GPA)')
    plt.savefig(str(output_dir / 'pre_gpa.png'), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved pre-GPA samples to {output_dir / 'pre_gpa.png'}")

    # Load real MNIST data for GPA
    print("Loading MNIST data...")
    from torchvision import datasets, transforms
    dataset = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.ToTensor())
    all_images = torch.stack([img for img, _ in dataset]).to(device)  # (60000, 1, 28, 28)
    n_real = len(all_images)
    real_flat = all_images.reshape(n_real, 784)

    # GPA refinement (unconditional — no y conditioning)
    # Discriminator works on flat 784-dim vectors, no y input
    print(f"GPA: K={args.K}, eta={args.eta}, L={args.L}, fstar={args.fstar}, lip={args.lip_mode}")

    # Build a simple unconditional discriminator
    disc = UnconditionalDisc(784, hidden=256, n_layers=3).to(device)
    use_gp = args.lip_mode == 'gp'
    if not use_gp:
        _project_disc_weights_uncond(disc, args.L)
    opt_disc = optim.Adam(disc.parameters(), lr=args.disc_lr)

    def compute_L_dual(phi_gen, phi_real):
        if args.fstar == 'reverse_kl':
            return phi_gen.mean() - torch.logsumexp(phi_real.squeeze(), dim=0) + np.log(phi_real.shape[0])
        else:
            return phi_gen.mean() - torch.exp(phi_real - 1).mean()

    t_start = time.time()
    for k in range(args.K):
        for _ in range(args.disc_steps):
            idx_fake = torch.randint(0, args.n_samples, (args.batch_size,))
            idx_real = torch.randint(0, n_real, (args.batch_size,))
            p_batch = particles_flat[idx_fake].detach()
            r_batch = real_flat[idx_real]
            phi_fake = disc(p_batch)
            phi_real = disc(r_batch)
            L_dual = compute_L_dual(phi_fake, phi_real)
            if use_gp:
                gp = _gradient_penalty_uncond(disc, r_batch, p_batch, args.L)
                disc_loss = -L_dual + args.gp_lambda * gp
            else:
                disc_loss = -L_dual
            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()
            if not use_gp:
                _project_disc_weights_uncond(disc, args.L)

        # Particle update
        particles_flat.requires_grad_(True)
        phi = disc(particles_flat)
        grad = torch.autograd.grad(phi.sum(), particles_flat, create_graph=False)[0]
        with torch.no_grad():
            particles_flat = (particles_flat - args.eta * grad).detach()

        if (k + 1) % 100 == 0:
            print(f"  [GPA {k+1}/{args.K}] L_dual={L_dual.item():.4f} ({time.time()-t_start:.0f}s)")

    elapsed = time.time() - t_start
    print(f"GPA done in {elapsed:.1f}s")

    # Visualize post-GPA samples
    particles_img = particles_flat.reshape(-1, 1, 28, 28).cpu().numpy()
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))
    for i in range(10):
        axes[0, i].imshow(particles_img[i, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
    for i in range(10):
        axes[1, i].imshow(particles_img[10+i, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
    plt.suptitle(f'Post-GPA (K={args.K}, eta={args.eta})')
    plt.savefig(str(output_dir / 'post_gpa.png'), dpi=100, bbox_inches='tight')
    plt.close()

    result = {
        'name': args.name,
        'K': args.K, 'eta': args.eta, 'L': args.L,
        'fstar': args.fstar, 'lip_mode': args.lip_mode,
        'n_samples': args.n_samples,
        'elapsed_s': elapsed,
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Done: {output_dir}")


class UnconditionalDisc(nn.Module):
    """Simple unconditional MLP discriminator for 784-dim MNIST."""

    def __init__(self, input_dim, hidden=256, n_layers=3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden))
            layers.append(nn.LeakyReLU(0.2))
            in_d = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _project_disc_weights_uncond(disc, L):
    """Hard spectral norm projection for unconditional disc."""
    D = sum(1 for m in disc.net if isinstance(m, nn.Linear))
    scale = L ** (1.0 / D)
    with torch.no_grad():
        for m in disc.net:
            if isinstance(m, nn.Linear):
                sigma = torch.linalg.norm(m.weight, ord=2)
                if sigma > scale:
                    m.weight.mul_(scale / sigma)


def _gradient_penalty_uncond(disc, real, fake, L):
    """Gradient penalty for unconditional disc."""
    alpha = torch.rand(real.shape[0], 1, device=real.device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = disc(interp)
    grads = torch.autograd.grad(d_interp.sum(), interp, create_graph=True)[0]
    grad_sq = (grads ** 2).sum(dim=1)
    return F.relu(grad_sq - L ** 2).mean()


if __name__ == '__main__':
    main()
