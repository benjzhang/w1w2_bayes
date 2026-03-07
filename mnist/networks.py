"""Network architectures for MNIST inpainting.

Self-contained: does not import from parent package.
"""

import math
import torch
import torch.nn as nn


class MollifiedReLU(nn.Module):
    """Smooth C3 ReLU approximation with Lipschitz constant 1.

    From Gu et al.: ReLU_s^eps(x) =
        0                                          if x <= 0
        x^2/(4*eps) + eps*(cos(pi*x/eps) - 1)/(2*pi^2)   if 0 < x < 2*eps
        x - eps                                    if x >= 2*eps
    """

    def __init__(self, eps: float = 0.5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = self.eps
        mask_neg = x <= 0
        mask_mid = (x > 0) & (x < 2 * eps)
        mid_val = (
            x.pow(2) / (4 * eps)
            + eps * (torch.cos(math.pi * x / eps) - 1) / (2 * math.pi ** 2)
        )
        return torch.where(
            mask_neg, torch.zeros_like(x),
            torch.where(mask_mid, mid_val, x - eps)
        )


class VelocityMLP(nn.Module):
    """Velocity field v(t, theta; y) for conditional flow.

    Input: [t, theta, y] = 1 + theta_dim + y_dim
    Output: theta_dim

    Args:
        theta_dim: Dimension of theta.
        y_dim: Dimension of conditioning variable y.
        hidden: Hidden layer width.
        n_layers: Number of layers (minimum 2).
    """

    def __init__(self, theta_dim: int = 392, y_dim: int = 392,
                 hidden: int = 512, n_layers: int = 4):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim

        input_dim = 1 + theta_dim + y_dim
        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.SiLU()])
        layers.append(nn.Linear(hidden, theta_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, theta: torch.Tensor,
                y: torch.Tensor) -> torch.Tensor:
        """Compute velocity v(t, theta; y).

        Args:
            t: Time, shape (batch, 1).
            theta: Position, shape (batch, theta_dim).
            y: Conditioning, shape (batch, y_dim).

        Returns:
            Velocity, shape (batch, theta_dim).
        """
        inp = torch.cat([t, theta, y], dim=1)
        return self.net(inp)


class DiscriminatorMLP(nn.Module):
    """Discriminator phi(theta, y) for W1 dual / GPA.

    Uses MollifiedReLU activation for Lipschitz compatibility
    with hard spectral norm projection.

    Input: [theta, y] = theta_dim + y_dim
    Output: scalar

    Args:
        theta_dim: Dimension of theta.
        y_dim: Dimension of conditioning variable y.
        hidden: Hidden layer width.
        n_layers: Number of layers (minimum 2).
    """

    def __init__(self, theta_dim: int = 392, y_dim: int = 392,
                 hidden: int = 512, n_layers: int = 4):
        super().__init__()
        self.theta_dim = theta_dim
        self.y_dim = y_dim

        input_dim = theta_dim + y_dim
        act_cls = lambda: MollifiedReLU(eps=0.5)
        layers = [nn.Linear(input_dim, hidden), act_cls()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), act_cls()])
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, theta: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute phi(theta, y).

        Args:
            theta: shape (batch, theta_dim).
            y: shape (batch, y_dim).

        Returns:
            Scalar output, shape (batch, 1).
        """
        inp = torch.cat([theta, y], dim=1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# CNN architectures
#
# Following Gu et al. appendix A.3: simple conv encoder → flatten → FC head.
# No U-Net, no decoder — just extract features then MLP.
# ---------------------------------------------------------------------------

class VelocityCNN(nn.Module):
    """CNN potential/velocity for MNIST inpainting.

    Architecture follows Gu et al. A.3 "CNN Potential":
      - 7x7 Conv, 1×1 stride (1 → ch)  + ReLU + 2×2 max pool
      - 7x7 Conv, 1×1 stride (ch → ch) + ReLU + 2×2 max pool
      - Spatial flatten (concat with t scalar)
      - FC layers: flat_dim+1 → 512 → 512 → 512 → theta_dim

    Input is full 28×28 image (top=y, bottom=theta) as 1 channel.
    t is concatenated after the conv flatten.
    Output is velocity for the bottom half (392 dims).

    Args:
        ch: Number of conv channels.
        fc_hidden: FC hidden width.
    """

    def __init__(self, ch: int = 8, fc_hidden: int = 512):
        super().__init__()
        self.theta_dim = 392
        self.y_dim = 392

        self.convs = nn.Sequential(
            nn.Conv2d(1, ch, 7, stride=1, padding=3),   # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                    # 28x28 → 14x14
            nn.Conv2d(ch, ch, 7, stride=1, padding=3),   # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),                    # 14x14 → 7x7
        )

        flat_dim = ch * 7 * 7  # after 2 pool layers on 28×28
        # +1 for the time scalar t
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + 1, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, fc_hidden), nn.ReLU(),
            nn.Linear(fc_hidden, 392),  # output velocity for bottom half
        )

    def forward(self, t, theta, y):
        """
        Args:
            t: (batch, 1)
            theta: (batch, 392) — bottom half
            y: (batch, 392) — top half
        Returns:
            velocity: (batch, 392)
        """
        B = theta.shape[0]
        top = y.view(B, 1, 14, 28)
        bottom = theta.view(B, 1, 14, 28)
        img = torch.cat([top, bottom], dim=2)  # (B, 1, 28, 28)

        features = self.convs(img)              # (B, ch, 7, 7)
        flat = features.view(B, -1)             # (B, ch*49)
        flat_t = torch.cat([flat, t], dim=1)    # (B, ch*49 + 1)
        return self.fc(flat_t)


class DiscriminatorCNN(nn.Module):
    """CNN discriminator for MNIST inpainting.

    Architecture follows Gu et al. A.3 table (b) "CNN Discriminator (MNIST)":
      - 5×5 Conv SN, 2×2 stride (1 → ch1) + leaky ReLU + dropout
      - 5×5 Conv SN, 2×2 stride (ch1 → ch2) + leaky ReLU + dropout
      - 5×5 Conv SN, 2×2 stride (ch2 → ch3) + leaky ReLU + dropout
      - Flatten → FC with SN → ReLU → FC with SN → linear

    Uses MollifiedReLU in FC head for Lipschitz compatibility with
    hard spectral norm projection. Conv layers use LeakyReLU (Lip ≤ 1).

    Args:
        ch1, ch2, ch3: Channel counts for conv layers.
        fc_hidden: FC hidden width.
        dropout: Dropout rate.
    """

    def __init__(self, ch1: int = 16, ch2: int = 32, ch3: int = 64,
                 fc_hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.theta_dim = 392
        self.y_dim = 392

        # 28x28 → 14x14 → 7x7 → 3x3
        self.convs = nn.Sequential(
            nn.Conv2d(1, ch1, 5, stride=2, padding=2),    # 28→14
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch1, ch2, 5, stride=2, padding=2),  # 14→7
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
            nn.Conv2d(ch2, ch3, 5, stride=2, padding=1),  # 7→3
            nn.LeakyReLU(0.2),
            nn.Dropout2d(dropout),
        )

        flat_dim = ch3 * 3 * 3
        self.net = nn.Sequential(
            nn.Linear(flat_dim, fc_hidden),
            MollifiedReLU(eps=0.5),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, theta, y):
        """
        Args:
            theta: (batch, 392) — bottom half
            y: (batch, 392) — top half
        Returns:
            scalar: (batch, 1)
        """
        B = theta.shape[0]
        top = y.view(B, 1, 14, 28)
        bottom = theta.view(B, 1, 14, 28)
        img = torch.cat([top, bottom], dim=2)  # (B, 1, 28, 28)

        features = self.convs(img)  # (B, 4*bc, 3, 3)
        flat = features.view(B, -1)
        return self.net(flat)


def build_networks(arch_type, theta_dim=392, y_dim=392, hidden=512,
                   n_layers=4, cnn_channels=32):
    """Factory to build velocity net + discriminator by architecture type.

    Args:
        arch_type: 'mlp' or 'cnn'
        hidden: MLP hidden width (also used as CNN fc_hidden)
        n_layers: MLP layers (ignored for CNN)
        cnn_channels: CNN conv channel multiplier (ignored for MLP).
            For CNN disc: ch1=cnn_channels/2, ch2=cnn_channels, ch3=cnn_channels*2.
            For CNN vel: ch=cnn_channels/4 (conv feature channels).

    Returns:
        (vel_net, disc)
    """
    if arch_type == 'mlp':
        vel = VelocityMLP(theta_dim, y_dim, hidden, n_layers)
        disc = DiscriminatorMLP(theta_dim, y_dim, hidden, n_layers)
    elif arch_type == 'cnn':
        vel_ch = max(8, cnn_channels // 4)
        vel = VelocityCNN(ch=vel_ch, fc_hidden=hidden)
        disc = DiscriminatorCNN(
            ch1=max(8, cnn_channels // 2),
            ch2=cnn_channels,
            ch3=cnn_channels * 2,
            fc_hidden=min(hidden, 256),
        )
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")
    return vel, disc
