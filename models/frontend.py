import torch
import torch.nn as nn

from .physics_bank import PhysicsBank
from .enhancement import ResidualEnhancement


class SPD(nn.Module):
    """
    Space-to-Depth.

    [B,C,H,W] -> [B,4C,H/2,W/2]
    """

    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        b, c, h, w = x.shape
        s = self.scale

        if h % s != 0 or w % s != 0:
            raise ValueError(
                f"SPD requires H and W divisible by {s}. "
                f"Received {h}x{w}."
            )

        x = x.view(b, c, h // s, s, w // s, s)
        x = x.permute(0, 1, 3, 5, 2, 4)

        return x.reshape(b, c * s * s, h // s, w // s)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, stride=stride, padding=1,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class FrontEnd(nn.Module):
    def __init__(self):
        super().__init__()

        self.physics_bank = PhysicsBank()
        self.enhancement = ResidualEnhancement(3)

        self.projection = nn.Sequential(
            nn.Conv2d(9, 18, kernel_size=1, bias=False),
            nn.BatchNorm2d(18),
            nn.SiLU()
        )

        self.spd = SPD(scale=2)

        self.depthwise_conv = DepthwiseSeparableConv(72, 64)

    def forward(self, x):
        rgb = x

        physics = self.physics_bank(rgb)
        residual = self.enhancement(rgb)

        fused = torch.cat(
            [rgb, physics, residual],
            dim=1
        )

        x = self.projection(fused)
        x = self.spd(x)
        x = self.depthwise_conv(x)

        return x