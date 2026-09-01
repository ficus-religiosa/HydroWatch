import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEnhancement(nn.Module):
    """
    Deterministic, parameter-free residual enhancement transform.

    A fixed Gaussian blur estimates the low-frequency component.
    The high-frequency residual is:
        residual = x - GaussianBlur(x)

    Input:
        [B, 3, H, W]

    Output:
        [B, 3, H, W]

    No trainable parameters are used.
    """

    def __init__(self, channels=3, kernel_size=5, sigma=1.0):
        super().__init__()

        if channels != 3:
            raise ValueError("ResidualEnhancement is defined for RGB input (3 channels).")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")

        radius = kernel_size // 2
        coords = torch.arange(
            -radius, radius + 1, dtype=torch.float32
        )
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        kernel = torch.exp(
            -(xx.pow(2) + yy.pow(2)) / (2 * sigma * sigma)
        )
        kernel = kernel / kernel.sum()

        self.register_buffer(
            "gaussian_kernel",
            kernel.view(1, 1, kernel_size, kernel_size).repeat(
                channels, 1, 1, 1
            )
        )

        self.channels = channels
        self.padding = radius

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [B,{self.channels},H,W], got {tuple(x.shape)}"
            )

        low_frequency = F.conv2d(
            x,
            self.gaussian_kernel.to(dtype=x.dtype),
            padding=self.padding,
            groups=self.channels
        )

        return x - low_frequency
