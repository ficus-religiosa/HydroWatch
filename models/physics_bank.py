import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsBank(nn.Module):
    """
    Deterministic underwater-image feature bank.

    Input:
        x: [B, 3, H, W]

    Output:
        physics: [B, 3, H, W]

    Channels:
        0 -> Dark channel
        1 -> Red attenuation
        2 -> Gradient energy
    """

    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor(
            [
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]
            ],
            dtype=torch.float32
        )

        sobel_y = torch.tensor(
            [
                [-1., -2., -1.],
                [0., 0., 0.],
                [1., 2., 1.]
            ],
            dtype=torch.float32
        )

        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def forward(self, x):
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        dark = torch.min(
            torch.cat([r, g, b], dim=1),
            dim=1,
            keepdim=True
        )[0]

        red_attenuation = 1.0 - (
            r / (r + g + b + 1e-6)
        )

        gray = 0.299 * r + 0.587 * g + 0.114 * b

        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)

        gradient_energy = torch.sqrt(
            gx.pow(2) + gy.pow(2) + 1e-6
        )

        return torch.cat(
            [dark, red_attenuation, gradient_energy],
            dim=1
        )