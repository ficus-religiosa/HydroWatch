import torch.nn as nn


class ResidualEnhancement(nn.Module):
    """
    Lightweight learned enhancement transform.

    Input:
        [B, 3, H, W]

    Output:
        [B, 3, H, W]
    """

    def __init__(self, channels=3):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),

            nn.Conv2d(16, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.body(x)