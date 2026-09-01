import torch.nn as nn

from .blocks import Conv, C3k2, SPPF, C2PSA


class YOLO11SBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.p2 = nn.Sequential(
            Conv(64, 64, 3, 2),
            C3k2(64, 64, n=2)
        )

        self.p3 = nn.Sequential(
            Conv(64, 128, 3, 2),
            C3k2(128, 128, n=2)
        )

        self.p4 = nn.Sequential(
            Conv(128, 256, 3, 2),
            C3k2(256, 256, n=2)
        )

        self.p5 = nn.Sequential(
            Conv(256, 512, 3, 2),
            C3k2(512, 512, n=2),
            SPPF(512, 512),
            C2PSA(512, n=2)
        )

    def forward(self, x):
        p2 = self.p2(x)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)

        return {
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "p5": p5
        }