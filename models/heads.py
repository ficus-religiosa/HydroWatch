import torch.nn as nn
import torch.nn.functional as F


class DetectionHead(nn.Module):
    def __init__(self, channels, num_classes, reg_max=16):
        super().__init__()

        self.reg_max = reg_max
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        self.box = nn.Conv2d(channels, 4 * reg_max, 1)
        self.objectness = nn.Conv2d(channels, 1, 1)
        self.classification = nn.Conv2d(channels, num_classes, 1)

    def forward(self, x):
        x = self.stem(x)

        return {
            "box": self.box(x),
            "objectness": self.objectness(x),
            "class": self.classification(x)
        }


class MultiScaleDetectionHead(nn.Module):
    def __init__(self, num_classes, reg_max=16):
        super().__init__()

        self.p2 = DetectionHead(64, num_classes, reg_max)
        self.p3 = DetectionHead(128, num_classes, reg_max)
        self.p4 = DetectionHead(256, num_classes, reg_max)
        self.p5 = DetectionHead(512, num_classes, reg_max)

    def forward(self, features):
        return {
            "p2": self.p2(features["p2"]),
            "p3": self.p3(features["p3"]),
            "p4": self.p4(features["p4"]),
            "p5": self.p5(features["p5"])
        }


class AuxiliaryMaskHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, p2, output_size):
        mask_logits = self.body(p2)

        return F.interpolate(
            mask_logits,
            size=output_size,
            mode="bilinear",
            align_corners=False
        )