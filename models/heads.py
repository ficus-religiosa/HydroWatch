import torch
import torch.nn as nn
import torch.nn.functional as F


class DFLDecoder(nn.Module):
    """
    Converts 4 x reg_max distribution logits into
    four continuous distances: left, top, right, bottom.
    """

    def __init__(self, reg_max=16):
        super().__init__()

        self.reg_max = reg_max

        self.register_buffer(
            "project",
            torch.arange(
                reg_max,
                dtype=torch.float32
            ).view(1, reg_max, 1, 1)
        )

    def forward(self, logits):
        b, _, h, w = logits.shape

        logits = logits.view(
            b, 4, self.reg_max, h, w
        )

        probs = torch.softmax(logits, dim=2)

        return (
            probs * self.project.to(
                device=logits.device,
                dtype=logits.dtype
            ).view(1, 1, self.reg_max, 1, 1)
        ).sum(dim=2)


def make_grid(h, w, stride, device, dtype):
    y, x = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij"
    )

    centers = torch.stack(
        [
            (x + 0.5) * stride,
            (y + 0.5) * stride
        ],
        dim=-1
    )

    return centers


def distances_to_xyxy(distances, stride, image_size):
    """
    distances: [B,4,H,W] in feature-cell units.
    Returns absolute-pixel xyxy boxes: [B,H,W,4].
    """
    b, _, h, w = distances.shape
    ih, iw = image_size

    centers = make_grid(
        h, w, stride,
        distances.device,
        distances.dtype
    )

    cx = centers[..., 0]
    cy = centers[..., 1]

    l = distances[:, 0] * stride
    t = distances[:, 1] * stride
    r = distances[:, 2] * stride
    bot = distances[:, 3] * stride

    x1 = l.mul(-1) + cx.view(1, h, w)
    y1 = t.mul(-1) + cy.view(1, h, w)
    x2 = r + cx.view(1, h, w)
    y2 = bot + cy.view(1, h, w)

    x1 = x1.clamp(0, iw)
    y1 = y1.clamp(0, ih)
    x2 = x2.clamp(0, iw)
    y2 = y2.clamp(0, ih)

    return torch.stack([x1, y1, x2, y2], dim=-1)


class DetectionHead(nn.Module):
    def __init__(self, channels, num_classes, reg_max=16):
        super().__init__()

        self.reg_max = reg_max
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(
                channels, channels, 3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        self.box = nn.Conv2d(
            channels, 4 * reg_max, 1
        )
        self.objectness = nn.Conv2d(
            channels, 1, 1
        )
        self.classification = nn.Conv2d(
            channels, num_classes, 1
        )

        self.dfl = DFLDecoder(reg_max)

    def forward(self, x, stride=None, image_size=None):
        x = self.stem(x)

        box_logits = self.box(x)
        objectness_logits = self.objectness(x)
        class_logits = self.classification(x)

        output = {
            "box": box_logits,
            "objectness": objectness_logits,
            "class": class_logits
        }

        if stride is not None and image_size is not None:
            distances = self.dfl(box_logits)

            boxes = distances_to_xyxy(
                distances,
                stride,
                image_size
            )

            output["distances"] = distances
            output["boxes"] = boxes

        return output


class MultiScaleDetectionHead(nn.Module):
    def __init__(self, num_classes, reg_max=16):
        super().__init__()

        self.strides = {
            "p2": 4,
            "p3": 8,
            "p4": 16,
            "p5": 32
        }

        self.p2 = DetectionHead(
            64, num_classes, reg_max
        )
        self.p3 = DetectionHead(
            128, num_classes, reg_max
        )
        self.p4 = DetectionHead(
            256, num_classes, reg_max
        )
        self.p5 = DetectionHead(
            512, num_classes, reg_max
        )

    def forward(self, features, image_size=None):
        if image_size is None:
            image_size = (
                features["p2"].shape[-2] * self.strides["p2"],
                features["p2"].shape[-1] * self.strides["p2"]
            )

        return {
            "p2": self.p2(
                features["p2"],
                self.strides["p2"],
                image_size
            ),
            "p3": self.p3(
                features["p3"],
                self.strides["p3"],
                image_size
            ),
            "p4": self.p4(
                features["p4"],
                self.strides["p4"],
                image_size
            ),
            "p5": self.p5(
                features["p5"],
                self.strides["p5"],
                image_size
            )
        }


class AuxiliaryMaskHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                64, 64, 3,
                padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(
                64, 32, 3,
                padding=1, bias=False
            ),
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
