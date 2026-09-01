import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv, C3k2


class DySample(nn.Module):
    """
    Dynamic upsampling module.

    The predicted offsets are resized to the exact target
    feature-map size so odd spatial dimensions are handled
    correctly for the 480x360 input.
    """

    def __init__(self, channels, scale=2):
        super().__init__()

        self.scale = scale

        self.offset = nn.Conv2d(
            channels,
            2 * scale * scale,
            kernel_size=1
        )

    def forward(self, x, target_size=None):
        scale = self.scale
        b, c, h, w = x.shape

        if target_size is None:
            target_h = h * scale
            target_w = w * scale
        else:
            target_h, target_w = target_size

        upsampled = F.interpolate(
            x,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )

        offsets = self.offset(x)

        offsets = F.interpolate(
            offsets,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False
        )

        offsets = offsets.view(
            b, 2, scale * scale, target_h, target_w
        ).mean(dim=2)

        yy, xx = torch.meshgrid(
            torch.linspace(
                -1, 1, target_h,
                device=x.device, dtype=x.dtype
            ),
            torch.linspace(
                -1, 1, target_w,
                device=x.device, dtype=x.dtype
            ),
            indexing="ij"
        )

        grid = torch.stack([xx, yy], dim=-1)
        grid = grid.unsqueeze(0).expand(b, -1, -1, -1).clone()

        grid[..., 0] += offsets[:, 0] / max(target_w, 1)
        grid[..., 1] += offsets[:, 1] / max(target_h, 1)

        return F.grid_sample(
            upsampled,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True
        )


class PANFPN(nn.Module):
    def __init__(self):
        super().__init__()

        self.up_p5 = DySample(512, scale=2)
        self.fuse_p4 = C3k2(512 + 256, 256, n=2)

        self.up_p4 = DySample(256, scale=2)
        self.fuse_p3 = C3k2(256 + 128, 128, n=2)

        self.up_p3 = DySample(128, scale=2)
        self.fuse_p2 = C3k2(128 + 64, 64, n=2)

        self.down_p2 = Conv(64, 128, 3, 2)
        self.fuse_p3_bottom = C3k2(128 + 128, 128, n=2)

        self.down_p3 = Conv(128, 256, 3, 2)
        self.fuse_p4_bottom = C3k2(256 + 256, 256, n=2)

        self.down_p4 = Conv(256, 512, 3, 2)
        self.fuse_p5_bottom = C3k2(512 + 512, 512, n=2)

    def forward(self, features):
        p2 = features["p2"]
        p3 = features["p3"]
        p4 = features["p4"]
        p5 = features["p5"]

        p4_td = self.fuse_p4(
            torch.cat(
                [self.up_p5(p5, p4.shape[-2:]), p4],
                dim=1
            )
        )

        p3_td = self.fuse_p3(
            torch.cat(
                [self.up_p4(p4_td, p3.shape[-2:]), p3],
                dim=1
            )
        )

        p2_td = self.fuse_p2(
            torch.cat(
                [self.up_p3(p3_td, p2.shape[-2:]), p2],
                dim=1
            )
        )

        p3_out = self.fuse_p3_bottom(
            torch.cat(
                [self.down_p2(p2_td), p3_td],
                dim=1
            )
        )

        p4_out = self.fuse_p4_bottom(
            torch.cat(
                [self.down_p3(p3_out), p4_td],
                dim=1
            )
        )

        p5_out = self.fuse_p5_bottom(
            torch.cat(
                [self.down_p4(p4_out), p5],
                dim=1
            )
        )

        return {
            "p2": p2_td,
            "p3": p3_out,
            "p4": p4_out,
            "p5": p5_out
        }