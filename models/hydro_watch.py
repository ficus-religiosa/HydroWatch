import torch.nn as nn

from .frontend import FrontEnd
from .backbone import YOLO11SBackbone
from .neck import PANFPN
from .heads import MultiScaleDetectionHead, AuxiliaryMaskHead


class HydroWatch(nn.Module):
    """
    Complete Marine Debris detection architecture.

    Input:
        RGB [B,3,H,W]

    Output:
        Multi-scale detection predictions
        + optional auxiliary mask prediction.
    """

    def __init__(
        self,
        num_classes=1,
        reg_max=16,
        enable_mask_head=True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.enable_mask_head = enable_mask_head

        self.frontend = FrontEnd()
        self.backbone = YOLO11SBackbone()
        self.neck = PANFPN()

        self.detection_head = MultiScaleDetectionHead(
            num_classes=num_classes,
            reg_max=reg_max
        )

        self.mask_head = (
            AuxiliaryMaskHead()
            if enable_mask_head
            else None
        )

    def forward(self, x, return_masks=None):
        if return_masks is None:
            return_masks = (
                self.training and self.enable_mask_head
            )

        input_size = x.shape[-2:]

        x = self.frontend(x)
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        detections = self.detection_head(neck_features)

        output = {
            "detections": detections,
            "features": neck_features
        }

        if return_masks and self.mask_head is not None:
            output["mask_logits"] = self.mask_head(
                neck_features["p2"],
                input_size
            )

        return output