import torch.nn as nn

from .frontend import FrontEnd
from .backbone import YOLO11SBackbone
from .neck import PANFPN
from .heads import MultiScaleDetectionHead


class HydroWatch(nn.Module):
    """HydroWatch lightweight multi-scale marine-debris detector."""

    def __init__(self, num_classes=1, reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        self.frontend = FrontEnd()
        self.backbone = YOLO11SBackbone()
        self.neck = PANFPN()
        self.detection_head = MultiScaleDetectionHead(
            num_classes=num_classes,
            reg_max=reg_max,
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.frontend(x)
        backbone_features = self.backbone(x)
        neck_features = self.neck(backbone_features)
        detections = self.detection_head(
            neck_features,
            image_size=input_size,
        )
        return {
            "detections": detections,
            "features": neck_features,
        }
