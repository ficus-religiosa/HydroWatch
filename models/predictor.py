from pathlib import Path

import torch
from PIL import Image
import numpy as np

from .hydro_watch import HydroWatch
from .postprocess import decode_detections


class HydroWatchPredictor:
    """
    Simple inference wrapper around HydroWatch.

    Input:
        PIL.Image, numpy HWC RGB image, or torch tensor [3,H,W]/[B,3,H,W].

    The model itself expects tensors. This wrapper performs only the
    basic RGB -> tensor conversion and returns detections in the model's
    input coordinate system.

    It does not perform dataset-specific preprocessing.
    """

    def __init__(
        self,
        model,
        device=None,
        confidence_threshold=0.25,
        iou_threshold=0.5,
        max_detections=300
    ):
        self.device = torch.device(
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model = model.to(self.device)
        self.model.eval()

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

    @staticmethod
    def _to_tensor(image):
        if isinstance(image, Image.Image):
            array = np.asarray(
                image.convert("RGB")
            )
            tensor = torch.from_numpy(
                array
            ).permute(2, 0, 1).float() / 255.0

        elif isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(
                    "NumPy input must be HWC RGB."
                )

            tensor = torch.from_numpy(
                image
            ).permute(2, 0, 1).float()

            if tensor.max() > 1:
                tensor = tensor / 255.0

        elif torch.is_tensor(image):
            tensor = image.float()

            if tensor.ndim == 3:
                if tensor.shape[0] != 3:
                    raise ValueError(
                        "Tensor input must be [3,H,W]."
                    )
            elif tensor.ndim == 4:
                if tensor.shape[1] != 3:
                    raise ValueError(
                        "Tensor input must be [B,3,H,W]."
                    )
            else:
                raise ValueError(
                    "Tensor input must be [3,H,W] or [B,3,H,W]."
                )

            if tensor.max() > 1:
                tensor = tensor / 255.0

            return tensor

        else:
            raise TypeError(
                "Input must be PIL.Image, NumPy array, or torch tensor."
            )

        return tensor

    @torch.no_grad()
    def predict(self, image):
        tensor = self._to_tensor(image)

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device,
            non_blocking=True
        )

        output = self.model(tensor)

        detections = decode_detections(
            output["detections"],
            confidence_threshold=self.confidence_threshold,
            iou_threshold=self.iou_threshold,
            max_detections=self.max_detections
        )

        return detections


def load_model(
    checkpoint_path,
    num_classes,
    device=None,
    reg_max=16,
    confidence_threshold=0.25,
    iou_threshold=0.5,
    max_detections=300
):
    """
    Construct HydroWatch and load a checkpoint produced by
    models.checkpoint.save_checkpoint().
    """
    from .checkpoint import load_checkpoint

    model = HydroWatch(
        num_classes=num_classes,
        reg_max=reg_max
    )

    load_checkpoint(
        checkpoint_path,
        model=model,
        device=device
    )

    return HydroWatchPredictor(
        model,
        device=device,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections
    )
