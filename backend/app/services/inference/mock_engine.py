"""
MOCK INFERENCE ENGINE
=====================
This module provides placeholder object detection results for the HydroWatch
marine debris detector while the actual PyTorch model architecture is being
finalized by the research team.

When the real model is ready, implement a new BaseInferenceEngine and select it
through the inference factory.

Do NOT use this module in production inference.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from app.services.inference.base import BaseInferenceEngine
from app.utils.logger import logger


# ---------------------------------------------------------------------------
# Catalogue of marine debris classes matched to the dataset vocabulary
# ---------------------------------------------------------------------------
_DEBRIS_CATALOGUE: List[Dict[str, Any]] = [
    # label               conf   size      nx    ny    nw    nh   (normalized 0-1)
    {"label": "Plastic Bottle",   "confidence": 96.2, "size": "medium", "nx": 0.18, "ny": 0.32, "nw": 0.18, "nh": 0.24},
    {"label": "Fishing Net",      "confidence": 88.5, "size": "large",  "nx": 0.38, "ny": 0.28, "nw": 0.24, "nh": 0.27},
    {"label": "Plastic Bag",      "confidence": 93.1, "size": "small",  "nx": 0.58, "ny": 0.46, "nw": 0.15, "nh": 0.18},
    {"label": "Plastic Container","confidence": 91.4, "size": "medium", "nx": 0.71, "ny": 0.38, "nw": 0.17, "nh": 0.21},
    {"label": "Other Debris",     "confidence": 86.8, "size": "large",  "nx": 0.49, "ny": 0.63, "nw": 0.20, "nh": 0.23},
    {"label": "Plastic Bottle",   "confidence": 94.7, "size": "small",  "nx": 0.12, "ny": 0.68, "nw": 0.16, "nh": 0.17},
]


class MockInferenceEngine(BaseInferenceEngine):
    """
    [MOCK] Deterministic marine debris detection engine.

    Produces structurally correct, frame-varied detections based on a stable
    hash of the image path so results are reproducible for the same input.

    Contract:
    ---------
    Each returned detection dict contains:
        id          : str   – globally unique detection ID
        label       : str   – debris class name
        confidence  : float – score 0–100 (percentage form)
        size        : str   – "small" | "medium" | "large"
        bbox        : dict  – normalized bounding box {x, y, width, height} in [0, 1]
        x, y, width, height : float – same as bbox values (percentage form 0–100)
                              for backward compatibility with existing frontend
    """

    @property
    def model_info(self) -> Dict[str, Any]:
        return {"name": "mock", "is_mock": True, "status": "deterministic_placeholder"}

    def detect_image(
        self, image_path: str, confidence_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        [MOCK] Generate placeholder detections for one image or video frame.

        Parameters
        ----------
        image_path:
            Absolute path to the image / extracted frame JPEG on disk.
        confidence_threshold:
            Detections below this threshold (expressed as 0–1) are filtered out.
            A threshold of 0.5 means keep detections with confidence >= 50.

        Returns
        -------
        List of detection dicts conforming to the BACKEND_HANDOFF.md contract.
        """
        # Derive a stable integer seed from the path so every frame in a video
        # produces a slightly different but reproducible detection set.
        path_hash = int(hashlib.md5(image_path.encode()).hexdigest(), 16)
        seed = path_hash % len(_DEBRIS_CATALOGUE)

        # Try to read actual image dimensions so bbox coords stay inside frame
        img_w, img_h = 640, 480  # sensible defaults
        try:
            path_obj = Path(image_path)
            if path_obj.exists():
                with Image.open(path_obj) as img:
                    img_w, img_h = img.size
        except Exception:
            pass  # fall back to defaults silently

        detections: List[Dict[str, Any]] = []
        for i, template in enumerate(_DEBRIS_CATALOGUE):
            # Skip entry deterministically so different frames have different counts
            if (i + seed) % 3 == 0:
                continue

            confidence: float = round(template["confidence"] / 100.0, 4)
            # Normalized bbox values (0–1, relative to frame dimensions)
            nx: float = round(template["nx"], 4)
            ny: float = round(template["ny"], 4)
            nw: float = round(template["nw"], 4)
            nh: float = round(template["nh"], 4)

            # Pixel-percentage form (0–100) for frontend CSS positioning
            px: float = round(nx * 100.0, 2)
            py: float = round(ny * 100.0, 2)
            pw: float = round(nw * 100.0, 2)
            ph: float = round(nh * 100.0, 2)

            # The pipeline assigns the contract ID after the source frame is known.
            det_id = f"engine_{i + 1:02d}"

            detections.append(
                {
                    "id": det_id,
                    "label": template["label"],
                    "confidence": confidence,
                    "size": template["size"],
                    # Normalized bbox (0–1) — primary contract field
                    "bbox": {"x": nx, "y": ny, "width": nw, "height": nh},
                    # Percentage form (0–100) — backward-compat with App.jsx / DetectionTable.jsx
                    "x": px,
                    "y": py,
                    "width": pw,
                    "height": ph,
                }
            )

        logger.debug(
            "[MOCK] detect_image('%s') → %d detections (seed=%d)",
            Path(image_path).name,
            len(detections),
            seed,
        )
        return detections


