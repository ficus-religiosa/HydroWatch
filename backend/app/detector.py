"""Loads the trained HydroNet checkpoint once and runs it on images (BGR numpy arrays)."""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import BACKEND_DIR, settings

# The checkpoint stores the HydroNet model class by module name "hydronet", so that module must be importable.
sys.path.insert(0, str(BACKEND_DIR))
import hydronet  # noqa: E402,F401

# model class key -> (label shown in the website, tier)
CLASS_INFO = {
    "bottle": ("Bottle", "debris"),
    "can_cup": ("Can / cup", "debris"),
    "soft_debris": ("Soft debris", "debris"),
    "pipe_tube": ("Pipe / tube", "debris"),
    "rope_net": ("Rope / net", "debris"),
    "rubber": ("Rubber", "debris"),
    "wreckage": ("Wreckage", "debris"),
    "wood": ("Wood", "debris"),
    "other_debris": ("Other debris", "debris"),
    "fish": ("Fish", "marine life"),
    "invertebrate": ("Invertebrate", "marine life"),
    "plant": ("Plant", "marine life"),
}
DEBRIS_COLOR = (0, 165, 255)   # BGR orange
LIFE_COLOR = (120, 220, 90)    # BGR green


class Detector:
    def __init__(self):
        import torch
        from ultralytics import YOLO

        path = Path(settings.MODEL_WEIGHTS)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {path}. Copy your trained best.pt there, or set MODEL_WEIGHTS "
                f"in backend/.env (see backend/README.md).")
        self.weights = path
        self.model = YOLO(str(path))
        dev = settings.DEVICE
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.half = settings.HALF and dev.startswith("cuda")
        self.gpu_name = torch.cuda.get_device_name(0) if dev.startswith("cuda") else None
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.predict([np.zeros((720, 1280, 3), dtype=np.uint8)], 0.5)   # warm-up: first call is slow

    def info(self):
        return {
            "name": "HydroNet (YOLO11-s + P2, physics-conditioned)",
            "weights": self.weights.name,
            "device": self.gpu_name or self.device,
            "fp16": self.half,
            "imgsz": settings.IMGSZ,
            "classes": [CLASS_INFO.get(n, (n, "debris"))[0] for n in self.names.values()],
            "note": settings.MODEL_NOTE,
            "is_mock": False,
        }

    def predict(self, images, conf):
        """images: list of BGR arrays -> list (per image) of detection dicts with normalised boxes."""
        extra = {"quantize": 16} if self.half else {}
        results = self.model.predict(images, imgsz=settings.IMGSZ, conf=conf, device=self.device,
                                     max_det=300, verbose=False, **extra)
        small, medium = settings.SIZE_CUTS
        out = []
        for r, im in zip(results, images):
            h, w = im.shape[:2]
            dets = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.float().cpu().numpy()
                confs = r.boxes.conf.float().cpu().numpy()
                cls = r.boxes.cls.int().cpu().numpy()
                for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls):
                    x1, x2 = max(0.0, float(x1)), min(float(w), float(x2))
                    y1, y2 = max(0.0, float(y1)), min(float(h), float(y2))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    key = self.names.get(int(k), str(k))
                    label, tier = CLASS_INFO.get(key, (key, "debris"))
                    geo = math.sqrt((x2 - x1) * (y2 - y1)) / max(w, h)
                    dets.append({
                        "class_key": key, "label": label, "tier": tier,
                        "confidence": round(float(c), 4),
                        "size": "Small" if geo < small else "Medium" if geo < medium else "Large",
                        "bbox": {"x": round(x1 / w, 5), "y": round(y1 / h, 5),
                                 "width": round((x2 - x1) / w, 5), "height": round((y2 - y1) / h, 5)},
                    })
            out.append(dets)
        return out


def draw(image, detections):
    """Copy of image with boxes and labels drawn in pixel space."""
    img = image.copy()
    h, w = img.shape[:2]
    t = max(2, round(max(h, w) / 640))
    fs = max(0.5, max(h, w) / 1600)
    for d in detections:
        b = d["bbox"]
        x1, y1 = int(b["x"] * w), int(b["y"] * h)
        x2, y2 = int((b["x"] + b["width"]) * w), int((b["y"] + b["height"]) * h)
        color = DEBRIS_COLOR if d["tier"] == "debris" else LIFE_COLOR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
        text = f"{d['label']} {d['confidence'] * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        ty = y1 - 4 if y1 - th - 8 > 0 else y1 + th + 8
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + base - 2), color, -1)
        cv2.putText(img, text, (x1 + 3, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), max(1, t // 2),
                    cv2.LINE_AA)
    return img
