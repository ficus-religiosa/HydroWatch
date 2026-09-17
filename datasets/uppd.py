from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from preprocessing.letterbox import (
    letterbox_image,
    transform_boxes
)
from preprocessing.annotations import yolo_to_xyxy


class UPPDDataset(Dataset):
    """
    Underwater Plastic Pollution Detection dataset.

    Expected layout:
        root/
            images/
            labels/

    Also accepts common train/images + train/labels layouts.

    YOLO annotation:
        class_id xc yc width height
    """

    def __init__(
        self,
        root,
        target_size=(480, 360),
        class_mapping=None
    ):
        self.root = Path(root)
        self.target_size = target_size
        self.class_mapping = class_mapping or {}

        self.image_paths = self._find_images()

        if not self.image_paths:
            raise RuntimeError(
                f"No images found under {self.root}"
            )

    def _find_images(self):
        image_exts = {
            ".jpg", ".jpeg", ".png",
            ".bmp", ".webp"
        }

        return sorted(
            p for p in self.root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in image_exts
        )

    def _label_path(self, image_path):
        candidates = [
            image_path.with_suffix(".txt"),
            self.root / "labels" / (
                image_path.stem + ".txt"
            )
        ]

        for p in candidates:
            if p.exists():
                return p

        # Preserve nested train/val/test structure if present.
        parts = image_path.parts
        if "images" in parts:
            i = len(parts) - 1 - parts[::-1].index("images")
            candidate = Path(
                *parts[:i],
                "labels",
                *parts[i + 1:]
            ).with_suffix(".txt")

            if candidate.exists():
                return candidate

        return None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        src_w, src_h = image.size

        label_path = self._label_path(
            image_path
        )

        rows = []

        if label_path is not None:
            for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines():
                line = line.strip()
                if line:
                    rows.append(
                        line.split()
                    )

        boxes, raw_labels = yolo_to_xyxy(
            rows,
            src_w,
            src_h
        )

        if self.class_mapping:
            labels = []
            keep = []

            for i, raw in enumerate(raw_labels):
                raw_key = str(int(raw))

                if raw_key not in self.class_mapping:
                    continue

                labels.append(
                    self.class_mapping[raw_key]
                )
                keep.append(i)

            boxes = boxes[keep]
            raw_labels = labels
        else:
            # Raw IDs are returned only for inspection.
            raw_labels = raw_labels.tolist()

        image, scale, pad_x, pad_y = letterbox_image(
            image,
            self.target_size
        )

        boxes = transform_boxes(
            boxes,
            scale,
            pad_x,
            pad_y,
            self.target_size
        )

        tensor = torch.from_numpy(
            np.asarray(image)
        ).permute(2, 0, 1).float() / 255.0

        labels = torch.as_tensor(
            raw_labels,
            dtype=torch.long
        )

        return {
            "image": tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": labels,
            "mask": None,
            "image_id": str(image_path),
            "dataset": "uppd"
        }
