from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

from preprocessing.letterbox import (
    letterbox_image,
    transform_boxes,
    transform_mask
)


class SeaClearDataset(Dataset):
    """
    SeaClear COCO adapter.

    Expected:
        root/
            images...
            annotations/*.json

    Or a direct COCO JSON path can be supplied as annotation_file.
    """

    def __init__(
        self,
        root,
        annotation_file=None,
        target_size=(480, 360),
        class_mapping=None
    ):
        self.root = Path(root)
        self.target_size = target_size
        self.class_mapping = class_mapping or {}

        if annotation_file is None:
            jsons = sorted(
                self.root.rglob("*.json")
            )

            if not jsons:
                raise RuntimeError(
                    f"No COCO JSON found under {self.root}"
                )

            # Prefer files whose names indicate annotations.
            preferred = [
                p for p in jsons
                if "annot" in p.name.lower()
                or "instances" in p.name.lower()
            ]

            annotation_file = (
                preferred[0]
                if preferred
                else jsons[0]
            )

        self.annotation_file = Path(
            annotation_file
        )

        data = json.loads(
            self.annotation_file.read_text(
                encoding="utf-8"
            )
        )

        self.images = {
            int(x["id"]): x
            for x in data["images"]
        }

        self.categories = {
            int(x["id"]): x["name"]
            for x in data["categories"]
        }

        self.annotations_by_image = {}

        for ann in data["annotations"]:
            image_id = int(
                ann["image_id"]
            )
            self.annotations_by_image.setdefault(
                image_id,
                []
            ).append(ann)

        self.image_ids = sorted(
            self.images.keys()
        )

    def _find_image(self, file_name):
        candidates = [
            self.annotation_file.parent / file_name,
            self.root / file_name,
            self.root / "images" / file_name
        ]

        for p in candidates:
            if p.exists():
                return p

        matches = list(
            self.root.rglob(
                Path(file_name).name
            )
        )

        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"SeaClear image {file_name!r} not found."
        )

    @staticmethod
    def _polygon_mask(
        segmentation,
        width,
        height
    ):
        mask = Image.new(
            "L",
            (width, height),
            0
        )

        draw = ImageDraw.Draw(mask)

        if isinstance(segmentation, list):
            for polygon in segmentation:
                if not polygon:
                    continue

                points = [
                    (
                        polygon[i],
                        polygon[i + 1]
                    )
                    for i in range(
                        0,
                        len(polygon) - 1,
                        2
                    )
                ]

                if len(points) >= 3:
                    draw.polygon(
                        points,
                        fill=1
                    )

        return mask

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        info = self.images[image_id]

        image_path = self._find_image(
            info["file_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        src_w, src_h = image.size

        boxes = []
        labels = []
        object_masks = []

        for ann in self.annotations_by_image.get(
            image_id,
            []
        ):
            category_id = int(
                ann["category_id"]
            )
            raw_name = self.categories[
                category_id
            ]

            if self.class_mapping:
                if raw_name not in self.class_mapping:
                    continue

                label = self.class_mapping[
                    raw_name
                ]
            else:
                label = raw_name

            x, y, w, h = map(
                float,
                ann["bbox"]
            )

            boxes.append(
                [
                    x,
                    y,
                    x + w,
                    y + h
                ]
            )
            labels.append(label)

            if "segmentation" in ann:
                object_masks.append(
                    self._polygon_mask(
                        ann["segmentation"],
                        src_w,
                        src_h
                    )
                )

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

        # Build one foreground mask from all available polygons.
        mask = None

        if object_masks:
            combined = np.zeros(
                (src_h, src_w),
                dtype=np.uint8
            )

            for m in object_masks:
                combined = np.maximum(
                    combined,
                    np.asarray(m)
                )

            mask_img = transform_mask(
                combined,
                scale,
                pad_x,
                pad_y,
                self.target_size
            )

            mask = torch.from_numpy(
                np.asarray(mask_img)
            ).float()

        tensor = torch.from_numpy(
            np.asarray(image)
        ).permute(2, 0, 1).float() / 255.0

        if self.class_mapping:
            labels = torch.tensor(
                labels,
                dtype=torch.long
            )

        return {
            "image": tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": labels,
            "mask": mask,
            "image_id": str(image_path),
            "dataset": "seaclear"
        }
