from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from preprocessing.letterbox import (
    letterbox_image,
    transform_boxes
)


class TrashCanDataset(Dataset):
    """
    TrashCan 1.0 adapter.

    Supports Pascal-VOC XML annotations, which are present
    in the supplied TrashCan examples.

    Expected flexible layout:
        root/
            images...
            *.xml

    Class names are preserved unless class_mapping is supplied.
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

        self.xml_paths = sorted(
            self.root.rglob("*.xml")
        )

        if not self.xml_paths:
            raise RuntimeError(
                f"No XML annotations found under {self.root}"
            )

    @staticmethod
    def _parse_xml(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext(
            "filename"
        )

        size = root.find("size")

        width = int(
            size.findtext("width")
        )
        height = int(
            size.findtext("height")
        )

        objects = []

        for obj in root.findall("object"):
            name = obj.findtext("name")

            box = obj.find("bndbox")

            if box is None:
                continue

            objects.append(
                {
                    "name": name,
                    "bbox": {
                        "xmin": float(
                            box.findtext("xmin")
                        ),
                        "ymin": float(
                            box.findtext("ymin")
                        ),
                        "xmax": float(
                            box.findtext("xmax")
                        ),
                        "ymax": float(
                            box.findtext("ymax")
                        )
                    }
                }
            )

        return (
            filename,
            width,
            height,
            objects
        )

    def _find_image(self, xml_path, filename):
        candidates = [
            xml_path.parent / filename,
            self.root / "images" / filename
        ]

        for p in candidates:
            if p.exists():
                return p

        matches = list(
            self.root.rglob(filename)
        )

        if matches:
            return matches[0]

        raise FileNotFoundError(
            f"Image {filename!r} for {xml_path} not found."
        )

    def __len__(self):
        return len(self.xml_paths)

    def __getitem__(self, index):
        xml_path = self.xml_paths[index]

        (
            filename,
            src_w,
            src_h,
            objects
        ) = self._parse_xml(xml_path)

        image_path = self._find_image(
            xml_path,
            filename
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        # Use actual image dimensions when available.
        actual_w, actual_h = image.size
        src_w, src_h = actual_w, actual_h

        boxes = []
        labels = []

        for obj in objects:
            raw_name = obj["name"]

            if self.class_mapping:
                if raw_name not in self.class_mapping:
                    continue
                label = self.class_mapping[
                    raw_name
                ]
            else:
                # Raw labels cannot be converted to model IDs
                # until the unified taxonomy is defined.
                label = raw_name

            b = obj["bbox"]
            boxes.append(
                [
                    b["xmin"],
                    b["ymin"],
                    b["xmax"],
                    b["ymax"]
                ]
            )
            labels.append(label)

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

        if self.class_mapping:
            labels = torch.tensor(
                labels,
                dtype=torch.long
            )
        else:
            labels = labels

        return {
            "image": tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": labels,
            "mask": None,
            "image_id": str(image_path),
            "dataset": "trashcan"
        }
