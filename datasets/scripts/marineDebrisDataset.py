import base64
import io
import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


class AnnotationParser:
    """
    Converts the raw annotation JSON into a standardized
    representation used by our detection dataset.
    """

    def __init__(self, target_classes):
        self.target_classes = target_classes

    @staticmethod
    def decode_bitmap(bitmap_data):
        """
        Decode the compressed bitmap stored inside the annotation JSON.

        Returns:
            RGBA numpy array: [H, W, 4]
        """

        compressed = base64.b64decode(bitmap_data)

        png_bytes = zlib.decompress(compressed)

        image = Image.open(
            io.BytesIO(png_bytes)
        ).convert("RGBA")

        return np.asarray(image)

    @staticmethod
    def reconstruct_mask(bitmap_data, origin, image_width, image_height):
        """
        Reconstruct the local bitmap into the full-image mask.
        """

        rgba = AnnotationParser.decode_bitmap(bitmap_data)

        # Alpha channel contains the object mask.
        local_mask = rgba[:, :, 3] > 0

        origin_x, origin_y = origin

        local_height, local_width = local_mask.shape

        full_mask = np.zeros((image_height, image_width), dtype=np.uint8)

        # Prevent accidental overflow.
        x1 = max(0, origin_x)
        y1 = max(0, origin_y)

        x2 = min(origin_x + local_width, image_width)

        y2 = min(origin_y + local_height, image_height)

        if x2 <= x1 or y2 <= y1:
            return full_mask

        src_x1 = x1 - origin_x
        src_y1 = y1 - origin_y

        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        full_mask[y1:y2, x1:x2] = (local_mask[src_y1:src_y2, src_x1:src_x2].astype(np.uint8))

        return full_mask

    @staticmethod
    def mask_to_bbox(mask):
        """
        Convert binary mask to pixel-coordinate bounding box.

        Returns:
            [x_min, y_min, x_max, y_max]
        """

        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            return None

        return [
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max())
        ]

    @staticmethod
    def bbox_to_yolo(bbox, image_width, image_height):
        """
        Convert pixel bbox to normalized:

            x_center
            y_center
            width
            height
        """

        x_min, y_min, x_max, y_max = bbox

        width = x_max - x_min
        height = y_max - y_min

        x_center = x_min + width / 2
        y_center = y_min + height / 2

        return [
            x_center / image_width,
            y_center / image_height,
            width / image_width,
            height / image_height
        ]

    def parse(self, annotation_path):
        """
        Parse one annotation JSON.

        Returns a dictionary containing all target objects.
        """

        annotation_path = Path(annotation_path)

        with annotation_path.open("r") as f:
            annotation = json.load(f)

        image_width = annotation["size"]["width"]
        image_height = annotation["size"]["height"]

        objects = []

        for obj in annotation.get("objects", []):

            class_title = obj.get("classTitle")

            # Ignore non-target classes such as ROV/unknown
            # unless explicitly included in target_classes.
            if class_title not in self.target_classes:
                continue

            class_index = self.target_classes[class_title]

            if obj.get("geometryType") != "bitmap":
                continue

            bitmap = obj["bitmap"]

            mask = self.reconstruct_mask(
                bitmap_data=bitmap["data"],
                origin=bitmap["origin"],
                image_width=image_width,
                image_height=image_height
            )

            bbox = self.mask_to_bbox(mask)

            if bbox is None:
                continue

            yolo_bbox = self.bbox_to_yolo(
                bbox,
                image_width,
                image_height
            )

            # Preserve all source attributes.
            attributes = {}

            for tag in obj.get("tags", []):
                attributes[tag["name"]] = tag.get("value")

            objects.append({
                "class_id": class_index,
                "class_name": class_title,

                "bbox": bbox,
                "yolo_bbox": yolo_bbox,

                "mask": mask,

                "attributes": attributes,

                # Preserve original dataset ID.
                "original_class_id": obj.get("classId"),
                "original_object_id": obj.get("id")
            })

        return {
            "width": image_width,
            "height": image_height,
            "objects": objects
        }