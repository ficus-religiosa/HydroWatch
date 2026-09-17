import numpy as np


def yolo_to_xyxy(
    rows,
    width,
    height
):
    """
    YOLO rows:
        class_id xc yc w h
    where coordinates are normalized to [0,1].

    Returns:
        boxes: float32 [N,4] xyxy
        labels: int64 [N]
    """
    boxes = []
    labels = []

    for row in rows:
        if len(row) < 5:
            continue

        cls = int(float(row[0]))
        xc, yc, bw, bh = map(
            float,
            row[1:5]
        )

        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height

        boxes.append(
            [x1, y1, x2, y2]
        )
        labels.append(cls)

    return (
        np.asarray(
            boxes,
            dtype=np.float32
        ).reshape(-1, 4),
        np.asarray(
            labels,
            dtype=np.int64
        )
    )


def voc_to_xyxy(objects):
    boxes = []
    labels = []

    for obj in objects:
        name = obj["name"]
        b = obj["bbox"]

        boxes.append(
            [
                float(b["xmin"]),
                float(b["ymin"]),
                float(b["xmax"]),
                float(b["ymax"])
            ]
        )
        labels.append(name)

    return (
        np.asarray(
            boxes,
            dtype=np.float32
        ).reshape(-1, 4),
        labels
    )
