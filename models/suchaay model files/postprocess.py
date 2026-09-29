import torch


def box_iou(boxes1, boxes2, eps=1e-7):
    """
    Pairwise IoU for xyxy boxes.

    boxes1: [N,4]
    boxes2: [M,4]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros(
            (boxes1.shape[0], boxes2.shape[0])
        )

    lt = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2]
    )
    rb = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:]
    )

    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0)
        * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    )

    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0)
        * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - inter
    )

    return inter / (union + eps)


def nms(boxes, scores, iou_threshold=0.5):
    """
    Pure PyTorch NMS.

    Returns indices of boxes to keep.
    """
    if boxes.numel() == 0:
        return torch.empty(
            0,
            dtype=torch.long,
            device=boxes.device
        )

    order = scores.argsort(
        descending=True
    )

    keep = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        ious = box_iou(
            boxes[current].unsqueeze(0),
            boxes[order[1:]]
        ).squeeze(0)

        order = order[1:][
            ious <= iou_threshold
        ]

    return torch.stack(keep)


@torch.no_grad()
def decode_detections(
    detections,
    confidence_threshold=0.25,
    iou_threshold=0.5,
    max_detections=300
):
    """
    Decode all P2-P5 outputs into final per-image detections.

    Returns a list of dictionaries, one per batch item:

        boxes:  [N,4] absolute input-pixel xyxy
        scores: [N]
        labels: [N]

    Confidence:
        objectness_probability * class_probability

    This function does not resize boxes. The model's decoded boxes
    are already expressed in the input image coordinate system.
    """
    levels = [
        "p2", "p3", "p4", "p5"
    ]

    batch_size = detections["p2"]["box"].shape[0]

    outputs = []

    for b in range(batch_size):
        level_boxes = []
        level_scores = []
        level_labels = []

        for level in levels:
            pred = detections[level]

            boxes = pred["boxes"][b].reshape(
                -1, 4
            )

            objectness = torch.sigmoid(
                pred["objectness"][b]
            ).reshape(-1)

            class_probs = torch.sigmoid(
                pred["class"][b]
            ).permute(1, 2, 0).reshape(
                -1,
                pred["class"].shape[1]
            )

            scores = (
                objectness[:, None]
                * class_probs
            )

            best_scores, best_labels = scores.max(
                dim=1
            )

            keep = best_scores >= confidence_threshold

            if keep.any():
                level_boxes.append(
                    boxes[keep]
                )
                level_scores.append(
                    best_scores[keep]
                )
                level_labels.append(
                    best_labels[keep]
                )

        if not level_boxes:
            outputs.append(
                {
                    "boxes": torch.empty(
                        (0, 4),
                        device=detections["p2"]["box"].device
                    ),
                    "scores": torch.empty(
                        (0,),
                        device=detections["p2"]["box"].device
                    ),
                    "labels": torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=detections["p2"]["box"].device
                    )
                }
            )
            continue

        boxes = torch.cat(
            level_boxes,
            dim=0
        )
        scores = torch.cat(
            level_scores,
            dim=0
        )
        labels = torch.cat(
            level_labels,
            dim=0
        )

        # Class-aware NMS: boxes of different classes do not suppress
        # each other.
        keep_all = []

        for cls in labels.unique():
            cls_indices = torch.nonzero(
                labels == cls,
                as_tuple=False
            ).flatten()

            cls_keep = nms(
                boxes[cls_indices],
                scores[cls_indices],
                iou_threshold
            )

            keep_all.append(
                cls_indices[cls_keep]
            )

        keep = torch.cat(
            keep_all,
            dim=0
        )

        keep = keep[
            scores[keep].argsort(
                descending=True
            )
        ]

        keep = keep[:max_detections]

        outputs.append(
            {
                "boxes": boxes[keep],
                "scores": scores[keep],
                "labels": labels[keep]
            }
        )

    return outputs
