import torch
import torch.nn as nn
import torch.nn.functional as F

from .assignment import NWDAssigner


def xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)

    return torch.stack(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            (x2 - x1).clamp(min=0),
            (y2 - y1).clamp(min=0)
        ],
        dim=-1
    )


def bbox_iou(box1, box2, eps=1e-7):
    x1 = torch.maximum(
        box1[..., 0],
        box2[..., 0]
    )
    y1 = torch.maximum(
        box1[..., 1],
        box2[..., 1]
    )
    x2 = torch.minimum(
        box1[..., 2],
        box2[..., 2]
    )
    y2 = torch.minimum(
        box1[..., 3],
        box2[..., 3]
    )

    inter = (
        (x2 - x1).clamp(min=0)
        * (y2 - y1).clamp(min=0)
    )

    area1 = (
        (box1[..., 2] - box1[..., 0]).clamp(min=0)
        * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    )

    area2 = (
        (box2[..., 2] - box2[..., 0]).clamp(min=0)
        * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    )

    union = area1 + area2 - inter

    return inter / (union + eps)


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    CIoU loss for absolute-pixel xyxy boxes.
    """
    iou = bbox_iou(
        pred_boxes,
        target_boxes,
        eps
    )

    p_cx = (
        pred_boxes[..., 0]
        + pred_boxes[..., 2]
    ) / 2
    p_cy = (
        pred_boxes[..., 1]
        + pred_boxes[..., 3]
    ) / 2

    t_cx = (
        target_boxes[..., 0]
        + target_boxes[..., 2]
    ) / 2
    t_cy = (
        target_boxes[..., 1]
        + target_boxes[..., 3]
    ) / 2

    center_distance = (
        (p_cx - t_cx).pow(2)
        + (p_cy - t_cy).pow(2)
    )

    enc_x1 = torch.minimum(
        pred_boxes[..., 0],
        target_boxes[..., 0]
    )
    enc_y1 = torch.minimum(
        pred_boxes[..., 1],
        target_boxes[..., 1]
    )
    enc_x2 = torch.maximum(
        pred_boxes[..., 2],
        target_boxes[..., 2]
    )
    enc_y2 = torch.maximum(
        pred_boxes[..., 3],
        target_boxes[..., 3]
    )

    diagonal = (
        (enc_x2 - enc_x1).pow(2)
        + (enc_y2 - enc_y1).pow(2)
        + eps
    )

    pw = (
        pred_boxes[..., 2]
        - pred_boxes[..., 0]
    ).clamp(min=eps)

    ph = (
        pred_boxes[..., 3]
        - pred_boxes[..., 1]
    ).clamp(min=eps)

    tw = (
        target_boxes[..., 2]
        - target_boxes[..., 0]
    ).clamp(min=eps)

    th = (
        target_boxes[..., 3]
        - target_boxes[..., 1]
    ).clamp(min=eps)

    v = (
        4 / torch.pi ** 2
    ) * (
        torch.atan(tw / th)
        - torch.atan(pw / ph)
    ).pow(2)

    alpha = (
        v / (1 - iou + v + eps)
    )

    ciou = (
        iou
        - center_distance / diagonal
        - alpha * v
    )

    return 1 - ciou


def dice_loss(
    logits,
    targets,
    eps=1e-6
):
    targets = targets.to(
        device=logits.device,
        dtype=logits.dtype
    )

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    probs = torch.sigmoid(logits)

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        probs * targets
    ).sum(dim=1)

    denominator = (
        probs.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2 * intersection + eps
    ) / (denominator + eps)

    return 1 - dice.mean()


def distribution_focal_loss(
    pred,
    target,
    reg_max=16
):
    """
    pred:
        [N, reg_max]
    target:
        [N], continuous distance in feature-cell units.
    """
    target = target.clamp(
        0,
        reg_max - 1 - 1e-6
    )

    left = target.floor().long()
    right = (left + 1).clamp(
        max=reg_max - 1
    )

    weight_right = (
        target - left.float()
    )
    weight_left = 1 - weight_right

    loss_left = F.cross_entropy(
        pred,
        left,
        reduction="none"
    )

    loss_right = F.cross_entropy(
        pred,
        right,
        reduction="none"
    )

    return (
        loss_left * weight_left
        + loss_right * weight_right
    )


def flatten_level_prediction(
    pred,
    stride
):
    """
    Converts one detection level into N x ... tensors.
    """
    box_logits = pred["box"]
    obj_logits = pred["objectness"]
    cls_logits = pred["class"]
    boxes = pred["boxes"]

    b, _, h, w = box_logits.shape

    # [B,4*R,H,W] -> [B,H,W,4*R]
    box_logits = box_logits.permute(
        0, 2, 3, 1
    ).contiguous()

    obj_logits = obj_logits.permute(
        0, 2, 3, 1
    ).contiguous()

    cls_logits = cls_logits.permute(
        0, 2, 3, 1
    ).contiguous()

    boxes = boxes.contiguous()

    distances = pred["distances"].permute(
        0, 2, 3, 1
    ).contiguous()

    yy, xx = torch.meshgrid(
        torch.arange(
            h,
            device=box_logits.device,
            dtype=box_logits.dtype
        ),
        torch.arange(
            w,
            device=box_logits.device,
            dtype=box_logits.dtype
        ),
        indexing="ij"
    )

    centers = torch.stack(
        [
            (xx + 0.5) * stride,
            (yy + 0.5) * stride
        ],
        dim=-1
    )

    return (
        box_logits.view(b, h * w, -1),
        obj_logits.view(b, h * w),
        cls_logits.view(b, h * w, -1),
        boxes.view(b, h * w, 4),
        distances.view(b, h * w, 4),
        centers.view(h * w, 2)
    )


class DetectionLoss(nn.Module):
    """
    Full training loss:

        lambda_box * CIoU
      + lambda_cls * BCE classification
      + lambda_obj * BCE objectness
      + lambda_dfl * DFL
      + lambda_mask * Dice

    Expected target format:
        targets = [
            {
                "boxes": Tensor[N,4] in absolute input-pixel xyxy,
                "labels": Tensor[N] with integer class IDs,
                "mask": optional Tensor[H,W] or [1,H,W]
            },
            ...
        ]

    The model must be called in a way that supplies decoded
    per-level boxes and DFL distances.
    """

    def __init__(
        self,
        reg_max=16,
        lambda_box=7.5,
        lambda_cls=0.5,
        lambda_obj=1.0,
        lambda_dfl=1.5,
        lambda_mask=1.0,
        top_k=10,
        nwd_scale=12.8
    ):
        super().__init__()

        self.reg_max = reg_max

        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_dfl = lambda_dfl
        self.lambda_mask = lambda_mask

        self.assigner = NWDAssigner(
            top_k=top_k,
            nwd_scale=nwd_scale
        )

    def forward(
        self,
        predictions,
        targets,
        mask_logits=None
    ):
        device = mask_logits.device if mask_logits is not None else next(
            iter(predictions["p2"]["box"].parameters()),
            None
        )

        if not isinstance(device, torch.Tensor):
            device = predictions["p2"]["box"].device

        total_box = torch.zeros(
            (), device=device
        )
        total_cls = torch.zeros(
            (), device=device
        )
        total_obj = torch.zeros(
            (), device=device
        )
        total_dfl = torch.zeros(
            (), device=device
        )

        level_strides = {
            "p2": 4,
            "p3": 8,
            "p4": 16,
            "p5": 32
        }

        # Flatten all scales.
        flat = {}

        for level in [
            "p2", "p3", "p4", "p5"
        ]:
            flat[level] = flatten_level_prediction(
                predictions[level],
                level_strides[level]
            )

        batch_size = predictions["p2"]["box"].shape[0]

        for batch_index in range(batch_size):
            target = targets[batch_index]

            gt_boxes = target["boxes"].to(
                device=device,
                dtype=torch.float32
            )

            gt_labels = target["labels"].to(
                device=device,
                dtype=torch.long
            )

            if gt_boxes.numel() == 0:
                gt_boxes = gt_boxes.reshape(0, 4)
                gt_labels = gt_labels.reshape(0)

            # Build one global prediction set for NWD assignment.
            all_boxes = []
            all_scores = []
            all_meta = []

            for level in [
                "p2", "p3", "p4", "p5"
            ]:
                (
                    box_logits,
                    obj_logits,
                    cls_logits,
                    boxes,
                    distances,
                    centers
                ) = flat[level]

                all_boxes.append(
                    boxes[batch_index]
                )

                scores = torch.sigmoid(
                    cls_logits[batch_index]
                )

                all_scores.append(scores)

                all_meta.extend(
                    [
                        (
                            level,
                            i,
                            centers[i]
                        )
                        for i in range(
                            centers.shape[0]
                        )
                    ]
                )

            all_boxes = torch.cat(
                all_boxes,
                dim=0
            )

            all_scores = torch.cat(
                all_scores,
                dim=0
            )

            assigned = self.assigner.assign(
                all_boxes.detach(),
                all_scores.detach(),
                gt_boxes,
                gt_labels
            )

            # Objectness target is 1 for assigned positives.
            positive_global = (
                assigned >= 0
            )

            global_offset = 0

            for level in [
                "p2", "p3", "p4", "p5"
            ]:
                (
                    box_logits,
                    obj_logits,
                    cls_logits,
                    boxes,
                    distances,
                    centers
                ) = flat[level]

                n = boxes.shape[1]
                local_assigned = assigned[
                    global_offset:
                    global_offset + n
                ]

                obj_target = (
                    local_assigned >= 0
                ).to(obj_logits.dtype)

                total_obj = total_obj + F.binary_cross_entropy_with_logits(
                    obj_logits[batch_index],
                    obj_target
                )

                pos = torch.nonzero(
                    local_assigned >= 0,
                    as_tuple=False
                ).flatten()

                if pos.numel() > 0:
                    gt_indices = local_assigned[pos]

                    pred_boxes_pos = boxes[
                        batch_index, pos
                    ]

                    target_boxes_pos = gt_boxes[
                        gt_indices
                    ]

                    total_box = total_box + ciou_loss(
                        pred_boxes_pos,
                        target_boxes_pos
                    ).mean()

                    class_target = torch.zeros(
                        (
                            pos.numel(),
                            cls_logits.shape[-1]
                        ),
                        device=device,
                        dtype=cls_logits.dtype
                    )

                    class_target[
                        torch.arange(
                            pos.numel(),
                            device=device
                        ),
                        gt_labels[gt_indices]
                    ] = 1.0

                    total_cls = total_cls + F.binary_cross_entropy_with_logits(
                        cls_logits[
                            batch_index, pos
                        ],
                        class_target
                    )

                    # DFL target: distances from grid center
                    # to GT edges, expressed in feature cells.
                    stride = level_strides[level]

                    centers_pos = centers[pos]

                    ltrb_pixels = torch.stack(
                        [
                            centers_pos[:, 0]
                            - target_boxes_pos[:, 0],
                            centers_pos[:, 1]
                            - target_boxes_pos[:, 1],
                            target_boxes_pos[:, 2]
                            - centers_pos[:, 0],
                            target_boxes_pos[:, 3]
                            - centers_pos[:, 1]
                        ],
                        dim=-1
                    ).clamp(min=0)

                    dfl_target = (
                        ltrb_pixels / stride
                    ).clamp(
                        0,
                        self.reg_max - 1 - 1e-6
                    )

                    raw_box = (
                        predictions[level]["box"]
                        [batch_index, :, :, :]
                        .permute(1, 2, 0)
                        .contiguous()
                    )

                    raw_box = raw_box.view(
                        raw_box.shape[0] *
                        raw_box.shape[1],
                        4,
                        self.reg_max
                    )

                    raw_box_pos = raw_box[pos]

                    dfl_losses = []

                    for side in range(4):
                        dfl_losses.append(
                            distribution_focal_loss(
                                raw_box_pos[:, side, :],
                                dfl_target[:, side],
                                self.reg_max
                            )
                        )

                    total_dfl = total_dfl + torch.stack(
                        dfl_losses
                    ).mean()

                global_offset += n

        # Normalize by batch to keep the loss scale stable.
        total_box = total_box / batch_size
        total_cls = total_cls / batch_size
        total_obj = total_obj / batch_size
        total_dfl = total_dfl / batch_size

        mask_component = torch.zeros(
            (), device=device
        )

        if (
            mask_logits is not None
            and any(
                "mask" in t
                and t["mask"] is not None
                for t in targets
            )
        ):
            mask_targets = torch.stack(
                [
                    t["mask"].to(
                        device=device,
                        dtype=mask_logits.dtype
                    )
                    for t in targets
                ],
                dim=0
            )

            if mask_targets.ndim == 3:
                mask_targets = mask_targets.unsqueeze(1)

            if mask_targets.shape[-2:] != mask_logits.shape[-2:]:
                mask_targets = F.interpolate(
                    mask_targets,
                    size=mask_logits.shape[-2:],
                    mode="nearest"
                )

            mask_component = dice_loss(
                mask_logits,
                mask_targets
            )

        total = (
            self.lambda_box * total_box
            + self.lambda_cls * total_cls
            + self.lambda_obj * total_obj
            + self.lambda_dfl * total_dfl
            + self.lambda_mask * mask_component
        )

        return {
            "total": total,
            "box": total_box,
            "classification": total_cls,
            "objectness": total_obj,
            "dfl": total_dfl,
            "mask": mask_component
        }
