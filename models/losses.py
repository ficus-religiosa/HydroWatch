import torch
import torch.nn as nn
import torch.nn.functional as F


def xywh_to_xyxy(box):
    x, y, w, h = box.unbind(-1)

    return torch.stack(
        [
            x - w / 2,
            y - h / 2,
            x + w / 2,
            y + h / 2
        ],
        dim=-1
    )


def bbox_iou(box1, box2, eps=1e-7):
    x1 = torch.maximum(box1[..., 0], box2[..., 0])
    y1 = torch.maximum(box1[..., 1], box2[..., 1])
    x2 = torch.minimum(box1[..., 2], box2[..., 2])
    y2 = torch.minimum(box1[..., 3], box2[..., 3])

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
    pred = xywh_to_xyxy(pred_boxes)
    target = xywh_to_xyxy(target_boxes)

    iou = bbox_iou(pred, target, eps)

    center_distance = (
        pred_boxes[..., :2] - target_boxes[..., :2]
    ).pow(2).sum(dim=-1)

    x1 = torch.minimum(pred[..., 0], target[..., 0])
    y1 = torch.minimum(pred[..., 1], target[..., 1])
    x2 = torch.maximum(pred[..., 2], target[..., 2])
    y2 = torch.maximum(pred[..., 3], target[..., 3])

    diagonal = (
        (x2 - x1).pow(2)
        + (y2 - y1).pow(2)
        + eps
    )

    pw = pred_boxes[..., 2]
    ph = pred_boxes[..., 3]
    tw = target_boxes[..., 2]
    th = target_boxes[..., 3]

    v = (
        4 / torch.pi ** 2
    ) * (
        torch.atan(tw / (th + eps))
        - torch.atan(pw / (ph + eps))
    ).pow(2)

    alpha = v / (1 - iou + v + eps)

    ciou = (
        iou
        - center_distance / diagonal
        - alpha * v
    )

    return 1 - ciou


def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)

    probs = probs.flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(dim=1)

    denominator = (
        probs.sum(dim=1) + targets.sum(dim=1)
    )

    dice = (
        2 * intersection + eps
    ) / (denominator + eps)

    return 1 - dice.mean()


def distribution_focal_loss(pred, target, reg_max=16):
    target = target.clamp(
        0,
        reg_max - 1 - 1e-6
    )

    left = target.floor().long()

    right = (left + 1).clamp(
        max=reg_max - 1
    )

    weight_right = target - left.float()
    weight_left = 1 - weight_right

    loss_left = F.cross_entropy(
        pred, left, reduction="none"
    )

    loss_right = F.cross_entropy(
        pred, right, reduction="none"
    )

    return (
        loss_left * weight_left
        + loss_right * weight_right
    ).mean()


class DetectionLoss(nn.Module):
    def __init__(
        self,
        reg_max=16,
        lambda_box=7.5,
        lambda_cls=0.5,
        lambda_dfl=1.5,
        lambda_obj=1.0,
        lambda_mask=1.0
    ):
        super().__init__()

        self.reg_max = reg_max
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_dfl = lambda_dfl
        self.lambda_obj = lambda_obj
        self.lambda_mask = lambda_mask

    def forward(
        self,
        pred,
        target,
        mask_logits=None,
        mask_targets=None
    ):
        if mask_logits is not None:
            device = mask_logits.device
        elif isinstance(pred, dict):
            device = next(iter(pred.values())).device
        else:
            device = target.device

        total = torch.zeros((), device=device)

        if mask_logits is not None and mask_targets is not None:
            total = total + self.lambda_mask * dice_loss(
                mask_logits,
                mask_targets
            )

        return total
