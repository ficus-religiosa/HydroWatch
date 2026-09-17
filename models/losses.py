import torch
import torch.nn as nn
import torch.nn.functional as F

from .assignment import NWDAssigner


def bbox_iou(box1, box2, eps=1e-7):
    x1 = torch.maximum(box1[..., 0], box2[..., 0])
    y1 = torch.maximum(box1[..., 1], box2[..., 1])
    x2 = torch.minimum(box1[..., 2], box2[..., 2])
    y2 = torch.minimum(box1[..., 3], box2[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    iou = bbox_iou(pred_boxes, target_boxes, eps)

    p_cx = (pred_boxes[..., 0] + pred_boxes[..., 2]) / 2
    p_cy = (pred_boxes[..., 1] + pred_boxes[..., 3]) / 2
    t_cx = (target_boxes[..., 0] + target_boxes[..., 2]) / 2
    t_cy = (target_boxes[..., 1] + target_boxes[..., 3]) / 2

    center_distance = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    enc_x1 = torch.minimum(pred_boxes[..., 0], target_boxes[..., 0])
    enc_y1 = torch.minimum(pred_boxes[..., 1], target_boxes[..., 1])
    enc_x2 = torch.maximum(pred_boxes[..., 2], target_boxes[..., 2])
    enc_y2 = torch.maximum(pred_boxes[..., 3], target_boxes[..., 3])
    diagonal = (enc_x2 - enc_x1).pow(2) + (enc_y2 - enc_y1).pow(2) + eps

    pw = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp(min=eps)
    ph = (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp(min=eps)
    tw = (target_boxes[..., 2] - target_boxes[..., 0]).clamp(min=eps)
    th = (target_boxes[..., 3] - target_boxes[..., 1]).clamp(min=eps)

    v = (4 / torch.pi**2) * (
        torch.atan(tw / th) - torch.atan(pw / ph)
    ).pow(2)
    alpha = v / (1 - iou + v + eps)

    return 1 - (iou - center_distance / diagonal - alpha * v)


def distribution_focal_loss(pred, target, reg_max=16):
    target = target.clamp(0, reg_max - 1 - 1e-6)
    left = target.floor().long()
    right = (left + 1).clamp(max=reg_max - 1)
    weight_right = target - left.float()
    weight_left = 1 - weight_right

    return (
        F.cross_entropy(pred, left, reduction="none") * weight_left
        + F.cross_entropy(pred, right, reduction="none") * weight_right
    )


def flatten_level_prediction(pred, stride):
    box_logits = pred["box"]
    obj_logits = pred["objectness"]
    cls_logits = pred["class"]
    boxes = pred["boxes"]
    distances = pred["distances"]
    b, _, h, w = box_logits.shape

    yy, xx = torch.meshgrid(
        torch.arange(h, device=box_logits.device, dtype=box_logits.dtype),
        torch.arange(w, device=box_logits.device, dtype=box_logits.dtype),
        indexing="ij",
    )
    centers = torch.stack([(xx + 0.5) * stride, (yy + 0.5) * stride], dim=-1)

    return (
        box_logits.permute(0, 2, 3, 1).contiguous().view(b, h * w, -1),
        obj_logits.permute(0, 2, 3, 1).contiguous().view(b, h * w),
        cls_logits.permute(0, 2, 3, 1).contiguous().view(b, h * w, -1),
        boxes.view(b, h * w, 4),
        distances.permute(0, 2, 3, 1).contiguous().view(b, h * w, 4),
        centers.view(h * w, 2),
    )


class DetectionLoss(nn.Module):
    """Detection-only loss: CIoU + classification BCE + objectness BCE + DFL."""

    def __init__(
        self,
        reg_max=16,
        lambda_box=7.5,
        lambda_cls=0.5,
        lambda_obj=1.0,
        lambda_dfl=1.5,
        top_k=10,
        nwd_scale=12.8,
    ):
        super().__init__()
        self.reg_max = reg_max
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_obj = lambda_obj
        self.lambda_dfl = lambda_dfl
        self.assigner = NWDAssigner(top_k=top_k, nwd_scale=nwd_scale)

    def forward(self, predictions, targets):
        device = predictions["p2"]["box"].device
        levels = ["p2", "p3", "p4", "p5"]
        strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        flat = {level: flatten_level_prediction(predictions[level], strides[level]) for level in levels}

        batch_size = predictions["p2"]["box"].shape[0]
        total_box = torch.zeros((), device=device)
        total_cls = torch.zeros((), device=device)
        total_obj = torch.zeros((), device=device)
        total_dfl = torch.zeros((), device=device)

        for b in range(batch_size):
            gt_boxes = targets[b]["boxes"].to(device=device, dtype=torch.float32).reshape(-1, 4)
            gt_labels = targets[b]["labels"].to(device=device, dtype=torch.long).reshape(-1)

            all_boxes, all_scores, all_centers = [], [], []
            for level in levels:
                _, _, cls_logits, boxes, _, centers = flat[level]
                all_boxes.append(boxes[b])
                all_scores.append(torch.sigmoid(cls_logits[b]))
                all_centers.append(centers)

            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_centers = torch.cat(all_centers, dim=0)

            assigned = self.assigner.assign(
                all_boxes.detach(), all_scores.detach(), all_centers,
                gt_boxes, gt_labels
            )

            offset = 0
            for level in levels:
                box_logits, obj_logits, cls_logits, boxes, distances, centers = flat[level]
                n = boxes.shape[1]
                local = assigned[offset:offset + n]

                obj_target = (local >= 0).to(obj_logits.dtype)
                total_obj = total_obj + F.binary_cross_entropy_with_logits(
                    obj_logits[b], obj_target
                )

                pos = torch.nonzero(local >= 0, as_tuple=False).flatten()
                if pos.numel() == 0:
                    offset += n
                    continue

                gt_idx = local[pos]
                pred_pos = boxes[b, pos]
                target_pos = gt_boxes[gt_idx]
                total_box = total_box + ciou_loss(pred_pos, target_pos).mean()

                class_target = torch.zeros(
                    (pos.numel(), cls_logits.shape[-1]),
                    device=device,
                    dtype=cls_logits.dtype,
                )
                class_target[
                    torch.arange(pos.numel(), device=device), gt_labels[gt_idx]
                ] = 1.0
                total_cls = total_cls + F.binary_cross_entropy_with_logits(
                    cls_logits[b, pos], class_target
                )

                # Ground-truth l/t/r/b distances from grid center, in cell units.
                c = centers[pos]
                stride = strides[level]
                target_ltrb = torch.stack(
                    [
                        c[:, 0] - target_pos[:, 0],
                        c[:, 1] - target_pos[:, 1],
                        target_pos[:, 2] - c[:, 0],
                        target_pos[:, 3] - c[:, 1],
                    ],
                    dim=-1,
                ).clamp(min=0) / stride
                target_ltrb = target_ltrb.clamp(0, self.reg_max - 1 - 1e-6)

                raw = box_logits[b, pos].view(pos.numel(), 4, self.reg_max)
                side_losses = [
                    distribution_focal_loss(raw[:, side], target_ltrb[:, side], self.reg_max)
                    for side in range(4)
                ]
                total_dfl = total_dfl + torch.stack(side_losses, dim=1).mean()

                offset += n

        total_box = total_box / batch_size
        total_cls = total_cls / batch_size
        total_obj = total_obj / batch_size
        total_dfl = total_dfl / batch_size

        total = (
            self.lambda_box * total_box
            + self.lambda_cls * total_cls
            + self.lambda_obj * total_obj
            + self.lambda_dfl * total_dfl
        )
        return {
            "total": total,
            "box": total_box,
            "classification": total_cls,
            "objectness": total_obj,
            "dfl": total_dfl,
        }
