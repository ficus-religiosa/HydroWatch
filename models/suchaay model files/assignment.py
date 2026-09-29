import torch


def xyxy_to_xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            (x2 - x1).clamp(min=0),
            (y2 - y1).clamp(min=0),
        ],
        dim=-1,
    )


def nwd_distance(pred_boxes, target_boxes, eps=1e-7):
    """Normalized Wasserstein-style distance for xyxy boxes."""
    pred = xyxy_to_xywh(pred_boxes)
    target = xyxy_to_xywh(target_boxes)

    p_center, p_wh = pred[..., :2], pred[..., 2:4]
    t_center, t_wh = target[..., :2], target[..., 2:4]

    p_var = p_wh.pow(2) / 12.0
    t_var = t_wh.pow(2) / 12.0

    mean_distance = (
        p_center[:, None, :] - t_center[None, :, :]
    ).pow(2).sum(dim=-1)

    variance_distance = (
        torch.sqrt(p_var[:, None, :] + eps)
        - torch.sqrt(t_var[None, :, :] + eps)
    ).pow(2).sum(dim=-1)

    return torch.sqrt(mean_distance + variance_distance + eps)


def nwd_similarity(pred_boxes, target_boxes, scale=12.8):
    return torch.exp(-nwd_distance(pred_boxes, target_boxes) / scale)


class NWDAssigner:
    """Center-aware top-k assignment using NWD and class confidence.

    A prediction is eligible for a GT when its feature-grid center lies
    inside that GT. Among eligible predictions, the top-k quality scores
    are selected. If a GT has no eligible center, the nearest predictions
    are used as a fallback so every GT can still receive supervision.
    """

    def __init__(self, top_k=10, nwd_scale=12.8):
        self.top_k = top_k
        self.nwd_scale = nwd_scale

    @torch.no_grad()
    def assign(
        self,
        pred_boxes,
        pred_scores,
        pred_centers,
        target_boxes,
        target_labels,
    ):
        n = pred_boxes.shape[0]
        assigned_gt = torch.full(
            (n,), -1, dtype=torch.long, device=pred_boxes.device
        )

        if target_boxes.numel() == 0 or n == 0:
            return assigned_gt

        similarity = nwd_similarity(
            pred_boxes, target_boxes, scale=self.nwd_scale
        )

        if pred_scores is not None:
            if pred_scores.ndim != 2:
                raise ValueError("pred_scores must have shape [N,C].")
            if pred_scores.shape[1] <= int(target_labels.max()):
                raise ValueError("target label exceeds prediction class count.")
            class_score = pred_scores[:, target_labels]
            quality = similarity * class_score
        else:
            quality = similarity

        px = pred_centers[:, 0, None]
        py = pred_centers[:, 1, None]
        x1 = target_boxes[None, :, 0]
        y1 = target_boxes[None, :, 1]
        x2 = target_boxes[None, :, 2]
        y2 = target_boxes[None, :, 3]

        inside = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)

        for gt_index in range(target_boxes.shape[0]):
            candidates = torch.nonzero(
                inside[:, gt_index], as_tuple=False
            ).flatten()

            if candidates.numel() == 0:
                # Fallback: nearest grid centers to the GT center.
                gt_center = (target_boxes[gt_index, :2] + target_boxes[gt_index, 2:4]) / 2
                distance = (pred_centers - gt_center).pow(2).sum(dim=-1)
                k = min(self.top_k, n)
                candidates = torch.topk(distance, k=k, largest=False).indices

            k = min(self.top_k, candidates.numel())
            local_scores = quality[candidates, gt_index]
            top = candidates[torch.topk(local_scores, k=k).indices]

            for idx in top.tolist():
                previous = assigned_gt[idx].item()
                if previous < 0:
                    assigned_gt[idx] = gt_index
                elif quality[idx, gt_index] > quality[idx, previous]:
                    assigned_gt[idx] = gt_index

        return assigned_gt
