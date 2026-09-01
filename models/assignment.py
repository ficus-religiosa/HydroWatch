import torch


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


def xywh_to_gaussian(boxes):
    center = boxes[..., :2]
    width_height = boxes[..., 2:4]

    variance = width_height.pow(2) / 12.0

    return center, variance


def nwd_distance(pred_boxes, target_boxes, eps=1e-7):
    pred_boxes = xyxy_to_xywh(pred_boxes)
    target_boxes = xyxy_to_xywh(target_boxes)

    p_mean, p_var = xywh_to_gaussian(pred_boxes)
    t_mean, t_var = xywh_to_gaussian(target_boxes)

    mean_distance = (
        p_mean[:, None, :] - t_mean[None, :, :]
    ).pow(2).sum(dim=-1)

    variance_distance = (
        torch.sqrt(p_var[:, None, :] + eps)
        - torch.sqrt(t_var[None, :, :] + eps)
    ).pow(2).sum(dim=-1)

    return torch.sqrt(
        mean_distance + variance_distance + eps
    )


def nwd_similarity(
    pred_boxes,
    target_boxes,
    scale=12.8
):
    distance = nwd_distance(
        pred_boxes,
        target_boxes
    )

    return torch.exp(
        -distance / scale
    )


class NWDAssigner:
    """
    Top-k NWD assignment.

    One prediction is assigned to at most one GT.
    If two GTs compete for the same prediction,
    the higher NWD similarity wins.
    """

    def __init__(
        self,
        top_k=10,
        nwd_scale=12.8
    ):
        self.top_k = top_k
        self.nwd_scale = nwd_scale

    @torch.no_grad()
    def assign(
        self,
        pred_boxes,
        pred_scores,
        target_boxes,
        target_labels
    ):
        num_predictions = pred_boxes.shape[0]

        assigned_gt = torch.full(
            (num_predictions,),
            -1,
            dtype=torch.long,
            device=pred_boxes.device
        )

        if target_boxes.numel() == 0:
            return assigned_gt

        similarity = nwd_similarity(
            pred_boxes,
            target_boxes,
            scale=self.nwd_scale
        )

        if pred_scores is not None:
            if pred_scores.ndim != 2:
                raise ValueError(
                    "pred_scores must be [N,C]."
                )

            gt_scores = pred_scores[
                :,
                target_labels
            ]

            combined = similarity * gt_scores
        else:
            combined = similarity

        for gt_index in range(
            target_boxes.shape[0]
        ):
            scores = combined[:, gt_index]
            k = min(
                self.top_k,
                num_predictions
            )

            _, indices = torch.topk(
                scores,
                k=k,
                largest=True
            )

            for idx in indices.tolist():
                previous = assigned_gt[idx].item()

                if previous < 0:
                    assigned_gt[idx] = gt_index
                else:
                    if (
                        similarity[idx, gt_index]
                        > similarity[idx, previous]
                    ):
                        assigned_gt[idx] = gt_index

        return assigned_gt
