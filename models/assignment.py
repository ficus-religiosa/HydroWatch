import torch


def xywh_to_gaussian(boxes):
    center = boxes[..., :2]
    width_height = boxes[..., 2:4]
    variance = width_height.pow(2) / 12.0
    return center, variance


def nwd_distance(pred_boxes, target_boxes, eps=1e-7):
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


def nwd_similarity(pred_boxes, target_boxes, scale=12.8):
    distance = nwd_distance(pred_boxes, target_boxes)
    return torch.exp(-distance / scale)


class NWDAssigner:
    def __init__(self, top_k=10, nwd_scale=12.8):
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
            self.nwd_scale
        )

        if pred_scores.dim() == 2:
            class_conf = pred_scores[:, target_labels]
        else:
            class_conf = pred_scores

        if class_conf.dim() == 2:
            cls_score = class_conf.clamp(min=1e-6).mean(dim=-1)
            combined = similarity * cls_score[:, None]
        else:
            combined = similarity

        for gt_index in range(target_boxes.shape[0]):
            scores = combined[:, gt_index]
            k = min(self.top_k, num_predictions)

            _, indices = torch.topk(scores, k=k)

            for idx in indices:
                current = assigned_gt[idx]

                if current < 0:
                    assigned_gt[idx] = gt_index
                else:
                    current_score = similarity[idx, current]
                    new_score = similarity[idx, gt_index]

                    if new_score > current_score:
                        assigned_gt[idx] = gt_index

        return assigned_gt
