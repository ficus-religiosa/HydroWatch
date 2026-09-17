import time

import torch


def count_parameters(model):
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


def parameter_groups(model):
    groups = {}

    for name, module in model.named_children():
        groups[name] = sum(
            p.numel()
            for p in module.parameters()
        )

    return groups


@torch.no_grad()
def output_sanity(model, input_tensor):
    """
    Verify that the main model outputs are finite and that all
    decoded boxes are ordered and inside the input image bounds.
    """
    model.eval()

    output = model(input_tensor)

    h, w = input_tensor.shape[-2:]

    for level, pred in output["detections"].items():
        for key in [
            "box",
            "objectness",
            "class",
            "boxes",
            "distances"
        ]:
            if not torch.isfinite(
                pred[key]
            ).all():
                raise RuntimeError(
                    f"{level}/{key} contains NaN or Inf."
                )

        boxes = pred["boxes"]

        if (boxes[..., 0] > boxes[..., 2]).any():
            raise RuntimeError(
                f"{level} contains x1 > x2 boxes."
            )

        if (boxes[..., 1] > boxes[..., 3]).any():
            raise RuntimeError(
                f"{level} contains y1 > y2 boxes."
            )

        if (
            boxes[..., 0].min() < 0
            or boxes[..., 1].min() < 0
            or boxes[..., 2].max() > w
            or boxes[..., 3].max() > h
        ):
            raise RuntimeError(
                f"{level} contains boxes outside image bounds."
            )

    print("Output sanity: PASS")


def benchmark(
    model,
    input_tensor,
    warmup=10,
    iterations=50
):
    """
    Measure forward latency only.

    CUDA synchronization is used so GPU timings are meaningful.
    """
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

        if input_tensor.is_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(iterations):
            model(input_tensor)

        if input_tensor.is_cuda:
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    ms = elapsed * 1000 / iterations
    fps = 1000 / ms

    return {
        "milliseconds_per_image": ms,
        "images_per_second": fps
    }
