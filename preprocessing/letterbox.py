from PIL import Image
import numpy as np


def letterbox_image(
    image,
    target_size=(480, 360),
    fill=(114, 114, 114)
):
    """
    Resize while preserving aspect ratio, then center-pad.

    target_size is (width, height).

    Returns:
        output_image, scale, pad_x, pad_y
    """
    target_w, target_h = target_size
    src_w, src_h = image.size

    scale = min(
        target_w / src_w,
        target_h / src_h
    )

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = image.resize(
        (new_w, new_h),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (target_w, target_h),
        fill
    )

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2

    canvas.paste(
        resized,
        (pad_x, pad_y)
    )

    return canvas, scale, pad_x, pad_y


def transform_boxes(
    boxes,
    scale,
    pad_x,
    pad_y,
    target_size=(480, 360)
):
    """
    Transform xyxy boxes from source image coordinates to
    letterboxed image coordinates.
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    boxes = np.asarray(
        boxes,
        dtype=np.float32
    ).copy()

    boxes[:, [0, 2]] *= scale
    boxes[:, [1, 3]] *= scale

    boxes[:, [0, 2]] += pad_x
    boxes[:, [1, 3]] += pad_y

    target_w, target_h = target_size

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]], 0, target_w
    )
    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]], 0, target_h
    )

    return boxes


def transform_mask(
    mask,
    scale,
    pad_x,
    pad_y,
    target_size=(480, 360)
):
    """
    Transform a binary mask using nearest-neighbor interpolation.
    """
    target_w, target_h = target_size

    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(
            np.asarray(mask).astype(np.uint8) * 255
        )

    src_w, src_h = mask.size

    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))

    resized = mask.resize(
        (new_w, new_h),
        Image.Resampling.NEAREST
    )

    canvas = Image.new(
        "L",
        (target_w, target_h),
        0
    )

    canvas.paste(
        resized,
        (pad_x, pad_y)
    )

    return canvas
