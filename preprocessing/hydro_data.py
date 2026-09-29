"""
hydro_data.py - read split files in training and evaluation code. Copy this one file into your
training project; it needs only numpy and Pillow (plus torch for HydroDataset).

A split file is self-contained: it holds image paths AND labels, already mapped to the unified
classes and scaled to the stored image. This reader never opens master.json during training.

    from hydro_data import HydroDataset, collate, unletterbox, drop_in_ignore

    base = r"D:\\zaynu\\Documents\\coding\\PS\\3-1\\HydroWatch\\datasets"
    splits = base + r"\\splits\\r1280_s0_v2-debris-plus-biota"

    baseline = HydroDataset(["L_A"], splits, base)              # Baseline arm
    oracle   = HydroDataset(["L_A", "U_1000"], splits, base)    # Oracle arm at U = 1000
    test     = HydroDataset(["TEST"], splits, base)

    loader = torch.utils.data.DataLoader(oracle, batch_size=8, shuffle=True,
                                         num_workers=4, collate_fn=collate)
    # Windows: a DataLoader with num_workers > 0 must be created under  if __name__ == "__main__":

One sample (dict):
    image      3 x R x R uint8, RGB. Letterboxed: resized so the long side is R, then padded to square
    boxes      N x 4 float32, x1 y1 x2 y2 in letterboxed pixels
    labels     N int64, class index starting at 0 (= category_id - 1)
    masks      N x (R/s) x (R/s) uint8 (s = mask_stride), one binary mask per object, for the aux mask head
    mask_valid N bool, False where the object has no usable polygon (skip it in the mask loss)
    ignore     M x 4 float32, don't-care zones (x1 y1 x2 y2). Your loss must give no reward and no
               penalty to predictions whose centre falls inside one; otherwise they act as background
    size_bin   N int64: 0 small, 1 medium, 2 large
    image_id   int, the same id as in master.json
    meta       file_name, stored size, load scale, padding. unletterbox() uses it to map predictions
               back to the split file's coordinates for scoring

FC-V2's physics channels (dark channel, red attenuation, gradient energy) should be computed from
`image` AFTER any augmentation, inside a transform or as the model's first step. Do not cache them:
colour augmentation changes the pixels they come from.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RS = getattr(Image, "Resampling", Image)   # Pillow < 9.1 has no Image.Resampling

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # the numpy parts still work without torch
    torch = None
    _TorchDataset = object

BIN_INDEX = {"small": 0, "medium": 1, "large": 2}
PAD_VALUE = 114


# ------------------------------------------------------------------ image helpers
def read_image(path):
    """RGB uint8 H x W x 3, raw pixel orientation (no EXIF rotation, matching the annotations)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def letterbox(img, R, pad_value=PAD_VALUE):
    """Resize so the long side is R (shrink: area average, enlarge: bicubic), pad to R x R.
    Returns the padded image, (sx, sy) scale applied, (dx, dy) padding offset."""
    h, w = img.shape[:2]
    r = R / max(h, w)
    nw, nh = max(1, round(w * r)), max(1, round(h * r))
    if (nw, nh) != (w, h):
        method = RS.BOX if r < 1 else RS.BICUBIC
        img = np.asarray(Image.fromarray(img).resize((nw, nh), method))
    dx, dy = (R - nw) // 2, (R - nh) // 2
    out = np.full((R, R, 3), pad_value, dtype=np.uint8)
    out[dy:dy + nh, dx:dx + nw] = img
    return out, (nw / w, nh / h), (dx, dy)


def _xywh_to_xyxy(boxes, s, d):
    if not boxes:
        return np.zeros((0, 4), np.float32)
    b = np.asarray(boxes, dtype=np.float32)
    x1 = b[:, 0] * s[0] + d[0]
    y1 = b[:, 1] * s[1] + d[1]
    x2 = (b[:, 0] + b[:, 2]) * s[0] + d[0]
    y2 = (b[:, 1] + b[:, 3]) * s[1] + d[1]
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


# ------------------------------------------------------------------ split reading
class SplitSet:
    """One or more split files loaded together (for example L_A + U_1000)."""

    def __init__(self, names, split_dir, base):
        names = [names] if isinstance(names, str) else list(names)
        self.base, self.split_dir, self.names = Path(base), Path(split_dir), names
        self.images, self.anns, self.info = [], {}, []
        self.categories, ref, seen = None, None, set()
        for n in names:
            path = self.split_dir / (n if n.endswith(".json") else f"{n}.json")
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            key = (d["info"]["R"], d["info"]["taxonomy"])
            if ref is None:
                ref, self.categories = key, d["categories"]
            elif key != ref or d["categories"] != self.categories:
                raise ValueError(f"cannot combine splits built with different R/taxonomy: {names}")
            for im in d["images"]:
                if im["id"] in seen:
                    raise ValueError(f"image {im['id']} appears in more than one of {names}")
                seen.add(im["id"])
                self.images.append(im)
            for a in d["annotations"]:
                self.anns.setdefault(a["image_id"], []).append(a)
            self.info.append(d["info"])
        self.R = ref[0]
        self.class_names = [c["name"] for c in self.categories]
        self.class_tiers = [c.get("tier") for c in self.categories]

    def __len__(self):
        return len(self.images)

    def path(self, im):
        return self.base / im["file_name"]

    def get_sample(self, i, with_masks=True, mask_stride=4, pad_value=PAD_VALUE):
        im = self.images[i]
        anns = self.anns.get(im["id"], [])
        img, s, d = letterbox(read_image(self.path(im)), self.R, pad_value)
        sample = {
            "image": img,
            "boxes": _xywh_to_xyxy([a["bbox"] for a in anns], s, d),
            "labels": np.asarray([a["category_id"] - 1 for a in anns], dtype=np.int64),
            "size_bin": np.asarray([BIN_INDEX[a["size_bin"]] for a in anns], dtype=np.int64),
            "ignore": _xywh_to_xyxy(im.get("ignore", []), s, d),
            "mask_valid": np.asarray([bool(a.get("has_mask")) for a in anns], dtype=bool),
            "image_id": im["id"],
            "meta": {"file_name": im["file_name"], "stored_size": (im["width"], im["height"]),
                     "load_scale": s, "pad": d, "R": self.R, "group": im.get("group")},
        }
        if with_masks:
            m = self.R // mask_stride
            masks = np.zeros((len(anns), m, m), dtype=np.uint8)
            for k, a in enumerate(anns):
                if not a.get("segmentation"):
                    continue
                canvas = Image.new("L", (m, m), 0)
                draw = ImageDraw.Draw(canvas)
                for poly in a["segmentation"]:
                    p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    p[:, 0] = (p[:, 0] * s[0] + d[0]) / mask_stride
                    p[:, 1] = (p[:, 1] * s[1] + d[1]) / mask_stride
                    if len(p) >= 3:
                        draw.polygon([tuple(q) for q in p], fill=1)
                masks[k] = np.asarray(canvas)
            sample["masks"] = masks
        else:
            sample["masks"] = None
        return sample


class HydroDataset(_TorchDataset):
    """PyTorch dataset over one or more split files.
    transform: optional function(sample_dict) -> sample_dict, applied to numpy arrays before they
    become tensors. If it moves pixels (flip, crop, scale) it must move boxes, masks and ignore too."""

    def __init__(self, names, split_dir, base, with_masks=True, mask_stride=4, transform=None):
        if torch is None:
            raise ImportError("HydroDataset needs PyTorch; use SplitSet.get_sample() without it")
        self.set = SplitSet(names, split_dir, base)
        self.with_masks, self.mask_stride, self.transform = with_masks, mask_stride, transform
        self.R, self.class_names = self.set.R, self.set.class_names

    def __len__(self):
        return len(self.set)

    def __getitem__(self, i):
        s = self.set.get_sample(i, self.with_masks, self.mask_stride)
        if self.transform is not None:
            s = self.transform(s)
        s["image"] = torch.from_numpy(np.ascontiguousarray(s["image"].transpose(2, 0, 1)))
        for k in ("boxes", "labels", "size_bin", "ignore", "mask_valid"):
            s[k] = torch.from_numpy(np.ascontiguousarray(s[k]))
        if s["masks"] is not None:
            s["masks"] = torch.from_numpy(np.ascontiguousarray(s["masks"]))
        return s


def collate(batch):
    """Stack images; keep per-image lists for everything whose length varies."""
    out = {"image": torch.stack([b["image"] for b in batch])}
    for k in batch[0]:
        if k != "image":
            out[k] = [b[k] for b in batch]
    return out


# ------------------------------------------------------------------ evaluation helpers
def unletterbox(boxes_xyxy, meta):
    """Letterboxed x1 y1 x2 y2 -> coordinates of the stored image, i.e. the split file's frame."""
    b = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4).copy()
    (sx, sy), (dx, dy) = meta["load_scale"], meta["pad"]
    W, H = meta["stored_size"]
    b[:, [0, 2]] = np.clip((b[:, [0, 2]] - dx) / sx, 0, W)
    b[:, [1, 3]] = np.clip((b[:, [1, 3]] - dy) / sy, 0, H)
    return b


def xyxy_to_xywh(b):
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def drop_in_ignore(pred_xyxy, ignore_xyxy, min_overlap=0.5):
    """Boolean keep-mask: False for predictions that lie mostly (>= min_overlap of their own area)
    inside a don't-care zone. Apply before scoring so ROV parts neither help nor hurt."""
    p = np.asarray(pred_xyxy, dtype=np.float32).reshape(-1, 4)
    g = np.asarray(ignore_xyxy, dtype=np.float32).reshape(-1, 4)
    if len(p) == 0 or len(g) == 0:
        return np.ones(len(p), dtype=bool)
    ix1 = np.maximum(p[:, None, 0], g[None, :, 0])
    iy1 = np.maximum(p[:, None, 1], g[None, :, 1])
    ix2 = np.minimum(p[:, None, 2], g[None, :, 2])
    iy2 = np.minimum(p[:, None, 3], g[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.clip(p[:, 2] - p[:, 0], 1e-9, None) * np.clip(p[:, 3] - p[:, 1], 1e-9, None)
    return (inter / area[:, None]).max(axis=1) < min_overlap


# ------------------------------------------------------------------ exploring one folder
@lru_cache(maxsize=2)
def _master(base):
    with open(Path(base) / "manifest" / "master.json", encoding="utf-8") as fh:
        m = json.load(fh)
    anns = {}
    for a in m["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    return {im["id"]: im for im in m["images"]}, anns, {c["id"]: c for c in m["categories"]}


def iter_group(group, base):
    """Loop over one folder from master.json (original classes, original images):
        for path, image, annotations in iter_group("seaclear/Lokrum/Bluerobotics HD", base): ...
    Each annotation gets an extra 'class_name' like 'seaclear:can_metal'. For exploring, not training."""
    base = str(Path(base))
    with open(Path(base) / "manifest" / "groups.json", encoding="utf-8") as fh:
        groups = json.load(fh)
    if group not in groups:
        raise KeyError(f"unknown group {group!r}; known: {sorted(groups)}")
    images, anns, cats = _master(base)
    for iid in groups[group]["image_ids"]:
        im = images[iid]
        out = [dict(a, class_name=f"{cats[a['category_id']]['source']}:{cats[a['category_id']]['name']}")
               for a in anns.get(iid, [])]
        yield Path(base) / im["file_name"], im, out
