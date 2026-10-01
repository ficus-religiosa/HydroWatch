"""
prepare_data.py - turn the HydroWatch split files into a YOLO dataset. Run once.

    python prepare_data.py

Training set : L_B (L_A + all TrashCan) + U_all          -> every labelled training image we have
Validation   : ~8% of the SeaClear training images, taken in whole duplicate groups
TEST         : NOT copied here. It is only touched by eval_test.py at the very end.

Images are hard-linked (instant, no extra disk) when the output is on the same drive, else copied.
ROV parts (ignore zones) are left as background, which is what you want in a deployed detector.
"""
import argparse
import json
import os
import random
import shutil
import sys

import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

BASE = r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets"
SPLITS = BASE + r"\splits\r1280_s0_v2-debris-plus-biota"
OUT = r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\yolo_ds"


def load_split(split_dir, name):
    with open(Path(split_dir) / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def place(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


FALLBACKS = {"rect": 0, "poly": 0}


def seg_points(a, W, H):
    """Object outline in stored pixels: the polygon if usable (multi-part objects merged), else the box."""
    seg = a.get("segmentation")
    if a.get("has_mask", True) and isinstance(seg, list) and seg and all(
            isinstance(p, list) and len(p) >= 6 for p in seg):
        if len(seg) > 1:
            from ultralytics.data.converter import merge_multi_segment
            pts = np.concatenate([np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in merge_multi_segment(seg)])
        else:
            pts = np.asarray(seg[0], dtype=np.float64).reshape(-1, 2)
        FALLBACKS["poly"] += 1
    else:
        x, y, w, h = a["bbox"]
        pts = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
        FALLBACKS["rect"] += 1
    pts[:, 0] = pts[:, 0].clip(0, W)
    pts[:, 1] = pts[:, 1].clip(0, H)
    return pts


def yolo_lines(anns, W, H):
    """YOLO segment format 'cls x1 y1 x2 y2 ...' normalised to 0..1. Detection reads the box from it;
    the aux mask head and copy-paste use the outline."""
    lines = []
    for a in anns:
        pts = seg_points(a, W, H)
        if np.ptp(pts[:, 0]) < 1 or np.ptp(pts[:, 1]) < 1:  # degenerate after clipping
            continue
        pts /= np.array([W, H], dtype=np.float64)
        lines.append(f"{a['category_id'] - 1} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--train", default="L_B,U_all", help="comma-separated split names")
    ap.add_argument("--add-control", action="store_true",
                    help="also train on Marseille (CONTROL). Off by default: huge objects, "
                         "heavy duplication, 4 classes - may skew scale more than it helps")
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--val-split", default=None,
                    help="use this split file (e.g. VAL) as validation instead of carving one from training")
    ap.add_argument("--seaclear-repeat", type=int, default=2,
                    help="list each SeaClear image N times so TrashCan (66%% of images) "
                         "doesn't dominate training. 1 = no oversampling")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    base, out = Path(a.base).resolve(), Path(a.out).resolve()
    names = [s.strip() for s in a.train.split(",") if s.strip()]
    if a.add_control:
        names.append("CONTROL")

    images, anns, categories = {}, defaultdict(list), None
    for n in names:
        d = load_split(a.splits, n)
        categories = categories or d["categories"]
        for im in d["images"]:
            if im["id"] in images:
                print(f"  note: image {im['id']} appears in more than one split, kept once")
                continue
            images[im["id"]] = im
        for an in d["annotations"]:
            anns[an["image_id"]].append(an)
        print(f"loaded {n:8s} {len(d['images']):6d} images")

    if a.val_split:  # ready-made validation split (e.g. from resplit_seaclear.py)
        d = load_split(a.splits, a.val_split)
        for im in d["images"]:
            if im["id"] in images:
                sys.exit(f"image {im['id']} is in both training and {a.val_split} - split is leaky")
            images[im["id"]] = im
        for an in d["annotations"]:
            anns[an["image_id"]].append(an)
        val_ids = {im["id"] for im in d["images"]}
        tr_groups = {im.get("leak_group") for i, im in images.items() if i not in val_ids} - {None}
        shared = tr_groups & {im.get("leak_group") for im in d["images"]}
        if shared:
            sys.exit(f"{len(shared)} near-duplicate groups are in both training and {a.val_split} - split is leaky")
        print(f"loaded {a.val_split:8s} {len(d['images']):6d} images (validation)")
    # ---- validation: whole duplicate groups of SeaClear training images (never Marseille)
    def is_val_candidate(im):
        return im.get("source") == "seaclear" and "Marseille" not in str(im.get("group", ""))

    groups = defaultdict(list)
    for iid, im in images.items():
        if not a.val_split and is_val_candidate(im):
            groups[im.get("leak_group") or f"img:{iid}"].append(iid)
    keys = sorted(groups)
    random.Random(a.seed).shuffle(keys)
    target = a.val_frac * sum(len(v) for v in groups.values())
    val_ids = val_ids if a.val_split else set()
    for k in ([] if a.val_split else keys):
        if len(val_ids) >= target:
            break
        val_ids.update(groups[k])
    train_ids = [i for i in images if i not in val_ids]

    # ---- write
    for sub in ("images", "labels"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": Counter(), "val": Counter()}
    n_files = {"train": 0, "val": 0}
    missing = 0
    for split, ids in (("train", train_ids), ("val", sorted(val_ids))):
        for iid in ids:
            im = images[iid]
            src = base / im["file_name"]
            if not src.exists():
                missing += 1
                continue
            lines = yolo_lines(anns.get(iid, []), im["width"], im["height"])
            reps = a.seaclear_repeat if (split == "train" and im.get("source") == "seaclear") else 1
            for r in range(reps):
                stem = f"{iid}" if r == 0 else f"{iid}_r{r}"
                place(src, out / "images" / split / f"{stem}{src.suffix.lower()}")
                (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                n_files[split] += 1
            for ln in lines:
                counts[split][int(ln.split()[0])] += 1

    cls_names = [c["name"] for c in sorted(categories, key=lambda c: c["id"])]
    yaml_lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val",
                  f"hydrowatch_splits: {Path(a.splits).resolve().as_posix()}  # scoring uses this split set's TEST",
                  "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(cls_names)]
    (out / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"\ntrain: {len(train_ids)} unique images -> {n_files['train']} files "
          f"(SeaClear x{a.seaclear_repeat})")
    print(f"val  : {len(val_ids)} images" + ("" if a.val_split else f" from {len(keys)} groups"))
    n = FALLBACKS["poly"] + FALLBACKS["rect"]
    print(f"outlines: {FALLBACKS['poly']} polygons, {FALLBACKS['rect']} box-shaped fallbacks "
          f"({100 * FALLBACKS['rect'] / max(n, 1):.1f}% - objects with no usable mask)")
    if missing:
        print(f"WARNING: {missing} image files not found under {base} - check BASE")
    print(f"\n{'class':14s}{'train objs':>11s}{'val objs':>10s}")
    for i, n in enumerate(cls_names):
        print(f"{n:14s}{counts['train'][i]:11d}{counts['val'][i]:10d}")
    print(f"\nwrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()