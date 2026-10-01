"""
eval_test.py - score a trained model on TEST using the project's protocol.

    python eval_test.py                                  # scores runs/oracle_p2_1280/weights/best.pt
    python eval_test.py --weights path\\to\\best.pt

Steps:
  1. predict every TEST image at 1280, confidence cut 0.001 (mAP needs the full ranking)
  2. drop predictions lying mostly inside ROV ignore zones - they neither help nor hurt
  3. COCO evaluation with the project's size bins (tertiles of TEST, from split_meta.json)
  4. report all classes, debris only, all-without-fish, and per-class AP (all sizes and small)
Saved next to the weights as test_eval.json.
"""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path

import hydronet  # noqa: F401  (lets saved HydroNet checkpoints load)

import numpy as np

BASE = r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets"
SPLITS = BASE + r"\splits\r1280_s0_v2-debris-plus-biota"
HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "runs" / "hydronet" / "weights" / "best.pt"


def default_splits():
    """The split set yolo_ds was built from (recorded by prepare_data.py), so scoring can never
    use a TEST whose images are in the current training data."""
    try:
        import yaml
        from prepare_data import OUT
        d = yaml.safe_load((Path(OUT) / "data.yaml").read_text(encoding="utf-8"))
        if d.get("hydrowatch_splits"):
            return d["hydrowatch_splits"]
    except Exception:
        pass
    return SPLITS


def xywh_to_xyxy(b):
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]], axis=1)


def drop_in_ignore(pred, ign, min_overlap=0.5):
    """keep-mask: False where >= min_overlap of a prediction's own area is inside an ignore zone"""
    p, g = np.asarray(pred, np.float32).reshape(-1, 4), np.asarray(ign, np.float32).reshape(-1, 4)
    if len(p) == 0 or len(g) == 0:
        return np.ones(len(p), dtype=bool)
    iw = np.clip(np.minimum(p[:, None, 2], g[None, :, 2]) - np.maximum(p[:, None, 0], g[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(p[:, None, 3], g[None, :, 3]) - np.maximum(p[:, None, 1], g[None, :, 1]), 0, None)
    area = np.clip((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]), 1e-9, None)
    return (iw * ih / area[:, None]).max(axis=1) < min_overlap


def coco_eval(gt, dt, img_ids, cat_ids, cuts, max_det):
    from pycocotools.cocoeval import COCOeval
    E = COCOeval(gt, dt, "bbox")
    E.params.imgIds, E.params.catIds = list(img_ids), list(cat_ids)
    E.params.maxDets = [1, 10, 100]  # keep 100: pycocotools hard-codes it for the headline mAP
    E.params.areaRng = [[0, 1e10], [0, cuts[0]], [cuts[0], cuts[1]], [cuts[1], 1e10]]
    E.params.areaRngLbl = ["all", "small", "medium", "large"]
    with contextlib.redirect_stdout(io.StringIO()):
        E.evaluate()
        E.accumulate()
        E.summarize()
    s = E.stats
    prec = E.eval["precision"]  # [IoU, recall, class, area, maxDet]

    def ap(k, a):
        p = prec[:, :, k, a, -1]
        p = p[p > -1]
        return float(p.mean()) if p.size else float("nan")

    return {"mAP50-95": s[0], "mAP50": s[1], "mAP75": s[2],
            "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
            "AR_small": s[9], "AR_medium": s[10], "AR_large": s[11],
            "per_class": {int(c): {"AP": ap(k, 0), "AP_small": ap(k, 1)} for k, c in enumerate(cat_ids)}}


def evaluate(weights=DEFAULT_WEIGHTS, base=BASE, splits=None, imgsz=1280, limit=None, max_det=300, test_json=None):
    import torch
    from pycocotools.coco import COCO
    from ultralytics import YOLO

    weights, base, splits = Path(weights), Path(base), Path(splits or default_splits())
    print(f"scoring against: {splits.name}")
    test_json = Path(test_json) if test_json else splits / "TEST.json"
    meta = json.loads((splits / "split_meta.json").read_text(encoding="utf-8"))
    raw = json.loads(test_json.read_text(encoding="utf-8"))
    L = max(max(im["width"], im["height"]) for im in raw["images"])  # common frame for size bins
    fac = {im["id"]: L / max(im["width"], im["height"]) for im in raw["images"]}
    norm = {"images": [{**im, "width": round(im["width"] * fac[im["id"]]),
                        "height": round(im["height"] * fac[im["id"]])} for im in raw["images"]],
            "annotations": [], "categories": raw["categories"]}
    for an in raw["annotations"]:
        f = fac[an["image_id"]]
        x, y, w, h = (v * f for v in an["bbox"])
        norm["annotations"].append({"id": an["id"], "image_id": an["image_id"], "category_id": an["category_id"],
                                    "bbox": [x, y, w, h], "area": w * h, "iscrowd": an.get("iscrowd", 0)})
    gt = COCO()
    gt.dataset = norm
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()

    images = sorted(raw["images"], key=lambda im: im["id"])  # original records: frame + ignore zones
    if limit:
        images = images[:limit]
    if len({max(im["width"], im["height"]) for im in images}) > 1:
        print(f"note: TEST mixes image sizes; all boxes are scaled to a {L}-px long side for size bins")
    g1, g2 = meta["size_bins"]["geo_cuts"]          # geo = sqrt(box area) / long side
    cuts = ((g1 * L) ** 2, (g2 * L) ** 2)           # -> box-area cut-offs in stored pixels

    cuda = torch.cuda.is_available()
    model = YOLO(str(weights))
    dets, n_dropped, rescaled = [], 0, 0
    print(f"Predicting {len(images)} TEST images ...")
    for i in range(0, len(images), 16):
        chunk = images[i:i + 16]
        res = model.predict([str(base / im["file_name"]) for im in chunk], imgsz=imgsz, conf=0.001,
                            iou=0.7, max_det=max_det, quantize=16 if cuda else None, device=0 if cuda else "cpu",
                            verbose=False)
        for im, r in zip(chunk, res):
            xyxy = r.boxes.xyxy.cpu().numpy()
            h0, w0 = r.orig_shape  # size of the file actually read
            sx, sy = im["width"] / w0, im["height"] / h0
            if abs(sx - 1) > 1e-3 or abs(sy - 1) > 1e-3:  # file on disk != annotation frame
                xyxy = xyxy * np.array([sx, sy, sx, sy], dtype=np.float32)
                rescaled += 1
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            keep = drop_in_ignore(xyxy, xywh_to_xyxy(im.get("ignore", [])))
            n_dropped += int((~keep).sum())
            f = fac[im["id"]]  # record frame -> common frame (same scaling as the ground truth)
            for (x1, y1, x2, y2), c, k in zip(xyxy[keep] * f, conf[keep], cls[keep]):
                dets.append({"image_id": im["id"], "category_id": int(k) + 1,
                             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                             "score": float(c)})
    if rescaled:
        print(f"WARNING: {rescaled}/{len(images)} images on disk differ in size from their annotation record.\n"
              "         Predictions were rescaled into the annotation frame. Run diagnose_test.py to see why.")
    if not dets:
        print("No detections at all - the model has not learned anything yet.")
        return None

    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
    cats = {c["id"]: c for c in gt.dataset["categories"]}
    all_ids = sorted(cats)
    debris_ids = [c for c in all_ids if cats[c].get("tier") == "debris"]
    no_fish = [c for c in all_ids if cats[c]["name"] != "fish"]
    img_ids = [im["id"] for im in images]

    report = {"weights": str(weights), "n_images": len(images), "n_detections": len(dets),
              "dropped_in_ignore": n_dropped, "rescaled_images": rescaled, "area_cuts_px": cuts,
              "all": coco_eval(gt, dt, img_ids, all_ids, cuts, max_det),
              "debris_only": coco_eval(gt, dt, img_ids, debris_ids, cuts, max_det),
              "without_fish": coco_eval(gt, dt, img_ids, no_fish, cuts, max_det)}

    def fmt(x):
        return "  -  " if x is None or (isinstance(x, float) and (math.isnan(x) or x < 0)) else f"{x:.3f}"

    print(f"\nTEST  ({len(images)} images, {len(dets)} detections, {n_dropped} dropped in ROV zones)")
    print(f"size bins by box area: small < {cuts[0]:.0f} px, medium < {cuts[1]:.0f} px\n")
    print(f"{'subset':14s}{'mAP50-95':>9s}{'mAP50':>8s}{'small':>8s}{'medium':>8s}{'large':>8s}")
    for key in ("all", "debris_only", "without_fish"):
        r = report[key]
        print(f"{key:14s}{fmt(r['mAP50-95']):>9s}{fmt(r['mAP50']):>8s}{fmt(r['AP_small']):>8s}"
              f"{fmt(r['AP_medium']):>8s}{fmt(r['AP_large']):>8s}")
    print(f"\n{'class':14s}{'AP':>8s}{'AP_small':>10s}")
    for c in all_ids:
        r = report["all"]["per_class"][c]
        print(f"{cats[c]['name']:14s}{fmt(r['AP']):>8s}{fmt(r['AP_small']):>10s}")

    stem = "test" if test_json.name == "TEST.json" else test_json.stem.lower()  # never overwrite TEST results
    out = weights.parent / (f"{stem}_eval_sample.json" if limit else f"{stem}_eval.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nsaved {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--splits", default=None, help="default: the split set yolo_ds was built from")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N images (quick check)")
    a = ap.parse_args()
    evaluate(a.weights, a.base, a.splits, a.imgsz, a.limit)