"""
diagnose_test.py - is the TEST score real, or an evaluation problem?  (~5 min)

    python diagnose_test.py
    python diagnose_test.py --weights runs\\hydronet_more\\weights\\best.pt

1. FRAME   - do TEST.json's image sizes match the files on disk, and do its boxes fit inside those sizes?
2. SCORER  - run eval_test's scoring on your VALIDATION images. It should land near Ultralytics' own
             validation number (~0.65). If it doesn't, the scorer is wrong, not the model.
3. PICTURES - ground truth (green) and predictions (red) on a few TEST images -> diag_test\\
"""
import argparse
import json
import random
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import hydronet  # noqa: F401  (lets saved HydroNet checkpoints load)
from eval_test import BASE, DEFAULT_WEIGHTS, SPLITS, evaluate
from prepare_data import OUT

HERE = Path(__file__).resolve().parent


def check_frame(test, base):
    print("1. FRAME")
    ims = {im["id"]: im for im in test["images"]}
    first = test["images"][0]
    print(f"   first record: file_name={first['file_name']!r}  width={first['width']}  height={first['height']}"
          + "".join(f"  {k}={first[k]}" for k in ("scale", "orig_size", "orig_file_name") if k in first))
    mismatch = []
    for im in test["images"]:
        with Image.open(base / im["file_name"]) as f:
            if f.size != (im["width"], im["height"]):
                mismatch.append((im["file_name"], (im["width"], im["height"]), f.size))
    if mismatch:
        print(f"   PROBLEM: {len(mismatch)}/{len(ims)} files differ in size from their record, e.g.")
        for fn, rec, real in mismatch[:3]:
            print(f"     {fn}: record says {rec}, file is {real}")
    else:
        print(f"   OK: all {len(ims)} files match their recorded size")
    ex = np.array([[(a["bbox"][0] + a["bbox"][2]) / ims[a["image_id"]]["width"],
                    (a["bbox"][1] + a["bbox"][3]) / ims[a["image_id"]]["height"]] for a in test["annotations"]])
    mx = ex.max(0)
    print(f"   boxes reach {mx[0]:.2f} of the width and {mx[1]:.2f} of the height (expect ~1.00)")
    if mx.max() < 0.8:
        print("   PROBLEM: boxes never reach the far edges - they are probably in a smaller image's coordinates")
    if mx.max() > 1.05:
        print("   PROBLEM: boxes extend past the image - they are probably in a larger image's coordinates")
    return bool(mismatch) or mx.max() < 0.8 or mx.max() > 1.05


def check_scorer(weights, test, splits, base, imgsz):
    print("\n2. SCORER on validation images (compare with Ultralytics' validation mAP50-95)")
    val_dir = Path(OUT) / "images" / "val"
    ids = {int(p.stem) for p in val_dir.iterdir() if "_r" not in p.stem}
    images, anns = {}, []
    for name in ("L_B", "U_all"):
        d = json.loads((splits / f"{name}.json").read_text(encoding="utf-8"))
        for im in d["images"]:
            if im["id"] in ids:
                images[im["id"]] = {**im, "ignore": im.get("ignore", [])}
        anns += [a for a in d["annotations"] if a["image_id"] in ids]
    val = {"images": list(images.values()), "annotations": anns, "categories": test["categories"]}
    tmp = Path(tempfile.mkdtemp()) / "VAL_AS_TEST.json"
    tmp.write_text(json.dumps(val), encoding="utf-8")
    rep = evaluate(weights, base=base, splits=splits, imgsz=imgsz, test_json=tmp)
    return None if rep is None else rep["all"]["mAP50-95"]


def draw(weights, test, base, imgsz, n=6):
    import cv2
    import torch
    from ultralytics import YOLO
    print(f"\n3. PICTURES -> {HERE / 'diag_test'}")
    out = HERE / "diag_test"
    out.mkdir(exist_ok=True)
    by_img = {}
    for a in test["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    names = {c["id"] - 1: c["name"] for c in test["categories"]}
    pick = random.Random(0).sample([im for im in test["images"] if im["id"] in by_img], n)
    model = YOLO(str(weights))
    cuda = torch.cuda.is_available()
    for im in pick:
        path = base / im["file_name"]
        img = cv2.imread(str(path))
        r = model.predict(str(path), imgsz=imgsz, conf=0.25, device=0 if cuda else "cpu", verbose=False)[0]
        fy, fx = img.shape[0] / im["height"], img.shape[1] / im["width"]  # annotation frame -> file
        for a in by_img[im["id"]]:
            x, y, w, h = a["bbox"]
            cv2.rectangle(img, (int(x * fx), int(y * fy)), (int((x + w) * fx), int((y + h) * fy)), (0, 255, 0), 2)
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{names[int(b.cls)]} {float(b.conf):.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(str(out / f"{im['id']}.jpg"), img)
        print(f"   {im['id']}: {len(by_img[im['id']])} true objects (green), {len(r.boxes)} predictions (red)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()
    base, splits = Path(a.base), Path(a.splits)
    test = json.loads((splits / "TEST.json").read_text(encoding="utf-8"))
    frame_bad = check_frame(test, base)
    val_map = check_scorer(a.weights, test, splits, base, a.imgsz)
    draw(a.weights, test, base, a.imgsz)
    print("\nVERDICT")
    if frame_bad:
        print("  TEST.json and its image files disagree about coordinates. The TEST score was measuring a")
        print("  coordinate mismatch, not the model. Re-run eval_test.py (it now corrects file-size mismatches).")
    elif val_map is not None and val_map < 0.3:
        print(f"  The scorer gives {val_map:.3f} on validation images Ultralytics scores ~0.65: the SCORER is wrong.")
    else:
        print(f"  Frame is consistent and the scorer reproduces validation ({val_map:.3f}). The TEST score is REAL:")
        print("  the model genuinely fails on the TEST sites. Look at diag_test\\ to see how it fails.")


if __name__ == "__main__":
    main()
