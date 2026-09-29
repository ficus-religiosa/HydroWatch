"""
predict_image.py - run the trained model on one picture (any size - it is letterboxed to 1280).

    python predict_image.py                   # opens a file picker
    python predict_image.py path\\to\\img.jpg
    python predict_image.py --conf 0.35       # stricter: fewer, surer boxes

Saves an annotated copy to runs/predict/ and opens it.
"""
import argparse
import os
from collections import Counter
from pathlib import Path

import hydronet  # noqa: F401  (lets saved HydroNet checkpoints load)

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "runs" / "hydronet" / "weights" / "best.pt"


def pick_file():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an underwater image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
    root.destroy()
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO

    path = a.image or pick_file()
    if not path:
        raise SystemExit("No image selected.")
    cuda = torch.cuda.is_available()
    model = YOLO(a.weights)
    r = model.predict(path, imgsz=a.imgsz, conf=a.conf, iou=0.7, quantize=16 if cuda else None,
                      device=0 if cuda else "cpu", verbose=False)[0]

    names = r.names
    print(f"\n{Path(path).name}: {len(r.boxes)} detections (conf >= {a.conf})")
    for b in sorted(r.boxes, key=lambda b: -float(b.conf)):
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        print(f"  {names[int(b.cls)]:14s} {float(b.conf):.2f}   box ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})")
    counts = Counter(names[int(c)] for c in r.boxes.cls.tolist())
    if counts:
        print("  totals: " + ", ".join(f"{k} x{v}" for k, v in counts.most_common()))

    out = HERE / "runs" / "predict"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"{Path(path).stem}_pred.jpg"
    cv2.imwrite(str(dst), r.plot(line_width=2))
    print(f"\nsaved {dst}")
    if os.name == "nt":
        os.startfile(dst)


if __name__ == "__main__":
    main()
