"""
resize_cache.py - store copies of the images at the training size, so training never shrinks
full-size images on the fly.

    python resize_cache.py
    python resize_cache.py --config path/to/hydro_config.yaml

Reads   manifest/master.json, config (R, resize_sources, jpeg_quality, workers)
Writes  resized_<R>/<same relative path as the original>.jpg
        resized_<R>/resize_index.json   image id -> stored file, width, height
Only sources listed in config.resize_sources are stored (default: seaclear). TrashCan is left as is
and enlarged when loaded, which is faster than reading a stored enlarged copy.
Resumable: copies that already exist at the right size are kept, so an interrupted run can simply
be started again.
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

RS = getattr(Image, "Resampling", Image)   # Pillow < 9.1 has no Image.Resampling

from preprocessing.hydro_common import Report, fail, load_config, load_json, now, save_json, sha256_file


def target_size(w, h, R):
    r = R / max(w, h)
    return max(1, round(w * r)), max(1, round(h * r)), r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to hydro_config.yaml (default: next to this script)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    base = cfg["_base"]
    R = int(cfg["R"])
    if R % 32:
        fail(f"R={R} must be a multiple of 32 (the detector's deepest level has stride 32)")
    quality = int(cfg.get("jpeg_quality", 95))
    sources = set(cfg.get("resize_sources", ["seaclear"]))
    master_path = base / "manifest" / "master.json"
    if not master_path.exists():
        fail("manifest/master.json not found - run build_manifest.py first")
    master = load_json(master_path)
    out_root = base / f"resized_{R}"
    rep = Report()
    rep(f"resize_cache  {now()}   R={R}  sources={sorted(sources)}  -> {out_root}")

    todo = [im for im in master["images"] if im["source"] in sources]
    if not todo:
        fail(f"no images from sources {sorted(sources)} in master.json")

    def work(im):
        src = base / im["file_name"]
        rel = Path(im["file_name"]).with_suffix(".jpg").as_posix()
        dst = out_root / rel
        nw, nh, r = target_size(im["width"], im["height"], R)
        if dst.exists():
            try:
                with Image.open(dst) as done:
                    if done.size == (nw, nh):
                        return im["id"], f"resized_{R}/{rel}", nw, nh, "kept"
            except OSError:
                pass                                   # unreadable partial file: rewrite it
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as pic:
            pic = pic.convert("RGB")                   # raw pixel orientation, no EXIF rotation
            if r < 1:
                pic = pic.resize((nw, nh), RS.BOX)      # area average: best for shrinking
            elif r > 1:
                pic = pic.resize((nw, nh), RS.BICUBIC)
            tmp = dst.with_name(dst.name + ".tmp")
            # 4:4:4 keeps full colour detail; underwater colour (red loss) feeds FC-V2's physics channels
            pic.save(tmp, "JPEG", quality=quality, subsampling=0)
        os.replace(tmp, dst)
        return im["id"], f"resized_{R}/{rel}", nw, nh, "written"

    workers = int(cfg.get("workers", os.cpu_count() or 4))
    index, counts = {}, {"written": 0, "kept": 0}
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, (iid, rel, w, h, status) in enumerate(ex.map(work, todo), 1):
            index[str(iid)] = {"file_name": rel, "width": w, "height": h}
            counts[status] += 1
            if k % 500 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)}  ({time.time() - t1:.0f}s)")

    save_json({"R": R, "created": now(), "sources": sorted(sources),
               "method": {"shrink": "box (area average)", "enlarge": "bicubic"},
               "jpeg_quality": quality, "chroma_subsampling": "4:4:4",
               "master_sha256": sha256_file(master_path), "images": index},
              out_root / "resize_index.json")
    rep(f"written {counts['written']}, already present {counts['kept']}, total {len(index)}")
    rep(f"index: resized_{R}/resize_index.json   finished in {time.time() - t0:.0f}s")
    rep.save(out_root / "resize_report.txt")


if __name__ == "__main__":
    main()
