r"""
fathomnet_download.py - check, then download, FathomNet's debris imagery.

    pip install fathomnet requests

STEP 1 - discover (metadata only, no images, a few minutes):
    python fathomnet_download.py --discover
  Prints every trash-related concept FathomNet really has, how many boxes each has,
  and how deep the images are. Writes discover_report.json. Read it before step 2.

STEP 2 - download (images + all their boxes):
    python fathomnet_download.py --download
    python fathomnet_download.py --download --max-depth 50          # shallow only
    python fathomnet_download.py --download --concepts trash plastic  # pick concepts yourself
  Safe to re-run: finished files are skipped, so it resumes after an interruption.

Output (default HydroWatch\datasets\FathomNet):
    images\<uuid>.jpg      the images
    fathomnet_raw.json     every image record as FathomNet returned it (depth, time, position, all boxes)
    fathomnet_coco.json    COCO-style boxes for ALL concepts on those images (not only trash)
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from fathomnet.api import boundingboxes, images, taxa

OUT = Path(r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets\FathomNet")
ROOT = "trash"
# name patterns that suggest debris; matched as whole words, printed for YOU to confirm
KEYWORDS = r"trash|debris|litter|garbage|plastic|bottle|can|bag|wrapper|cup|rope|net|netting|line|" \
           r"fishing|tire|tyre|rubber|glass|metal|scrap|fabric|cloth|clothing|tarp|wreck|wreckage|container|bucket"
DEPTH_BINS = [(10, "<=10 m"), (30, "10-30 m"), (100, "30-100 m"), (500, "100-500 m"), (1e9, ">500 m")]


def as_dict(obj):
    return obj.to_dict() if hasattr(obj, "to_dict") else dict(vars(obj))


def retry(fn, *a, tries=4, **k):
    for i in range(tries):
        try:
            return fn(*a, **k)
        except Exception as e:  # network hiccups: back off and retry
            if i == tries - 1:
                raise
            print(f"   retry {i + 1} after error: {e}")
            time.sleep(2 * (i + 1))


def descendants_of_root():
    """Concepts under 'trash' according to each taxonomy provider FathomNet exposes."""
    found = {}
    for prov in retry(taxa.list_taxa_providers):
        try:
            names = [t.name for t in retry(taxa.find_taxa, prov, ROOT)]
        except Exception:
            names = []
        if names:
            found[prov] = names
    return found


def depth_profile(imgs):
    c = Counter()
    for im in imgs:
        d = im.get("depthMeters")
        c["unknown" if d is None else next(label for lim, label in DEPTH_BINS if d <= lim)] += 1
    return {label: c.get(label, 0) for _, label in DEPTH_BINS} | {"unknown": c.get("unknown", 0)}


def fetch_images(concept):
    return [as_dict(im) for im in retry(images.find_by_concept, concept)]


def discover(out):
    print("1) Concepts under 'trash' in FathomNet's own taxonomy:")
    tree = descendants_of_root()
    for prov, names in tree.items():
        print(f"   provider '{prov}': {', '.join(names)}")
    if not tree:
        print("   (no provider returned a subtree - relying on the name search below)")

    print("2) Concepts whose NAME looks like debris (check these by eye - some may be animals):")
    pat = re.compile(rf"\b({KEYWORDS})\b", re.I)
    by_name = sorted(c for c in retry(boundingboxes.find_concepts) if pat.search(c))
    print("   " + ", ".join(by_name))

    candidates = sorted(set(sum(tree.values(), [])) | set(by_name) | {ROOT})
    print(f"3) Box counts and depths for {len(candidates)} candidate concepts (metadata only)...")
    rows, all_imgs = [], {}
    for c in candidates:
        n = retry(boundingboxes.count_by_concept, c).count
        imgs = fetch_images(c) if n else []
        for im in imgs:
            all_imgs[im["uuid"]] = im
        rows.append({"concept": c, "boxes": n, "images": len(imgs), "in_trash_tree": c in sum(tree.values(), []),
                     "depth": depth_profile(imgs)})
    rows.sort(key=lambda r: -r["boxes"])
    print(f"\n   {'concept':32s} {'boxes':>7s} {'images':>7s}  depth <=30 m")
    for r in rows:
        shallow = r["depth"]["<=10 m"] + r["depth"]["10-30 m"]
        tag = " *" if r["in_trash_tree"] else ""
        print(f"   {r['concept'][:32]:32s} {r['boxes']:7d} {r['images']:7d}  {shallow:7d}{tag}")
    total = depth_profile(all_imgs.values())
    print(f"\n   ALL candidate images (deduplicated): {len(all_imgs)}   depth: {total}")
    print("   * = listed under 'trash' by FathomNet's taxonomy")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"trash_tree": tree, "name_matches": by_name, "concepts": rows, "total_images": len(all_imgs),
           "total_depth": total}
    (out / "discover_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'discover_report.json'} - decide which concepts to keep, then run --download")


def download_one(im, img_dir):
    url = im.get("url")
    ext = Path(url.split("?")[0]).suffix.lower() if url else ""
    ext = ext if ext in (".jpg", ".jpeg", ".png") else ".jpg"
    path = img_dir / f"{im['uuid']}{ext}"
    if path.exists() and path.stat().st_size > 0:
        return path, "skipped"
    r = retry(requests.get, url, timeout=60)
    r.raise_for_status()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(r.content)
    tmp.replace(path)
    if im.get("sha256") and hashlib.sha256(r.content).hexdigest() != im["sha256"]:
        return path, "hash-mismatch"
    return path, "ok"


def download(out, concepts, max_depth, workers):
    rep_path = out / "discover_report.json"
    if not concepts:
        if not rep_path.exists():
            raise SystemExit("Run --discover first, or pass --concepts explicitly.")
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        concepts = sorted(set(sum(rep["trash_tree"].values(), [])) | {ROOT})
        print(f"Using FathomNet's own trash subtree ({len(concepts)} concepts). "
              "Add name-matched concepts with --concepts if you want them.")
    imgs = {}
    for c in concepts:
        got = fetch_images(c)
        for im in got:
            imgs[im["uuid"]] = im
        print(f"   {c}: {len(got)} images")
    if max_depth is not None:
        before = len(imgs)
        imgs = {u: im for u, im in imgs.items() if im.get("depthMeters") is not None and im["depthMeters"] <= max_depth}
        print(f"   depth <= {max_depth} m keeps {len(imgs)} of {before} images (unknown depth dropped)")
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (out / "fathomnet_raw.json").write_text(json.dumps(list(imgs.values()), indent=0, default=str), encoding="utf-8")

    status, files = Counter(), {}
    print(f"Downloading {len(imgs)} images with {workers} threads...")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(download_one, im, img_dir): u for u, im in imgs.items()}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                path, st = f.result()
                files[futs[f]] = path.name
            except Exception as e:
                st = "failed"
                print(f"   failed {futs[f]}: {e}")
            status[st] += 1
            if i % 200 == 0 or i == len(futs):
                print(f"   {i}/{len(futs)}  {dict(status)}")

    cats, anns, coco_imgs = {}, [], []
    for u, im in imgs.items():
        if u not in files:
            continue
        iid = len(coco_imgs) + 1
        coco_imgs.append({"id": iid, "file_name": f"images/{files[u]}", "width": im.get("width"),
                          "height": im.get("height"), "uuid": u, "depthMeters": im.get("depthMeters"),
                          "timestamp": im.get("timestamp"), "latitude": im.get("latitude"),
                          "longitude": im.get("longitude"), "imagingType": im.get("imagingType"),
                          "contributorsEmail": im.get("contributorsEmail")})
        for b in im.get("boundingBoxes") or []:
            if b.get("rejected"):
                continue
            cid = cats.setdefault(b["concept"], len(cats) + 1)
            anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": cid,
                         "bbox": [b["x"], b["y"], b["width"], b["height"]], "area": b["width"] * b["height"],
                         "iscrowd": 0, "occluded": b.get("occluded"), "truncated": b.get("truncated"),
                         "altConcept": b.get("altConcept")})
    coco = {"images": coco_imgs, "annotations": anns,
            "categories": [{"id": i, "name": n} for n, i in sorted(cats.items(), key=lambda kv: kv[1])]}
    (out / "fathomnet_coco.json").write_text(json.dumps(coco), encoding="utf-8")
    top = Counter(a["category_id"] for a in anns)
    names = {i: n for n, i in cats.items()}
    print(f"\nDone: {len(coco_imgs)} images, {len(anns)} boxes across {len(cats)} concepts. Most common:")
    for cid, n in top.most_common(15):
        print(f"   {names[cid]:32s} {n}")
    print(f"Status: {dict(status)}  ->  {out}")


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--discover", action="store_true")
    mode.add_argument("--download", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--max-depth", type=float, default=None, help="keep only images at or above this depth (m)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    if a.discover:
        discover(out)
    else:
        download(out, a.concepts, a.max_depth, a.workers)


if __name__ == "__main__":
    main()
