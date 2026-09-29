"""
build_manifest.py - merge TrashCan + SeaClear into one master file with unique IDs.

Run once (rerun only if source data or the ID / duplicate settings change):
    python build_manifest.py
    python build_manifest.py --config path/to/hydro_config.yaml

Reads   DS1 TrashCan train+val JSONs and images, DS3 SeaClear dataset.json and images.
Writes  manifest/master.json         every image + annotation, ORIGINAL classes, cleaned boxes
        manifest/groups.json         folder ("group") -> image ids
        manifest/dup_clusters.json   near-duplicate clusters with 2+ images
        manifest/phash_cache.json    image fingerprints, reused on reruns
        manifest/dup_review/*.png    side-by-side pairs near the duplicate threshold, to check by eye
        manifest/build_report.txt    every check and count from this run
The original datasets are only read, never modified.
"""
import argparse
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

RS = getattr(Image, "Resampling", Image)   # Pillow < 9.1 has no Image.Resampling

try:
    import imagehash
except ImportError:  # pragma: no cover
    raise SystemExit("imagehash is required:  pip install imagehash")

from preprocessing.hydro_common import (IMG_EXT, Report, config_snapshot, fail, load_config, load_json, now,
                          save_json, sha256_file, to_posix_rel)

ID_GAP = 1_000_000          # original ids must stay below this, so offsets never collide
CLIP_RE = re.compile(r"^(vid_\d+)_frame\d+", re.IGNORECASE)


# ------------------------------------------------------------------ geometry
def polygon_area(poly):
    xs, ys = poly[0::2], poly[1::2]
    n = len(xs)
    return abs(sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))) / 2.0


def clean_bbox(b, W, H):
    """Clip [x, y, w, h] to the image. Returns (box or None, was_clipped)."""
    try:
        x, y, w, h = (float(v) for v in b[:4])
    except (TypeError, ValueError):
        return None, False
    if any(v != v for v in (x, y, w, h)):          # NaN check
        return None, False
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(W), x + w), min(float(H), y + h)
    clipped = (x1, y1, x2, y2) != (x, y, x + w, y + h)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None, clipped
    return [x1, y1, x2 - x1, y2 - y1], clipped


def clean_segmentation(seg, W, H):
    """Keep valid polygons, clip them to the image. Returns (polygons, n_dropped, was_rle)."""
    if isinstance(seg, dict):                       # run-length mask: not used by these datasets
        return [], 0, True
    kept, dropped = [], 0
    for poly in seg or []:
        if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2:
            dropped += 1
            continue
        pts = [min(max(float(v), 0.0), float(W if i % 2 == 0 else H)) for i, v in enumerate(poly)]
        if polygon_area(pts) < 1e-6:
            dropped += 1
            continue
        kept.append(pts)
    return kept, dropped, False


# ------------------------------------------------------------------ duplicates
def hex_to_u64(hexes):
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


def close_pairs(h64, max_dist):
    """All index pairs (i, j, distance) whose 64-bit fingerprints differ in <= max_dist bits."""
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = []
    n = len(h64)
    for i in range(n - 1):
        x = np.ascontiguousarray(h64[i] ^ h64[i + 1:]).view(np.uint8).reshape(-1, 8)
        d = lut[x].sum(axis=1, dtype=np.int64)
        for j in np.nonzero(d <= max_dist)[0]:
            out.append((i, i + 1 + int(j), int(d[j])))
    return out


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def review_image(base, rec_a, rec_b, dist, threshold, out_path, height=300):
    tiles = []
    for rec in (rec_a, rec_b):
        with Image.open(base / rec["file_name"]) as im:
            im = im.convert("RGB")
            w = max(1, round(im.width * height / im.height))
            tiles.append(im.resize((w, height), RS.BILINEAR))
    gap, top = 10, 34
    canvas = Image.new("RGB", (tiles[0].width + gap + tiles[1].width, height + top), "white")
    canvas.paste(tiles[0], (0, top))
    canvas.paste(tiles[1], (tiles[0].width + gap, top))
    verdict = "SAME CLUSTER" if dist <= threshold else "separate"
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 2), f"distance {dist} (threshold {threshold}) -> {verdict}", fill="black")
    draw.text((4, 17), f"{rec_a['group']}/{Path(rec_a['file_name']).name}   |   "
                       f"{rec_b['group']}/{Path(rec_b['file_name']).name}", fill="black")
    canvas.save(out_path)


# ------------------------------------------------------------------ loading
def walk_seaclear(root, base):
    """filename -> (relative path, group). Images must sit in <Site>/<Camera>/ folders."""
    found, dups, shallow = {}, defaultdict(list), []
    for dirpath, _, files in os.walk(root):
        parts = os.path.relpath(dirpath, root).split(os.sep)
        for f in files:
            if not f.lower().endswith(IMG_EXT):
                continue
            full = os.path.join(dirpath, f)
            if len(parts) < 2 or parts[0] == ".":
                shallow.append(to_posix_rel(full, base))
                continue
            rec = (to_posix_rel(full, base), f"seaclear/{parts[0]}/{parts[1]}")
            if f in found:
                dups[f].append(rec[0])
            else:
                found[f] = rec
    return found, dups, shallow


def load_sources(cfg, rep):
    base = cfg["_base"]
    img_off, cat_off = cfg["id_offsets"], cfg["category_offsets"]
    offs = [img_off["trashcan_train"], img_off["trashcan_val"], img_off["seaclear"]]
    if len(set(offs)) != 3 or min(abs(a - b) for a in offs for b in offs if a != b) < ID_GAP:
        fail(f"id_offsets must be distinct and at least {ID_GAP} apart: {offs}")

    images, anns, cats = [], [], []
    sources_meta, stats = {}, Counter()
    missing_files = []

    # ---------- TrashCan (train + val share one category list)
    tc = cfg["sources"]["trashcan"]
    ref_cats = None
    for split in ("train", "val"):
        jpath = base / tc["jsons"][split]
        if not jpath.exists():
            fail(f"TrashCan {split} json not found: {jpath}")
        d = load_json(jpath)
        sources_meta[to_posix_rel(jpath, base)] = sha256_file(jpath)
        rep(f"trashcan {split:5}: {len(d['images']):>6} images {len(d['annotations']):>7} annotations"
            f"  ({to_posix_rel(jpath, base)})")
        these = sorted((c["id"], c["name"]) for c in d["categories"])
        if ref_cats is None:
            ref_cats = these
            for c in d["categories"]:
                cats.append({"id": cat_off["trashcan"] + c["id"], "name": c["name"], "source": "trashcan",
                             "supercategory": c.get("supercategory", c["name"]), "orig_id": c["id"]})
        elif these != ref_cats:
            fail("TrashCan train and val category lists differ - cannot merge them safely")

        off = img_off[f"trashcan_{split}"]
        img_dir = tc["image_dirs"][split].rstrip("/\\").replace("\\", "/")
        max_img = max((i["id"] for i in d["images"]), default=0)
        max_ann = max((a["id"] for a in d["annotations"]), default=0)
        if max_img >= ID_GAP or max_ann >= ID_GAP:
            fail(f"TrashCan {split} has ids >= {ID_GAP}; offsets would collide")
        for im in d["images"]:
            rel = f"{img_dir}/{im['file_name']}"
            if not (base / rel).exists():
                missing_files.append(rel)
            m = CLIP_RE.match(im["file_name"])
            images.append({"id": off + im["id"], "file_name": rel,
                           "width": int(im["width"]), "height": int(im["height"]),
                           "source": "trashcan", "group": "trashcan",
                           "orig_id": im["id"], "orig_split": split,
                           "clip": m.group(1) if m else None})
        for a in d["annotations"]:
            anns.append(dict(a, _src="trashcan", _id=off + a["id"], _img=off + a["image_id"],
                             _cat=cat_off["trashcan"] + a["category_id"]))

    # ---------- SeaClear (file_name in the json is bare; the folder tree gives site/camera)
    sc = cfg["sources"]["seaclear"]
    root = base / sc["image_root"]
    jpath = base / sc["json"]
    if not jpath.exists():
        fail(f"SeaClear json not found: {jpath}")
    found, dups, shallow = walk_seaclear(root, base)
    if dups:
        ex = "; ".join(f"{k}: {v}" for k, v in list(dups.items())[:5])
        fail(f"{len(dups)} SeaClear filenames exist in more than one folder, so the json cannot be "
             f"matched to files unambiguously. Examples: {ex}")
    if shallow:
        rep(f"WARNING: {len(shallow)} SeaClear images not inside <Site>/<Camera>/ were skipped, e.g. {shallow[:3]}")
    d = load_json(jpath)
    sources_meta[to_posix_rel(jpath, base)] = sha256_file(jpath)
    rep(f"seaclear     : {len(d['images']):>6} images {len(d['annotations']):>7} annotations"
        f"  ({to_posix_rel(jpath, base)})")
    off = img_off["seaclear"]
    if max((i["id"] for i in d["images"]), default=0) >= ID_GAP or \
            max((a["id"] for a in d["annotations"]), default=0) >= ID_GAP:
        fail(f"SeaClear has ids >= {ID_GAP}; offsets would collide")
    for c in d["categories"]:
        cats.append({"id": cat_off["seaclear"] + c["id"], "name": c["name"], "source": "seaclear",
                     "supercategory": c.get("supercategory", c["name"]), "orig_id": c["id"]})
    for im in d["images"]:
        hit = found.get(im["file_name"])
        if hit is None:
            missing_files.append(f"{sc['image_root']}/?/?/{im['file_name']}")
            continue
        rel, group = hit
        images.append({"id": off + im["id"], "file_name": rel,
                       "width": int(im["width"]), "height": int(im["height"]),
                       "source": "seaclear", "group": group,
                       "orig_id": im["id"], "orig_split": None, "clip": None})
    for a in d["annotations"]:
        anns.append(dict(a, _src="seaclear", _id=off + a["id"], _img=off + a["image_id"],
                         _cat=cat_off["seaclear"] + a["category_id"]))

    if missing_files:
        fail(f"{len(missing_files)} images listed in the source jsons are missing on disk. "
             f"First ones: {missing_files[:10]}")
    return images, anns, cats, sources_meta


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to hydro_config.yaml (default: next to this script)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    base = cfg["_base"]
    out = base / "manifest"
    out.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep(f"build_manifest  {now()}   base = {base}")

    # ---------------- load
    rep.section("1. SOURCES")
    images, raw_anns, cats, sources_meta = load_sources(cfg, rep)
    img_by_id = {}
    for im in images:
        if im["id"] in img_by_id:
            fail(f"duplicate image id after offsets: {im['id']}")
        img_by_id[im["id"]] = im
    cat_ids = {c["id"] for c in cats}
    if len(cat_ids) != len(cats):
        fail("duplicate category id after offsets")

    # ---------------- probe every image: real size, and fingerprint for duplicate detection
    rep.section("2. IMAGE CHECKS + FINGERPRINTS")
    dcfg = cfg["duplicates"]
    hash_sources = set(dcfg.get("sources", ["seaclear"]))
    cache_path = out / "phash_cache.json"
    cache = load_json(cache_path) if cache_path.exists() else {}

    def key_for(im):
        st = os.stat(base / im["file_name"])
        return f"{im['file_name']}|{st.st_size}|{st.st_mtime_ns}"

    def probe(im):
        path = base / im["file_name"]
        want = im["source"] in hash_sources
        key = key_for(im) if want else None
        with Image.open(path) as pic:
            w, h = pic.size
            ph = None
            if want:
                ph = cache.get(key)
                if ph is None:
                    pic.draft("L", (256, 256))       # JPEG: decode at reduced size, much faster
                    ph = str(imagehash.phash(pic))
        return im["id"], w, h, key, ph

    workers = int(cfg.get("workers", os.cpu_count() or 4))
    results, t1 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, r in enumerate(ex.map(probe, images), 1):
            results.append(r)
            if k % 1000 == 0 or k == len(images):
                print(f"  probed {k}/{len(images)}  ({time.time() - t1:.0f}s)")
    mismatched, new_cache = [], {}
    for iid, w, h, key, ph in results:
        im = img_by_id[iid]
        if (w, h) != (im["width"], im["height"]):
            mismatched.append(f"{im['file_name']}: json {im['width']}x{im['height']}, file {w}x{h}")
        if ph is not None:
            im["_phash"] = ph
            new_cache[key] = ph
    if mismatched:
        fail(f"{len(mismatched)} images differ in size from their json entry, so their boxes would be "
             f"in the wrong frame. First ones: {mismatched[:10]}")
    save_json(new_cache, cache_path)
    rep(f"all {len(images)} images exist and match their declared size")
    rep(f"fingerprinted {len(new_cache)} images from {sorted(hash_sources)}"
        f"  ({len(set(new_cache) & set(cache))} reused from cache)")

    # ---------------- annotations: foreign keys + geometry clean-up
    rep.section("3. ANNOTATIONS")
    anns, st, zero_examples = [], Counter(), []
    seen_ann = set()
    for a in raw_anns:
        im = img_by_id.get(a["_img"])
        if im is None:
            st["dropped: image id not found"] += 1
            continue
        if a["_cat"] not in cat_ids:
            st["dropped: category id not found"] += 1
            continue
        if a["_id"] in seen_ann:
            fail(f"duplicate annotation id after offsets: {a['_id']}")
        seen_ann.add(a["_id"])
        W, H = im["width"], im["height"]
        box, clipped = clean_bbox(a.get("bbox", []), W, H)
        if box is None:
            st["dropped: zero-size or invalid box"] += 1
            if len(zero_examples) < 10:
                zero_examples.append(f"{im['file_name']} ann {a.get('id')} bbox {a.get('bbox')}")
            continue
        st["boxes clipped to image edge"] += clipped
        polys, n_bad, was_rle = clean_segmentation(a.get("segmentation"), W, H)
        st["polygons dropped (fewer than 3 points or zero area)"] += n_bad
        st["run-length masks dropped (box kept)"] += was_rle
        crowd = int(a.get("iscrowd", 0) or 0)
        st["crowd annotations (become ignore zones in splits)"] += crowd
        area = a.get("area")
        anns.append({"id": a["_id"], "image_id": im["id"], "category_id": a["_cat"],
                     "bbox": box, "segmentation": polys,
                     "area": float(area) if area is not None else box[2] * box[3],
                     "iscrowd": crowd, "has_mask": bool(polys),
                     "geo": round((box[2] * box[3]) ** 0.5 / max(W, H), 6),
                     "orig_id": a["id"]})
    rep(f"kept {len(anns)} of {len(raw_anns)} annotations")
    for k, v in sorted(st.items()):
        rep(f"  {k}: {v}")
    for z in zero_examples:
        rep(f"    e.g. {z}")

    # ---------------- near-duplicate clusters (all hashed images at once, across folders)
    rep.section("4. NEAR-DUPLICATE CLUSTERS")
    thr = int(dcfg.get("threshold", 3))
    lo, hi = dcfg.get("review_range", [2, 4])
    hashed = sorted((im for im in images if "_phash" in im), key=lambda r: r["id"])
    h64 = hex_to_u64([im["_phash"] for im in hashed])
    t2 = time.time()
    pairs = close_pairs(h64, max(thr, hi)) if len(hashed) > 1 else []
    uf = UnionFind(len(hashed))
    for i, j, dist in pairs:
        if dist <= thr:
            uf.union(i, j)
    comps = defaultdict(list)
    for i in range(len(hashed)):
        comps[uf.find(i)].append(hashed[i]["id"])
    ordered = sorted(comps.values(), key=lambda ids: min(ids))
    multi = {}
    for cid, ids in enumerate(ordered, 1):
        for iid in ids:
            img_by_id[iid]["dup_cluster"] = cid
            img_by_id[iid]["dup_size"] = len(ids)
        if len(ids) > 1:
            multi[str(cid)] = {"size": len(ids),
                               "groups": sorted({img_by_id[i]["group"] for i in ids}),
                               "image_ids": sorted(ids)}
    in_multi = sum(v["size"] for v in multi.values())
    rep(f"compared {len(hashed)} images ({len(pairs)} pairs within distance {max(thr, hi)}) "
        f"in {time.time() - t2:.1f}s")
    rep(f"threshold {thr}: {len(multi)} clusters of 2+ images covering {in_multi} images; "
        f"{len(hashed) - in_multi} images have no near-duplicate")
    size_hist = Counter(v["size"] for v in multi.values())
    rep("cluster sizes: " + ", ".join(f"{s} imgs x{n}" for s, n in sorted(size_hist.items())))
    for g, n in Counter(img_by_id[i]["group"] for v in multi.values() for i in v["image_ids"]).most_common():
        rep(f"  {g:44} {n:>5} images in clusters")
    cross = {k: v for k, v in multi.items() if len(v["groups"]) > 1}
    rep(f"clusters spanning more than one folder: {len(cross)}")
    for k, v in list(cross.items())[:20]:
        rep(f"  cluster {k}: {v['size']} images across {v['groups']}")
    if cross:
        rep("  -> make_splits.py stops if one of these crosses a train/test boundary (see config: cross_role_duplicates)")

    # review images around the threshold
    rev_dir = out / "dup_review"
    rev_dir.mkdir(exist_ok=True)
    for old in rev_dir.glob("pair_*.png"):
        old.unlink()
    band = [(i, j, dd) for i, j, dd in pairs if lo <= dd <= hi]
    rng = random.Random(f"{cfg.get('seed', 0)}:review")
    pick = rng.sample(band, min(int(dcfg.get("review_pairs", 20)), len(band)))
    for n, (i, j, dist) in enumerate(sorted(pick, key=lambda p: p[2]), 1):
        review_image(base, hashed[i], hashed[j], dist, thr, rev_dir / f"pair_{n:02d}_d{dist}.png")
    rep(f"wrote {len(pick)} review images (distance {lo}-{hi}) to manifest/dup_review/")

    # ---------------- leak group: the unit that must never be split across train and test
    for im in images:
        if im["source"] == "trashcan":
            im["leak_group"] = f"clip:{im['clip']}" if im["clip"] else f"img:{im['id']}"
            im.setdefault("dup_cluster", None)
            im.setdefault("dup_size", None)
        elif "dup_cluster" in im:
            im["leak_group"] = f"dup:{im['dup_cluster']}"
        else:
            im["leak_group"] = f"img:{im['id']}"
            im.setdefault("dup_cluster", None)
            im.setdefault("dup_size", None)
        im.pop("_phash", None)

    # ---------------- groups + counts
    rep.section("5. GROUPS")
    ann_count = Counter(a["image_id"] for a in anns)
    groups = defaultdict(lambda: {"source": None, "n_images": 0, "n_annotations": 0, "image_ids": []})
    for im in sorted(images, key=lambda r: r["id"]):
        g = groups[im["group"]]
        g["source"] = im["source"]
        g["n_images"] += 1
        g["n_annotations"] += ann_count[im["id"]]
        g["image_ids"].append(im["id"])
    rep(f"{'group':46}{'images':>8}{'annotations':>13}")
    for name in sorted(groups):
        rep(f"{name:46}{groups[name]['n_images']:>8}{groups[name]['n_annotations']:>13}")
    for src, want in (cfg.get("expected_images") or {}).items():
        got = sum(1 for im in images if im["source"] == src)
        if got != want:
            rep(f"WARNING: expected {want} {src} images, found {got}")
        else:
            rep(f"expected {src} image count matches ({got})")

    # ---------------- write
    rep.section("6. OUTPUT")
    master = {
        "info": {"description": "HydroWatch master manifest - original classes, cleaned geometry",
                 "created": now(), "source_files_sha256": sources_meta,
                 "id_offsets": cfg["id_offsets"], "category_offsets": cfg["category_offsets"],
                 "duplicates": {"method": "pHash 64-bit (JPEG draft decode)", "threshold": thr,
                                "sources": sorted(hash_sources)},
                 "config": config_snapshot(cfg)},
        "images": sorted(images, key=lambda r: r["id"]),
        "annotations": sorted(anns, key=lambda r: r["id"]),
        "categories": sorted(cats, key=lambda r: r["id"]),
    }
    save_json(master, out / "master.json")
    save_json(dict(sorted(groups.items())), out / "groups.json", pretty=True)
    save_json({"threshold": thr, "method": "pHash 64-bit (JPEG draft decode)",
               "n_clusters_2plus": len(multi), "n_images_in_clusters": in_multi,
               "n_singletons": len(hashed) - in_multi, "clusters": multi},
              out / "dup_clusters.json", pretty=True)
    rep(f"manifest/master.json        {len(images)} images, {len(anns)} annotations, {len(cats)} categories")
    rep("manifest/groups.json        manifest/dup_clusters.json   manifest/phash_cache.json")
    rep(f"finished in {time.time() - t0:.0f}s")
    rep.save(out / "build_report.txt")


if __name__ == "__main__":
    main()
