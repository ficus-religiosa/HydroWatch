r"""
make_oracle_split.py - leak-free, in-distribution split for the best possible oracle model.

    pip install ImageHash
    python make_oracle_split.py

Reads the existing split files (L_B, U_all, TEST), pools every usable image, and writes
datasets\splits\oracle_v3\ :
    TRAIN.json  VAL.json  TEST.json     same record format as before
    split_meta.json                     size-bin cuts recomputed from the new TEST
    split_report.txt                    counts per folder / class / size bin, and the leakage audit
    leakage_audit.csv                   TEST/VAL images with a TRAIN image at Hamming distance 3-4 (look at these)
    phash_cache.json                    hashes, so a re-run takes seconds

Rules
  SeaClear  near-duplicate clusters = pHash Hamming distance <= 2, chained. Every camera folder (one video)
            sends ~20% of its clusters to TEST, ~10% to VAL, ~70% to TRAIN, balancing objects per class and
            per size bin. Folders with too few clusters go to TRAIN. TEST and VAL use the 1920 originals.
  TrashCan  TRAIN only. Per clip, near-duplicates (<= 2) removed, then at most --tc-cap frames kept,
            chosen to be as different from each other as possible.
  Excluded  Marseille, Jakljan/Paralenz (not in the source splits anyway; guarded).
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

BASE = r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets"
SRC = BASE + r"\splits\r1280_s0_v2-debris-plus-biota"
DST = BASE + r"\splits\oracle_v3"
RESIZED = "resized_1280/"
SPLITS = ("TEST", "VAL", "TRAIN")
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ----------------------------------------------------------------------------- hashing
def _phash(path):
    import imagehash
    with Image.open(path) as im:
        return int(str(imagehash.phash(im)), 16)


def hash_all(paths, cache_file):
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    todo = [p for p in paths if p not in cache]
    if todo:
        print(f"hashing {len(todo)} images ({len(paths) - len(todo)} cached)...")
        with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1)) as ex:
            for i, (p, h) in enumerate(zip(todo, ex.map(_phash, todo, chunksize=32)), 1):
                cache[p] = format(h, "016x")
                if i % 1000 == 0:
                    print(f"   {i}/{len(todo)}")
        cache_file.write_text(json.dumps(cache))
    return {p: int(cache[p], 16) for p in paths}


def hamming(a, b):
    """a: (n,) uint64, b: (m,) uint64 -> (n, m) bit distances."""
    x = np.bitwise_xor(a[:, None], b[None, :])
    return POP[x.view(np.uint8).reshape(len(a), len(b), 8)].sum(-1)


def clusters_by_hash(hashes, thr):
    """Single-link (chained) clusters: any pair within thr ends up in one cluster."""
    n = len(hashes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    h = np.array(hashes, dtype=np.uint64)
    for s in range(0, n, 512):
        d = hamming(h[s:s + 512], h)
        for i, j in zip(*np.nonzero(d <= thr)):
            a, b = find(s + i), find(j)
            if a != b:
                parent[a] = b
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


# ----------------------------------------------------------------------------- records
def load(src, name):
    d = json.loads((src / f"{name}.json").read_text(encoding="utf-8"))
    return d


def orig_of(im, base):
    """Original (full-resolution) file for a SeaClear record, and stored/original scale."""
    fn = im["file_name"]
    rel = im.get("orig_file_name") or (fn[len(RESIZED):] if fn.startswith(RESIZED) else fn)
    if im.get("orig_size"):
        w0, h0 = im["orig_size"]
    elif rel == fn:
        w0, h0 = im["width"], im["height"]
    else:
        with Image.open(base / rel) as f:
            w0, h0 = f.size
    return rel, w0, h0, im["width"] / w0, im["height"] / h0


def to_original(im, anns, base):
    rel, w0, h0, sx, sy = orig_of(im, base)
    if (rel, w0, h0) == (im["file_name"], im["width"], im["height"]):
        return im, anns
    im = {**im, "file_name": rel, "width": w0, "height": h0, "scale": [1.0, 1.0], "orig_size": [w0, h0],
          "orig_file_name": rel, "ignore": [[x / sx, y / sy, w / sx, h / sy] for x, y, w, h in im.get("ignore", [])]}
    out = []
    for a in anns:
        x, y, w, h = a["bbox"]
        b = {**a, "bbox": [x / sx, y / sy, w / sx, h / sy]}
        if "area" in a:
            b["area"] = a["area"] / (sx * sy)
        if isinstance(a.get("segmentation"), list):
            b["segmentation"] = [[v / (sx if k % 2 == 0 else sy) for k, v in enumerate(p)] for p in a["segmentation"]]
        out.append(b)
    return im, out


def geo(a, im):
    return (a["bbox"][2] * a["bbox"][3]) ** 0.5 / max(im["width"], im["height"])


# ----------------------------------------------------------------------------- assignment
def assign(folder_clusters, stats, ratios, seed, min_clusters, report):
    """Greedy stratified assignment inside each folder; returns {cluster_index: split}."""
    rng = random.Random(seed)
    out = {}
    glob = Counter()
    for v in stats:
        glob.update(v["cls"])
    for folder, cl in sorted(folder_clusters.items()):
        if len(cl) < min_clusters:
            out.update({c: "TRAIN" for c in cl})
            report.append(f"   {folder}: only {len(cl)} clusters -> all TRAIN")
            continue
        tot_img = sum(stats[c]["n"] for c in cl)
        tot_cls, tot_bin = Counter(), Counter()
        for c in cl:
            tot_cls.update(stats[c]["cls"])
            tot_bin.update(stats[c]["bin"])
        cur = {s: {"n": 0, "cls": Counter(), "bin": Counter()} for s in SPLITS}

        def rarity(c):
            k = stats[c]["cls"]
            return min((glob[x] for x in k), default=10 ** 9)

        order = sorted(cl, key=lambda c: (rarity(c), -stats[c]["n"], rng.random()))
        for c in order:
            best, best_score = "TRAIN", -1e9
            for s in SPLITS:
                need_img = ratios[s] * tot_img - cur[s]["n"]
                if need_img <= 0 and s != "TRAIN":
                    continue
                sc = need_img / max(ratios[s] * tot_img, 1)
                for k in stats[c]["cls"]:
                    t = ratios[s] * tot_cls[k]
                    sc += (t - cur[s]["cls"][k]) / max(t, 1)
                for k in stats[c]["bin"]:
                    t = ratios[s] * tot_bin[k]
                    sc += 0.5 * (t - cur[s]["bin"][k]) / max(t, 1)
                if sc > best_score:
                    best, best_score = s, sc
            out[c] = best
            cur[best]["n"] += stats[c]["n"]
            cur[best]["cls"].update(stats[c]["cls"])
            cur[best]["bin"].update(stats[c]["bin"])
        report.append(f"   {folder}: {len(cl)} clusters, {tot_img} images -> "
                      + ", ".join(f"{s} {cur[s]['n']} ({100 * cur[s]['n'] / max(tot_img, 1):.0f}%)" for s in SPLITS))
    return out


def trashcan_keep(clip_idx, hashes, thr, cap):
    """Per clip: drop frames within thr of a kept one, keep up to cap maximally different frames."""
    keep = []
    for clip, idx in clip_idx.items():
        idx = sorted(idx)
        chosen = [idx[len(idx) // 2]]
        h = np.array([hashes[i] for i in idx], dtype=np.uint64)
        while len(chosen) < cap:
            d = hamming(h, np.array([hashes[c] for c in chosen], dtype=np.uint64)).min(1)
            k = int(d.argmax())
            if d[k] <= thr:
                break
            chosen.append(idx[k])
        keep += chosen
    return set(keep)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--src", default=SRC, help="folder with the current L_B / U_all / TEST split files")
    ap.add_argument("--out", default=DST)
    ap.add_argument("--threshold", type=int, default=2, help="pHash Hamming distance that counts as a duplicate")
    ap.add_argument("--test", type=float, default=0.20)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--min-clusters", type=int, default=5, help="folders with fewer clusters go wholly to TRAIN")
    ap.add_argument("--tc-cap", type=int, default=8, help="max TrashCan frames kept per clip")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base, src, out = Path(a.base), Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ratios = {"TEST": a.test, "VAL": a.val, "TRAIN": 1 - a.test - a.val}
    rep = []

    # ---- pool every usable image once
    images, anns, cats, seen_ann = {}, defaultdict(list), None, set()
    for name in ("L_B", "U_all", "TEST"):
        d = load(src, name)
        cats = cats or d["categories"]
        for im in d["images"]:
            images.setdefault(im["id"], im)
        for x in d["annotations"]:
            if (x["image_id"], x["id"]) not in seen_ann:
                seen_ann.add((x["image_id"], x["id"]))
                anns[x["image_id"]].append(x)
    meta_src = json.loads((src / "split_meta.json").read_text(encoding="utf-8"))
    g1, g2 = meta_src["size_bins"]["geo_cuts"]

    def bad(im):
        g = str(im.get("group", ""))
        return "Marseille" in g or ("Jakljan" in g and "Paralenz" in g)

    sea = sorted(i for i, im in images.items() if im.get("source") == "seaclear" and not bad(im))
    tc = sorted(i for i, im in images.items() if im.get("source") == "trashcan")
    print(f"pool: {len(sea)} SeaClear images, {len(tc)} TrashCan images")

    # ---- hashes (SeaClear from the originals, so every image is hashed at the same resolution)
    sea_path = {i: orig_of(images[i], base)[0] for i in sea}
    tc_path = {i: images[i]["file_name"] for i in tc}
    H = hash_all([str(base / p) for p in list(sea_path.values()) + list(tc_path.values())], out / "phash_cache.json")
    sh = [H[str(base / sea_path[i])] for i in sea]
    th = {i: H[str(base / tc_path[i])] for i in tc}

    # ---- SeaClear clusters and their statistics
    cl = clusters_by_hash(sh, a.threshold)
    sizes = sorted((len(c) for c in cl), reverse=True)
    rep.append(f"SeaClear: {len(sea)} images -> {len(cl)} clusters at distance <= {a.threshold}; "
               f"largest {sizes[:5]}, singletons {sum(1 for s in sizes if s == 1)}")
    stats, folder_clusters = [], defaultdict(list)
    for ci, members in enumerate(cl):
        ids = [sea[k] for k in members]
        cls, bins = Counter(), Counter()
        for i in ids:
            for x in anns[i]:
                cls[x["category_id"]] += 1
                gg = geo(x, images[i])
                bins["small" if gg < g1 else "medium" if gg < g2 else "large"] += 1
        stats.append({"ids": ids, "n": len(ids), "cls": cls, "bin": bins})
        home = Counter(images[i].get("group", "?") for i in ids).most_common(1)[0][0]
        folder_clusters[home].append(ci)
    rep.append(f"folders (camera = one video): {len(folder_clusters)}")
    rep.append("\nassignment per folder:")
    choice = assign(folder_clusters, stats, ratios, a.seed, a.min_clusters, rep)

    split_of = {}
    for ci, s in choice.items():
        for i in stats[ci]["ids"]:
            split_of[i] = s
    leak = {i: f"dupv3:{ci}" for ci, v in enumerate(stats) for i in v["ids"]}

    # ---- TrashCan: TRAIN only, de-duplicated per clip
    clip_idx = defaultdict(list)
    for i in tc:
        clip_idx[images[i].get("leak_group") or f"img:{i}"].append(i)
    kept = trashcan_keep(clip_idx, th, a.threshold, a.tc_cap)
    rep.append(f"TrashCan: {len(tc)} frames in {len(clip_idx)} clips -> kept {len(kept)} "
               f"(<= {a.tc_cap} per clip, near-duplicates <= {a.threshold} removed)")

    # ---- leakage audit: every TEST/VAL image vs every SeaClear TRAIN image
    idx = {i: k for k, i in enumerate(sea)}
    tr = np.array([sh[idx[i]] for i in sea if split_of[i] == "TRAIN"], dtype=np.uint64)
    tr_ids = [i for i in sea if split_of[i] == "TRAIN"]
    audit, near = Counter(), []
    for s in ("TEST", "VAL"):
        ids = [i for i in sea if split_of[i] == s]
        for k in range(0, len(ids), 256):
            chunk = ids[k:k + 256]
            d = hamming(np.array([sh[idx[i]] for i in chunk], dtype=np.uint64), tr)
            j = d.argmin(1)
            for i, jj, dd in zip(chunk, j, d[np.arange(len(chunk)), j]):
                band = "<=2" if dd <= 2 else "3-4" if dd <= 4 else "5-6" if dd <= 6 else ">6"
                audit[(s, band)] += 1
                if dd <= 4:
                    near.append((s, int(dd), sea_path[i], sea_path[tr_ids[jj]]))
    assert audit[("TEST", "<=2")] == 0 and audit[("VAL", "<=2")] == 0, "a duplicate crossed splits - bug"
    with open(out / "leakage_audit.csv", "w", encoding="utf-8") as f:
        f.write("split,distance,image,nearest_train_image\n")
        for r in sorted(near, key=lambda r: r[1]):
            f.write(",".join(map(str, r)) + "\n")

    # ---- write splits (TEST/VAL converted to 1920 originals; TRAIN left as stored)
    outd = {s: {"images": [], "annotations": []} for s in SPLITS}
    for i in sea:
        s = split_of[i]
        im, an = images[i], anns[i]
        if s != "TRAIN":
            im, an = to_original(im, an, base)
        outd[s]["images"].append({**im, "leak_group": leak[i]})
        outd[s]["annotations"] += an
    for i in sorted(kept):
        outd["TRAIN"]["images"].append(images[i])
        outd["TRAIN"]["annotations"] += anns[i]
    for s in SPLITS:
        d = {"info": {"split": s, "made_by": "make_oracle_split.py", "threshold": a.threshold, "seed": a.seed},
             "images": outd[s]["images"], "annotations": outd[s]["annotations"], "categories": cats}
        (out / f"{s}.json").write_text(json.dumps(d), encoding="utf-8")

    # ---- size bins from the new TEST (tertiles of geo = sqrt(box area) / long side)
    tim = {im["id"]: im for im in outd["TEST"]["images"]}
    gs = np.array([geo(x, tim[x["image_id"]]) for x in outd["TEST"]["annotations"]])
    n1, n2 = (float(v) for v in np.quantile(gs, [1 / 3, 2 / 3]))
    L = max(max(im["width"], im["height"]) for im in tim.values())
    meta = {**meta_src, "description": "oracle_v3: in-distribution, leak-free (pHash <= %d clusters), "
                                       "folder-stratified; TrashCan TRAIN only" % a.threshold,
            "size_bins": {"geo_cuts": [n1, n2], "reference_long_side": L,
                          "area_px_cuts": [(n1 * L) ** 2, (n2 * L) ** 2],
                          "definition": "geo = sqrt(box area) / image long side; tertiles of TEST"}}
    (out / "split_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # ---- report
    names = {c["id"]: c["name"] for c in cats}
    rep.append("\nimages / objects per split:")
    for s in SPLITS:
        src_n = Counter(im.get("source") for im in outd[s]["images"])
        rep.append(f"   {s:5s} {len(outd[s]['images']):6d} images {dict(src_n)}  {len(outd[s]['annotations'])} objects")
    rep.append("\nobjects per class (TEST / VAL / TRAIN-SeaClear / TRAIN-TrashCan):")
    sea_ids = set(sea)
    for cid in sorted(names):
        t = sum(1 for x in outd["TEST"]["annotations"] if x["category_id"] == cid)
        v = sum(1 for x in outd["VAL"]["annotations"] if x["category_id"] == cid)
        trs = sum(1 for x in outd["TRAIN"]["annotations"] if x["category_id"] == cid and x["image_id"] in sea_ids)
        trt = sum(1 for x in outd["TRAIN"]["annotations"] if x["category_id"] == cid and x["image_id"] not in sea_ids)
        flag = "   <- under 50 in TEST" if t < 50 else ""
        rep.append(f"   {names[cid]:14s} {t:6d} {v:6d} {trs:6d} {trt:6d}{flag}")
    rep.append(f"\nnew size bins (TEST tertiles): geo {n1:.4f} / {n2:.4f}  = boxes under "
               f"{n1 * 1280:.0f} / {n2 * 1280:.0f} px across at 1280")
    rep.append("\nleakage audit - nearest SeaClear TRAIN image for every TEST/VAL image:")
    for s in ("TEST", "VAL"):
        rep.append(f"   {s:5s} " + "  ".join(f"{b}: {audit[(s, b)]}" for b in ("<=2", "3-4", "5-6", ">6")))
    rep.append("   <=2 must be 0. Open a few of the 3-4 pairs listed in leakage_audit.csv: if they look like the")
    rep.append("   same scene, re-run with --threshold 4.")
    txt = "\n".join(rep)
    (out / "split_report.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":  # required on Windows (worker processes)
    main()
