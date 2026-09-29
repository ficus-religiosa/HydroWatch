"""
make_splits.py - turn manifest/master.json into self-contained split files for training and scoring.

    python make_splits.py
    python make_splits.py --config path/to/hydro_config.yaml

Reads   hydro_config.yaml (folder roles, R, seed, sizes), the taxonomy file, manifest/master.json,
        resized_<R>/resize_index.json (if present; otherwise split files point at the originals)
Writes  splits/r<R>_s<seed>_<taxonomy version>/
            L_A.json  L_B.json  U_0500.json ... U_all.json  TEST.json  CONTROL.json
            split_meta.json    sizes, size-bin cut-offs, fingerprints, per-class counts
            split_report.txt   everything printed during the run
Takes seconds and never touches images. Rerun whenever the config or the taxonomy changes.

Folder roles (set per folder in the config):
  POOL_L   L_A is drawn from here; the rest of the folder goes to U
  POOL     goes to U only
  L_EXTRA  added to L_A to form L_B (TrashCan)
  TEST     scored only, never trained on
  CONTROL  scored separately, never trained on (Marseille)
  EXCL     not used anywhere
"""
import argparse
import random
import re
import time
from collections import Counter, defaultdict

import numpy as np

from preprocessing.hydro_common import (EVAL_ROLES, ROLES, TRAIN_ROLES, Report, config_snapshot, fail,
                          fingerprint_ids, fingerprint_obj, load_config, load_json, load_taxonomy,
                          now, resolve_class, save_json, sha256_file, tier_of)

BINS = ("small", "medium", "large")
SPLIT_FILE = re.compile(r"^(L_A|L_B|U_\d{4}|U_all|TEST|CONTROL)\.json$")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="path to hydro_config.yaml (default: next to this script)")
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config(args.config)
    base = cfg["_base"]
    R, seed = int(cfg["R"]), cfg.get("seed", 0)
    L_size, min_support = int(cfg["L_size"]), int(cfg.get("min_support", 50))
    master_path = base / "manifest" / "master.json"
    if not master_path.exists():
        fail("manifest/master.json not found - run build_manifest.py first")
    master = load_json(master_path)
    tax_path = base / cfg["taxonomy"]
    tax = load_taxonomy(tax_path)
    unified = tax["unified_classes"]
    rep = Report()
    rep(f"make_splits  {now()}   R={R}  seed={seed}  taxonomy={tax['version']}")

    imgs = {im["id"]: im for im in master["images"]}
    cats = {c["id"]: c for c in master["categories"]}
    anns_by_img = defaultdict(list)
    for a in master["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    # ---------------------------------------------------------------- 1. taxonomy vs master
    rep.section("1. TAXONOMY")
    used = Counter(a["category_id"] for a in master["annotations"])
    kind_of, uncovered = {}, []
    for cid, c in cats.items():
        kind, k = resolve_class(tax, c["source"], c["name"])
        if kind is None:
            if used[cid]:
                uncovered.append(f"{c['source']}.{c['name']} ({used[cid]} annotations)")
            kind = "background"
        kind_of[cid] = (kind, k)
    if uncovered:
        fail("these classes have annotations but are not in the taxonomy (add them to map, ignore "
             "or background): " + ", ".join(uncovered))
    names = defaultdict(set)
    for c in cats.values():
        names[c["source"]].add(c["name"])
    placed = defaultdict(list)
    for section in ("map", "ignore", "background"):
        for src, entries in tax[section].items():
            for n in (entries.keys() if isinstance(entries, dict) else entries):
                placed[(src, n)].append(section)
    twice = [f"{s}.{n} in {sec}" for (s, n), sec in placed.items() if len(sec) > 1]
    if twice:
        fail("classes listed in more than one taxonomy section: " + ", ".join(twice))
    typos = [f"{s}.{n}" for (s, n) in placed if n not in names.get(s, set())]
    if typos:
        rep(f"WARNING: taxonomy names classes that do not exist in master (typo?): {typos}")
    n_kind = Counter()
    for cid, (kind, _) in kind_of.items():
        n_kind[kind] += used[cid]
    rep(f"{len(unified)} classes; annotations -> class: {n_kind['map']}, ignore zone: {n_kind['ignore']}, "
        f"background: {n_kind['background']}")

    # ---------------------------------------------------------------- 2. folder roles
    rep.section("2. FOLDER ROLES")
    roles_cfg = cfg["groups"]
    in_master = sorted({im["group"] for im in imgs.values()})
    missing = [g for g in in_master if g not in roles_cfg]
    unknown = [g for g in roles_cfg if g not in in_master]
    bad = [f"{g}: {r}" for g, r in roles_cfg.items() if r not in ROLES]
    if missing:
        fail(f"folders with no role in the config: {missing}")
    if unknown:
        fail(f"config lists folders that are not in master.json (typo?): {unknown}")
    if bad:
        fail(f"unknown roles {bad}; allowed: {ROLES}")
    role = {iid: roles_cfg[im["group"]] for iid, im in imgs.items()}
    for need in ("POOL_L", "TEST"):
        if need not in roles_cfg.values():
            fail(f"no folder has role {need}")
    per_group = Counter(im["group"] for im in imgs.values())
    for g in in_master:
        rep(f"  {g:46} {roles_cfg[g]:8} {per_group[g]:>6} imgs")

    # ---------------------------------------------------------------- 3. leak groups
    rep.section("3. LEAK PROTECTION (near-duplicates and TrashCan videos stay on one side)")
    members = defaultdict(list)
    for iid, im in imgs.items():
        members[im["leak_group"]].append(iid)
    dropped = {}
    violations = []
    for lg, ids in members.items():
        rs = {role[i] for i in ids} - {"EXCL"}
        if rs & TRAIN_ROLES and rs & EVAL_ROLES:
            violations.append((lg, ids, rs))
    if violations:
        rep(f"{len(violations)} near-duplicate clusters contain both training and evaluation images:")
        for lg, ids, rs in violations[:30]:
            rep(f"  {lg}: roles {sorted(rs)}  " + ", ".join(imgs[i]["file_name"] for i in ids[:4]))
        mode = cfg.get("cross_role_duplicates", "error")
        if mode == "error":
            rep.save(base / "splits" / "last_failed_split_report.txt")
            fail("training and evaluation images are near-duplicates of each other (listed above; also "
                 "see manifest/dup_review). Check them, then set  cross_role_duplicates: drop_from_train "
                 "in the config to remove the training-side copies.")
        if mode != "drop_from_train":
            fail(f"cross_role_duplicates must be 'error' or 'drop_from_train', got {mode!r}")
        for lg, ids, _ in violations:
            for i in ids:
                if role[i] in TRAIN_ROLES:
                    dropped[i] = f"near-duplicate of an evaluation image ({lg})"
    for lg, ids in members.items():
        rs = {role[i] for i in ids if i not in dropped}
        if "L_EXTRA" in rs and rs & {"POOL", "POOL_L"}:
            for i in ids:
                if role[i] in ("POOL", "POOL_L") and i not in dropped:
                    dropped[i] = f"near-duplicate of an L_EXTRA image ({lg})"
    rep(f"training images removed as duplicates of evaluation/L_EXTRA images: {len(dropped)}")

    def live(i):
        return i not in dropped

    # ---------------------------------------------------------------- 4. L_A, U, TEST, CONTROL
    rep.section("4. DRAWING THE SPLITS")
    eligible, mixed = [], 0
    for lg in sorted(members):
        ids = sorted(i for i in members[lg] if live(i) and role[i] != "EXCL")
        rs = {role[i] for i in ids}
        if rs == {"POOL_L"}:
            eligible.append((lg, ids))
        elif "POOL_L" in rs:
            mixed += 1
    random.Random(f"{seed}:L_A").shuffle(eligible)
    if sum(len(ids) for _, ids in eligible) < L_size:
        fail(f"L_size={L_size} but only {sum(len(ids) for _, ids in eligible)} POOL_L images are eligible")
    L_A, filled = set(), 0
    for lg, ids in eligible:
        if filled + len(ids) <= L_size:
            L_A.update(ids)
            filled += len(ids)
            if filled == L_size:
                break
    rep(f"L_A: {len(L_A)} images drawn from POOL_L in whole near-duplicate clusters "
        f"({mixed} clusters shared with POOL folders were sent to U whole)")
    if len(L_A) != L_size:
        rep(f"WARNING: could not reach exactly {L_size}; got {len(L_A)}")

    U_all = {i for i, r in role.items() if r in ("POOL_L", "POOL") and live(i) and i not in L_A}
    u_groups = defaultdict(list)
    for i in U_all:
        u_groups[imgs[i]["leak_group"]].append(i)
    order = sorted(u_groups)
    random.Random(f"{seed}:U").shuffle(order)
    levels = [str(x) for x in cfg.get("U_levels", ["all"])]
    numeric = sorted(int(x) for x in levels if x != "all")
    if numeric and numeric[-1] > len(U_all):
        fail(f"U level {numeric[-1]} is larger than the whole U pool ({len(U_all)} images)")
    U_sets, chosen, remaining = {}, set(), order
    for lvl in numeric:
        rest = []
        for lg in remaining:
            ids = u_groups[lg]
            if len(chosen) + len(ids) <= lvl:
                chosen.update(ids)
            else:
                rest.append(lg)
        remaining = rest
        U_sets[f"U_{lvl:04d}"] = set(chosen)
        if len(chosen) != lvl:
            rep(f"WARNING: U level {lvl} reached {len(chosen)} images")
    U_sets["U_all"] = set(U_all)          # always written: the teacher labels the full pool later

    TEST = {i for i, r in role.items() if r == "TEST" and live(i)}
    CONTROL = {i for i, r in role.items() if r == "CONTROL" and live(i)}
    L_EXTRA = {i for i, r in role.items() if r == "L_EXTRA" and live(i)}
    rep(f"U: {len(U_all)} images, levels " + ", ".join(f"{k}={len(v)}" for k, v in U_sets.items()))
    rep(f"TEST: {len(TEST)}   CONTROL: {len(CONTROL)}   L_EXTRA: {len(L_EXTRA)}")

    # ---------------------------------------------------------------- 5. size bins
    rep.section("5. SIZE BINS (object size = sqrt(box area) / image long side, called 'geo')")
    test_geo = [a["geo"] for i in TEST for a in anns_by_img[i]
                if kind_of[a["category_id"]][0] == "map" and not a.get("iscrowd")]
    if not test_geo:
        fail("TEST has no scored objects")
    sc = cfg.get("size_cuts", "auto")
    if sc == "auto":
        cuts = [float(v) for v in np.percentile(test_geo, [100 / 3, 200 / 3])]
        cut_source = "auto: tertiles of TEST"
    else:
        cuts = [float(v) for v in sc]
        cut_source = "config"
    cuts = [round(c, 6) for c in cuts]    # the stored cut and the cut used for binning must be identical
    if len(cuts) != 2 or not cuts[0] < cuts[1]:
        fail(f"size_cuts must be two increasing numbers, got {cuts}")

    def size_bin(g):
        return "small" if g < cuts[0] else ("medium" if g < cuts[1] else "large")

    rep(f"cuts {cuts[0]:.4f} / {cuts[1]:.4f}  ({cut_source})")
    rep(f"at input size R={R}: small < {cuts[0] * R:.0f} px, medium {cuts[0] * R:.0f}-{cuts[1] * R:.0f} px, "
        f"large > {cuts[1] * R:.0f} px  (object side length)")
    if cut_source.startswith("auto"):
        rep(f"to freeze these for all future runs, set in the config:  size_cuts: [{cuts[0]:.4f}, {cuts[1]:.4f}]")

    # ---------------------------------------------------------------- 6. stored images
    ridx_path = base / f"resized_{R}" / "resize_index.json"
    ridx = {}
    if ridx_path.exists():
        rj = load_json(ridx_path)
        if int(rj["R"]) != R:
            fail(f"resize_index.json was built for R={rj['R']}, config says R={R}")
        ridx = rj["images"]
    else:
        rep(f"WARNING: {ridx_path.name} not found - split files will point at the original images "
            f"(run resize_cache.py for faster loading)")
    want_resized = set(cfg.get("resize_sources", ["seaclear"])) if ridx else set()
    all_used = L_A | L_EXTRA | U_all | TEST | CONTROL
    gaps = [i for i in all_used if imgs[i]["source"] in want_resized and str(i) not in ridx]
    if gaps:
        fail(f"{len(gaps)} images have no resized copy (resize_cache.py interrupted?) - rerun it")

    def stored(im):
        e = ridx.get(str(im["id"]))
        if e:
            return e["file_name"], int(e["width"]), int(e["height"])
        return im["file_name"], im["width"], im["height"]

    # ---------------------------------------------------------------- 7. write split files
    categories_out = [{"id": k, "name": n, "tier": tier_of(tax, n)} for k, n in enumerate(unified, 1)]
    out_name = f"r{R}_s{seed}_" + re.sub(r"[^A-Za-z0-9_.-]", "_", tax["version"])
    out_dir = base / "splits" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def build(name, ids, role_desc, labels_desc):
        images_out, anns_out = [], []
        per_class, per_bin, class_bin = Counter(), Counter(), Counter()
        n_ignore = n_empty = 0
        for i in sorted(ids):
            im = imgs[i]
            fn, w2, h2 = stored(im)
            sx, sy = w2 / im["width"], h2 / im["height"]
            ignore, kept = [], 0
            for a in anns_by_img[i]:
                kind, k = kind_of[a["category_id"]]
                if kind == "background":
                    continue
                x, y, w, h = a["bbox"]
                box = [round(x * sx, 2), round(y * sy, 2), round(w * sx, 2), round(h * sy, 2)]
                if kind == "ignore" or a.get("iscrowd"):
                    ignore.append(box)
                    continue
                segs = [[round(v * (sx if j % 2 == 0 else sy), 2) for j, v in enumerate(p)]
                        for p in a["segmentation"]]
                b = size_bin(a["geo"])
                c = cats[a["category_id"]]
                anns_out.append({"id": a["id"], "image_id": i, "category_id": k, "bbox": box,
                                 "segmentation": segs, "area": round(box[2] * box[3], 2), "iscrowd": 0,
                                 "has_mask": bool(segs), "geo": a["geo"], "size_bin": b,
                                 "orig_category": f"{c['source']}:{c['name']}"})
                kept += 1
                per_class[k] += 1
                per_bin[b] += 1
                class_bin[(k, b)] += 1
            n_ignore += len(ignore)
            n_empty += kept == 0
            images_out.append({"id": i, "file_name": fn, "width": w2, "height": h2,
                               "scale": [round(sx, 6), round(sy, 6)],
                               "orig_file_name": im["file_name"], "orig_size": [im["width"], im["height"]],
                               "source": im["source"], "group": im["group"],
                               "leak_group": im["leak_group"], "ignore": ignore})
        content = {"images": images_out, "annotations": anns_out}
        info = {"split": name, "role": role_desc, "labels": labels_desc, "R": R, "seed": seed,
                "taxonomy": tax["version"], "n_images": len(images_out), "n_annotations": len(anns_out),
                "n_ignore_zones": n_ignore, "n_images_without_objects": n_empty,
                "image_fingerprint": fingerprint_ids(ids), "content_fingerprint": fingerprint_obj(content),
                "created": now()}
        save_json({"info": info, **content, "categories": categories_out}, out_dir / f"{name}.json")
        return info, per_class, per_bin, class_bin

    plan = [("L_A", L_A, "labelled set for teacher and students", "real")]
    if L_EXTRA:
        plan.append(("L_B", L_A | L_EXTRA, "L_A plus L_EXTRA folders", "real"))
    for name, s in U_sets.items():
        plan.append((name, s, "pool whose labels are hidden in pseudo-label arms", "real (Oracle arm)"))
    plan.append(("TEST", TEST, "evaluation only", "real"))
    if CONTROL:
        plan.append(("CONTROL", CONTROL, "separate evaluation only", "real"))
    prev_meta = out_dir / "split_meta.json"
    prev_fp = {}
    if prev_meta.exists():
        prev_fp = {k: v.get("content_fingerprint") for k, v in load_json(prev_meta).get("splits", {}).items()}
    for old in out_dir.glob("*.json"):
        if SPLIT_FILE.match(old.name):
            old.unlink()                 # never leave a stale split file from an earlier config
    results = {}
    for name, ids, rd, ld in plan:
        results[name] = build(name, ids, rd, ld)

    # ---------------------------------------------------------------- 8. verification
    rep.section("6. VERIFICATION")
    parts = {"L_A": L_A, "L_EXTRA": L_EXTRA, "U": U_all, "TEST": TEST, "CONTROL": CONTROL}
    names_p = list(parts)
    for a_i, a in enumerate(names_p):
        for b in names_p[a_i + 1:]:
            both = parts[a] & parts[b]
            if both:
                fail(f"{len(both)} images are in both {a} and {b}")
    where = defaultdict(set)
    for pname, s in parts.items():
        for i in s:
            where[imgs[i]["leak_group"]].add(pname)
    allowed = ({"L_A", "L_EXTRA"}, {"TEST", "CONTROL"})
    split_lgs = {lg: p for lg, p in where.items() if len(p) > 1 and p not in allowed}
    if split_lgs:
        ex = list(split_lgs.items())[:10]
        fail(f"{len(split_lgs)} near-duplicate clusters / videos ended up in more than one split: {ex}")
    prev = set()
    for name in sorted(k for k in U_sets if k != "U_all"):
        if not prev <= U_sets[name]:
            fail(f"{name} does not contain the smaller U level")
        prev = U_sets[name]
    if not prev <= U_sets["U_all"]:
        fail("U_all does not contain every U level")
    missing_files = [stored(imgs[i])[0] for i in all_used if not (base / stored(imgs[i])[0]).exists()]
    if missing_files:
        fail(f"{len(missing_files)} split images are missing on disk, e.g. {missing_files[:5]}")
    rep("no image in two splits; no near-duplicate cluster or TrashCan video crosses a split; "
        "U levels are nested; every image file exists")

    # ---------------------------------------------------------------- 9. report tables
    rep.section("7. SPLIT SIZES")
    rep(f"{'split':10}{'images':>8}{'objects':>9}{'ignore':>8}{'no-object imgs':>16}")
    for name, (info, *_rest) in results.items():
        rep(f"{name:10}{info['n_images']:>8}{info['n_annotations']:>9}{info['n_ignore_zones']:>8}"
            f"{info['n_images_without_objects']:>16}")

    rep.section(f"8. CLASS SUPPORT (flag = fewer than {min_support} in L_A or TEST)")
    has_lb = "L_B" in results
    rep(f"{'#':>3} {'class':14}{'tier':>8}{'L_A':>7}{'L_B':>7}{'U_all':>7}{'TEST':>7}  flag")
    for k, n in enumerate(unified, 1):
        la = results["L_A"][1][k]
        lb = results["L_B"][1][k] if has_lb else 0
        ua = results["U_all"][1][k]
        te = results["TEST"][1][k]
        flag = "LOW" if la < min_support or te < min_support else ""
        rep(f"{k:>3} {n:14}{(tier_of(tax, n) or '-'):>8}{la:>7}{lb:>7}{ua:>7}{te:>7}  {flag}")

    rep.section("9. OBJECTS PER SIZE BIN")
    rep(f"{'split':10}" + "".join(f"{b:>9}" for b in BINS))
    for name, (_i, _c, per_bin, _cb) in results.items():
        rep(f"{name:10}" + "".join(f"{per_bin[b]:>9}" for b in BINS))
    rep("capture efficiency divides by (Oracle - Baseline); a U level with few small objects makes the "
        "small-bin ratio unreliable")

    rep.section("10. TEST: CLASS x SIZE BIN (cells under 30 are too thin for per-class size claims)")
    cb = results["TEST"][3]
    rep(f"{'class':14}" + "".join(f"{b:>9}" for b in BINS))
    for k, n in enumerate(unified, 1):
        rep(f"{n:14}" + "".join(f"{cb[(k, b)]:>9}" for b in BINS))

    rep.section("11. FINGERPRINTS (identical fingerprints = identical data)")
    for name, (info, *_r) in results.items():
        rep(f"{name:10} images {info['image_fingerprint'][:23]}...  content {info['content_fingerprint'][:23]}...")

    # ---------------------------------------------------------------- 10. meta
    test_long = {max(stored(imgs[i])[1], stored(imgs[i])[2]) for i in TEST}
    if len(test_long) == 1:
        L = test_long.pop()
        area_px = {"small": [0, round((cuts[0] * L) ** 2, 1)],
                   "medium": [round((cuts[0] * L) ** 2, 1), round((cuts[1] * L) ** 2, 1)],
                   "large": [round((cuts[1] * L) ** 2, 1), 1e10]}
        area_note = f"box area in stored TEST image pixels (long side {L}); use for COCOeval areaRng"
    else:
        area_px, area_note = None, "TEST images have different long sides; use each object's size_bin field"
    meta = {
        "created": now(), "R": R, "seed": seed, "output_folder": f"splits/{out_name}",
        "taxonomy": {"version": tax["version"], "file": cfg["taxonomy"], "sha256": sha256_file(tax_path)},
        "master_sha256": sha256_file(master_path),
        "categories": categories_out,
        "size_bins": {"geo_cuts": [round(c, 6) for c in cuts], "source": cut_source,
                      "object_side_px_at_R": [round(cuts[0] * R, 1), round(cuts[1] * R, 1)],
                      "area_px": area_px, "area_note": area_note},
        "splits": {name: dict(info,
                              per_class={unified[k - 1]: v for k, v in sorted(pc.items())},
                              per_bin={b: pb[b] for b in BINS})
                   for name, (info, pc, pb, _cb) in results.items()},
        "dropped_images": [{"id": i, "file_name": imgs[i]["file_name"], "reason": r}
                           for i, r in sorted(dropped.items())],
        "config": config_snapshot(cfg),
    }
    save_json(meta, out_dir / "split_meta.json", pretty=True)
    changed = [n for n, (info, *_r) in results.items()
               if prev_fp and prev_fp.get(n) not in (None, info["content_fingerprint"])]
    if changed and (out_dir / "arms").exists():
        rep(f"WARNING: splits {changed} changed since the last run; label files in arms/ were built from "
            f"the old splits and must be rebuilt")
    rep(f"\nwritten to splits/{out_name}/  in {time.time() - t0:.1f}s")
    rep.save(out_dir / "split_report.txt")


if __name__ == "__main__":
    main()
