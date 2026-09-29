"""
count_instances.py - per-class instance counts across TrashCan + SeaClear,
broken down by the roles defined in ROLES.

Views
  1. FINE     original class names, per source              (always)
  2. UNIFIED  classes folded through a taxonomy JSON         (if taxonomy_path is set)
  3. DETAIL   per-folder breakdown for chosen classes        (if class_filter is set)

Jupyter:
  %run count_instances.py
  or
  from count_instances import main
  main(taxonomy_path=TAX, class_filter=["tire_rubber"])
"""
import json
import os
from collections import Counter, defaultdict

import numpy as np

# ------------------------------------------------------------------ CONFIG
BASE = os.environ.get(
    "HYDRO_BASE", r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets"
)
TRASHCAN_JSONS = [
    os.path.join(BASE, "DS1", "instance_version", "instances_train_trashcan.json"),
    os.path.join(BASE, "DS1", "instance_version", "instances_val_trashcan.json"),
]
SEACLEAR_ROOT = os.path.join(BASE, "DS3")
SEACLEAR_JSON = os.path.join(SEACLEAR_ROOT, "dataset.json")

ROLES = {
    "POOL": [
        "Bistrina/Bluerobotics HD",
        "Bistrina/Paralenz Vaquita Gen 2",
        "Bistrina/SIP-E323CV",
    ],
    "TEST": [
        "Jakljan/Bluerobotics HD",
        "Lokrum/Bluerobotics HD",
        "Lokrum/Paralenz Vaquita Gen 2",
        "Lokrum/SIP-E323CV",
        "Slano/Bluerobotics HD",
        "Slano/Paralenz Vaquita",
    ],
    "EXCL": [
        "Marseille/SIP-E323CV",
        "Jakljan/Paralenz Vaquita",
    ],
}
L_SOURCE_GROUP = "Bistrina/Paralenz Vaquita Gen 2"   # L_A is drawn from this folder
L_SIZE = 1000                                        # images in L_A

TAXONOMY_PATH = None      # e.g. os.path.join(BASE, "taxonomy_v0.json")
CLASS_FILTER = None       # e.g. ["tire_rubber", "net_plastic"]
MIN_SUPPORT = 50          # instances needed in L and in TEST
UNSEEN_MIN = 10           # source class with fewer than this in L_A counts as unseen
IMG_EXT = (".jpg", ".jpeg", ".png")


# ------------------------------------------------------------------ HELPERS
def _geo(ann, img):
    """sqrt(w*h) / long side: object size as a fraction of the letterboxed input."""
    b = ann.get("bbox")
    if not b or len(b) < 4:
        return None
    w, h = float(b[2]), float(b[3])
    if w <= 0 or h <= 0:
        return None
    return (w * h) ** 0.5 / max(img["width"], img["height"])


def _med(vals):
    vals = [v for v in vals if v is not None]
    return f"{np.median(vals):.4f}" if vals else "     -"


def _seaclear_groups(root):
    """filename -> 'Site/Camera' by walking the folder tree."""
    fmap, dup = {}, 0
    for dp, _, files in os.walk(root):
        parts = os.path.relpath(dp, root).split(os.sep)
        if len(parts) < 2:
            continue
        group = f"{parts[0]}/{parts[1]}"
        for f in files:
            if f.lower().endswith(IMG_EXT):
                if f in fmap:
                    dup += 1
                fmap[f] = group
    if dup:
        print(f"WARNING: {dup} SeaClear filenames exist in more than one folder - group map is ambiguous")
    return fmap


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ------------------------------------------------------------------ COLLECT
def collect():
    recs, warn = [], Counter()
    n_img = Counter()
    role_of = {g: r for r, gs in ROLES.items() for g in gs}

    for p in TRASHCAN_JSONS:
        d = _load(p)
        imgs = {i["id"]: i for i in d["images"]}
        cats = {c["id"]: c["name"] for c in d["categories"]}
        tag = os.path.splitext(os.path.basename(p))[0]
        n_img["trashcan"] += len(imgs)
        for a in d["annotations"]:
            im = imgs.get(a["image_id"])
            if im is None:
                warn["trashcan: annotation points to missing image"] += 1
                continue
            if a["category_id"] not in cats:
                warn["trashcan: annotation points to missing category"] += 1
                continue
            recs.append(dict(source="trashcan", group="trashcan", role="TRASHCAN",
                             cls=cats[a["category_id"]], img=(tag, a["image_id"]),
                             geo=_geo(a, im)))

    fmap = _seaclear_groups(SEACLEAR_ROOT)
    d = _load(SEACLEAR_JSON)
    imgs = {i["id"]: i for i in d["images"]}
    cats = {c["id"]: c["name"] for c in d["categories"]}
    img_group = {}
    for iid, im in imgs.items():
        g = fmap.get(im["file_name"])
        if g is None:
            warn["seaclear: image in json but not on disk"] += 1
            g = "?"
        img_group[iid] = g
        n_img[g] += 1
    for a in d["annotations"]:
        im = imgs.get(a["image_id"])
        if im is None:
            warn["seaclear: annotation points to missing image"] += 1
            continue
        if a["category_id"] not in cats:
            warn["seaclear: annotation points to missing category"] += 1
            continue
        g = img_group[a["image_id"]]
        recs.append(dict(source="seaclear", group=g, role=role_of.get(g, "UNASSIGNED"),
                         cls=cats[a["category_id"]], img=("seaclear", a["image_id"]),
                         geo=_geo(a, im)))

    unassigned = sorted({r["group"] for r in recs if r["role"] == "UNASSIGNED"})
    if unassigned:
        print("WARNING: SeaClear groups not listed in ROLES (shown as UNASSIGNED):", unassigned)
    for k, v in warn.items():
        print(f"WARNING: {k}: {v}")
    return recs, n_img


# ------------------------------------------------------------------ VIEWS
def _banner(title):
    print("\n" + "=" * 86 + f"\n{title}\n" + "=" * 86)


def summary(n_img):
    _banner("IMAGES PER ROLE")
    for role, groups in ROLES.items():
        total = sum(n_img.get(g, 0) for g in groups)
        print(f"  {role:6} {total:>6} imgs")
        for g in groups:
            print(f"         {g:38} {n_img.get(g, 0):>6}")
    print(f"  {'TC':6} {n_img.get('trashcan', 0):>6} imgs (TrashCan train+val)")


def fine_view(recs, la_frac, min_support):
    _banner("TRASHCAN - original classes (all of it goes to L_B)")
    tc = [r for r in recs if r["source"] == "trashcan"]
    cnt = Counter(r["cls"] for r in tc)
    imgs, geos = defaultdict(set), defaultdict(list)
    for r in tc:
        imgs[r["cls"]].add(r["img"])
        geos[r["cls"]].append(r["geo"])
    print(f"{'class':28}{'anns':>7}{'imgs':>7}{'med_geo':>9}")
    for c, n in cnt.most_common():
        print(f"{c:28}{n:>7}{len(imgs[c]):>7}{_med(geos[c]):>9}")
    print(f"{'TOTAL':28}{sum(cnt.values()):>7}")

    _banner("SEACLEAR - original classes by role   (L_A~ = proportional estimate)")
    sc = [r for r in recs if r["source"] == "seaclear"]
    by, la, geos = defaultdict(Counter), Counter(), defaultdict(list)
    for r in sc:
        by[r["cls"]][r["role"]] += 1
        if r["group"] == L_SOURCE_GROUP:
            la[r["cls"]] += 1
        if r["role"] in ("POOL", "TEST"):
            geos[r["cls"]].append(r["geo"])
    order = sorted(by, key=lambda c: -(by[c]["POOL"] + by[c]["TEST"]))
    print(f"{'class':28}{'L_A~':>7}{'POOL':>7}{'TEST':>7}{'EXCL':>7}{'med_geo':>9}  flag")
    for c in order:
        lae = la[c] * la_frac
        ok = lae >= min_support and by[c]["TEST"] >= min_support
        print(f"{c:28}{lae:>7.0f}{by[c]['POOL']:>7}{by[c]['TEST']:>7}{by[c]['EXCL']:>7}"
              f"{_med(geos[c]):>9}  {'OK' if ok else 'low'}")


def unified_view(recs, la_frac, tax, min_support, unseen_min):
    m = tax.get("map", {})
    ign = {s: set(v) for s, v in tax.get("ignore", {}).items()}
    bg = {s: set(v) for s, v in tax.get("background", {}).items()}
    order = tax["unified_classes"]
    tiers = tax.get("tiers")
    tier_of = {c: t for t, cs in tiers.items() for c in cs} if tiers else {}
    scored = set(tiers["target"]) if tiers and "target" in tiers else set(order)

    # ---- validation: every source class must be mapped or explicitly sent to background
    bad = False
    seen = defaultdict(set)
    for r in recs:
        seen[r["source"]].add(r["cls"])
    for s, classes in sorted(seen.items()):
        mapped, dropped, ignored = set(m.get(s, {})), bg.get(s, set()), ign.get(s, set())
        missing = sorted(classes - mapped - dropped - ignored)
        if missing:
            bad = True
            print(f"ERROR: {s} classes not in map, ignore or background: {missing}")
        both = sorted(mapped & dropped)
        if both:
            bad = True
            print(f"ERROR: {s} classes both mapped and in background: {both}")
        ign_bg = sorted(ignored & dropped)
        if ign_bg:
            bad = True
            print(f"ERROR: {s} classes both ignored and in background: {ign_bg}")
    for s, mm in m.items():
        for src_c, tgt in mm.items():
            if tgt not in order:
                bad = True
                print(f"ERROR: {s}.{src_c} -> '{tgt}' is not in unified_classes")
    if tiers:
        untiered = [c for c in order if c not in tier_of]
        stray = [c for c in tier_of if c not in order]
        if untiered or stray:
            bad = True
            print(f"ERROR: tiers mismatch - untiered: {untiered}, not in unified_classes: {stray}")
    if bad:
        print("Fix the taxonomy before trusting the table below.")

    # ---- per-subclass L_A presence, for the unseen check
    la_sub = Counter(r["cls"] for r in recs
                     if r["source"] == "seaclear" and r["group"] == L_SOURCE_GROUP)

    agg, geos, bg_ct, ign_ct = defaultdict(Counter), defaultdict(list), Counter(), Counter()
    kept_test_geo = []
    for r in recs:
        s, c = r["source"], r["cls"]
        if c in bg.get(s, set()):
            bg_ct[r["role"]] += 1
            continue
        if c in ign.get(s, set()):
            if c in m.get(s, {}):                 # old style: crowd region inside a class
                agg[m[s][c]]["IGN"] += 1
            else:                                 # v2 style: class-agnostic don't-care zone
                ign_ct[r["role"]] += 1
            continue
        u = m.get(s, {}).get(c, "!!UNMAPPED")
        if s == "trashcan":
            agg[u]["TC"] += 1
            continue
        agg[u][r["role"]] += 1
        if r["group"] == L_SOURCE_GROUP:
            agg[u]["LA_raw"] += 1
        if r["role"] == "TEST":
            geos[u].append(r["geo"])
            if r["geo"] is not None and u in scored:
                kept_test_geo.append(r["geo"])
            if la_sub[c] * la_frac < unseen_min:
                agg[u]["UNSEEN"] += 1

    _banner(f"UNIFIED - taxonomy '{tax.get('version', '?')}'   "
            f"A/B = usable with L_A/L_B (>= {min_support} in L and TEST)")
    print(f"{'class':14}{'tier':>5}{'TC':>7}{'L_A~':>7}{'L_B~':>7}{'POOL':>7}{'TEST':>7}"
          f"{'EXCL':>7}{'IGN':>6}{'unseen':>8}{'geoTEST':>9}{'A':>3}{'B':>3}")
    rows = order + sorted(k for k in agg if k not in order)
    for u in rows:
        a = agg[u]
        lae = a["LA_raw"] * la_frac
        lbe = lae + a["TC"]
        pa = "Y" if lae >= min_support and a["TEST"] >= min_support else "-"
        pb = "Y" if lbe >= min_support and a["TEST"] >= min_support else "-"
        uns = f"{100 * a['UNSEEN'] / a['TEST']:.0f}%" if a["TEST"] else "-"
        tier = (tier_of.get(u) or "-")[:4]
        print(f"{u:14}{tier:>5}{a['TC']:>7}{lae:>7.0f}{lbe:>7.0f}{a['POOL']:>7}{a['TEST']:>7}"
              f"{a['EXCL']:>7}{a['IGN']:>6}{uns:>8}{_med(geos[u]):>9}{pa:>3}{pb:>3}")

    judged = [u for u in order if u in scored]
    n_a = sum(1 for u in judged if agg[u]["LA_raw"] * la_frac >= min_support
              and agg[u]["TEST"] >= min_support)
    print(f"\n{n_a}/{len(judged)} scored classes usable with L_A.")
    print(f"unseen = share of a class's TEST instances whose source class has < {unseen_min} "
          f"instances in L_A (teacher effectively never saw it).")
    print(f"background (annotation dropped, pixels become background): "
          f"POOL {bg_ct['POOL']}, TEST {bg_ct['TEST']}, EXCL {bg_ct['EXCL']}, TrashCan {bg_ct['TRASHCAN']}")
    print(f"ignore zones (no class, no reward, no penalty): "
          f"POOL {ign_ct['POOL']}, TEST {ign_ct['TEST']}, EXCL {ign_ct['EXCL']}, TrashCan {ign_ct['TRASHCAN']}")

    if kept_test_geo:
        g = np.array(kept_test_geo)
        p = {q: np.percentile(g, q) for q in (5, 10, 25, 33.3, 50, 66.7)}
        scope = "SCORED"
        print(f"\nTEST geo percentiles for {scope} classes (n={len(g)}) - redo the R choice on these:")
        print("  " + "  ".join(f"p{q:g}={v:.4f}" for q, v in p.items()))
        r_need = 8 / p[10]
        print(f"  R for p10 on 2 cells at stride 4: {r_need:.0f} -> round up to {int(np.ceil(r_need / 32) * 32)}")
        t1, t2 = np.percentile(g, [100 / 3, 200 / 3])       # same rule make_splits.py uses
        print(f"  size-bin tertile cuts (normalized): {t1:.4f} / {t2:.4f}")


def detail(recs, names):
    for name in names:
        rs = [r for r in recs if r["cls"] == name]
        _banner(f"DETAIL - {name}")
        if not rs:
            print("  no instances in any source")
            continue
        geos, imgs, role = defaultdict(list), defaultdict(set), {}
        for r in rs:
            k = f"{r['source']}:{r['group']}"
            geos[k].append(r["geo"])
            imgs[k].add(r["img"])
            role[k] = r["role"]
        print(f"  {'source:group':44}{'role':>10}{'anns':>7}{'imgs':>7}{'med_geo':>9}")
        for k in sorted(geos, key=lambda x: -len(geos[x])):
            print(f"  {k:44}{role[k]:>10}{len(geos[k]):>7}{len(imgs[k]):>7}{_med(geos[k]):>9}")
        print(f"  {'TOTAL':44}{'':>10}{len(rs):>7}")


# ------------------------------------------------------------------ MAIN
def main(taxonomy_path=TAXONOMY_PATH, class_filter=CLASS_FILTER, min_support=MIN_SUPPORT,
         unseen_min=UNSEEN_MIN):
    recs, n_img = collect()
    n_src = n_img.get(L_SOURCE_GROUP, 0)
    la_frac = L_SIZE / n_src if n_src else 0.0
    if la_frac > 1.0:
        print(f"WARNING: L_SIZE={L_SIZE} > {n_src} imgs in {L_SOURCE_GROUP}; L_A~ capped at the whole folder")
        la_frac = 1.0
    print(f"Loaded {len(recs)} annotations. L_A~ = counts in {L_SOURCE_GROUP} x {L_SIZE}/{n_src}")

    summary(n_img)
    fine_view(recs, la_frac, min_support)
    if taxonomy_path:
        unified_view(recs, la_frac, _load(taxonomy_path), min_support, unseen_min)
    if class_filter:
        detail(recs, class_filter)


if __name__ == "__main__":
    main()
