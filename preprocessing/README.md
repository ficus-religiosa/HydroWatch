# HydroWatch data pipeline

Put every file from this folder into `datasets\` (next to `DS1\` and `DS3\`).

Install once: `pip install numpy pillow imagehash pyyaml` (PyTorch only where you train).

## Run order

| Step | Command | When | Check afterwards |
|---|---|---|---|
| 1 | `python build_manifest.py` | once | `manifest\build_report.txt`, and open the images in `manifest\dup_review\` |
| 2 | `python resize_cache.py` | once per R | `resized_1280\resize_report.txt` |
| 3 | `python make_splits.py` | whenever the config or taxonomy changes (seconds) | `splits\<run>\split_report.txt` |

After the first successful step 3, copy the printed `size_cuts: [..., ...]` line into `hydro_config.yaml`
so the small / medium / large bins never move again.

If step 3 stops with "training and evaluation images are near-duplicates", look at the listed files.
If they really are copies, set `cross_role_duplicates: drop_from_train` and rerun: the training-side copies are
removed and listed in `split_meta.json`. Test images are never removed.

## What you edit

- `hydro_config.yaml`: which folder goes where (`groups:`), R, seed, L_A size, U sizes.
- `taxonomy_v2.json`: how original classes map to the 12 classes, and which become ignore zones.

Originals in `DS1\` and `DS3\` are only ever read.

## Folder layout after running

```
datasets\
  DS1\ DS3\                       originals, untouched
  hydro_config.yaml  taxonomy_v2.json  taxonomy_v2_debris_only.json
  manifest\                       step 1
    master.json                   every image + annotation, original classes, cleaned boxes, unique ids
    groups.json                   folder -> image ids
    dup_clusters.json             near-duplicate clusters (2+ images)
    phash_cache.json              image fingerprints, reused on reruns
    dup_review\*.png              pairs near the duplicate threshold, to check by eye
    build_report.txt
  resized_1280\DS3\...            step 2: SeaClear at long side 1280 (TrashCan is enlarged when loaded)
    resize_index.json
  splits\r1280_s0_v2-debris-plus-biota\   step 3
    L_A.json L_B.json U_0500.json U_1000.json U_1800.json U_all.json TEST.json CONTROL.json
    split_meta.json               size-bin cuts, COCOeval area ranges, fingerprints, per-class counts
    split_report.txt
```

Ids: the leading digit tells the source. 1xxxxxx = TrashCan train, 2xxxxxx = TrashCan val, 3xxxxxx = SeaClear.

## Using the splits in training

Copy `hydro_data.py` into the training project. Split files are self-contained (paths + labels); the reader
never opens `master.json`.

```python
from hydro_data import HydroDataset, collate, unletterbox, drop_in_ignore

base   = r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets"
splits = base + r"\splits\r1280_s0_v2-debris-plus-biota"

baseline = HydroDataset(["L_A"], splits, base)            # Baseline arm
oracle   = HydroDataset(["L_A", "U_1000"], splits, base)  # Oracle arm, U = 1000
baseline_b = HydroDataset(["L_B"], splits, base)          # Baseline with TrashCan added
test     = HydroDataset(["TEST"], splits, base)
```

Arms (experiment variants) are combinations of split files. Pseudo and Random-drop arms need a trained teacher,
so they come later from `make_arms.py`, over the same U image lists.

Two things your training code must do:

1. **Ignore zones.** Each sample has `ignore` boxes (ROV parts). The loss must give no reward and no penalty to
   predictions whose centre lies inside one. If it skips this, ROV areas silently become background.
2. **Scoring.** Map predictions back with `unletterbox()`, remove those inside ignore zones with
   `drop_in_ignore()`, then run COCOeval with `areaRng` from `split_meta.json` (`size_bins.area_px`), or group by
   each object's `size_bin` field. Report every result with and without `fish`.

## The "do animals hurt debris detection" check

Freeze `size_cuts` in the config first, so both runs use identical size bins. Then set
`taxonomy: "taxonomy_v2_debris_only.json"`, run `make_splits.py` (it writes a separate folder), train one Baseline on each, compare debris scores. The only difference between the two is whether
animals and plants are labelled.

## count_instances.py

Quick class and size counts straight from the original files, for trying a taxonomy before making splits:
`python -c "from count_instances import main; main(taxonomy_path='taxonomy_v2.json')"`
