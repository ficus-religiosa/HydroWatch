# HydroNet — run guide

**What it is:** COCO-pretrained YOLO11-s + a stride-4 (P2) head, with four underwater additions. Every
addition starts as an exact no-op, so at step 0 the model *is* the pretrained network, and each part only
changes it as far as training finds useful.

| Part | What it does | Cost at inference |
|---|---|---|
| **SPD stem** | Replaces layer 0 with space-to-depth + 3×3 conv, initialised to reproduce the pretrained layer 0 exactly, but able to use the full 2×2 pixel detail — more capacity for tiny objects | ~+2% |
| **Physics conditioning** | 3 macro maps (veiling from G/B only, absorption over ~124 px, blur via Laplacian energy) + global colour cast → tiny depthwise-separable encoder → per-position scale/shift on backbone P2/P3/P4 | ~+1% |
| **Aux mask head** | Training-only foreground mask at stride 4 — rope, net and bag boxes are mostly seabed, masks say which pixels are object | **zero** (not run) |
| **Adaptive NWD** | Assignment scores matches by max(CIoU, scale-adaptive NWD): much kinder to tiny boxes, identical to IoU on large ones | **zero** (training only) |
| Data | Scale augmentation 0.3–1.7× (TEST objects are smaller than training objects); copy-paste within the same image | — |

Optional candidates, off by default: `--dysample` (learned P3→P2 upsampling), `--ema-attn` (attention on P2/P3 outputs).

**What I verified** on the same torch 2.6.0 / torchvision 0.21 / Ultralytics 8.4.160 stack (on CPU):
the no-op starts (physics and attention exactly zero change; SPD stem within float rounding, ≤0.03 px),
every physics map responding to its own effect and not the others, full training with a crash and resume,
fp16/bf16 mixed-precision training with gradients reaching every new part, candidate modules, the
all-off ablation, saving/reloading, prediction (mask head confirmed not running), and TEST scoring.
**Not verified:** actual CUDA execution — the smoke test is that check.

---

## Oracle split (v3) — in-distribution, leak-free

```
pip install ImageHash
python make_oracle_split.py
python prepare_data.py --splits D:\zaynu\Documents\coding\PS\3-1\HydroWatch\datasets\splits\oracle_v3 --train TRAIN --val-split VAL
python train.py --name oracle_v3 --hours 21
```
Read `split_report.txt` before training. NWD is now off by default (`--nwd` turns it on).
After this, `eval_test.py` scores the v3 TEST automatically. To re-score an OLD run on its own TEST,
pass `--splits ...\splits\r1280_s0_v2-debris-plus-biota`.

## 1. Rebuild the dataset — required

The labels are now object **outlines** (the mask head and copy-paste need them):
```
python prepare_data.py
```
Check the new line `outlines: N polygons, M box-shaped fallbacks (x%)`. Under ~5% is fine. Much more means
many objects have no usable mask, and the mask head learns box-shaped masks for them.

## 2. Smoke test (~5 min)
```
python train.py --smoke
```
Look for:
- `HydroNet parts: physics=True, spd=True, aux=True, nwd=True, ...`
- `HydroNet: calibrating physics normalisation on 64 training images...` then one line of statistics
- a `mask_loss` column in the training table
- `GPU_mem` — expect ~3.5–4.5 GB at batch 2. If it's near 5.5 GB, use `--batch 1` or free VRAM
- ends with `SMOKE TEST PASSED`

## 3. Overnight
```
python train.py                      # 8 h time budget
python train.py --epochs 40          # or a fixed epoch count
```
Results print at the end and go to `runs\hydronet\weights\test_eval.json`.

**Compare against your baseline** (`runs\oracle_p2_1280`) on TEST **debris-only** numbers, at the same time
budget. Validation is the training site, so don't judge on it.

## 4. Ablations — to learn which parts earn their place
Add `--seed 1`, `--seed 2` to repeat a configuration: the spread between seeds is the smallest difference you can trust.
Each switches off exactly one part; use a new `--name` for each:
```
python train.py --no-physics --name abl_nophys
python train.py --no-aux     --name abl_noaux
python train.py --no-nwd     --name abl_nonwd
python train.py --no-spd     --name abl_nospd
python train.py --dysample --ema-attn --name cand
```
**Keep a part** if removing it lowers debris-only mAP50-95 by ≥ 0.5 without AP_large dropping by more than 0.5.
With limited nights, use `--hours 4` and compare runs at the same budget.

## 5. Keep training a finished run

`--resume` only continues **unfinished** runs (crash, power cut, Ctrl+C). When a run finishes, Ultralytics
deletes its optimizer state, so there is nothing to resume. To train it further, start a new run from it:
```
python train.py --init runs\hydronet\weights\last.pt --name hydronet_more --hours 10
```
This keeps the model's parts and physics calibration, and restarts gently: learning rate 0.002 instead of
0.01, short warmup, no bias-LR spike — the finished run ended at 0.0001, and jumping back to 0.01 would undo
part of what it learned. Worth it only if `runs\hydronet\results.csv` shows validation mAP still rising at the end.

## 6. Speed

Measure first (3–5 min, no training):
```
python bench.py
```
It times CPU data loading against one GPU training step, and the GPU cost of each HydroNet part, then says
which one is holding training back. Training runs at the speed of the slower one.

| Verdict | What helps | Effect on results |
|---|---|---|
| GPU-bound | Free VRAM (unplug monitors / Hybrid mode), then `--batch 3` | None |
| Data-bound | `--set cache=disk` (stores decoded images; needs ~20+ GB free on D:) | None |
| Data-bound | `--set mixup=0` (MixUp builds a whole second mosaic for 10% of samples) | Small |
| Data-bound | `--set copy_paste=0.1` | Small |
| Either | Only if ablation shows the mask head doesn't help: `--no-aux --set copy_paste=0` (boxes only, lighter loading) | Loses that part |

cuDNN kernel autotuning is now always on (free; training images are always 1280×1280). With `--hours`,
a faster setup gives you **more epochs in the same night**, not a shorter night.

## 7. If TEST looks far worse than validation
```
python diagnose_test.py --weights runs\hydronet_more\weights\best.pt
```
Checks whether TEST.json's image sizes and boxes agree with the files on disk, re-runs the scorer on your
validation images (it should reproduce Ultralytics' validation number), and draws truth (green) vs
predictions (red) on a few TEST images into `diag_test\`. Its verdict says whether the gap is an
evaluation problem or a real cross-site failure.

## 8. Resume, predict, re-score
```
python train.py --resume             # after a crash (unfinished runs only)
python predict_image.py
python eval_test.py --weights runs\hydronet\weights\best.pt
```
Checkpoints need `hydronet.py` next to the scripts to load.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `torchvision cannot run on the GPU` | `pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124` |
| Out of memory | `--batch 1`, or `--no-spd`, or `--imgsz 1024` |
| Much slower than the old model | Outlines make data loading heavier — try `--workers 6`; check CPU isn't at 100% |
| `already has a run` | `--resume`, or a new `--name` |
| Worker / BrokenPipe errors | `--workers 0` |

## For Paper 2
Still treats ROV parts as background and uses the Ultralytics trainer, so this is **not** a Paper 2 arm.
