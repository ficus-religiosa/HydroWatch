r"""
train.py - overnight training of the HydroWatch showcase detector.

    python train.py --smoke      # ~5-10 min end-to-end check. Run this first.
    python train.py              # overnight: 8 hours, then scores TEST automatically
    python train.py --hours 6
    python train.py --resume     # continue after a crash or power cut (unfinished runs only)
    python train.py --init runs\hydronet\weights\last.pt --name hydronet_more   # keep training a FINISHED run
    python train.py --set mixup=0 cache=disk scale=0.5                          # override any training setting

Model : HydroNet = COCO-pretrained YOLO11-s + P2 head, plus SPD stem, physics conditioning, aux mask head
        and adaptive-NWD assignment (see hydronet.py). 1280 input, fp16.
Ablate: --no-physics --no-spd --no-aux --no-nwd   Candidates: --dysample --ema-attn
Data  : yolo_ds/data.yaml made by prepare_data.py. TEST is never seen until eval_test.py.
"""
import argparse
import yaml
import shutil
import sys
import time
from pathlib import Path

import torch

from hydronet import DEFAULT_OPTS, HydroTrainer

HERE = Path(__file__).resolve().parent
DATA = r"D:\zaynu\Documents\coding\PS\3-1\HydroWatch\yolo_ds\data.yaml"
MODEL_CFG = HERE / "yolo11s-p2.yaml"
PRETRAINED = "yolo11s.pt"          # downloaded automatically on the first run

HYP = dict(
    optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=5e-4,
    cos_lr=True, warmup_epochs=1.0, close_mosaic=3,
    # ROV footage: colour casts vary a lot between sites; the camera is never upside down
    hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=0.0, translate=0.1,
    scale=0.7,        # 0.3-1.7x: TEST objects are several times smaller than training objects
    fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.1,
    copy_paste=0.3,   # "flip" mode pastes objects within the same image, so the water always matches
)


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def as_batch(b):
    return int(b) if b >= 1 else float(b)   # < 1 means "this fraction of VRAM" (AutoBatch)


def train(a, name, overrides=None):
    """Train with automatic fallback to smaller batches if the GPU runs out of memory."""
    from ultralytics import YOLO

    run_dir = HERE / "runs" / name
    last = run_dir / "weights" / "last.pt"
    resume = a.resume
    if resume and not last.exists():
        sys.exit(f"--resume: nothing to resume at {last}")
    if resume:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if ck.get("epoch", -1) == -1 or ck.get("optimizer") is None:
            sys.exit(f"{last} belongs to a run that already FINISHED, so it can't be resumed.\n"
                     f"To keep training it:  python train.py --init {last} --name {name}_more")
    if not resume and last.exists():
        sys.exit(f"{run_dir} already has a run. Use --resume to continue it, or --name something_new.")

    plan = [a.batch] + [b for b in (2, 1) if a.batch < 1 or b < a.batch]
    for b in plan:   # Ultralytics itself halves the batch on OOM, but only in epoch 1
        try:
            if resume:
                print(f"\nResuming {last} with batch={as_batch(b)}")
                YOLO(str(last)).train(resume=True, batch=as_batch(b), workers=a.workers, trainer=HydroTrainer)
            else:
                # --init: start from a finished HydroNet run (keeps its parts and calibration)
                model = YOLO(a.init) if a.init else YOLO(str(MODEL_CFG)).load(PRETRAINED)
                args = dict(data=a.data, imgsz=a.imgsz, epochs=a.epochs or 300,
                            time=None if a.epochs else a.hours, patience=100,
                            batch=as_batch(b), workers=a.workers, device=0, amp=True, cache=False,
                            project=str(HERE / "runs"), name=name, exist_ok=True, seed=a.seed,
                            deterministic=False, plots=True, **HYP)
                if a.init:  # already trained: gentler restart (no big LR jump, no bias-LR spike)
                    args.update(lr0=0.002, warmup_epochs=0.5, warmup_bias_lr=0.0)
                args.update(a.overrides)
                args.update(overrides or {})
                model.train(trainer=HydroTrainer, **args)
            return run_dir / "weights"
        except Exception as e:
            if not is_oom(e):
                raise
            torch.cuda.empty_cache()
            print(f"\n*** GPU out of memory at batch={as_batch(b)} - retrying with a smaller batch ***\n")
            resume = last.exists()
    sys.exit("Still out of memory at batch 2. Free VRAM (see RUN.md) or run with --imgsz 1024.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=float, default=2,
                    help="images per step. 2 fits 6 GB at 1280 with room to spare; try 3 only with VRAM freed. "
                         "(AutoBatch is misled by Windows spilling VRAM into RAM, so it is not used)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default="hydronet")
    ap.add_argument("--epochs", type=int, default=None, help="fixed epoch count instead of a time budget")
    for k in ("physics", "spd", "aux", "nwd"):
        ap.add_argument(f"--no-{k}", action="store_true", help=f"ablation: turn off {k}")
    ap.add_argument("--dysample", action="store_true", help="candidate: learned upsampling P3->P2")
    ap.add_argument("--ema-attn", action="store_true", help="candidate: EMA attention on P2/P3 outputs")
    ap.add_argument("--mask-weight", type=float, default=DEFAULT_OPTS["mask_weight"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="start a new run from a finished HydroNet checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override training settings, e.g. --set mixup=0 cache=disk scale=0.5 lr0=0.005")
    ap.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("PyTorch cannot see the GPU. Check: python -c \"import torch; print(torch.cuda.is_available())\"")
    a.overrides = {}
    for kv in a.set:
        if "=" not in kv:
            sys.exit(f"--set expects KEY=VALUE, got '{kv}'")
        k, v = kv.split("=", 1)
        a.overrides[k] = yaml.safe_load(v)
    if a.init and not Path(a.init).exists():
        sys.exit(f"--init: {a.init} not found")
    torch.backends.cudnn.benchmark = True  # training images are always 1280x1280: let cuDNN pick its fastest kernels
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {p.name} | {free / 2**30:.1f} of {total / 2**30:.1f} GB free | torch {torch.__version__}")
    if free / total < 0.85:
        print("WARNING: something else is using VRAM (browser, external monitor on the GPU?). "
              "Close it for a faster, safer run - see RUN.md.")
    try:  # Windows can silently spill VRAM into system RAM: no error, just 10-50x slower training
        probe = torch.empty(int(total * 1.15), dtype=torch.uint8, device="cuda")
        del probe
        torch.cuda.empty_cache()
        print("WARNING: the GPU driver spills into system RAM instead of reporting out-of-memory.\n"
              "  If memory ever fills, training will crawl instead of recovering. Fix (1 min, see RUN.md):\n"
              "  NVIDIA Control Panel > Manage 3D settings > CUDA - Sysmem Fallback Policy > Prefer No Sysmem Fallback")
    except RuntimeError:
        torch.cuda.empty_cache()   # good: a real out-of-memory error, which train() recovers from
    try:  # a CPU-only torchvision breaks GPU NMS, but only once validation starts - catch it now
        import torchvision
        torchvision.ops.nms(torch.rand(8, 4, device="cuda") * 100, torch.rand(8, device="cuda"), 0.5)
    except Exception as e:
        sys.exit(f"torchvision cannot run on the GPU ({type(e).__name__}). Fix:\n"
                 "  pip install torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124")
    if not Path(a.data).exists():
        sys.exit(f"{a.data} not found - run  python prepare_data.py  first.")

    HydroTrainer.hydro_opts = {**DEFAULT_OPTS, "physics": not a.no_physics, "spd": not a.no_spd,
                               "aux": not a.no_aux, "nwd": not a.no_nwd, "dysample": a.dysample,
                               "ema_attn": a.ema_attn, "mask_weight": a.mask_weight}
    if a.init:
        print(f"Continuing from {a.init}: parts and calibration come from that checkpoint (--no-* flags ignored)")
    else:
        print("HydroNet parts:", ", ".join(f"{k}={v}" for k, v in HydroTrainer.hydro_opts.items()))
    if a.overrides:
        print("Overrides:", a.overrides)
    t0 = time.time()
    if a.smoke:
        shutil.rmtree(HERE / "runs" / "smoke", ignore_errors=True)
        a.resume = False
        weights = train(a, "smoke", {"epochs": 1, "time": None, "fraction": 0.03, "plots": False})
        from eval_test import evaluate
        evaluate(weights / "last.pt", limit=40)
        print(f"\nSMOKE TEST PASSED in {(time.time() - t0) / 60:.1f} min. Now run:  python train.py")
        return

    weights = train(a, a.name)
    print(f"\nTraining finished in {(time.time() - t0) / 3600:.2f} h. Weights: {weights}")
    from eval_test import evaluate
    evaluate(weights / "best.pt")


if __name__ == "__main__":      # required on Windows (dataloader workers)
    main()
