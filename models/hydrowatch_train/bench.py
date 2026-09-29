"""
bench.py - where does training time go? (~3-5 minutes, no training happens)

    python bench.py

Measures, at your real settings (1280, batch 2, full augmentation):
  1. DATA   - how fast the CPU workers can produce batches (mosaic, outlines, copy-paste...)
  2. GPU    - how long one training step takes on the GPU (synthetic batch, so no waiting on data),
              for full HydroNet and with each part switched off
Training can only go as fast as the slower of the two.
"""
import argparse
import time

import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from hydronet import DEFAULT_OPTS, HydroDetectionModel
from train import DATA, HYP, MODEL_CFG


def data_time(args, task, device):
    cfg = get_cfg(overrides=dict(data=args.data, imgsz=args.imgsz, batch=args.batch, task=task, **HYP))
    data = check_det_dataset(args.data)
    ds = build_yolo_dataset(cfg, data["train"], args.batch, data, mode="train", stride=32)
    dl = build_dataloader(ds, args.batch, args.workers, shuffle=True, device=device)
    it = iter(dl)
    for _ in range(max(3, args.workers)):  # let every worker start up
        next(it)
    t = time.perf_counter()
    for _ in range(args.steps * 3):
        next(it)
    return (time.perf_counter() - t) / (args.steps * 3)


def gpu_time(args, opts, device):
    m = HydroDetectionModel(str(MODEL_CFG), nc=12, verbose=False).hydro_setup(**opts)
    m.args = get_cfg(overrides=HYP)
    m.to(device).train()
    cuda = device.type == "cuda"
    opt = torch.optim.SGD(m.parameters(), lr=1e-5, momentum=0.9)
    scaler = torch.amp.GradScaler(device.type, enabled=cuda)
    b, s, n = args.batch, args.imgsz, 12
    g = torch.Generator().manual_seed(0)
    batch = {"img": torch.rand(b, 3, s, s, generator=g).to(device),
             "batch_idx": torch.arange(b).repeat_interleave(n).float().to(device),
             "cls": torch.randint(0, 12, (b * n, 1), generator=g).float().to(device),
             "bboxes": (torch.rand(b * n, 4, generator=g) * torch.tensor([0.8, 0.8, 0.15, 0.15]) + 0.05).to(device),
             "masks": (torch.rand(b, s // 4, s // 4, generator=g) > 0.97).float().to(device)}

    def step():
        with torch.autocast(device.type, enabled=cuda):
            loss, _ = m.loss(batch)
        scaler.scale(loss.sum()).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.steps):
        step()
    if cuda:
        torch.cuda.synchronize()
    out = (time.perf_counter() - t) / args.steps
    del m, opt
    if cuda:
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=15)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device {device} | imgsz {args.imgsz} | batch {args.batch} | workers {args.workers}\n")

    print("DATA (CPU workers, full augmentation)")
    d_seg = data_time(args, "segment", device)
    print(f"  with outlines    {d_seg:.3f} s/batch   (what training uses)")
    d_det = data_time(args, "detect", device)
    print(f"  boxes only       {d_det:.3f} s/batch   (no mask head, no copy-paste)\n")

    print("GPU (one training step)")
    configs = [("full HydroNet", {}), ("- physics", {"physics": False}), ("- SPD stem", {"spd": False}),
               ("- mask head", {"aux": False}), ("- NWD", {"nwd": False}),
               ("all parts off", {"physics": False, "spd": False, "aux": False, "nwd": False})]
    times = {}
    for name, off in configs:
        times[name] = gpu_time(args, {**DEFAULT_OPTS, **off}, device)
        print(f"  {name:14s} {times[name]:.3f} s/step")
    full = times["full HydroNet"]
    print(f"\n  HydroNet parts together cost {100 * (full / times['all parts off'] - 1):+.0f}% GPU time")

    print("\nVERDICT")
    if d_seg > full * 1.1:
        print(f"  DATA-BOUND: the GPU waits {d_seg - full:.3f} s per step for the CPU "
              f"(~{100 * (1 - full / d_seg):.0f}% idle).")
        print("  Speed-ups that help: --set cache=disk, --set mixup=0, fewer outlines (see RUN.md).")
    else:
        print("  GPU-BOUND: data keeps up. Speed-ups that help: free VRAM and use --batch 3;")
        print("  CPU-side changes won't help.")
    print(f"  Expected: ~{max(d_seg, full) * 13930 / args.batch / 60:.0f} min per epoch")


if __name__ == "__main__":  # required on Windows (dataloader workers)
    main()
