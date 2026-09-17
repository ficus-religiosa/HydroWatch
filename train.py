import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from models import HydroWatch
from models.checkpoint import save_checkpoint
from models.losses import DetectionLoss
from datasets import (
    TrashCanDataset,
    SeaClearDataset,
    UPPDDataset,
    UnifiedMarineDebrisDataset,
    collate_detection_batch,
)


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_config(config):
    classes = config.get("unified_classes", [])
    if not classes:
        raise RuntimeError(
            "Configure unified_classes and class_mapping in dataset_config.json "
            "before training. Run inspect_datasets.py first."
        )

    mappings = config.get("class_mapping", {})
    for name in ("trashcan", "seaclear", "uppd"):
        if not mappings.get(name):
            raise RuntimeError(f"Missing class_mapping for {name}.")
        for raw, mapped in mappings[name].items():
            if not isinstance(mapped, int) or not 0 <= mapped < len(classes):
                raise RuntimeError(f"Invalid mapping {name}:{raw} -> {mapped}.")
    return len(classes)


def build_dataset(config):
    size = tuple(config["input_size"])
    mapping = config["class_mapping"]

    return UnifiedMarineDebrisDataset([
        TrashCanDataset(
            config["datasets"]["trashcan"]["root"],
            target_size=size,
            class_mapping=mapping["trashcan"],
        ),
        SeaClearDataset(
            config["datasets"]["seaclear"]["root"],
            annotation_file=config["datasets"]["seaclear"].get("annotation_file"),
            target_size=size,
            class_mapping=mapping["seaclear"],
        ),
        UPPDDataset(
            config["datasets"]["uppd"]["root"],
            target_size=size,
            class_mapping=mapping["uppd"],
        ),
    ])


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    totals = {"total": 0.0, "box": 0.0, "classification": 0.0, "objectness": 0.0, "dfl": 0.0}

    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        targets = batch["targets"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=scaler is not None,
        ):
            output = model(images)
            losses = criterion(output["detections"], targets)

        if training:
            if scaler is not None:
                scaler.scale(losses["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

        for key in totals:
            totals[key] += float(losses[key].detach().item())

    count = max(len(loader), 1)
    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train HydroWatch detector.")
    parser.add_argument("--config", default="dataset_config.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = load_config(args.config)
    num_classes = validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)
    print("Classes:", config["unified_classes"])

    dataset = build_dataset(config)
    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_detection_batch,
    )

    model = HydroWatch(num_classes=num_classes, reg_max=16).to(device)
    criterion = DetectionLoss(reg_max=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    best_val = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {len(dataset)} | train: {train_size} | val: {val_size}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:03d} "
            f"train={train_metrics['total']:.5f} "
            f"val={val_metrics['total']:.5f} "
            f"box={val_metrics['box']:.5f} "
            f"cls={val_metrics['classification']:.5f} "
            f"obj={val_metrics['objectness']:.5f} "
            f"dfl={val_metrics['dfl']:.5f}"
        )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer=optimizer,
            epoch=epoch,
            extra={"train": train_metrics, "val": val_metrics},
        )

        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer=optimizer,
                epoch=epoch,
                extra={"train": train_metrics, "val": val_metrics},
            )

    print("Training complete.")
    print("Best checkpoint:", checkpoint_dir / "best.pt")


if __name__ == "__main__":
    main()
