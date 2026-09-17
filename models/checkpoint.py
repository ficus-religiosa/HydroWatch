from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    extra=None
):
    """
    Save a reproducible HydroWatch checkpoint.

    The architecture metadata is stored alongside the state dict so
    a checkpoint cannot silently be loaded with the wrong class count.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_classes": int(model.num_classes),
            "reg_max": int(model.reg_max)
        }
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = (
            optimizer.state_dict()
        )

    if epoch is not None:
        checkpoint["epoch"] = int(epoch)

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(
        checkpoint,
        path
    )


def load_checkpoint(
    path,
    model,
    optimizer=None,
    device=None,
    strict=True
):
    """
    Load a HydroWatch checkpoint.

    Architecture metadata is checked before loading weights.
    """
    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location=device or "cpu"
    )

    saved_config = checkpoint.get(
        "model_config",
        {}
    )

    expected = {
        "num_classes": int(model.num_classes),
        "reg_max": int(model.reg_max)
    }

    for key, value in expected.items():
        if key in saved_config and saved_config[key] != value:
            raise ValueError(
                f"Checkpoint {key}={saved_config[key]} "
                f"does not match model {value}."
            )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict
    )

    if optimizer is not None and (
        "optimizer_state_dict" in checkpoint
    ):
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return checkpoint
