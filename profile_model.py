import torch

from models import HydroWatch
from models.profile import (
    benchmark,
    count_parameters,
    parameter_groups,
    output_sanity
)
from models.checkpoint import (
    save_checkpoint,
    load_checkpoint
)


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    num_classes = 6

    model = HydroWatch(
        num_classes=num_classes,
        reg_max=16
    ).to(device)

    x = torch.rand(
        1, 3, 360, 480,
        device=device
    )

    total, trainable = count_parameters(
        model
    )

    print("Device:", device)
    print("Input:", tuple(x.shape))
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")

    print("\nParameter groups:")
    for name, count in parameter_groups(model).items():
        print(
            f"  {name}: {count:,}"
        )

    output_sanity(
        model,
        x
    )

    result = benchmark(
        model,
        x,
        warmup=3,
        iterations=10
    )

    print("\nForward benchmark:")
    print(
        "  ms/image:",
        f"{result['milliseconds_per_image']:.3f}"
    )
    print(
        "  images/sec:",
        f"{result['images_per_second']:.2f}"
    )

    # Checkpoint round-trip.
    checkpoint = "hydrowatch_model_test.pt"

    save_checkpoint(
        checkpoint,
        model,
        extra={
            "purpose": "model smoke-test checkpoint"
        }
    )

    restored = HydroWatch(
        num_classes=num_classes,
        reg_max=16
    ).to(device)

    saved = load_checkpoint(
        checkpoint,
        restored,
        device=device
    )

    print("\nCheckpoint round-trip: PASS")
    print(
        "  Saved config:",
        saved["model_config"]
    )

    # Compare one deterministic inference output.
    model.eval()
    restored.eval()

    with torch.no_grad():
        a = model(x)
        b = restored(x)

    for level in [
        "p2", "p3", "p4", "p5"
    ]:
        for key in [
            "box",
            "objectness",
            "class"
        ]:
            if not torch.equal(
                a["detections"][level][key],
                b["detections"][level][key]
            ):
                raise RuntimeError(
                    f"Checkpoint round-trip mismatch: "
                    f"{level}/{key}"
                )

    print(
        "Checkpoint output equality: PASS"
    )

    print("\nMODEL PROFILE PASSED.")


if __name__ == "__main__":
    main()
