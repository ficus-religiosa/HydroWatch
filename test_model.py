import torch

from models import HydroWatch
from models.losses import DetectionLoss


def print_feature_shapes(title, features):
    print(f"\n{title}:")
    for level, tensor in features.items():
        print(f"  {level}: {tuple(tensor.shape)}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # This is only a smoke test. The final class count must come
    # from the unified taxonomy of the three real datasets.
    num_classes = 6

    print("Device:", device)
    print("Synthetic test classes:", num_classes)

    model = HydroWatch(
        num_classes=num_classes,
        reg_max=16
    ).to(device)

    # ------------------------------------------------------------
    # Forward test
    # ------------------------------------------------------------
    x = torch.rand(
        1, 3, 360, 480,
        device=device
    )

    print("\nInput:", tuple(x.shape))

    model.train()

    output = model(x)

    print("\nFrontend:")
    with torch.no_grad():
        physics = model.frontend.physics_bank(x)
        residual = model.frontend.enhancement(x)
        fused = torch.cat(
            [x, physics, residual],
            dim=1
        )
        frontend_output = model.frontend(x)

    print("  RGB:", tuple(x.shape))
    print("  Physics:", tuple(physics.shape))
    print("  Enhancement:", tuple(residual.shape))
    print("  9-channel fusion:", tuple(fused.shape))
    print("  Frontend output:", tuple(frontend_output.shape))

    print_feature_shapes(
        "Backbone features",
        model.backbone(frontend_output)
    )

    print_feature_shapes(
        "Neck features",
        output["features"]
    )

    print("\nDetection outputs:")

    for level, pred in output["detections"].items():
        print(f"  {level}:")
        print("    box logits:", tuple(pred["box"].shape))
        print("    objectness:", tuple(pred["objectness"].shape))
        print("    class:", tuple(pred["class"].shape))
        print("    decoded boxes:", tuple(pred["boxes"].shape))
        print("    DFL distances:", tuple(pred["distances"].shape))


    # ------------------------------------------------------------
    # Synthetic ground truth
    # Absolute pixel-space xyxy boxes in 480x360 coordinates.
    # ------------------------------------------------------------
    targets = [
        {
            "boxes": torch.tensor(
                [
                    [90., 80., 210., 190.],
                    [270., 190., 390., 320.]
                ],
                device=device
            ),
            "labels": torch.tensor(
                [0, 5],
                dtype=torch.long,
                device=device
            ),
        }
    ]


    # ------------------------------------------------------------
    # Real loss path
    # ------------------------------------------------------------
    criterion = DetectionLoss(
        reg_max=16,
        lambda_box=7.5,
        lambda_cls=0.5,
        lambda_obj=1.0,
        lambda_dfl=1.5,
        top_k=10,
        nwd_scale=12.8
    ).to(device)

    losses = criterion(
        output["detections"],
        targets
    )

    print("\nLoss components:")
    for name, value in losses.items():
        print(
            f"  {name}: {value.detach().item():.6f}"
        )

    if not torch.isfinite(losses["total"]):
        raise RuntimeError(
            "Total loss is NaN or Inf."
        )

    # ------------------------------------------------------------
    # Backward + optimizer step
    # ------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()

    grad_count = sum(
        1
        for p in model.parameters()
        if p.grad is not None
    )

    if grad_count == 0:
        raise RuntimeError(
            "No parameter received a gradient."
        )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=10.0
    )

    optimizer.step()

    print(
        "\nBackward + optimizer.step() successful."
    )
    print(
        "Parameters with gradients:",
        grad_count
    )

    # ------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    enhancement_trainable = sum(
        p.numel()
        for p in model.frontend.enhancement.parameters()
        if p.requires_grad
    )

    physics_trainable = sum(
        p.numel()
        for p in model.frontend.physics_bank.parameters()
        if p.requires_grad
    )

    print("\nParameters:")
    print("  Total:", f"{total_params:,}")
    print("  Trainable:", f"{trainable_params:,}")
    print(
        "  Enhancement trainable:",
        enhancement_trainable
    )
    print(
        "  Physics Bank trainable:",
        physics_trainable
    )

    if physics_trainable != 0:
        raise RuntimeError(
            "Physics Bank unexpectedly has trainable parameters."
        )

    if enhancement_trainable != 0:
        raise RuntimeError(
            "Deterministic enhancement unexpectedly has "
            "trainable parameters."
        )

    # ------------------------------------------------------------
    # Inference test
    # ------------------------------------------------------------
    model.eval()

    with torch.no_grad():
        inference_output = model(x)

    print("\nInference successful.")
    print(
        "Inference levels:",
        list(
            inference_output["detections"].keys()
        )
    )

    print("\nALL MODEL TESTS PASSED.")


if __name__ == "__main__":
    main()
