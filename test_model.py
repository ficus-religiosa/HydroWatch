import torch

from models import HydroWatch


def print_feature_shapes(title, features):
    print(f"\n{title}:")
    for level, tensor in features.items():
        print(f"  {level}: {tuple(tensor.shape)}")


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = HydroWatch(
        num_classes=1,
        reg_max=16,
        enable_mask_head=True
    ).to(device)

    # Final model input: RGB, H=360, W=480.
    # rand() keeps the smoke test in an image-like [0, 1] range.
    x = torch.rand(
        1, 3, 360, 480,
        device=device
    )

    print("\nInput:", tuple(x.shape))

    # -----------------------------
    # Frontend diagnostics
    # -----------------------------
    model.train()

    with torch.no_grad():
        physics = model.frontend.physics_bank(x)
        residual = model.frontend.enhancement(x)
        fused = torch.cat([x, physics, residual], dim=1)
        frontend_output = model.frontend(x)

    print("\nFrontend:")
    print("  RGB:", tuple(x.shape))
    print("  Physics:", tuple(physics.shape))
    print("  Enhancement:", tuple(residual.shape))
    print("  9-channel fusion:", tuple(fused.shape))
    print("  Frontend output:", tuple(frontend_output.shape))

    # -----------------------------
    # Backbone diagnostics
    # -----------------------------
    with torch.no_grad():
        backbone_features = model.backbone(frontend_output)

    print_feature_shapes(
        "Backbone features",
        backbone_features
    )

    # -----------------------------
    # Neck diagnostics
    # -----------------------------
    with torch.no_grad():
        neck_features = model.neck(backbone_features)

    print_feature_shapes(
        "Neck features",
        neck_features
    )

    # -----------------------------
    # Full training-mode forward
    # -----------------------------
    output = model(x)

    print("\nDetection outputs:")

    for level, pred in output["detections"].items():
        print(f"  {level}:")
        print("    box:", tuple(pred["box"].shape))
        print("    objectness:", tuple(pred["objectness"].shape))
        print("    class:", tuple(pred["class"].shape))

    print(
        "\nMask:",
        tuple(output["mask_logits"].shape)
    )

    # -----------------------------
    # Backpropagation smoke test
    # -----------------------------
    # Use a scalar built from all model outputs so autograd
    # traverses the complete graph.
    loss = torch.zeros(
        (),
        device=device,
        requires_grad=True
    )

    for pred in output["detections"].values():
        loss = loss + pred["box"].mean()
        loss = loss + pred["objectness"].mean()
        loss = loss + pred["class"].mean()

    loss = loss + output["mask_logits"].mean()

    loss.backward()

    print("\nBackward pass successful.")

    # -----------------------------
    # Parameter count
    # -----------------------------
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nParameters:")
    print("  Total:", f"{total_params:,}")
    print("  Trainable:", f"{trainable_params:,}")

    # -----------------------------
    # Inference mode
    # -----------------------------
    model.eval()

    with torch.no_grad():
        inference_output = model(
            x,
            return_masks=False
        )

    print("\nInference successful.")
    print(
        "Inference levels:",
        list(inference_output["detections"].keys())
    )


if __name__ == "__main__":
    main()