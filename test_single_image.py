import tkinter as tk
from tkinter import filedialog

import torch
from PIL import Image

from models import HydroWatch
from models.losses import DetectionLoss


# ============================================================
# 1. SELECT IMAGE FROM YOUR COMPUTER
# ============================================================

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an underwater image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("All files", "*.*"),
    ],
)

if not image_path:
    raise RuntimeError("No image selected.")

print("Selected image:", image_path)


# ============================================================
# 2. LOAD THE IMAGE
# ============================================================

image = Image.open(image_path).convert("RGB")

print("Original image size:", image.size)

# Your model test uses 480 × 360
image = image.resize((480, 360))

print("Resized image size:", image.size)


# ============================================================
# 3. CONVERT IMAGE → PYTORCH TENSOR
# ============================================================

# PIL image → tensor
image_tensor = torch.from_numpy(
    __import__("numpy").array(image)
).float()

# [H, W, C] → [C, H, W]
image_tensor = image_tensor.permute(2, 0, 1)

# Pixel values:
# 0...255 → 0...1
image_tensor = image_tensor / 255.0

# Add batch dimension:
# [3, 360, 480] → [1, 3, 360, 480]
image_tensor = image_tensor.unsqueeze(0)


# ============================================================
# 4. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

image_tensor = image_tensor.to(device)


# ============================================================
# 5. CREATE YOUR EXISTING HYDROWATCH MODEL
# ============================================================

num_classes = 6

model = HydroWatch(
    num_classes=num_classes,
    reg_max=16
).to(device)

model.train()


# ============================================================
# 6. FORWARD PROPAGATION
# ============================================================

print("\nRunning forward propagation...")

output = model(image_tensor)

print("Forward propagation: SUCCESS")


# ============================================================
# 7. DUMMY GROUND-TRUTH BOXES
# ============================================================

# IMPORTANT:
# These are NOT detected boxes.
#
# They are fake ground-truth boxes used ONLY so that
# your existing DetectionLoss can calculate a loss.
#
# Format:
# [x1, y1, x2, y2]
#
# Coordinates are in pixels for a 480 × 360 image.

targets = [
    {
        "boxes": torch.tensor(
            [
                [90., 80., 210., 190.],
                [270., 190., 390., 320.]
            ],
            dtype=torch.float32,
            device=device,
        ),

        # These are dummy class IDs.
        # Must be < num_classes.
        "labels": torch.tensor(
            [0, 5],
            dtype=torch.long,
            device=device,
        ),
    }
]

print("\nDummy boxes:")
print(targets[0]["boxes"])

print("Dummy labels:")
print(targets[0]["labels"])


# ============================================================
# 8. YOUR EXISTING DETECTION LOSS
# ============================================================

criterion = DetectionLoss(
    reg_max=16,
    lambda_box=7.5,
    lambda_cls=0.5,
    lambda_obj=1.0,
    lambda_dfl=1.5,
    top_k=10,
    nwd_scale=12.8,
).to(device)


# ============================================================
# 9. CALCULATE LOSS
# ============================================================

print("\nCalculating loss...")

losses = criterion(
    output["detections"],
    targets,
)

print("\nLoss components:")

for name, value in losses.items():
    print(
        f"  {name}: {value.detach().item():.6f}"
    )


# ============================================================
# 10. CHECK LOSS
# ============================================================

if not torch.isfinite(losses["total"]):
    raise RuntimeError(
        "Loss is NaN or Inf!"
    )

print("\nLoss calculation: SUCCESS")


# ============================================================
# 11. BACKWARD PROPAGATION
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)

optimizer.zero_grad(set_to_none=True)

print("\nRunning backward propagation...")

losses["total"].backward()

print("Backward propagation: SUCCESS")


# ============================================================
# 12. CHECK GRADIENTS
# ============================================================

grad_count = 0

for name, parameter in model.named_parameters():

    if parameter.grad is not None:
        grad_count += 1

print(
    "\nParameters with gradients:",
    grad_count
)

if grad_count == 0:
    raise RuntimeError(
        "No model parameters received gradients!"
    )

print("Gradient check: SUCCESS")


# ============================================================
# 13. OPTIMIZER STEP
# ============================================================

optimizer.step()

print("Optimizer step: SUCCESS")


# ============================================================
# 14. FINAL RESULT
# ============================================================

print("\n======================================")
print("FORWARD + BACKWARD TEST PASSED")
print("======================================")