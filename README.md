# HydroWatch — Marine Debris Detection

HydroWatch is a lightweight underwater marine-debris object detector. It combines RGB information, physics-inspired underwater features, deterministic residual enhancement, and a multi-scale detection network.

## Project Structure

```text
HydroWatch/
├── datasets/
│   ├── trashcan.py
│   ├── seaclear.py
│   ├── uppd.py
│   └── unified_dataset.py
├── models/
│   ├── physics_bank.py
│   ├── enhancement.py
│   ├── frontend.py
│   ├── blocks.py
│   ├── backbone.py
│   ├── neck.py
│   ├── heads.py
│   ├── assignment.py
│   ├── losses.py
│   ├── postprocess.py
│   ├── hydro_watch.py
│   ├── predictor.py
│   └── checkpoint.py
├── preprocessing/
│   ├── annotations.py
│   └── letterbox.py
├── dataset_config.json
├── inspect_datasets.py
├── profile_model.py
├── test_model.py
├── train.py
└── README.md
```

## `datasets/`

- **`trashcan.py`** — Loads TrashCan images and annotations and converts them to the common detector format.
- **`seaclear.py`** — Loads and converts SeaClear data.
- **`uppd.py`** — Loads and converts UPPD data.
- **`unified_dataset.py`** — Provides one common dataset interface so the same training code can work with the different datasets.

## `preprocessing/`

- **`annotations.py`** — Processes/converts object-detection annotations and bounding boxes.
- **`letterbox.py`** — Resizes images while preserving aspect ratio and pads them to the detector's input size.

## `models/`

### Input processing
- **`physics_bank.py`** — Computes three deterministic underwater features: Dark Channel, Red Attenuation, and Gradient Energy.
- **`enhancement.py`** — Produces deterministic residual enhancement features to retain useful details affected by underwater degradation.
- **`frontend.py`** — Combines RGB + Physics + Residual into a 9-channel input and performs the initial feature processing.

### Feature extraction
- **`blocks.py`** — Contains reusable neural-network building blocks.
- **`backbone.py`** — Extracts hierarchical visual features from the input.
- **`neck.py`** — Fuses backbone features at multiple scales for detection.

### Detection
- **`heads.py`** — Converts multi-scale features into bounding-box, confidence, and class predictions.
- **`assignment.py`** — Matches predictions with ground-truth objects during training.
- **`losses.py`** — Calculates detection losses used for optimization.
- **`postprocess.py`** — Decodes raw predictions and removes redundant detections using post-processing/NMS.

### Model and inference
- **`hydro_watch.py`** — Connects all model components into the complete HydroWatch detector.
- **`predictor.py`** — Loads a trained checkpoint and performs inference on images.
- **`checkpoint.py`** — Saves and loads model checkpoints.

## Configuration

- **`dataset_config.json`** — Stores dataset paths, class definitions/mappings, and related dataset settings.

## Root Scripts

- **`inspect_datasets.py`** — Inspects dataset structure, images, annotations, and classes.
- **`profile_model.py`** — Profiles the model and reports its parameter count and computational information.
- **`test_model.py`** — Performs model smoke tests to verify that the network and major components execute correctly.
- **`train.py`** — Main training program: loads data, runs forward pass, assigns targets, computes losses, backpropagates, validates, and saves checkpoints.

## Overall Pipeline

```text
Underwater RGB
      │
      ├── Physics Bank ──────┐
      ├── Residual Enhancement│
      │                       │
      └───────────────────────┤
                              ↓
                         9-channel Fusion
                              ↓
                           Frontend
                              ↓
                           Backbone
                              ↓
                             Neck
                              ↓
                      Multi-scale Features
                              ↓
                        Detection Head
                              ↓
                         Decode + NMS
                              ↓
                         Final Detections
```

## Typical Workflow

```bash
python inspect_datasets.py
python test_model.py
python profile_model.py
python train.py
```

After training, use `predictor.py` with the saved checkpoint for inference.

## Current Status

The core code and architecture are implemented. The remaining major work is experimental:

1. Proper model training
2. Validation/testing
3. mAP, precision, recall and latency/FPS measurements
4. Ablation studies
5. YOLO baseline comparison
6. Final analysis and research conclusions

The architecture should not be substantially changed unless experiments reveal a concrete implementation or performance problem.
