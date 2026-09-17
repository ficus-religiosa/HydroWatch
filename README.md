# HydroWatch

Lightweight underwater marine-debris object detector.

## Pipeline

RGB → Physics Bank → Residual Enhancement → 9-channel fusion → lightweight backbone → multi-scale neck → detection heads → decoded boxes/NMS.

### Physics Bank
- Dark Channel
- Red Attenuation
- Gradient Energy

### Training
The project is detection-only. The auxiliary segmentation mask head was removed so the baseline focuses on the stated marine-debris detection objective.

Before training, configure `dataset_config.json` with:
- dataset paths
- the unified class taxonomy
- an explicit class mapping for TrashCan, SeaClear and UPPD

Run dataset inspection first:

```bash
python inspect_datasets.py
```

Then train:

```bash
python train.py --epochs 50 --batch-size 2
```

Check the model without a dataset:

```bash
python test_model.py
python profile_model.py
```

Training writes `checkpoints/last.pt` and `checkpoints/best.pt`.

## Research workflow

1. Finalize the common class taxonomy.
2. Verify dataset splits and duplicate leakage.
3. Run a clean baseline training.
4. Evaluate mAP50/mAP50-95, precision, recall and latency.
5. Run ablations for Physics Bank and Residual Enhancement.
6. Compare against a YOLO baseline under the same data split.
