# SAM799 External Patchwise Evaluation

This directory contains fixed-protocol external evaluation artifacts for a single DAFormer/HRDA reviewer-baseline checkpoint on SAM799-CVAT.

Protocol summary:
- Dataset root: `/home/jakeguan/dev/crack-detection/SAM799_CVAT`
- Split evaluated: `external_test`
- Images evaluated: `53`
- Patch size / stride: `512` / `384`
- Model input size: `360`
- Stitching: `probability_average`
- Training, validation, and threshold selection on SAM799-CVAT: `false`

Stage:
- `daformer`: `DAFormer (MiT-B5)` | threshold `0.6` | checkpoint `/home/jakeguan/dev/crack-detection/work_dirs/daformer_crack500_to_uav_mitb5_s0_360/iter_40000.pth` | config `/home/jakeguan/dev/crack-detection/configs/daformer/crack500_to_uav_daformer_mitb5_s0.py`

Generated files:
- `eval_config.json`
- `sam799_manifest.csv`
- `per_image_patch_counts.csv`
- `per_image_metrics.csv`
- `aggregate_metrics.csv`
- `predictions/` and `overlays/` (unless skipped)
