# SAM799 External Patchwise Evaluation

This directory contains fixed-protocol external evaluation artifacts for the self-collected SAM799-CVAT UAV crack dataset.

Protocol summary:
- Dataset root: `/home/jakeguan/dev/crack-detection/SAM799_CVAT`
- Split evaluated: `external_test`
- Images evaluated: `53`
- Patch size / stride: `512` / `384`
- Model input size: `360`
- Stitching: `probability_average`
- Training, validation, and threshold selection on SAM799-CVAT: `false`

Stages:
- `advent`: `ADVENT (DeepLabV3+)` | threshold `0.9` | checkpoint `/home/jakeguan/dev/crack-detection/checkpoints/advent_deeplabv3plus_crack500_to_uav.pth`

Generated files:
- `eval_config.json`
- `sam799_manifest.csv`
- `per_image_patch_counts.csv`
- `per_image_metrics.csv`
- `aggregate_metrics.csv`
- `profiling_summary.md`
- `predictions/` and `overlays/` (unless skipped)
