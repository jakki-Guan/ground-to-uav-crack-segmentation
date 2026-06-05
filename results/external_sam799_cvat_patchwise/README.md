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
- `source_only`: `Source-only` | threshold `0.5` | checkpoint `/home/jakeguan/dev/crack-detection/checkpoints/segformer_b2_plain_360.pth`
- `b1_selected`: `B1 selected` | threshold `0.6` | checkpoint `/home/jakeguan/dev/crack-detection/checkpoints/segformer_b2_b1_tsbank_thr080_mean082.pth`
- `b2_fs05`: `B2 fs05` | threshold `0.5` | checkpoint `/home/jakeguan/dev/crack-detection/checkpoints/segformer_b2_b2_fs05_seed42.pth`
- `b2_fs10`: `B2 fs10` | threshold `0.5` | checkpoint `/home/jakeguan/dev/crack-detection/checkpoints/segformer_b2_b2_fs10_seed42.pth`
- `b2_fs20`: `B2 fs20` | threshold `0.5` | checkpoint `/home/jakeguan/dev/crack-detection/checkpoints/segformer_b2_b2_fs20_seed42.pth`

Generated files:
- `eval_config.json`
- `sam799_manifest.csv`
- `per_image_patch_counts.csv`
- `per_image_metrics.csv`
- `aggregate_metrics.csv`
- `profiling_summary.md`
- `predictions/` and `overlays/` (unless skipped)
