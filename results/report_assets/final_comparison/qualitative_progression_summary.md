# Kaggle Qualitative Progression

Fixed Kaggle UAV qualitative comparison for the paper-facing story `source-only over-activation -> UDA partial relief -> B1 suppression -> B2 recovery`.

- Samples: `slide1290, slide1434, slide1089`
- Paper row labels: `Case 1 = slide1290, Case 2 = slide1434, Case 3 = slide1089`
- Columns: `Input / GT / Source-only / ADVENT / B1 / B2 fs10`
- Colors: `green = TP`, `red = FP`, `yellow = FN`
- Threshold note: Source-only, Selected B1, and B2 fs10 use thresholds `0.5`, `0.6`, and `0.5`, respectively; ADVENT-style uses the validation-selected threshold `0.9`.
- Backbone note: `ADVENT-style` uses `DeepLabV3+`; the `Source-only`, `Selected B1`, and `B2 fs10` columns use `SegFormer-B2`.

Paper caption:

> Representative qualitative comparison on the primary Kaggle UAV target. Columns show the input image, ground-truth mask, source-only prediction, ADVENT-style UDA prediction, selected B1 prediction, and B2 fs10 prediction. Green, red, and yellow denote true-positive, false-positive, and false-negative pixels, respectively. Source-only transfer produces structured false positives along pavement boundaries and crack-like background regions. ADVENT-style adaptation partially reduces false positives but remains conservative at a high validation-selected threshold, while selected B1 and B2 fs10 reduce excessive false-positive activation in these examples. Per-sample ordering may vary; aggregate trends are reported in Table XI.

Stage details:

- `Source-only SegFormer-B2`: checkpoint `checkpoints/segformer_b2_plain_360.pth`, threshold `0.5`
- `ADVENT-style DeepLabV3+`: checkpoint `checkpoints/advent_deeplabv3plus_crack500_to_uav.pth`, threshold `0.9`
- `Selected B1 SegFormer-B2`: checkpoint `checkpoints/segformer_b2_b1_tsbank_thr080_mean082.pth`, threshold `0.6`
- `B2 fs10 SegFormer-B2`: checkpoint `checkpoints/segformer_b2_b2_fs10_seed42.pth`, threshold `0.5`
