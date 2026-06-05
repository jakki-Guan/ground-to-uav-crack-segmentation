# Seed Stability Summary

This report summarizes fixed-hold-out `UAV test` reruns for selected `SegFormer-B2` settings.

- Seeds: 7, 13, 42
- Test split: `UAV_Crack_Segmentation_Kaggle/test` (`63` images)
- Purpose: quantify training stochasticity on the same fixed test split

## Mean ± Std

| Setting | Stage | IoU | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Source-only @ 0.5 | Source-only | 0.1542 ± 0.0072 | 0.2613 ± 0.0112 | 0.1652 ± 0.0113 | 0.7030 ± 0.0358 |
| B1 promoted TS-bank @ 0.6 | B1 | 0.3360 ± 0.0268 | 0.4903 ± 0.0321 | 0.4785 ± 0.0393 | 0.5701 ± 0.0757 |
| B2 fs05 @ 0.5 | B2 | 0.4814 ± 0.0185 | 0.6400 ± 0.0213 | 0.6029 ± 0.0166 | 0.7035 ± 0.0186 |
| B2 fs10 @ 0.5 | B2 | 0.5350 ± 0.0203 | 0.6907 ± 0.0181 | 0.6616 ± 0.0279 | 0.7313 ± 0.0172 |
| B2 fs20 @ 0.5 | B2 | 0.5881 ± 0.0087 | 0.7365 ± 0.0071 | 0.7116 ± 0.0113 | 0.7672 ± 0.0049 |

## Per-Seed IoU

- `Source-only @ 0.5`: 0.1625, 0.1508, 0.1494
- `B1 promoted TS-bank @ 0.6`: 0.3134, 0.3657, 0.3290
- `B2 fs05 @ 0.5`: 0.4728, 0.4686, 0.5026
- `B2 fs10 @ 0.5`: 0.5410, 0.5124, 0.5516
- `B2 fs20 @ 0.5`: 0.5968, 0.5793, 0.5882

## Suggested Paper Text

To assess training stochasticity, we repeated the selected settings with `3` random seeds for `SegFormer-B2` and report mean ± standard deviation on the fixed `UAV test` split. The cross-seed variance can then be compared directly against the between-setting gaps.
