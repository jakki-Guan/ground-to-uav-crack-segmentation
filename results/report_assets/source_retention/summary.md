# Source-Domain Retention Analysis

This report back-tests each frozen target-stage checkpoint on `Crack500 test` using the same operating point used for the target-side row.

- Source reference (`crack500_test`): `IoU 0.5988`, `F1 0.7491`
- Target reference (`uav_crack_segmentation_kaggle_test`): `IoU 0.1394`, `F1 0.2447`

## Main Findings

- `B1 promoted TS-bank @ 0.6` keeps `Crack500 IoU 0.5745` (95.9% of source-only; delta `-0.0243`) while raising `Kaggle IoU` to `0.3832` (gain `+0.2437` vs source-only; 95% bootstrap CI `[0.3035, 0.4592]`).
- `B1 legacy raw @ 0.7` keeps `Crack500 IoU 0.5792` (96.7% of source-only; delta `-0.0196`) while raising `Kaggle IoU` to `0.3539` (gain `+0.2145` vs source-only; 95% bootstrap CI `[0.2613, 0.4526]`).
- `B2 fs05 @ 0.5` keeps `Crack500 IoU 0.5458` (91.2% of source-only; delta `-0.0530`) while raising `Kaggle IoU` to `0.5165` (gain `+0.3770` vs source-only; 95% bootstrap CI `[0.4672, 0.5590]`).
- `B2 fs10 @ 0.5` keeps `Crack500 IoU 0.5281` (88.2% of source-only; delta `-0.0708`) while raising `Kaggle IoU` to `0.5506` (gain `+0.4112` vs source-only; 95% bootstrap CI `[0.5007, 0.5935]`).
- `B2 fs20 @ 0.5` keeps `Crack500 IoU 0.5173` (86.4% of source-only; delta `-0.0815`) while raising `Kaggle IoU` to `0.5764` (gain `+0.4370` vs source-only; 95% bootstrap CI `[0.5236, 0.6238]`).

## Combined Table

| Stage | Crack500 IoU | Retention | Kaggle IoU | Kaggle 95% CI | Kaggle F1 | Kaggle F1 95% CI |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Source-only @ 0.5 | 0.5988 | 100.0% | 0.1394 | [0.0982, 0.1929] | 0.2447 | [0.1789, 0.3234] |
| B1 promoted TS-bank @ 0.6 | 0.5745 | 95.9% | 0.3832 | [0.3035, 0.4592] | 0.5540 | [0.4657, 0.6294] |
| B1 legacy raw @ 0.7 | 0.5792 | 96.7% | 0.3539 | [0.2613, 0.4526] | 0.5228 | [0.4144, 0.6231] |
| B2 fs05 @ 0.5 | 0.5458 | 91.2% | 0.5165 | [0.4672, 0.5590] | 0.6811 | [0.6368, 0.7172] |
| B2 fs10 @ 0.5 | 0.5281 | 88.2% | 0.5506 | [0.5007, 0.5935] | 0.7102 | [0.6673, 0.7449] |
| B2 fs20 @ 0.5 | 0.5173 | 86.4% | 0.5764 | [0.5236, 0.6238] | 0.7313 | [0.6873, 0.7684] |

## Skeleton Metric

Skeleton scores are computed after a lightweight morphological skeletonization of both prediction and ground-truth masks. They are stricter than region IoU/F1 and mainly reflect centerline retention.

| Stage | Crack500 skeleton F1 | Kaggle skeleton F1 | Kaggle skeleton F1 95% CI |
| --- | ---: | ---: | --- |
| Source-only @ 0.5 | 0.1332 | 0.1838 | [0.1558, 0.2133] |
| B1 promoted TS-bank @ 0.6 | 0.1297 | 0.1847 | [0.1536, 0.2134] |
| B1 legacy raw @ 0.7 | 0.1346 | 0.2124 | [0.1742, 0.2482] |
| B2 fs05 @ 0.5 | 0.1363 | 0.3241 | [0.3034, 0.3451] |
| B2 fs10 @ 0.5 | 0.1346 | 0.3534 | [0.3301, 0.3771] |
| B2 fs20 @ 0.5 | 0.1325 | 0.3989 | [0.3754, 0.4212] |

## Method

- `Crack500 test` is used as the source-domain retention back-test.
- `UAV_Crack_Segmentation_Kaggle test` is resampled at the image level with replacement.
- Each bootstrap replicate recomputes dataset-level metrics from summed per-image `tp/fp/fn`.
- Point estimates in this report are recomputed from the same per-image counts for bootstrap consistency, so they can differ slightly from older batch-averaged rows in `results/experiments.csv`.
- No checkpoint is retrained for this report.
