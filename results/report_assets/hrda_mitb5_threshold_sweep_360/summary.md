# HRDA Threshold Selection Report

- Config: `configs/hrda/crack500_to_uav_hrda_mitb5_s0.py`
- Checkpoint: `work_dirs/hrda_crack500_to_uav_mitb5_s0_360/latest.pth`
- Validation split: `val` (63 samples)
- Test split: `test` (63 samples)
- Selection metric: `iou`
- Selected threshold: `0.30`

## Validation Selection

- IoU `0.0840`, F1 `0.1550`, precision `0.0863`, recall `0.7576`

## Confirmatory Test

- IoU `0.1143`, F1 `0.2052`, precision `0.1191`, recall `0.7390`

Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.
