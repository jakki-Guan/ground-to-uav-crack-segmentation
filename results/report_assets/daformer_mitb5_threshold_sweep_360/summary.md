# DAFormer Threshold Selection Report

- Config: `configs/daformer/crack500_to_uav_daformer_mitb5_s0.py`
- Checkpoint: `work_dirs/daformer_crack500_to_uav_mitb5_s0_360/latest.pth`
- Validation split: `val` (63 samples)
- Test split: `test` (63 samples)
- Selection metric: `iou`
- Selected threshold: `0.60`

## Validation Selection

- IoU `0.0842`, F1 `0.1553`, precision `0.0853`, recall `0.8718`

## Confirmatory Test

- IoU `0.1353`, F1 `0.2384`, precision `0.1388`, recall `0.8442`

Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.
