# Threshold Selection Report

- Checkpoint: `checkpoints/segformer_b2_plain_360.pth`
- Validation split: `UAV_Crack_Segmentation_Kaggle/val`
- Test split: `UAV_Crack_Segmentation_Kaggle/test`
- Selection metric: `iou`
- Selected threshold: `0.90`

## Validation Selection

- IoU `0.2226`, F1 `0.3381`, precision `0.2714`, recall `0.5266`

## Confirmatory Test

- IoU `0.1769`, F1 `0.2877`, precision `0.2050`, recall `0.5302`

Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.
