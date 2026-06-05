# Threshold Selection Report

- Checkpoint: `checkpoints/deeplabv3plus_plain_360.pth`
- Validation split: `UAV_Crack_Segmentation_Kaggle/val`
- Test split: `UAV_Crack_Segmentation_Kaggle/test`
- Selection metric: `iou`
- Selected threshold: `0.80`

## Validation Selection

- IoU `0.1354`, F1 `0.2230`, precision `0.1516`, recall `0.5691`

## Confirmatory Test

- IoU `0.1282`, F1 `0.2202`, precision `0.1434`, recall `0.5689`

Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.
