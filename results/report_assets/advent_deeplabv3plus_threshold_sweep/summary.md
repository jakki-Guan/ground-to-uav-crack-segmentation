# Threshold Selection Report

- Checkpoint: `checkpoints/advent_deeplabv3plus_crack500_to_uav.pth`
- Validation split: `UAV_Crack_Segmentation_Kaggle/val`
- Test split: `UAV_Crack_Segmentation_Kaggle/test`
- Selection metric: `iou`
- Selected threshold: `0.90`

## Local ADVENT Setup

- Training script: `train_advent.py`
- Reusable command: `scripts/experiments/run_advent_deeplabv3plus.sh`
- Source supervised training data: `CRACK500/train`
- Target adaptation data: `UAV_Crack_Segmentation_Kaggle/train`
- Target-label rule: target masks are ignored during ADVENT training
- Backbone: `DeepLabV3+` with `resnet34` encoder
- Initialization checkpoint: `checkpoints/deeplabv3plus_plain_360.pth`
- Deployment rule: test inference loads only the segmentation checkpoint; the entropy discriminator is training-only

## Validation Selection

- IoU `0.2036`, F1 `0.3157`, precision `0.2455`, recall `0.5003`

## Confirmatory Test

- IoU `0.2022`, F1 `0.3076`, precision `0.2427`, recall `0.4845`

Threshold selection used validation data only. The test split was evaluated once with the frozen selected threshold.

## Interpretation

ADVENT-style output-space adversarial adaptation improves the `DeepLabV3+` source-only baseline under the same validation-selected threshold protocol, but the selected threshold is highly conservative. This indicates that output-space alignment improves the high-confidence region of the prediction distribution while leaving substantial UAV false-positive pressure. The result is useful as a reviewer-facing UDA baseline, but it remains well below the promoted `DeepLabV3+ B1` result that uses explicit target-background / hard-negative exposure.
