# Baseline Results Log

This file records stable baseline runs that should be treated as reference points for later experiments.

## Run 001: U-Net on Crack500

**Date recorded:** 2026-04-22  
**Purpose:** First in-domain CNN baseline for the project

### Model

- Architecture: `U-Net`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Dataset: `Crack500`
- Image size: `360`
- Batch size: `16`
- Learning rate: `1e-4`
- Optimizer: `AdamW`
- Scheduler: `CosineAnnealingLR`
- Max epochs: `30`
- Early stopping patience: `5`
- `min_delta`: `1e-3`
- Loss: `BCE + Dice`

### Checkpoint

- Best checkpoint path: `checkpoints/best_model.pth`
- Early stopping triggered at epoch: `18`
- Best validation checkpoint: `epoch 13`
- Best validation IoU seen in the latest training log: `0.6481`

### Test Result

- Test loss: `0.4256`
- Test IoU: `0.5593`
- Test F1: `0.7116`

### Qualitative Review

- Visualization notebook: `notebooks/03_test_visualization.ipynb`
- Fixed-sample visualization is now in place for repeatable qualitative comparison.
- Failure-case review shows that not all low-IoU samples represent the same error type.
- The first two zero-prediction cases appear visually ambiguous even to a human observer; the labeled crack region looks more like a slightly darker area than a clearly visible crack.
- Other failure cases are more typical segmentation errors: the model follows the crack location but predicts a thinner mask and breaks the crack into disconnected segments.

### Notes

- This is now the frozen CNN baseline for comparison against `SegFormer-B2`.
- Do not keep tuning this model based on the test result. If more tuning is needed, use the validation set and record it as a new run.
- The baseline package now includes training, test-set evaluation, and qualitative visualization.
- Next major step: implement the `SegFormer-B2` baseline under the same evaluation protocol.
