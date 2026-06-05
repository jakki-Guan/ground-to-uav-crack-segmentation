# Baseline Results Log

This file records stable baseline runs that should be treated as reference points for later experiments.

Baseline reporting rule:

- Entries in this file refer to raw model predictions unless explicitly stated otherwise.
- Deployment-oriented postprocessed variants should be documented separately rather than replacing the raw baseline record.
- For runs logged before the `2026-04-26` experiment-log schema upgrade, this document should be treated as the authoritative record for `precision`, `recall`, and any newer configuration metadata that may be blank in older CSV rows.

## Final Frozen Result Index

Use this index first when citing paper-facing results. Earlier runs below remain important as historical build-up, calibration context, or validation-only screening, but they should not replace the frozen rows listed here.

### Final Paper-Facing Rows

- `Run 012`: `U-Net` source-only raw on the fixed `UAV test` hold-out (`IoU 0.1284`, `F1 0.2244`). This is the formal classical-reference row; no official `U-Net B1/B2` continuation is frozen.
- `Run 002`: `SegFormer-B2` source-only raw on the fixed `UAV test` hold-out (`IoU 0.1442`, `F1 0.2476`).
- `Run 006`: `DeepLabV3+` source-only raw on the fixed `UAV test` hold-out (`IoU 0.1230`, `F1 0.2152`).
- `Run 014`: `DeepLabV3+` ADVENT-style UDA reviewer baseline, validation-selected threshold `0.9` (`IoU 0.2022`, `F1 0.3076`).
- `Run 010`: promoted official `GT`-filtered `SegFormer-B2 B1` row `TS-bank thr080_mean082 @ 0.6` (`IoU 0.3775`, `F1 0.5317`).
- `Run 011`: promoted official `GT`-filtered `DeepLabV3+ B1` row `TS-bank area1200 @ 0.5` (`IoU 0.3860`, `F1 0.5470`).
- `Run 005`: `SegFormer-B2 B2` few-shot curve `fs05 / fs10 / fs20` (`IoU 0.5074 / 0.5420 / 0.5686`).
- `Run 008`: `DeepLabV3+ B2` few-shot curve `fs05_pat12 / fs10 / fs20` (`IoU 0.3599 / 0.4354 / 0.4760`).
- `Run 004`: `SegFormer-B2` UAV in-domain upper bound (`IoU 0.5879`, `F1 0.7369`).
- `Run 013`: `DeepLabV3+` UAV in-domain upper bound (`IoU 0.5085`, `F1 0.6693`).

### Historical Or Supporting Rows

- `Run 001`: in-domain `U-Net` baseline on `Crack500`.
- `Run 003`: first-generation `SegFormer-B2 B1` hold-out checkpoint kept for calibration and operating-point history; superseded by `Run 010`.
- `Run 007`: first-generation `DeepLabV3+ B1` hold-out checkpoint kept for historical comparison; superseded by `Run 011`.
- `Run 009`: temperature-scaled mining, audit, and mechanism follow-up log; use it to justify candidate selection and claim boundaries, not as a main-table row.
- `Run 015`: `HRDA/MiT-B5` reviewer UDA baseline under the unified `360 x 360` protocol; completed as a negative but still informative comparison row (`IoU 0.1143`, `F1 0.2052` at threshold `0.3`).

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

## Run 002: SegFormer-B2 Source-Only Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-04-25  
**Purpose:** Official raw cross-domain baseline on the fixed `UAV_Crack_Segmentation_Kaggle` hold-out split

### Model

- Architecture: `SegFormer-B2`
- Pretrained model: `nvidia/segformer-b2-finetuned-ade-512-512`
- Input channels: `3`
- Output classes: `1`

### Training Origin

- Source-domain checkpoint: `checkpoints/segformer_b2_plain_360.pth`
- Source training dataset: `Crack500`
- Source recipe: plain `360`, baseline augmentation, `BCE + Dice`

### Evaluation Setup

- Dataset: `UAV_Crack_Segmentation_Kaggle`
- Official hold-out split: `test`
- Dataset size: `63`
- Image size: `360`
- Batch size: `8`

### Test Result

- Raw prediction
  - Test loss: `0.9583`
  - Test IoU: `0.1442`
  - Test F1: `0.2476`
  - Test precision: `0.1495`
  - Test recall: `0.7727`
- Deployment-oriented postprocessed variant
  - Threshold: `0.9`
  - `min_area = 20`
  - `max_fill_ratio = 0.85`
  - Test IoU: `0.1784`
  - Test F1: `0.2900`
  - Test precision: `0.2099`
  - Test recall: `0.5081`

### Notes

- The raw prediction remains the official cross-domain baseline because it measures the model's own generalization under domain shift.
- The deployment-oriented variant is a separate operating point for cluttered UAV scenes and should not replace the raw baseline entry.
- `crossdomain_all` should still be used for exploratory diagnosis, while this `test` split is the formal hold-out reference.

## Run 003: SegFormer-B2 `B1` Hard-Negative-Mixed Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-04-25  
**Purpose:** First training-side mitigation result using a mined UAV hard-negative bank

### Model

- Architecture: `SegFormer-B2`
- Pretrained model: `nvidia/segformer-b2-finetuned-ade-512-512`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Main supervised dataset: `Crack500`
- Validation dataset: `UAV_Crack_Segmentation_Kaggle/val`
- Auxiliary negative bank: `generated/uav_hard_negatives_segformer_b2_train_thr080`
- Auxiliary negative repeat: `2`
- Image size: `360`
- Batch size: `8`
- Learning rate: `1e-4`
- Max epochs: `30`
- Early stopping: enabled

### Checkpoint

- Best checkpoint path: `checkpoints/segformer_b2_b1_negbank.pth`
- Best validation checkpoint: `epoch 20`
- Best validation IoU on `UAV val`: `0.3419`
- Best validation F1 on `UAV val`: `0.4812`
- Best validation precision on `UAV val`: `0.4296`
- Best validation recall on `UAV val`: `0.5944`

### Test Result

- Raw prediction at threshold `0.5`
  - Test loss: `0.6286`
  - Test IoU: `0.3222`
  - Test F1: `0.4677`
  - Test precision: `0.4660`
  - Test recall: `0.5662`
- Calibrated raw operating point at threshold `0.7`
  - Test IoU: `0.3325`
  - Test F1: `0.4727`
  - Test precision: `0.5374`
  - Test recall: `0.5133`
- High-precision raw operating point at threshold `0.9`
  - Test IoU: `0.3193`
  - Test F1: `0.4502`
  - Test precision: `0.6497`
  - Test recall: `0.4112`
- Deployment-oriented postprocessed variant
  - Threshold: `0.9`
  - `min_area = 20`
  - `max_fill_ratio = 0.85`
  - Test IoU: `0.3166`
  - Test F1: `0.4469`
  - Test precision: `0.6490`
  - Test recall: `0.4071`

### Notes

- This is the first stable evidence that training-side mitigation can improve raw cross-domain behavior directly rather than relying on post-hoc filtering.
- `B1 raw` is already stronger than the source-only deployment-oriented variant on the same fixed hold-out split.
- The calibrated raw operating point at threshold `0.7` is the best main-report setting for this checkpoint on the fixed hold-out split.
- The high-threshold raw operating point at `0.9` reaches essentially the same high-precision regime as the postprocessed deployment variant while slightly exceeding it on `IoU` and `F1`.
- After `B1`, the old deployment switch is therefore best treated as a comparison row rather than as the preferred deployment recipe.
- Subsequent confirmatory hold-out follow-up on `2026-05-03` then promoted `segformer_b2_b1_tsbank_thr080_mean082_test_thr060` to the official `GT`-filtered `SegFormer-B2 B1` row; see `Run 010` for the frozen paper-facing record.
- The older `B1 raw @ 0.7` row should therefore be retained as historical calibration context rather than as the final main-report `B1` result.

## Run 004: SegFormer-B2 In-Domain Upper Bound on UAV

**Date recorded:** 2026-04-26  
**Purpose:** Target-domain ceiling reference on the fixed `UAV_Crack_Segmentation_Kaggle` split

### Model

- Architecture: `SegFormer-B2`
- Pretrained model: `nvidia/segformer-b2-finetuned-ade-512-512`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Dataset: `UAV_Crack_Segmentation_Kaggle`
- Train split: `train`
- Validation split: `val`
- Test split: `test`
- Image size: `360`
- Batch size: `8`
- Learning rate: `1e-4`
- Augmentation: `baseline`
- Max epochs: `40`
- Early stopping: enabled

### Checkpoint

- Best checkpoint path: `checkpoints/segformer_b2_uav_indomain_plain_360.pth`
- Best validation checkpoint: `epoch 34`
- Best validation IoU on `UAV val`: `0.5701`
- Best validation F1 on `UAV val`: `0.7227`

### Test Result

- Raw prediction
  - Test loss: `0.3849`
  - Test IoU: `0.5879`
  - Test F1: `0.7369`
  - Test precision: `0.6970`
  - Test recall: `0.7830`

### Notes

- This run serves as the target-domain upper bound rather than as a cross-domain baseline.
- Its main role is to quantify how much performance is still missing after source-only transfer or `B1` mitigation.
- The large gap between this ceiling and the cross-domain results supports the interpretation that domain shift remains the main bottleneck.

## Run 005: SegFormer-B2 `B2` Few-Shot Fine-Tuning on UAV

**Date recorded:** 2026-04-26  
**Purpose:** Quantify how much limited target-domain supervision is needed to close the remaining domain gap

### Model

- Architecture: `SegFormer-B2`
- Pretrained model: `nvidia/segformer-b2-finetuned-ade-512-512`
- Input channels: `3`
- Output classes: `1`

### Fine-Tuning Setup

- Base initialization checkpoint: `checkpoints/segformer_b2_plain_360.pth`
- Target dataset: `UAV_Crack_Segmentation_Kaggle`
- Validation split: `val`
- Test split: `test`
- Image size: `360`
- Batch size: `4`
- Learning rate: `5e-5`
- Augmentation: `baseline`
- Max epochs: `50`
- Early stopping: enabled

### Few-Shot Splits

- `fs05`: `train_fs05_seed42` with `9` labeled UAV training samples
- `fs10`: `train_fs10_seed42` with `19` labeled UAV training samples
- `fs20`: `train_fs20_seed42` with `38` labeled UAV training samples

### Test Result

- `fs05`
  - Best validation epoch: `28`
  - Test loss: `0.4054`
  - Test IoU: `0.5074`
  - Test F1: `0.6695`
  - Test precision: `0.6232`
  - Test recall: `0.7277`
- `fs10`
  - Best validation epoch: `25`
  - Test loss: `0.3703`
  - Test IoU: `0.5420`
  - Test F1: `0.6988`
  - Test precision: `0.6762`
  - Test recall: `0.7251`
- `fs20`
  - Best validation epoch: `27`
  - Test loss: `0.3354`
  - Test IoU: `0.5686`
  - Test F1: `0.7209`
  - Test precision: `0.6724`
  - Test recall: `0.7826`

### Notes

- These runs show a strong supervision-efficiency trend: limited target-domain labeling already closes a large fraction of the remaining gap.
- Relative to the promoted `SegFormer-B2 B1 TS-bank @ 0.6` row (`IoU 0.3775`), the few-shot runs improve IoU to `0.5074`, `0.5420`, and `0.5686`.
- `fs20` reaches about `96.7%` of the full in-domain upper bound (`IoU 0.5879`), which indicates that the domain gap is highly correctable with modest target annotation effort.
- The earliest logged `fs10` and `fs20` test commands accidentally pointed to the `fs05` checkpoint; the official values above come from the corrected reruns and should be treated as authoritative.
- Together with `B1`, these results support a strong final interpretation:
  - zero-shot transfer is weak
  - target-style nuisance exposure helps substantially
  - limited target supervision closes most of the remaining gap

## Run 006: DeepLabV3+ Source-Only Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-04-27  
**Purpose:** Add an ASPP-style CNN source-only transfer reference on the fixed `UAV_Crack_Segmentation_Kaggle` hold-out split

### Model

- Architecture: `DeepLabV3+`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Training Origin

- Source-domain checkpoint: `checkpoints/deeplabv3plus_plain_360.pth`
- Source training dataset: `Crack500`
- Source recipe: plain `360`, baseline augmentation, `BCE + Dice`, early-stopping patience `30`

### Evaluation Setup

- Dataset: `UAV_Crack_Segmentation_Kaggle`
- Official hold-out split: `test`
- Dataset size: `63`
- Image size: `360`
- Batch size: `8`

### Test Result

- Raw prediction
  - Test loss: `1.0857`
  - Test IoU: `0.1230`
  - Test F1: `0.2152`
  - Test precision: `0.1303`
  - Test recall: `0.6913`
- Deployment-oriented postprocessed variant
  - Threshold: `0.9`
  - `min_area = 20`
  - `max_fill_ratio = 0.85`
  - Test IoU: `0.1220`
  - Test F1: `0.2090`
  - Test precision: `0.1447`
  - Test recall: `0.4711`

### Notes

- This run serves as the stable `DeepLabV3+` source-only transfer reference on the fixed UAV hold-out split.
- On `Crack500`, the current in-domain best `DeepLabV3+` recipe is `deeplabv3plus_fgcrop_360` with `test IoU = 0.5709`, slightly above the plain run (`0.5648`).
- On the fixed `UAV test` split, that same `foreground crop` intervention hurts transfer strongly:
  - `deeplabv3plus_fgcrop_360_uav_holdout_raw`
    - Test IoU: `0.0940`
    - Test F1: `0.1698`
    - Test precision: `0.0987`
    - Test recall: `0.6906`
- The transfer degradation comes mainly from a `precision` collapse at nearly unchanged `recall`, so future `DeepLabV3+` `B1/B2` work should start from `plain_360` rather than from the in-domain `fgcrop` winner.

## Run 007: DeepLabV3+ `B1` Hard-Negative-Mixed Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-04-27  
**Purpose:** Check whether the negative-bank `B1` strategy also improves an ASPP-style CNN on the fixed `UAV_Crack_Segmentation_Kaggle` hold-out split

### Model

- Architecture: `DeepLabV3+`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Main supervised dataset: `Crack500`
- Validation dataset: `UAV_Crack_Segmentation_Kaggle/val`
- Auxiliary negative bank: `generated/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080`
- Auxiliary negative repeat: `2`
- Image size: `360`
- Batch size: `8`
- Learning rate: `1e-4`
- Max epochs: `30`
- Early stopping: enabled

### Checkpoint

- Best checkpoint path: `checkpoints/deeplabv3plus_b1_negbank.pth`
- Best validation checkpoint: `epoch 9`
- Best validation IoU on `UAV val`: `0.2704`
- Best validation F1 on `UAV val`: `0.4009`
- Best validation precision on `UAV val`: `0.5418`
- Best validation recall on `UAV val`: `0.3750`

### Test Result

- Raw prediction at threshold `0.5`
  - Test loss: `0.6625`
  - Test IoU: `0.2830`
  - Test F1: `0.4143`
  - Test precision: `0.5820`
  - Test recall: `0.3767`
- Deployment-oriented postprocessed variant
  - Threshold: `0.9`
  - `min_area = 20`
  - `max_fill_ratio = 0.85`
  - Test IoU: `0.1996`
  - Test F1: `0.3123`
  - Test precision: `0.8021`
  - Test recall: `0.2130`

### Notes

- `DeepLabV3+ B1` more than doubles the source-only raw hold-out IoU (`0.1230 -> 0.2830`), so the target-style nuisance exposure idea transfers beyond `SegFormer-B2`.
- Unlike the promoted `SegFormer-B2 TS-bank` line, the initial `DeepLabV3+ B1` checkpoint did not need extra threshold retuning; the default `0.5` operating point remained the best raw setting for that original bank.
- The deployment-oriented switch reaches a very high-precision regime, but it is clearly worse than the raw `B1` row on `IoU` and `F1`.
- For `DeepLabV3+`, the learned mitigation is therefore the main story, while the postprocessed deployment row should remain comparison-only.
- Subsequent audit-derived hold-out follow-up on `2026-05-03` then promoted `deeplabv3plus_b1_tsbank_autofilter_area1200_test` to the official `GT`-filtered `DeepLabV3+ B1` row; see `Run 011` for the frozen paper-facing record.
- The old `deeplabv3plus_b1_holdout_raw_thr050` row should therefore be kept as historical comparison rather than as the final main-report `B1` result.

## Run 008: DeepLabV3+ `B2` Few-Shot Fine-Tuning Status on UAV

**Date recorded:** 2026-04-27  
**Purpose:** Track the current `DeepLabV3+` few-shot transfer curve while distinguishing between logged validation checkpoints and confirmatory fixed-hold-out test rows

### Model

- Architecture: `DeepLabV3+`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Fine-Tuning Setup

- Base initialization checkpoint: `checkpoints/deeplabv3plus_plain_360.pth`
- Target dataset: `UAV_Crack_Segmentation_Kaggle`
- Validation split: `val`
- Test split: `test`
- Image size: `360`
- Batch size: `4` for training, `8` for testing
- Learning rate: `5e-5`
- Augmentation: `baseline`
- Max epochs: `50`
- Early stopping: enabled

### Few-Shot Splits

- `fs05`: `train_fs05_seed42` with `9` labeled UAV training samples
- `fs10`: `train_fs10_seed42` with `19` labeled UAV training samples
- `fs20`: `train_fs20_seed42` with `38` labeled UAV training samples

### Current Logged Result

- `fs05`
  - default early-stopping run
  - Best validation epoch: `25`
  - Validation IoU: `0.3119`
  - Validation F1: `0.4502`
- `fs05` patience-extended rerun
  - experiment: `deeplabv3plus_b2_fs05_seed42_pat12`
  - Best validation epoch: `38`
  - Validation IoU: `0.3239`
  - Validation F1: `0.4656`
  - Confirmatory fixed-hold-out test loss: `0.5550`
  - Confirmatory fixed-hold-out test IoU: `0.3599`
  - Confirmatory fixed-hold-out test F1: `0.5226`
  - Confirmatory fixed-hold-out test precision: `0.4967`
  - Confirmatory fixed-hold-out test recall: `0.5998`
- `fs10`
  - Best validation epoch: `23`
  - Validation IoU: `0.3410`
  - Confirmatory fixed-hold-out test loss: `0.4799`
  - Confirmatory fixed-hold-out test IoU: `0.4354`
  - Confirmatory fixed-hold-out test F1: `0.6005`
  - Confirmatory fixed-hold-out test precision: `0.5608`
  - Confirmatory fixed-hold-out test recall: `0.6620`
- `fs20`
  - Best validation epoch: `42`
  - Validation IoU: `0.4511`
  - Validation F1: `0.6075`
  - Confirmatory fixed-hold-out test loss: `0.4340`
  - Confirmatory fixed-hold-out test IoU: `0.4760`
  - Confirmatory fixed-hold-out test F1: `0.6340`
  - Confirmatory fixed-hold-out test precision: `0.5903`
  - Confirmatory fixed-hold-out test recall: `0.7060`

### Notes

- The current `DeepLabV3+` few-shot evidence already shows a real upward trend from `fs05` through `fs20`, so limited target-domain supervision clearly helps this model beyond the `B1` setting.
- The smallest split is more sensitive to early stopping than the in-domain `Crack500` runs; increasing patience from `5` to `12` improved the `fs05` validation checkpoint and delayed the best epoch from `25` to `38`.
- `DeepLabV3+` few-shot training also needed a small loader safeguard so that a final singleton training batch is dropped only when necessary; otherwise `DeepLabV3+` BatchNorm can fail on tiny splits such as `fs05`.
- The confirmatory fixed-hold-out `B2` test rows are now recorded for `fs05_pat12`, `fs10`, and `fs20`, and they preserve the same upward trend on test `IoU` (`0.3599 -> 0.4354 -> 0.4760`).
- Even with the now-frozen few-shot trend, `DeepLabV3+` should still be treated as a comparative CNN baseline rather than as the main transfer model because its source-only and current few-shot results remain below the `SegFormer-B2` line.

## Run 009: Temperature-Scaled `B1` Hard-Negative Retuning Status on UAV

**Date recorded:** 2026-04-28  
**Purpose:** Track the first calibration-aware `B1` follow-up sweep on `UAV val` before any confirmatory fixed-hold-out promotion

### Calibration Setup

- Frozen source-domain checkpoints were calibrated on `UAV_Crack_Segmentation_Kaggle/val` with scalar temperature scaling.
- Probability definition during calibrated mining:
  - `calibrated_prob = sigmoid(logits / temperature)`
- Fitted temperatures and calibration metrics:
  - `SegFormer-B2`
    - checkpoint: `checkpoints/segformer_b2_plain_360.pth`
    - fitted `T`: `1.8958`
    - `BCE`: `0.1930 -> 0.1488`
    - `ECE`: `0.0412 -> 0.0231`
  - `DeepLabV3+`
    - checkpoint: `checkpoints/deeplabv3plus_plain_360.pth`
    - fitted `T`: `2.3733`
    - `BCE`: `0.3585 -> 0.2257`
    - `ECE`: `0.0705 -> 0.0385`

### Mining-Sweep Setup

- Hard negatives were mined on `UAV_Crack_Segmentation_Kaggle/train` using the calibrated probabilities.
- The first pass swept:
  - `component_threshold in {0.70, 0.75, 0.80}`
  - `min_component_mean_prob in {0.80, 0.82, 0.85}`
- Candidate banks promoted to mixed training:
  - `SegFormer-B2`
    - calibrated-bank baseline: `generated/uav_hard_negatives_segformer_b2_train_thr080_ts`
    - retuned balanced candidate: `tmp/ts_mining_sweep/segformer_b2_plain_360/thr080_mean082`
      - `191` crops from `100` UAV train images
    - retuned large-context control: `tmp/ts_mining_sweep/segformer_b2_plain_360/thr075_mean080`
      - `144` crops from `97` UAV train images
  - `DeepLabV3+`
    - calibrated-bank baseline: `generated/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080_ts`
    - retuned broad candidate: `tmp/ts_mining_sweep/deeplabv3plus_plain_360/thr080_mean080`
      - `158` crops from `89` UAV train images
    - retuned conservative candidate: `tmp/ts_mining_sweep/deeplabv3plus_plain_360/thr080_mean082`
      - `99` crops from `63` UAV train images

### Validation Results

- `SegFormer-B2`
  - original `B1` reference: `segformer_b1_negbank_uavval`
    - best epoch: `20`
    - validation `IoU`: `0.3419`
    - validation `F1`: `0.4812`
  - calibrated-bank baseline: `segformer_b1_tsbank_uavval`
    - best epoch: `5`
    - validation `IoU`: `0.3568`
    - validation `F1`: `0.5027`
    - validation `precision`: `0.5499`
    - validation `recall`: `0.5236`
  - retuned balanced candidate: `segformer_b2_b1_tsbank_thr080_mean082`
    - best epoch: `10`
    - validation `IoU`: `0.3693`
    - validation `F1`: `0.5227`
    - validation `precision`: `0.4996`
    - validation `recall`: `0.6096`
  - retuned large-context control: `segformer_b2_b1_tsbank_thr075_mean080`
    - best epoch: `10`
    - validation `IoU`: `0.2558`
    - validation `F1`: `0.3782`
    - validation `precision`: `0.3280`
    - validation `recall`: `0.6191`
- `DeepLabV3+`
  - original `B1` reference: `deeplabv3plus_b1_negbank_uavval`
    - best epoch: `9`
    - validation `IoU`: `0.2704`
    - validation `F1`: `0.4009`
  - calibrated-bank baseline: `deeplabv3plus_b1_tsbank_uavval`
    - best epoch: `11`
    - validation `IoU`: `0.3149`
    - validation `F1`: `0.4512`
    - validation `precision`: `0.5672`
    - validation `recall`: `0.4928`
  - retuned broad candidate: `deeplabv3plus_b1_tsbank_thr080_mean080`
    - best epoch: `13`
    - validation `IoU`: `0.3068`
    - validation `F1`: `0.4544`
    - validation `precision`: `0.4439`
    - validation `recall`: `0.5218`
  - retuned conservative candidate: `deeplabv3plus_b1_tsbank_thr080_mean082`
    - best epoch: `5`
    - validation `IoU`: `0.2444`
    - validation `F1`: `0.3803`
    - validation `precision`: `0.4056`
    - validation `recall`: `0.4497`

### Notes

- The rows above document the historical validation sweep, but confirmatory fixed-hold-out follow-up has now been completed on `2026-05-03`.
- `SegFormer-B2` shows the clearest support for the calibrated-bank idea:
  - even the default calibrated bank beats the original `B1` validation IoU
  - the retuned `thr080_mean082` bank improves it further to `0.3693`
  - the default-threshold TS-bank hold-out row reaches `IoU 0.3728`, `F1 0.5276`
  - the validation-selected `@ 0.6` TS-bank hold-out row reaches `IoU 0.3775`, `F1 0.5317`
- The `SegFormer-B2` large-context control `thr075_mean080` underperforms badly, so larger connected components or broader context are not automatically beneficial after temperature calibration.
- `DeepLabV3+` reacts less cleanly:
  - both the default calibrated bank and the broader `thr080_mean080` retune beat the original `B1` validation IoU
  - the more conservative `thr080_mean082` retune falls below the original `B1`
- On `2026-04-29`, a first `DeepLabV3+ TS-bank` qualitative audit package was generated at `results/hard_negative_audit/deeplabv3plus_tsbank_round1` using `scripts/banks/make_hard_negative_audit_assets.py`.
- The audit package contains `223` review cards plus `audit_samples.csv` covering:
  - `generated/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080`
  - `generated/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080_ts`
  - `tmp/ts_mining_sweep/deeplabv3plus_plain_360/thr080_mean080`
  - `tmp/ts_mining_sweep/deeplabv3plus_plain_360/thr080_mean082`
- The package was then upgraded on `2026-04-29` to a v4-compatible two-layer annotation schema:
  - layer 1 = sample-quality judgment such as `hard_fp / ambiguous / noise / bad_crop / gt_issue`
  - layer 2 = nuisance taxonomy such as `pavement_edge / shadow_dark_stripe / line_like_texture / surface_boundary / debris_object`
- At that stage, the diagnostic priority was to manually judge whether these selected crops were genuine crack-like hard false positives, ambiguous-but-useful negatives, or mostly `noise / bad_crop / gt_issue`, and then assign a nuisance category whenever the visual pattern was identifiable.
- Formal manual annotation for this audit package began on `2026-04-30`.

### Audit Completion Update

- On `2026-05-02`, the `DeepLabV3+ TS-bank` review was completed:
  - layer 1 completion: `223 / 223`
  - layer 2 completion: `222 / 223`
  - aggregate layer-1 distribution: `noise = 129`, `hard_fp = 66`, `ambiguous = 27`, `bad_crop = 1`
- Reviewed keepable shares (`hard_fp + ambiguous`) for the audited `DeepLabV3+` banks were:
  - calibrated-bank baseline `uav_hard_negatives_deeplabv3plus_plain_360_train_thr080_ts`: `23 / 43 = 53.5%`
  - original source-bank baseline `uav_hard_negatives_deeplabv3plus_plain_360_train_thr080`: `28 / 60 = 46.7%`
  - retuned broad candidate `thr080_mean080`: `23 / 60 = 38.3%`
  - retuned conservative candidate `thr080_mean082`: `19 / 60 = 31.7%`
- `scripts/banks/export_curated_hard_negative_bank.py` now converts the reviewed `audit_samples.csv` rows into trainable curated bank roots.
- `scripts/banks/export_auto_filtered_hard_negative_bank.py` now exports simple rule-filtered variants of the same mined bank for low-cost audit-derived follow-up.
- Curated-bank validation follow-up after the audit:
  - `deeplabv3plus_b1_tsbank_curated_hfpa_uavval`
    - curated bank: calibrated `TS-bank` filtered to `hard_fp + ambiguous`
    - best epoch: `9`
    - validation `IoU`: `0.2920`
    - validation `F1`: `0.4301`
    - validation `precision`: `0.5727`
    - validation `recall`: `0.4790`
  - `deeplabv3plus_b1_tsbank_curated_hardfp_uavval`
    - curated bank: calibrated `TS-bank` filtered to `hard_fp` only
    - best epoch: `10`
    - validation `IoU`: `0.3099`
    - validation `F1`: `0.4567`
    - validation `precision`: `0.5677`
    - validation `recall`: `0.4570`
- Threshold sweep on the curated `hard_fp` checkpoint:
  - `thr = 0.35`: `IoU = 0.3112`, `F1 = 0.4600`, `precision = 0.5162`, `recall = 0.5050`
  - `thr = 0.40`: `IoU = 0.3122`, `F1 = 0.4605`, `precision = 0.5340`, `recall = 0.4893`
  - `thr = 0.45`: `IoU = 0.3114`, `F1 = 0.4589`, `precision = 0.5518`, `recall = 0.4730`
  - `thr = 0.50`: `IoU = 0.3099`, `F1 = 0.4567`, `precision = 0.5677`, `recall = 0.4570`
- Audit-derived auto-filter follow-up on the calibrated `TS-bank`:
  - `deeplabv3plus_b1_tsbank_autofilter_span035_uavval`
    - auto-filter: keep crops with `span_ratio >= 0.35`
    - audited kept subset: `23 / 43` = `15 hard_fp`, `3 ambiguous`, `5 noise`
    - best explicit rerun epoch: `13`
    - validation `IoU`: `0.2783`
    - validation `F1`: `0.4086`
    - validation `precision`: `0.6296`
    - validation `recall`: `0.3660`
  - `deeplabv3plus_b1_tsbank_autofilter_area1200_uavval`
    - auto-filter: keep crops with `component_area >= 1200`
    - audited kept subset: `28 / 43` = `16 hard_fp`, `4 ambiguous`, `8 noise`
    - best epoch: `25`
    - validation `IoU`: `0.3564`
    - validation `F1`: `0.5136`
    - validation `precision`: `0.5420`
    - validation `recall`: `0.5329`
- Interpretation after audit completion:
  - the curated follow-up confirms that `ambiguous` crops dilute the calibrated bank for `DeepLabV3+`, because `hard_fp`-only filtering is clearly stronger than `hard_fp + ambiguous`
  - the `span_ratio >= 0.35` rule is too aggressive: although it strips out most audited `noise`, it also trims useful nuisance structure and underperforms both the raw calibrated bank and the curated `hard_fp` export
  - the `component_area >= 1200` rule works much better because it removes tiny junk while preserving all audited `hard_fp`; it therefore beats the raw calibrated bank (`0.3564 > 0.3149`) and the best curated operating point (`0.3564 > 0.3122`) on validation, then confirms on the fixed hold-out split at `IoU 0.3860`
  - the updated reading is more specific than the earlier purity-versus-diversity story: the best transfer bank is not the visually cleanest one, but the one that suppresses small noisy components while preserving larger target-domain nuisance diversity
- Matched random-background control follow-up:
  - first single-run `SegFormer-B2` control:
    - `segformer_b2_b1_random_bg_thr080_mean082_uavval`
    - validation `IoU`: `0.2747`
    - validation `F1`: `0.4040`
    - comparison target: mined `segformer_b2_b1_tsbank_thr080_mean082` at `IoU 0.3693`, `F1 0.5227`
  - paired `5-seed` `SegFormer-B2` rerun:
    - random-background seeds `42 / 7 / 13 / 21 / 99`: `IoU 0.3458, 0.3268, 0.3629, 0.3101, 0.3551`
    - random-background mean `IoU`: `0.3401 +- 0.0215`
    - random-background mean `F1`: `0.4791 +- 0.0237`
    - mined-bank seeds `42 / 7 / 13 / 21 / 99`: `IoU 0.3143, 0.3284, 0.3275, 0.3455, 0.2815`
    - mined-bank mean `IoU`: `0.3194 +- 0.0239`
    - mined-bank mean `F1`: `0.4584 +- 0.0247`
  - `deeplabv3plus_b1_random_bg_area1200_uavval`
    - validation `IoU`: `0.3023`
    - validation `F1`: `0.4368`
    - comparison target: mined `deeplabv3plus_b1_tsbank_autofilter_area1200_uavval` at `IoU 0.3564`, `F1 0.5136`
- `SegFormer-B2` audit-derived curated-bank follow-up:
  - a first `SegFormer-B2` audit package is now available at `results/hard_negative_audit/segformer_b2_tsbank_round1` with `180` reviewed cards across `thr080_mean082`, `thr080_mean080`, and `thr075_mean080`
  - reviewed keepable shares (`hard_fp + ambiguous`) were:
    - `thr080_mean080`: `20 / 60 = 33.3%`
    - `thr080_mean082`: `17 / 60 = 28.3%`
    - `thr075_mean080`: `12 / 60 = 20.0%`
  - curated-bank validation follow-up:
    - `segformer_b2_b1_thr080_mean080_curated_hfpa_seed042_uavval`
      - curated bank: `thr080_mean080` filtered to `hard_fp + ambiguous`
      - best epoch: `10`
      - validation `IoU`: `0.3439`
      - validation `F1`: `0.4830`
      - validation `precision`: `0.5142`
      - validation `recall`: `0.5111`
    - `segformer_b2_b1_thr080_mean080_curated_hardfp_seed042_uavval`
      - curated bank: `thr080_mean080` filtered to `hard_fp` only
      - best epoch: `11`
      - validation `IoU`: `0.3420`
      - validation `F1`: `0.4836`
      - validation `precision`: `0.4709`
      - validation `recall`: `0.5749`
    - `segformer_b2_b1_thr080_mean082_curated_hfpa_seed042_uavval`
      - curated bank: `thr080_mean082` filtered to `hard_fp + ambiguous`
      - best epoch: `9`
      - validation `IoU`: `0.3304`
      - validation `F1`: `0.4682`
      - validation `precision`: `0.4532`
      - validation `recall`: `0.5400`
    - `segformer_b2_b1_thr080_mean082_curated_hardfp_seed042_uavval`
      - curated bank: `thr080_mean082` filtered to `hard_fp` only
      - best epoch: `9`
      - validation `IoU`: `0.3535`
      - validation `F1`: `0.4993`
      - validation `precision`: `0.5711`
      - validation `recall`: `0.5222`
- Interpretation after the matched random-background control:
  - `DeepLabV3+` remains directionally consistent with useful mined nuisance signal, but that evidence is still weak because the control bank could keep same-image matching for only `3 / 28` crops
  - `SegFormer-B2` no longer supports the earlier strong mined-signal reading: once the comparison is rerun with paired `5-seed` controls, matched random background is at least competitive with the mined bank and averages slightly higher on `UAV val`
  - the `SegFormer-B2` curated audit follow-up is explanatory rather than promotable: the strongest curated export is `thr080_mean082 hard_fp`, but it still stays below the raw mined validation winner (`0.3535 < 0.3693`)
  - on `SegFormer-B2`, `ambiguous` crops clearly dilute the `thr080_mean082` bank (`0.3304 -> 0.3535`), while the `thr080_mean080` curated exports are nearly tied and do not isolate the same effect
  - the updated mechanism conclusion is therefore model-dependent rather than universal: generic target-background exposure explains a substantial share of the gain, and mined-bank-specific value is not yet established on the main `SegFormer-B2` backbone
- Updated promotion rule:
  - `SegFormer-B2`: freeze `segformer_b2_b1_tsbank_thr080_mean082_test_thr060` as the official `GT`-filtered `B1` row; keep `B1 raw @ 0.9` as the high-precision comparison point and keep the older raw/postprocess rows as historical calibration context
  - `DeepLabV3+`: freeze `deeplabv3plus_b1_tsbank_autofilter_area1200_test` as the official `GT`-filtered `B1` row; keep the curated-bank variants and `span_ratio` rule as diagnostic follow-ups rather than as the promoted row

## Run 010: SegFormer-B2 Promoted `TS-bank` `B1` Official Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-05-03  
**Purpose:** Freeze the official `GT`-filtered `SegFormer-B2 B1` row after validation-first bank selection and confirmatory fixed-hold-out testing

### Model

- Architecture: `SegFormer-B2`
- Pretrained model: `nvidia/segformer-b2-finetuned-ade-512-512`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Main supervised dataset: `Crack500`
- Validation dataset: `UAV_Crack_Segmentation_Kaggle/val`
- Auxiliary negative bank: `tmp/ts_mining_sweep/segformer_b2_plain_360/thr080_mean082`
- Auxiliary negative repeat: `2`
- Image size: `360`
- Batch size: `8`
- Learning rate: `1e-4`
- Augmentation: `baseline`
- Max epochs: `30`
- Loss: `BCE + Dice`

### Checkpoint

- Best checkpoint path: `checkpoints/segformer_b2_b1_tsbank_thr080_mean082.pth`
- Best validation checkpoint: `epoch 10`
- Best validation IoU on `UAV val`: `0.3693`
- Best validation F1 on `UAV val`: `0.5227`
- Best validation precision on `UAV val`: `0.4996`
- Best validation recall on `UAV val`: `0.6096`

### Test Result

- Raw promoted operating point at threshold `0.6`
  - Test loss: `0.5676`
  - Test IoU: `0.3775`
  - Test F1: `0.5317`
  - Test precision: `0.5257`
  - Test recall: `0.5779`

### Notes

- This is the official `GT`-filtered `SegFormer-B2 B1` row for paper-facing comparison tables.
- The promoted row is still a raw prediction line; it is threshold-selected on `UAV val`, not postprocessed on the hold-out split.
- `Run 003` remains important as the first-generation `B1` checkpoint and operating-point history, but it is no longer the final `B1` citation target.
- Audit-derived curated reruns and random-background controls remain supporting mechanism evidence in `Run 009`, not replacements for this frozen row.

## Run 011: DeepLabV3+ Promoted `TS-bank area1200` `B1` Official Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-05-03  
**Purpose:** Freeze the official `GT`-filtered `DeepLabV3+ B1` row after audit-derived bank selection and confirmatory fixed-hold-out testing

### Model

- Architecture: `DeepLabV3+`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Main supervised dataset: `Crack500`
- Validation dataset: `UAV_Crack_Segmentation_Kaggle/val`
- Auxiliary negative bank: `generated/auto_filtered_banks/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080_ts__area1200`
- Auxiliary negative repeat: `3`
- Image size: `360`
- Batch size: `8`
- Learning rate: `1e-4`
- Augmentation: `baseline`
- Max epochs: `40`
- Early stopping patience: `10`
- Loss: `BCE + Dice`

### Checkpoint

- Best checkpoint path: `checkpoints/deeplabv3plus_b1_tsbank_autofilter_area1200.pth`
- Best validation checkpoint: `epoch 25`
- Best validation IoU on `UAV val`: `0.3564`
- Best validation F1 on `UAV val`: `0.5136`
- Best validation precision on `UAV val`: `0.5420`
- Best validation recall on `UAV val`: `0.5329`

### Test Result

- Raw promoted operating point at threshold `0.5`
  - Test loss: `0.5403`
  - Test IoU: `0.3860`
  - Test F1: `0.5470`
  - Test precision: `0.6195`
  - Test recall: `0.5070`

### Notes

- This is the official `GT`-filtered `DeepLabV3+ B1` row for paper-facing comparison tables.
- The promoted bank is the audit-derived `component_area >= 1200` auto-filter export rather than the earlier raw mined bank or the curated-only exports.
- `Run 007` remains the first-generation `DeepLabV3+ B1` reference, but it is now historical comparison rather than the final `B1` citation target.
- The curated-bank variants and the `span_ratio` rule stay diagnostic and claim-bounding; they do not replace this frozen row.

## Run 012: U-Net Source-Only Cross-Domain Hold-Out on UAV

**Date recorded:** 2026-04-27  
**Purpose:** Formalize the fixed-hold-out `U-Net` source-only row used as the classical CNN reference in final comparison assets

### Model

- Architecture: `U-Net`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Training Origin

- Source-domain checkpoint: `checkpoints/unet_ablate_aug_mild_360_fgcrop.pth`
- Source training dataset: `Crack500`
- Source recipe: `360 + mild augmentation + foreground-aware crop (0.5) + BCE + Dice`

### Evaluation Setup

- Dataset: `UAV_Crack_Segmentation_Kaggle`
- Official hold-out split: `test`
- Dataset size: `63`
- Image size: `360`
- Batch size: `16`

### Test Result

- Raw prediction
  - Test loss: `1.0803`
  - Test IoU: `0.1284`
  - Test F1: `0.2244`
  - Test precision: `0.1330`
  - Test recall: `0.8040`

### Notes

- This run formalizes the `U-Net` source-only row already used in `results/report_assets/final_comparison/main_results.*`.
- `U-Net` is kept as a classical source-only reference rather than a full paper-story backbone; no official `U-Net B1`, `U-Net B2`, or `U-Net` target-domain upper-bound line is frozen here.
- If a tighter final manuscript table is needed, `U-Net` can move to an appendix or secondary comparison table without changing the main promoted-result backbone.

## Run 013: DeepLabV3+ In-Domain Upper Bound on UAV

**Date recorded:** 2026-05-07  
**Purpose:** Add the target-domain ceiling reference for the `DeepLabV3+` backbone on the fixed `UAV_Crack_Segmentation_Kaggle` split

### Model

- Architecture: `DeepLabV3+`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`

### Training Setup

- Dataset: `UAV_Crack_Segmentation_Kaggle`
- Train split: `train`
- Validation split: `val`
- Test split: `test`
- Image size: `360`
- Batch size: `8`
- Learning rate: `1e-4`
- Augmentation: `baseline`
- Max epochs: `40`
- Early stopping: enabled
- Loss: `BCE + Dice`

### Checkpoint

- Best checkpoint path: `checkpoints/deeplabv3plus_uav_indomain_plain_360.pth`
- Best validation checkpoint: `epoch 37`
- Best validation IoU on `UAV val`: `0.5068`
- Best validation F1 on `UAV val`: `0.6677`

### Test Result

- Raw prediction
  - Test loss: `0.4958`
  - Test IoU: `0.5085`
  - Test F1: `0.6693`
  - Test precision: `0.6187`
  - Test recall: `0.7295`

### Notes

- This run mirrors the existing `SegFormer-B2` UAV upper-bound protocol so the ceiling comparison remains architecture-consistent:
  - same dataset split family (`train / val / test`)
  - same image size (`360`)
  - same batch size (`8`)
  - same learning rate (`1e-4`)
  - same augmentation profile (`baseline`)
  - same maximum epochs (`40`)
- The result now closes an important comparison gap for the paper because `DeepLabV3+` already has frozen `source-only`, `B1`, and `B2` rows.
- Relative to the frozen `DeepLabV3+` `B2 fs20` row (`IoU 0.4760`), the in-domain ceiling is higher but not dramatically so (`0.5085`), which suggests that limited target supervision already recovers a large fraction of the remaining gap for this backbone as well.

## Run 014: DeepLabV3+ ADVENT-Style UDA Cross-Domain Baseline on UAV

**Date recorded:** 2026-05-24  
**Purpose:** Add a reviewer-facing unsupervised domain adaptation comparison for `Crack500 -> UAV_Crack_Segmentation_Kaggle`, covering the ADVENT family of output-space adversarial adaptation methods

### Model And Local Implementation

- Segmentation backbone: `DeepLabV3+`
- Encoder: `resnet34`
- Encoder weights: `imagenet`
- Input channels: `3`
- Output classes: `1`
- Local training entry point: `train_advent.py`
- Reusable experiment command: `scripts/experiments/run_advent_deeplabv3plus.sh`
- Threshold-selection/reporting entry point: `scripts/reports/run_threshold_sweep_report.py`
- ADVENT component: binary entropy-map discriminator trained on source-vs-target output entropy maps
- Deployment rule: only the segmentation checkpoint is used for validation/test inference; the discriminator is a training-only component and is not loaded at deployment time

This is an ADVENT-style adaptation implemented inside the repository's binary crack-segmentation protocol. It should be described as `ADVENT-style output-space adversarial adaptation` or `ADVENT adapted to our binary crack-segmentation setting`, not as a byte-for-byte reproduction of the official ADVENT DeepLabV2 implementation.

### Training Setup

- Source supervised dataset: `CRACK500/train`
- Target adaptation dataset: `UAV_Crack_Segmentation_Kaggle/train`
- Target-label rule: target masks are ignored during ADVENT training; target images contribute only through entropy/adversarial losses
- Validation dataset for checkpoint selection: `UAV_Crack_Segmentation_Kaggle/val`
- Final test dataset: `UAV_Crack_Segmentation_Kaggle/test`
- Initialization checkpoint: `checkpoints/deeplabv3plus_plain_360.pth`
- Output segmentation checkpoint: `checkpoints/advent_deeplabv3plus_crack500_to_uav.pth`
- Output discriminator checkpoint: `checkpoints/advent_deeplabv3plus_crack500_to_uav_discriminator.pth`
- Image size: `360`
- Training batch size: `4`
- Validation/test reporting batch size: `8`
- Learning rate: `1e-4`
- Discriminator learning rate: `1e-4`
- Target adversarial loss weight: `1e-3`
- Max epochs: `30`
- Early stopping patience: `8`
- Seed: `42`
- Loss for supervised source segmentation: `BCE + Dice`

### Checkpoint Selection

- Best checkpoint selected during ADVENT training: `epoch 17`
- Training-time validation metric at threshold `0.5`:
  - Validation IoU: `0.1974`
  - Validation F1: `0.2986`
  - Validation precision: `0.2255`
  - Validation recall: `0.6179`

### Threshold Selection Protocol

- Threshold selection was performed after checkpoint selection using `UAV val` only.
- Candidate thresholds: `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`
- Selection metric: validation `IoU`
- Selected threshold: `0.9`
- Report assets:
  - `results/report_assets/advent_deeplabv3plus_threshold_sweep/threshold_sweep_val.csv`
  - `results/report_assets/advent_deeplabv3plus_threshold_sweep/selected_threshold_test.csv`
  - `results/report_assets/advent_deeplabv3plus_threshold_sweep/summary.md`

Validation sweep for the selected checkpoint:

| Threshold | Val IoU | Val F1 | Val Precision | Val Recall |
| --- | ---: | ---: | ---: | ---: |
| `0.1` | `0.1458` | `0.2447` | `0.1511` | `0.7590` |
| `0.2` | `0.1593` | `0.2631` | `0.1665` | `0.7357` |
| `0.3` | `0.1677` | `0.2739` | `0.1767` | `0.7119` |
| `0.4` | `0.1744` | `0.2822` | `0.1854` | `0.6894` |
| `0.5` | `0.1804` | `0.2893` | `0.1938` | `0.6657` |
| `0.6` | `0.1858` | `0.2956` | `0.2021` | `0.6395` |
| `0.7` | `0.1908` | `0.3014` | `0.2112` | `0.6094` |
| `0.8` | `0.1962` | `0.3075` | `0.2232` | `0.5685` |
| `0.9` | `0.2036` | `0.3157` | `0.2455` | `0.5003` |

### Final Test Result

- Confirmatory raw test result at validation-selected threshold `0.9`
  - Test loss: `0.8968`
  - Test IoU: `0.2022`
  - Test F1: `0.3076`
  - Test precision: `0.2427`
  - Test recall: `0.4845`

For same-protocol context, `DeepLabV3+` source-only threshold selection on `UAV val` selected threshold `0.8` and reached:

- Source-only test at validation-selected threshold `0.8`
  - Test IoU: `0.1282`
  - Test F1: `0.2202`
  - Test precision: `0.1434`
  - Test recall: `0.5689`

### Interpretation

- ADVENT-style output-space adaptation improves over the same-backbone source-only baseline under a validation-selected threshold protocol (`IoU 0.1282 -> 0.2022`, `F1 0.2202 -> 0.3076`).
- The gain is driven primarily by better false-positive control (`precision 0.1434 -> 0.2427`) rather than better recall (`0.5689 -> 0.4845`).
- The selected threshold of `0.9` is itself diagnostic: the adapted model still needs a highly conservative operating point on UAV imagery, indicating that implicit output-space alignment does not fully remove target-domain false-positive pressure.
- Compared with the promoted `DeepLabV3+ B1` row (`IoU 0.3860`, `F1 0.5470` at threshold `0.5`), ADVENT-style adaptation helps but remains much weaker.
- This supports the paper's claim boundary: generic adversarial alignment can improve cross-domain calibration, but explicit target-background / hard-negative exposure is more effective for the structured false positives that dominate ground-to-UAV crack segmentation transfer.

## Run 015: HRDA/MiT-B5 Reviewer UDA Cross-Domain Baseline on UAV

**Date recorded:** 2026-05-27  
**Purpose:** Add a reviewer-facing `HRDA` baseline for `Crack500 -> UAV_Crack_Segmentation_Kaggle` under the same unified `360 x 360` protocol used for the DAFormer comparison

### Model And Local Implementation

- Segmentation model: `HRDAEncoderDecoder`
- Single-scale decoder family: `DAFormerHead`
- Decoder variant: `daformer_sepaspp_mitb5`
- Encoder: `MiT-B5`
- Input channels: `3`
- Output classes: `2` (`background`, `crack`)
- External code root: `external/HRDA`
- External snapshot checked locally: `c370b5b`
- Project-owned config: `configs/hrda/crack500_to_uav_hrda_mitb5_s0.py`
- Reusable runner: `scripts/experiments/run_hrda_crack500_to_uav.sh`
- Threshold-selection/reporting entry point: `scripts/daformer/evaluate_daformer_threshold_sweep.py`

This is HRDA adapted to the repository's local binary crack-segmentation protocol, not a claim of byte-for-byte reproduction of the original urban-scene benchmark stack.

### Training Setup

- Source supervised dataset: `Crack500/train`
- Target adaptation dataset: `UAV/train`
- Validation dataset for threshold selection: `UAV/val`
- Final test dataset: `UAV/test`
- MMSeg-format dataset root: `generated/hrda/crack500_to_uav`
- Dataset sizes:
  - source train: `1896`
  - target train: `189`
  - target val: `63`
  - target test: `63`
- Target-label rule: target masks are ignored during HRDA training and used only for validation/test reporting
- Backbone initialization: `external/HRDA/pretrained/mit_b5.pth`
- Outer image size: `360`
- HR detail crop size: `176`
- Feature scale: `0.5`
- Dual-scale settings:
  - `scales = [1, 0.5]`
  - `crop_coord_divisible = 8`
  - `attention_classwise = True`
  - `hr_loss_weight = 0.1`
  - `hr_slide_inference = True`
- Training batch size: `1`
- Validation/test reporting workers: `1`
- Seed: `42`

### Optimization And UDA Parameters

- UDA family: `DACS`
- Mixing mode: `class`
- EMA alpha: `0.999`
- Pseudo-label threshold: `0.968`
- ImageNet feature distance:
  - `lambda = 0.005`
  - classes `[1]`
  - `scale_min_ratio = 0.75`
- Rare-class sampling:
  - `min_pixels = 100`
  - `class_temp = 0.01`
  - `min_crop_ratio = 0.5`
- Additional UDA switches:
  - `blur = True`
  - `color_jitter_strength = 0.2`
  - `color_jitter_probability = 0.2`
- Optimizer: `AdamW`
- Learning rate: `6e-5`
- Weight decay: `0.01`
- Paramwise multipliers:
  - `head.lr_mult = 10.0`
  - `pos_block.decay_mult = 0.0`
  - `norm.decay_mult = 0.0`
- Schedule: `poly10warm`
- Warmup iters: `1500`
- Max iterations: `40000`
- Checkpoint interval: `4000`
- Validation interval: `4000`

### Threshold Selection Protocol

- Threshold selection was performed after training using `UAV val` only.
- Candidate thresholds: `0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`
- Selection metric: validation `IoU`
- Selected threshold: `0.3`
- Report assets:
  - `results/report_assets/hrda_mitb5_threshold_sweep_360/threshold_sweep_val.csv`
  - `results/report_assets/hrda_mitb5_threshold_sweep_360/selected_threshold_test.csv`
  - `results/report_assets/hrda_mitb5_threshold_sweep_360/summary.md`

Validation sweep for the selected checkpoint:

| Threshold | Val IoU | Val F1 | Val Precision | Val Recall |
| --- | ---: | ---: | ---: | ---: |
| `0.1` | `0.0782` | `0.1450` | `0.0794` | `0.8384` |
| `0.2` | `0.0824` | `0.1522` | `0.0841` | `0.7956` |
| `0.3` | `0.0840` | `0.1550` | `0.0863` | `0.7576` |
| `0.4` | `0.0839` | `0.1548` | `0.0867` | `0.7189` |
| `0.5` | `0.0832` | `0.1537` | `0.0866` | `0.6788` |
| `0.6` | `0.0822` | `0.1519` | `0.0863` | `0.6342` |
| `0.7` | `0.0799` | `0.1480` | `0.0848` | `0.5779` |
| `0.8` | `0.0757` | `0.1407` | `0.0818` | `0.5036` |
| `0.9` | `0.0685` | `0.1283` | `0.0766` | `0.3930` |

### Final Test Result

- Confirmatory raw test result at validation-selected threshold `0.3`
  - Test IoU: `0.1143`
  - Test F1: `0.2052`
  - Test precision: `0.1191`
  - Test recall: `0.7390`

For same-protocol context, the completed `DAFormer/MiT-B5` reviewer baseline selected threshold `0.6` and reached:

- `UAV test`: `IoU 0.1353`, `F1 0.2384`, `precision 0.1388`, `recall 0.8442`

### Interpretation

- Under the repository's unified `360 x 360` crack protocol, HRDA did not beat the simpler DAFormer reviewer baseline.
- The selected threshold `0.3` and final `precision 0.1191` indicate that the best validation operating point remains recall-heavy rather than precision-stable.
- This is therefore a completed negative reviewer baseline, but still useful evidence: dual-scale high-resolution fusion alone did not solve the structured UAV false-positive problem in this local setting.
