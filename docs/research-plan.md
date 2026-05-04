# Research Plan and Progress Summary

**Author:** Sizhe Guan  
**Original planning date:** 2026-04-20  
**Repository snapshot updated:** 2026-05-04

This document is an English project version of the original planning notes and is intended to serve as the long-term research reference for this repository.

## 1. Research Direction

### Research Question

How much does the performance of existing Transformer-based crack segmentation models, represented by `SegFormer`, degrade when trained on ground-level crack images and transferred to UAV imagery under cross-resolution and illumination disturbances? Which lightweight strategies that do not modify the backbone architecture can effectively mitigate this degradation?

### Background and Motivation

- Existing crack detection studies rely heavily on traditional ground-image datasets such as `Crack500`, `DeepCrack`, and `CFD`, while UAV-specific benchmarks remain fragmented.
- Models often perform well on benchmark datasets but degrade significantly under changes in lighting, material, and resolution. This is a well-recognized bottleneck in the field.
- `SegFormer` does not use fixed positional encoding and is therefore theoretically more robust to resolution changes, making it a strong candidate for systematic evaluation.

### Paper Type

Controlled benchmark / evaluation study.

The goal is not to introduce a new model architecture, but to systematically evaluate existing models in realistic UAV scenarios and provide a reusable recipe of lightweight improvements.

## 2. Experimental Design

### Datasets

| Dataset | Purpose | Status |
| --- | --- | --- |
| `Crack500` | Ground-domain source dataset | Downloaded |
| `DronePavSeg` | Preferred UAV target-domain dataset | Access request in progress |
| `UAV-PDD2023` | Backup UAV target-domain dataset | Publicly available |

### Models

- `SegFormer-B2` as the main model using the `MiT-B2` encoder, about `24.2M` parameters
- `U-Net` as the CNN baseline and a standard comparison model for crack segmentation
- `DeepLabV3+` as an additional ASPP-style CNN baseline for architecture-side comparison

### Experiment Matrix

- In-domain: `Crack500` train + test, and UAV dataset train + test
- Cross-domain: `Crack500` train -> UAV dataset test
- Perturbation testing on the UAV test set with:
  - brightness shifts
  - shadow changes
  - contrast changes
  - resolution downsampling

### Lightweight Improvement Strategies

These strategies keep the backbone architecture unchanged:

- Photometric augmentation:
  - brightness
  - contrast
  - shadow
  - blur
- Resolution strategy:
  - multi-scale training
  - progressively increasing training resolution
- Loss and sampling strategy:
  - `BCE + Dice`
  - foreground-aware sampling

### Ablation Protocol

All tuning after the frozen `U-Net` baseline should follow a one-change-at-a-time protocol.

- Keep `unet_baseline_360` as the frozen reference.
- Change only one factor per experiment when possible:
  - resolution
  - foreground-aware sampling
  - loss
  - photometric augmentation strength
- Use the validation set for model selection and reserve the test set for confirmatory reporting.
- Interpret negative results carefully when more than one intervention is changed in the same run.

### Evaluation Metrics

- `mIoU` as the main metric
- `F1-score`
- `Precision`
- `Recall`
- performance curves under each perturbation condition

### Reporting Principle

- Raw model predictions remain the default baseline for both in-domain and cross-domain reporting.
- This baseline is intended to measure the model's own generalization ability, so it should not be rewritten by postprocessing.
- Postprocessed outputs are reported separately as deployment-oriented variants when they are useful for cluttered target-domain scenes.
- A postprocessed variant is only promoted as a separate deployment result if its benefit is explained as a clear operating-point tradeoff rather than as a universal improvement.

## 3. Hardware and Environment

| Item | Configuration |
| --- | --- |
| GPU | `NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)` |
| System | `Windows 11 + WSL2 Ubuntu` |
| CUDA | `13.1` |
| Python environment | conda `crackdet`, Python `3.10` |
| Main dependencies | `PyTorch`, `HuggingFace Transformers`, `segmentation-models-pytorch`, `albumentations` |

## 4. Current Progress

The table below reflects the current repository state as of `2026-05-04`, which is now ahead of the original draft in several implementation items.

| Task | Status | Notes |
| --- | --- | --- |
| Research direction finalized | Done | Cross-domain robustness, `SegFormer` vs `U-Net` |
| Experimental design finalized | Done | Datasets, models, experiment matrix, and improvement strategies defined |
| WSL2 GPU environment setup | Done | PyTorch + CUDA verified |
| conda environment `crackdet` created | Done | Python `3.10` and dependencies installed |
| `Crack500` downloaded | Done | `traindata` / `valdata` / `testdata` structure available |
| Git repository initialized | Done | `~/dev/crack-detection` |
| UAV dataset application | In progress | Waiting for response from `DronePavSeg` authors |
| Data loading code | Done | Implemented in `dataset.py` |
| Dataset sanity checks | Done | Implemented in `check_dataset.py` |
| Generic segmentation baseline scaffolding | Done | CLI training and test scripts are now in place |
| `U-Net` baseline on `Crack500` | Done | First CNN baseline has been trained, checkpointed, and tested |
| Baseline qualitative visualization | Done | Fixed-sample and failure-case review notebook is now in place |
| Experiment logging and comparison dashboard | Done | CSV logging plus `notebooks/06_round1_experiments.ipynb` are now in place |
| Round-1 combined-recipe `U-Net` probe | Done | `512 + strong aug + foreground crop + Focal Tversky` underperformed vs the frozen baseline and motivated single-variable ablations |
| Single-variable `U-Net` ablation round on `Crack500` | Done | Resolution, sampling, loss, and augmentation were isolated; `360 + foreground crop + mild aug + BCE + Dice` is currently best |
| `SegFormer-B2` baseline training | Done | Plain `SegFormer-B2` baseline has now been trained and tested on `Crack500` |
| Initial `SegFormer-B2` augmentation transfer probe | Done | `mild` augmentation did not beat the plain baseline on validation and was not promoted to test |
| Initial `SegFormer-B2` foreground-sampling transfer probe | Done | Foreground crop improved validation slightly but did not improve test, so plain `SegFormer-B2` remains the working baseline |
| Initial `SegFormer-B2` loss transfer probe | Done | `Focal Tversky` underperformed the plain baseline on validation and was not promoted to test |
| Initial `SegFormer-B2` resolution transfer probe | Done | `512` resolution improved validation but reduced test IoU, so plain `360` remains the working baseline |
| Initial `DeepLabV3+` baseline and in-domain ablations on `Crack500` | Done | `plain / 512 / foreground crop / mild / Focal Tversky` are now recorded; `foreground crop` is currently best in-domain |
| `DeepLabV3+` source-only hold-out evaluation on `UAV` | Done | `plain` raw and deploy rows plus `foreground crop` transfer check are now recorded on the fixed `UAV test` split |
| `DeepLabV3+` UAV hard-negative bank mining | Done | A `DeepLabV3+`-specific all-background hard-negative bank is now available for future `B1`-style experiments |
| `DeepLabV3+` `B1` hard-negative-mixed hold-out evaluation on `UAV` | Done | `B1 raw @ 0.5` first reached `IoU 0.2830`; the promoted `TS-bank area1200 @ 0.5` hold-out update now raises the official `DeepLabV3+ B1` row to `0.3860` |
| `DeepLabV3+` `B2` few-shot fine-tuning on `UAV` | Done | `fs05/pat12`, `fs10`, and `fs20` validation checkpoints plus fixed-hold-out test rows are now logged; the `DeepLabV3+` supervision-efficiency curve is frozen on the official `UAV test` split |
| Temperature-scaled `B1` bank retuning on `UAV val` | Done | Confirmatory hold-out promotion is now complete: `SegFormer-B2 TS-bank thr080_mean082 @ 0.6` reaches `IoU 0.3775`, and `DeepLabV3+ TS-bank area1200 @ 0.5` reaches `IoU 0.3860` |
| Random-background control vs mined-bank `B1` check | Done | The final reading is now model-dependent: `DeepLabV3+` still shows `0.3023 < 0.3564`, but paired `5-seed` `SegFormer-B2` reruns reverse the average (`random_bg 0.3401 > mined 0.3194`), so `B1` cannot be explained by a single mined-signal story |
| `SegFormer-B2 TS-bank` qualitative audit package | Done | `make_hard_negative_audit_assets.py` exported a `180`-card review package across `thr080_mean082 / thr080_mean080 / thr075_mean080`; curated follow-up did not beat the raw mined winner (`best curated = thr080_mean082 hard_fp at IoU 0.3535 < 0.3693`), so this line remains diagnostic rather than promotable |
| `DeepLabV3+ TS-bank` qualitative audit package | Done | `make_hard_negative_audit_assets.py` exported the `223`-card review package, the manual review was completed on `2026-05-02`, and `export_curated_hard_negative_bank.py` now turns those audit decisions into trainable curated banks for follow-up validation |
| Cross-domain raw evaluation on `UAV_Crack_Segmentation_Kaggle` | Done | Raw prediction remains the official cross-domain baseline for both `U-Net` and `SegFormer-B2` |
| UAV postprocessed deployment-variant probe | Done | Optional connected-component filtering improves UAV precision but should be reported separately from the raw baseline |
| Official `UAV_Crack_Segmentation_Kaggle` hold-out split | Done | Fixed `train/val/test` and preserved `crossdomain_all` for exploratory diagnosis |
| `B1` hard-negative mining and mixed training | Done | `SegFormer-B2` now has a first target-aware training-side mitigation result on the fixed hold-out split |
| `UAV` in-domain upper bound for `SegFormer-B2` | Done | Fixed `UAV train/val/test` training now provides a target-domain ceiling reference near `IoU 0.60` |
| `B2` few-shot split family for `UAV` fine-tuning | Done | Reproducible `5% / 10% / 20%` subsets were generated from the official `UAV train` split |
| Paper writing | In progress | Core experimental claims are now frozen enough for drafting; an `AIC`-oriented outline plus `Introduction / Problem Setting / Method` draft, a separate `Drones`-oriented venue plan, and the `SegFormer` audit follow-up notebook are now in place |

### 4.1 U-Net Ablation Summary

The first single-variable `U-Net` ablation round now provides a much clearer picture than the earlier combined-recipe probe.

- Frozen reference `unet_baseline_360`
  - validation `IoU`: `0.6545`
  - test `IoU`: `0.5680`
- `512` only, while keeping baseline augmentation and `BCE + Dice`
  - validation `IoU`: `0.6471`
  - test `IoU`: `0.5737`
- `360 + foreground crop + BCE + Dice`
  - validation `IoU`: `0.6629`
  - test `IoU`: `0.5707`
- `360 + foreground crop + Focal Tversky`
  - validation `IoU`: `0.6397`
- `360 + foreground crop + mild augmentation + BCE + Dice`
  - validation `IoU`: `0.6681`
  - test `IoU`: `0.5747`
- `360 + foreground crop + strong augmentation + BCE + Dice`
  - latest validation `IoU`: `0.6627`

Interpretation:

- Increasing resolution alone does not beat the frozen baseline on validation.
- Foreground-aware sampling is helpful in this setting.
- `Focal Tversky` is harmful in this setting relative to `BCE + Dice`.
- `mild` photometric augmentation helps more than `strong`.
- The current best `U-Net` recipe is therefore:
  - image size `360`
  - foreground-aware crop probability `0.5`
  - augmentation profile `mild`
  - loss `BCE + Dice`

Experimental takeaway:

- The earlier negative combined-recipe result has now been decomposed into interpretable parts.
- The main gain came from foreground-aware sampling plus a moderate level of photometric augmentation.
- The present `U-Net` work can now be treated as a stabilized CNN reference recipe for later `SegFormer-B2` comparison.

### 4.2 Experiment Log Caveat

- `results/experiments.csv` currently contains repeated experiment keys such as `unet_ablate_aug_strong_360_fgcrop` and `deeplabv3plus_plain_360`.
- Any notebook or script that summarizes experiment outcomes should select the latest `timestamp_utc` row for repeated `(experiment_name, stage, split)` combinations.
- Otherwise, an older run may be mixed into the comparison and distort the reported ablation result.
- On `2026-04-26`, the experiment-log schema was expanded so that new runs now also store:
  - `train_split`, `val_dataset_root`, and `val_split`
  - `aux_negative_*` settings for `B1`
  - `init_checkpoint_path` for `B2`
  - `eval_threshold` and postprocessing parameters
  - `precision` and `recall` alongside `loss`, `IoU`, and `F1`
- The CSV header has already been migrated to this unified schema.
- Some older rows created before that migration may still have blank values in the newly added columns; for those historical runs, the curated values recorded in this document and in `docs/baseline-results.md` should be treated as authoritative.

### 4.3 SegFormer-B2 Baseline And Early Transferability Checks

The plain `SegFormer-B2` baseline on `Crack500` is now established, and the first architecture-transfer checks have been run.

- `segformer_b2_plain_360`
  - validation `IoU`: `0.6666`
  - test `IoU`: `0.5854`
- `segformer_b2_mild_360`
  - validation `IoU`: `0.6644`
  - not promoted to test because it did not beat the plain baseline on validation
- `segformer_b2_fgcrop_360`
  - validation `IoU`: `0.6703`
  - test `IoU`: `0.5816`
- `segformer_b2_ftversky_360`
  - validation `IoU`: `0.6460`
  - not promoted to test because it did not beat the plain baseline on validation
- `segformer_b2_512_plain`
  - validation `IoU`: `0.6726`
  - test `IoU`: `0.5713`

Interpretation:

- Plain `SegFormer-B2` is already competitive with the best tuned `U-Net` on validation.
- Plain `SegFormer-B2` is ahead of the current best `U-Net` on test.
- This supports the view that the Transformer baseline generalizes slightly better in the present in-domain setup.
- The `mild` augmentation that helped `U-Net` did not transfer cleanly to `SegFormer-B2`.
- Foreground-aware crop produced a small validation gain, but that gain did not hold on test.
- `Focal Tversky` also did not transfer cleanly to `SegFormer-B2`.
- Higher resolution produced a validation gain, but that gain did not hold on test either.
- Therefore, the present working `SegFormer-B2` baseline should remain the plain configuration rather than the foreground-crop variant.

Experimental takeaway:

- Helpful interventions are not automatically architecture-invariant.
- `SegFormer-B2` should now follow the same validation-first, one-change-at-a-time ablation logic rather than inheriting `U-Net` changes by default.
- When validation and test disagree on a small-margin ablation, the more conservative choice is to keep the simpler plain baseline as the active reference.
- After augmentation, sampling, loss, and resolution probes all failed to displace the plain baseline, in-domain tuning should pause and the study should move on to cross-domain evaluation.

### 4.4 DeepLabV3+ Baseline And Early Transferability Checks

`DeepLabV3+` has now been added as an additional CNN-style segmentation baseline and evaluated under the same validation-first protocol.

In-domain `Crack500` results:

- `deeplabv3plus_plain_360`
  - the earliest quick-stop run with `patience = 5` peaked too early at `epoch 3`
  - the stabilized reference rerun with `patience = 30` reached:
    - validation `IoU`: `0.6443`
    - test `IoU`: `0.5648`
- `deeplabv3plus_mild_360`
  - validation `IoU`: `0.6514`
  - not yet promoted to test because `fgcrop_360` later exceeded it on validation and became the stronger confirmatory DeepLab run
- `deeplabv3plus_fgcrop_360`
  - validation `IoU`: `0.6613`
  - test `IoU`: `0.5709`
- `deeplabv3plus_ftversky_360`
  - validation `IoU`: `0.6263`
  - not promoted to test because validation did not beat the stabilized plain baseline
- `deeplabv3plus_512_plain`
  - validation `IoU`: `0.6282`
  - not promoted to test because validation did not beat the stabilized plain baseline

Interpretation:

- `DeepLabV3+` needed a longer early-stopping window than the earliest quick-stop run.
- Foreground-aware sampling helps `DeepLabV3+` in-domain, similar to the later `U-Net` recipe.
- `mild` augmentation is promising on validation but still sits below the current DeepLab winner.
- `Focal Tversky` and higher resolution do not improve over the stabilized plain baseline under the current setup.
- The current in-domain `DeepLabV3+` recipe should therefore be:
  - image size `360`
  - baseline augmentation
  - foreground-aware crop probability `0.5`
  - loss `BCE + Dice`
  - early-stopping patience `30`

Fixed-hold-out `UAV test` results for source-only `DeepLabV3+`:

- `deeplabv3plus_plain_360_uav_holdout_raw`
  - `IoU`: `0.1230`
  - `F1`: `0.2152`
  - `precision`: `0.1303`
  - `recall`: `0.6913`
- `deeplabv3plus_plain_360_uav_holdout_deploy`
  - `IoU`: `0.1220`
  - `F1`: `0.2090`
  - `precision`: `0.1447`
  - `recall`: `0.4711`
- `deeplabv3plus_fgcrop_360_uav_holdout_raw`
  - `IoU`: `0.0940`
  - `F1`: `0.1698`
  - `precision`: `0.0987`
  - `recall`: `0.6906`

Transfer interpretation:

- Source-only `DeepLabV3+` is weaker than the current `U-Net` and `SegFormer-B2` source-only references on the fixed `UAV test` split.
- The deployment-style threshold plus connected-component filter increases `precision`, but unlike the `SegFormer-B2` source-only case it does not improve `IoU/F1`.
- Foreground-aware crop helps `DeepLabV3+` in-domain but hurts source-only transfer almost entirely through a `precision` collapse at nearly unchanged `recall`.
- Therefore:
  - freeze `deeplabv3plus_fgcrop_360` as the current in-domain `DeepLabV3+` recipe
  - freeze `deeplabv3plus_plain_360_uav_holdout_raw` as the `DeepLabV3+` source-only transfer reference
  - the current `DeepLabV3+ B1/B2` line should continue to start from `plain_360` rather than from `fgcrop_360`

DeepLab-specific transfer-prep status:

- A `DeepLabV3+` hard-negative bank has now been mined at `generated/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080`.
- The bank contains `115` all-background crops from `80` source UAV training images and is ready for future `B1`-style mixed-training experiments.

DeepLab target-aware mitigation status:

- `deeplabv3plus_b1_negbank_uavval`
  - best validation checkpoint on `UAV val`: `epoch 9`
  - validation `IoU`: `0.2704`
  - validation `F1`: `0.4009`
  - validation `precision`: `0.5418`
  - validation `recall`: `0.3750`
- `deeplabv3plus_b1_holdout_raw_thr050`
  - `IoU`: `0.2830`
  - `F1`: `0.4143`
  - `precision`: `0.5820`
  - `recall`: `0.3767`
- `deeplabv3plus_b1_holdout_deploy`
  - `IoU`: `0.1996`
  - `F1`: `0.3123`
  - `precision`: `0.8021`
  - `recall`: `0.2130`
- Temperature-scaled `B1` bank follow-up on `UAV val`
  - `deeplabv3plus_b1_tsbank_uavval`
    - best validation checkpoint: `epoch 11`
    - validation `IoU`: `0.3149`
    - validation `F1`: `0.4512`
    - validation `precision`: `0.5672`
    - validation `recall`: `0.4928`
  - `deeplabv3plus_b1_tsbank_thr080_mean080`
    - best validation checkpoint: `epoch 13`
    - validation `IoU`: `0.3068`
    - validation `F1`: `0.4544`
    - validation `precision`: `0.4439`
    - validation `recall`: `0.5218`
  - `deeplabv3plus_b1_tsbank_thr080_mean082`
    - best validation checkpoint: `epoch 5`
    - validation `IoU`: `0.2444`
    - validation `F1`: `0.3803`
    - validation `precision`: `0.4056`
    - validation `recall`: `0.4497`

Interpretation:

- `DeepLabV3+ B1` is now represented by the promoted hold-out row `deeplabv3plus_b1_tsbank_autofilter_area1200_test`, which raises source-only raw `IoU` from `0.1230` to `0.3860` and `F1` from `0.2152` to `0.5470`.
- For the promoted `DeepLabV3+ B1` checkpoint, the default threshold `0.5` remains the best official raw operating point, so this line does not require the extra threshold retuning that benefited `SegFormer-B2`.
- The deployment-oriented switch is clearly worse than the promoted raw `B1` row on `IoU / F1`, so for `DeepLabV3+` it should remain comparison-only rather than becoming the preferred reporting line.
- A single matched random-background control on `UAV val` reaches `IoU 0.3023`, `F1 0.4368`, which stays below the mined `component_area >= 1200` bank (`IoU 0.3564`, `F1 0.5136`). This is directionally consistent with useful mined nuisance structure for `DeepLabV3+`, but the evidence remains limited because only `3 / 28` random crops could stay on the same source image as their reference crop.
- Temperature-scaled bank mining is still mixed at the candidate level for `DeepLabV3+`, but the audit-derived `component_area >= 1200` rule confirms that the useful move is discarding tiny noisy components while preserving larger target-domain nuisance structures.
- The old raw `deeplabv3plus_b1_holdout_raw_thr050` row should therefore be kept as historical comparison rather than as the official zero-label `B1` result.

DeepLab few-shot `B2` status:

- `deeplabv3plus_b2_fs05_seed42`
  - default early-stopping run
  - validation `IoU`: `0.3119`
  - best epoch: `25`
- `deeplabv3plus_b2_fs05_seed42_pat12`
  - longer-patience rerun
  - validation `IoU`: `0.3239`
  - best epoch: `38`
  - fixed-hold-out test `IoU`: `0.3599`
  - fixed-hold-out test `F1`: `0.5226`
  - fixed-hold-out test `precision`: `0.4967`
  - fixed-hold-out test `recall`: `0.5998`
- `deeplabv3plus_b2_fs10_seed42`
  - validation `IoU`: `0.3410`
  - fixed-hold-out test `IoU`: `0.4354`
  - fixed-hold-out test `F1`: `0.6005`
  - fixed-hold-out test `precision`: `0.5608`
  - fixed-hold-out test `recall`: `0.6620`
- `deeplabv3plus_b2_fs20_seed42`
  - validation `IoU`: `0.4511`
  - best epoch: `42`
  - fixed-hold-out test `IoU`: `0.4760`
  - fixed-hold-out test `F1`: `0.6340`
  - fixed-hold-out test `precision`: `0.5903`
  - fixed-hold-out test `recall`: `0.7060`

Interpretation:

- Limited target-domain supervision clearly helps `DeepLabV3+` beyond the `B1` setting.
- The smallest `DeepLabV3+` few-shot split is more sensitive to early stopping than the in-domain `Crack500` runs; increasing patience from `5` to `12` improved the `fs05` validation checkpoint and delayed the best epoch from `25` to `38`.
- The fixed-hold-out `DeepLabV3+ B2` curve is now confirmed end to end and stays monotonic: `fs05_pat12 -> fs10 -> fs20` improves test `IoU` from `0.3599 -> 0.4354 -> 0.4760`.
- Even with the now-frozen few-shot curve, `DeepLabV3+` should still be treated as a comparative CNN baseline rather than as the main transfer model because its source-only and current few-shot results remain below the `SegFormer-B2` line.

### 4.5 Cross-Domain UAV Evaluation And Reporting Rule

The first `Crack500 -> UAV_Crack_Segmentation_Kaggle` cross-domain evaluation round is now available for both frozen source-domain models.

Raw cross-domain baseline:

- `U-Net` raw prediction
  - test `IoU`: `0.1389`
  - test `F1`: `0.2333`
- `SegFormer-B2` raw prediction
  - test `IoU`: `0.1781`
  - test `F1`: `0.2829`

These raw predictions should remain the official cross-domain baselines because their job is to measure true model generalization under domain shift.

Deployment-oriented postprocessed variant:

- A high-threshold plus connected-component filter was added as an optional deployment-oriented variant for cluttered UAV scenes.
- On the UAV test set, it improves false-positive control:
  - `U-Net`
    - `IoU`: `0.1389 -> 0.1560`
    - `precision`: `0.1458 -> 0.1783`
    - `recall`: `0.7685 -> 0.5953`
  - `SegFormer-B2`
    - `IoU`: `0.1781 -> 0.2010`
    - `precision`: `0.1873 -> 0.2560`
    - `recall`: `0.7506 -> 0.4855`

The gain is meaningful for UAV deployment, but it is not a universal improvement. On `Crack500` in-domain, the same filter reduces recall and main-task overlap quality:

- `SegFormer-B2`
  - test `IoU`: `0.5854 -> 0.5399`
  - `precision`: `0.6878 -> 0.7845`
  - `recall`: `0.8005 -> 0.6367`

Interpretation:

- The default baseline still uses raw predictions.
- The postprocessed result should be reported separately as a deployment-oriented variant.
- This produces a cleaner evaluation-study structure:
  - raw model performance = actual cross-domain generalization
  - postprocessed deployment variant = deployable target-domain performance after imposing a geometric prior to suppress false positives
- This distinction is especially important because the postprocessed result improves UAV clutter handling through a deliberate `precision-recall` tradeoff rather than by making the underlying model representation universally better.

### 4.6 Official Hold-Out Split And First Training-Side Mitigation

The UAV dataset now has two roles:

- `crossdomain_all` keeps all `315` labeled UAV images for exploratory diagnosis and qualitative analysis.
- `train / val / test = 189 / 63 / 63` provides a fixed hold-out protocol for formal target-domain reporting.

The first training-side mitigation experiment, `B1`, keeps `Crack500` as the main supervised source dataset but mixes in a mined UAV hard-negative bank during training. The goal is to reduce false positives on target-style nuisance objects without replacing the source-domain training problem by full target-domain retraining.

Official hold-out results for `SegFormer-B2` on `UAV test`:

- Source-only raw
  - `IoU`: `0.1442`
  - `F1`: `0.2476`
  - `precision`: `0.1495`
  - `recall`: `0.7727`
- Source-only deployment-oriented variant
  - `IoU`: `0.1784`
  - `F1`: `0.2900`
  - `precision`: `0.2099`
  - `recall`: `0.5081`
- `B1` raw at the default threshold `0.5`
  - `IoU`: `0.3222`
  - `F1`: `0.4677`
  - `precision`: `0.4660`
  - `recall`: `0.5662`
- `B1` calibrated raw at threshold `0.7`
  - `IoU`: `0.3325`
  - `F1`: `0.4727`
  - `precision`: `0.5374`
  - `recall`: `0.5133`
- `B1` high-precision raw at threshold `0.9`
  - `IoU`: `0.3193`
  - `F1`: `0.4502`
  - `precision`: `0.6497`
  - `recall`: `0.4112`
- `B1` deployment-oriented variant
  - `IoU`: `0.3166`
  - `F1`: `0.4469`
  - `precision`: `0.6490`
  - `recall`: `0.4071`

Interpretation:

- `B1` remains strongest when it improves the model's own raw cross-domain behavior rather than relying on post-hoc suppression.
- Even the original `B1 raw` row was already well above the source-only deployment-oriented variant, which means the main gain had moved from postprocessing into the learned representation itself.
- The promoted `TS-bank` confirmatory hold-out update is now the main reporting point for the training-side mitigation line:
  - `segformer_b2_b1_tsbank_thr080_mean082_test_thr060` reaches `IoU 0.3775`, `F1 0.5317`, `precision 0.5257`, `recall 0.5779`
  - this exceeds both the old `B1 raw @ 0.7` row (`IoU 0.3325`) and the default-threshold TS-bank hold-out row (`IoU 0.3728`)
- The first single random-background control (`seed42`) reached only `IoU 0.2747`, `F1 0.4040`, compared with `IoU 0.3693`, `F1 0.5227` for the mined `TS-bank thr080_mean082` validation row.
- However, a later paired `5-seed` rerun changed that reading:
  - matched random-background seeds `42 / 7 / 13 / 21 / 99` average `IoU 0.3401 +- 0.0215`, `F1 0.4791 +- 0.0237`
  - mined `TS-bank` seeds `42 / 7 / 13 / 21 / 99` average `IoU 0.3194 +- 0.0239`, `F1 0.4584 +- 0.0247`
- The current `SegFormer-B2` mechanism interpretation therefore has to stay cautious: generic target-background exposure explains a substantial share of the `B1` gain, and mined negatives are not yet proven to be uniquely stronger than a matched random-background control on this backbone.
- A first `SegFormer-B2 TS-bank` audit package was then generated at `results/hard_negative_audit/segformer_b2_tsbank_round1`, covering `thr080_mean082`, `thr080_mean080`, and `thr075_mean080` with `180` reviewed cards.
- Reviewed keepable shares (`hard_fp + ambiguous`) for the audited `SegFormer-B2` banks were:
  - `thr080_mean080`: `20 / 60 = 33.3%`
  - `thr080_mean082`: `17 / 60 = 28.3%`
  - `thr075_mean080`: `12 / 60 = 20.0%`
- Audit-derived curated-bank validation follow-up on `SegFormer-B2`:
  - `thr080_mean080 hard_fp + ambiguous`: `IoU 0.3439`, `F1 0.4830`
  - `thr080_mean080 hard_fp`: `IoU 0.3420`, `F1 0.4836`
  - `thr080_mean082 hard_fp + ambiguous`: `IoU 0.3304`, `F1 0.4682`
  - `thr080_mean082 hard_fp`: `IoU 0.3535`, `F1 0.4993`
- Interpretation after the `SegFormer-B2` audit follow-up:
  - the strongest curated export is `thr080_mean082 hard_fp`, but it still remains below the raw mined validation winner (`0.3535 < 0.3693`)
  - `ambiguous` crops dilute the `thr080_mean082` bank for `SegFormer-B2` (`0.3304 -> 0.3535`)
  - the `thr080_mean080` curated exports are nearly tied, so `ambiguous` is not the main issue on that bank
  - the current value of the `SegFormer-B2` audit is therefore explanatory rather than promotable: it clarifies sample quality and bank composition, but it has not yet produced a stronger `B1` candidate
- The old deployment switch is therefore no longer the main story after `B1`.
- `B1 raw @ 0.9` slightly exceeds the postprocessed deployment variant in `IoU / F1` while matching its high-precision behavior.
- The promoted `TS-bank @ 0.6` row now serves as the main zero-label mitigation result, while `B1 raw @ 0.9` remains the high-precision comparison point.
- This strengthens the overall study structure:
  - source-only raw baseline = natural cross-domain generalization
  - source-only deployment switch = heuristic false-positive suppression
  - promoted `TS-bank B1` raw = training-side mitigation through target-style nuisance exposure plus lightweight score calibration
  - `B1` high-threshold raw = optional operating point for stricter precision needs
  - `B1` deployment switch = comparison-only row that shows the postprocess heuristic has become largely redundant

Recommended immediate follow-up:

- Freeze `segformer_b2_b1_tsbank_thr080_mean082_test_thr060` as the main training-side mitigation result.
- Freeze `B1 raw @ 0.9` as the high-precision operating point.
- Keep `B1` deployment only as a comparison row rather than as the preferred deployment recipe.
- Keep `B2` as a separate, already-completed supervision-efficiency line rather than as an open dependency for validating `B1`.

Temperature-scaled hard-negative mining follow-up:

- Scalar temperature fitting on `UAV val` produced:
  - `SegFormer-B2`: `T = 1.8958`, `BCE 0.1930 -> 0.1488`, `ECE 0.0412 -> 0.0231`
  - `DeepLabV3+`: `T = 2.3733`, `BCE 0.3585 -> 0.2257`, `ECE 0.0705 -> 0.0385`
- First validation-only mixed-training results:
  - `segformer_b1_tsbank_uavval`: `IoU 0.3568`, `F1 0.5027`
  - `segformer_b2_b1_tsbank_thr080_mean082`: `IoU 0.3693`, `F1 0.5227`
  - `segformer_b2_b1_tsbank_thr075_mean080`: `IoU 0.2558`, `F1 0.3782`
  - `deeplabv3plus_b1_tsbank_uavval`: `IoU 0.3149`, `F1 0.4512`
  - `deeplabv3plus_b1_tsbank_thr080_mean080`: `IoU 0.3068`, `F1 0.4544`
  - `deeplabv3plus_b1_tsbank_thr080_mean082`: `IoU 0.2444`, `F1 0.3803`
- Interpretation:
  - `SegFormer-B2` shows a clear validation gain from the calibrated-bank idea, and that gain has now been confirmed on the fixed hold-out split:
    - default-threshold TS-bank test: `IoU 0.3728`, `F1 0.5276`
    - validation-selected `@ 0.6` TS-bank test: `IoU 0.3775`, `F1 0.5317`
    - first single matched random-background control: `IoU 0.2747`, `F1 0.4040`
    - paired `5-seed` random-background mean: `IoU 0.3401 +- 0.0215`, `F1 0.4791 +- 0.0237`
    - paired `5-seed` mined-bank mean: `IoU 0.3194 +- 0.0239`, `F1 0.4584 +- 0.0247`
    - audit-derived curated follow-up:
      - `thr080_mean080 hard_fp + ambiguous`: `IoU 0.3439`, `F1 0.4830`
      - `thr080_mean080 hard_fp`: `IoU 0.3420`, `F1 0.4836`
      - `thr080_mean082 hard_fp + ambiguous`: `IoU 0.3304`, `F1 0.4682`
      - `thr080_mean082 hard_fp`: `IoU 0.3535`, `F1 0.4993`
  - The alternative `SegFormer-B2` large-context bank `thr075_mean080` underperforms badly, so a larger crop context is not automatically beneficial after temperature calibration.
  - The `SegFormer-B2` audit clarifies bank composition without changing promotion: `thr080_mean082 hard_fp` is the strongest curated export, and removing `ambiguous` helps on that bank, but even this cleaner variant still stays below the raw mined validation winner (`0.3535 < 0.3693`).
  - `DeepLabV3+` reacts less cleanly at the candidate level: the calibrated-bank idea can help, but its gains are inconsistent across retuned mining thresholds, which is why the audit-derived filtering step matters.
    - single matched random-background control for the promoted `area1200` setting: `IoU 0.3023`, `F1 0.4368`, compared with `IoU 0.3564`, `F1 0.5136` for the mined bank
  - On `2026-04-29`, a first `DeepLabV3+ TS-bank` qualitative audit package was generated at `results/hard_negative_audit/deeplabv3plus_tsbank_round1`.
  - The package contains `223` review cards plus `audit_samples.csv` spanning the source bank, calibrated bank, and the two retuned `thr080_mean080 / thr080_mean082` candidates.
  - The package was then upgraded on `2026-04-29` to a v4-compatible two-layer review schema:
    - layer 1 = bank-quality judgment such as `hard_fp / ambiguous / noise / bad_crop / gt_issue`
    - layer 2 = paper-facing nuisance taxonomy such as `pavement_edge / shadow_dark_stripe / line_like_texture / surface_boundary / debris_object`
  - On `2026-05-02`, the audit was completed with `223 / 223` layer-1 labels and `222 / 223` layer-2 labels filled.
  - Aggregate layer-1 review distribution for the audited `DeepLabV3+` package:
    - `noise = 129`
    - `hard_fp = 66`
    - `ambiguous = 27`
    - `bad_crop = 1`
  - Reviewed keepable shares (`hard_fp + ambiguous`) were:
    - calibrated-bank baseline: `23 / 43 = 53.5%`
    - original source-bank baseline: `28 / 60 = 46.7%`
    - `thr080_mean080`: `23 / 60 = 38.3%`
    - `thr080_mean082`: `19 / 60 = 31.7%`
  - `export_curated_hard_negative_bank.py` now exports trainable curated banks directly from the reviewed `audit_samples.csv`.
  - Curated-bank validation follow-up on the calibrated `DeepLabV3+ TS-bank`:
    - `hard_fp + ambiguous`: `deeplabv3plus_b1_tsbank_curated_hfpa_uavval` reaches `IoU 0.2920`, `F1 0.4301`
    - `hard_fp` only: `deeplabv3plus_b1_tsbank_curated_hardfp_uavval` reaches `IoU 0.3099`, `F1 0.4567`
    - threshold sweep on the curated `hard_fp` checkpoint peaks at `IoU 0.3122` with threshold `0.40`
  - `export_auto_filtered_hard_negative_bank.py` now exports simple rule-filtered variants of the same calibrated bank for low-cost audit-derived follow-up.
  - Auto-filter validation follow-up on the calibrated `DeepLabV3+ TS-bank`:
    - `span_ratio >= 0.35`: `deeplabv3plus_b1_tsbank_autofilter_span035_uavval` keeps `23 / 43` audited crops (`15 hard_fp`, `3 ambiguous`, `5 noise`) and reaches `IoU 0.2783`, `F1 0.4086`
    - `component_area >= 1200`: `deeplabv3plus_b1_tsbank_autofilter_area1200_uavval` keeps `28 / 43` audited crops (`16 hard_fp`, `4 ambiguous`, `8 noise`) and reaches `IoU 0.3564`, `F1 0.5136`
  - Interpretation:
    - removing `ambiguous` crops helps for `DeepLabV3+`, which suggests that the ambiguous subset dilutes the bank
    - the `span_ratio >= 0.35` rule is too aggressive: it removes many `noise` crops, but it also prunes useful nuisance structure and underperforms both the raw calibrated bank and the curated `hard_fp` export
    - `component_area >= 1200` is the first simple audit-derived rule that clearly improves transfer, which suggests that the important move is suppressing tiny noisy components without collapsing larger target-domain nuisance diversity
  - On `2026-05-03`, `deeplabv3plus_b1_tsbank_autofilter_area1200_test` then confirmed this audit-derived rule on the fixed hold-out split at `IoU 0.3860`, `F1 0.5470`, promoting it to the official zero-label `DeepLabV3+ B1` row.

### 4.7 Target-Domain Upper Bound And `B2` Few-Shot Fine-Tuning

The first in-domain `SegFormer-B2` upper-bound run on the fixed `UAV train / val / test` split is now available.

Official in-domain `UAV test` result for `SegFormer-B2`:

- Best validation checkpoint on `UAV val`
  - validation `IoU`: `0.5701`
  - validation `F1`: `0.7227`
  - best epoch: `34`
- In-domain upper bound on `UAV test`
  - `IoU`: `0.5879`
  - `F1`: `0.7369`
  - `precision`: `0.6970`
  - `recall`: `0.7830`

Interpretation:

- The target-domain ceiling is much higher than the source-only cross-domain baseline (`0.1442`) and still clearly above the best `B1` calibrated result (`0.3325`).
- This means the UAV task itself is learnable at a substantially higher level than the current zero-shot or lightly mitigated transfer setting.
- Therefore, the main bottleneck is best interpreted as severe domain shift rather than as an inherently unsolved target-domain segmentation problem.

This result also sharpens the role of `B2`:

- `B1` asked whether target-style nuisance exposure could improve raw cross-domain behavior without target supervision.
- `B2` now asks how much target-domain supervision is needed to move from `0.33` toward the `0.59` ceiling.

To make that question reproducible, three few-shot labeled subsets were created from the official `UAV train` split using `seed = 42`:

- `train_fs05_seed42`: `9` samples
- `train_fs10_seed42`: `19` samples
- `train_fs20_seed42`: `38` samples

Recommended `B2` protocol:

- Initialize `SegFormer-B2` from the frozen source-domain checkpoint `segformer_b2_plain_360.pth`.
- Fine-tune separately on `train_fs05_seed42`, `train_fs10_seed42`, and `train_fs20_seed42`.
- Keep `UAV val` for checkpoint selection and `UAV test` for one-time confirmatory reporting.
- Compare each result against:
  - source-only raw
  - `B1` calibrated raw at `0.7`
  - full in-domain upper bound

Official `B2` few-shot results on the fixed `UAV test` split:

- `fs05` fine-tuning
  - train samples: `9`
  - best validation epoch: `28`
  - test `IoU`: `0.5074`
  - test `F1`: `0.6695`
  - test `precision`: `0.6232`
  - test `recall`: `0.7277`
- `fs10` fine-tuning
  - train samples: `19`
  - best validation epoch: `25`
  - test `IoU`: `0.5420`
  - test `F1`: `0.6988`
  - test `precision`: `0.6762`
  - test `recall`: `0.7251`
- `fs20` fine-tuning
  - train samples: `38`
  - best validation epoch: `27`
  - test `IoU`: `0.5686`
  - test `F1`: `0.7209`
  - test `precision`: `0.6724`
  - test `recall`: `0.7826`

Interpretation:

- Even `5%` labeled UAV supervision is enough to move performance far beyond the source-only and `B1` cross-domain settings.
- The progression `0.5074 -> 0.5420 -> 0.5686` confirms a strong supervision-efficiency trend rather than a one-off lucky run.
- `fs20` already reaches about `96.7%` of the full in-domain upper bound (`0.5879`), which means the remaining gap can be closed very quickly once limited target labels are available.
- This gives the study a stronger conclusion than a pure diagnosis paper:
  - the bottleneck is indeed domain shift
  - the bottleneck is highly correctable with modest target-domain annotation effort

## 5. Timeline

### Phase 1: Build the Foundation (now -> 6 months)

- Invest `8-10` hours per week
- Read papers in the morning on weekdays for about `30` minutes
- Spend `4-5` hours on weekends for coding and experiments
- Complete data loading, `SegFormer` baseline training, and initial experiments
- Reach out to target professors at `UBC Smart Structures Group` and `McGill CIM`

### Phase 2: Produce Results (7 -> 18 months)

- Complete the full experiment matrix
- Record all results in a reproducible way
- Write the paper with a target venue such as `IEEE ITSC` or an `ISPRS`-related venue
- Start faculty outreach and aim for early positive responses

### Phase 3: Apply (19 -> 24 months)

- Prepare the `SOP`, recommendation letters, and transcripts
- First priority: `UBC` (`Smart Structures Group`)
- Second priority: `McGill` (`CIM`, most friendly to part-time policy)
- Third priority: `Stanford SCPD` (as a part-time master's pathway)

## 6. Next Actions

1. Freeze `360 + foreground crop + mild augmentation + BCE + Dice` as the current best `U-Net` recipe.
2. Use notebook aggregation that deduplicates repeated experiment rows by latest `timestamp_utc` before comparing runs.
3. Freeze `segformer_b2_plain_360` as the current `SegFormer-B2` baseline.
4. Freeze `deeplabv3plus_fgcrop_360` as the current in-domain `DeepLabV3+` recipe.
5. Freeze `deeplabv3plus_plain_360_uav_holdout_raw` as the `DeepLabV3+` source-only transfer reference and keep `deeplabv3plus_plain_360_uav_holdout_deploy` as a separate deployment-only comparison.
6. Freeze `deeplabv3plus_b1_tsbank_autofilter_area1200_test` as the current `DeepLabV3+` training-side mitigation row and keep `deeplabv3plus_b1_holdout_deploy` as a comparison-only deployment row.
7. Keep any further `DeepLabV3+ B2` work rooted in `plain_360`; the current frozen hold-out curve is `fs05_pat12` test `IoU 0.3599`, `fs10` test `IoU 0.4354`, and `fs20` test `IoU 0.4760`.
8. Stop in-domain `SegFormer-B2` tuning unless a new hypothesis is substantially different from the ablations already run.
9. Keep all formal baseline tables and plots on raw predictions rather than postprocessed masks.
10. Report the UAV postprocessed results separately as deployment-oriented variants, with the `precision-recall` tradeoff made explicit.
11. Freeze the fixed `UAV test` reference rows for:
   - source-only raw
   - source-only deployment
   - historical raw `B1` calibration comparisons (`0.5 / 0.7 / 0.9`) where relevant
   - promoted `SegFormer-B2 B1` `TS-bank thr080_mean082 @ 0.6`
   - promoted `DeepLabV3+ B1` `TS-bank area1200 @ 0.5`
   - `B1` deployment comparison rows
12. Freeze the in-domain `SegFormer-B2` upper bound on the fixed `UAV test` split as the target-domain ceiling reference.
13. Freeze the `B2` few-shot fine-tuning results on:
   - `fs05`
   - `fs10`
   - `fs20`
14. Use the promoted `TS-bank` rows as the main training-side mitigation results:
   - `SegFormer-B2`: `segformer_b2_b1_tsbank_thr080_mean082_test_thr060`
   - `DeepLabV3+`: `deeplabv3plus_b1_tsbank_autofilter_area1200_test`
15. Keep `SegFormer-B2 B1 raw @ 0.9` as the high-precision operating point and keep the deployment variants as comparison-only rows.
16. Treat the old validation-only wording as retired: both promoted `TS-bank` lines have now completed confirmatory hold-out testing.
17. Build the final comparison assets:
   - main result table
   - supervision-scaling figure
   - representative qualitative examples
18. Expand the UAV qualitative diagnosis by collecting representative false-positive object categories and failure cases.
19. Treat the current code paths for training, testing, split generation, hard-negative mining, few-shot split creation, and experiment logging as frozen unless a new paper-facing requirement justifies reopening them.

## 7. Paper-Facing Companion Plan

This file remains the long-term research reference and experiment log for the repository.

Its role is to preserve:

- the original research direction
- the full experiment matrix
- implementation status
- historical ablations, caveats, and frozen result rows

The paper-facing interpretation has now been separated into [paper-plan-v4.md](./paper-plan-v4.md), which should be used for:

- submission positioning
- claim boundaries
- `AIC`-first versus `Drones`-second framing
- paper-priority experiments
- manuscript structure and venue-aware emphasis

In short:

- use this file as the experiment archive
- use [paper-plan-v4.md](./paper-plan-v4.md) as the publication roadmap
