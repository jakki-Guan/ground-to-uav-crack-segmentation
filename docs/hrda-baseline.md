# HRDA Reviewer Baseline Record

**Status updated:** 2026-05-27  
**Purpose:** Add a reviewer-facing `HRDA/MiT-B5` UDA comparison for `Crack500 -> UAV_Crack_Segmentation_Kaggle` under the repository's fixed `360 x 360` protocol.

## Scope

This is a reviewer-facing HRDA baseline, not a new method proposed by this repository.

- External code: `external/HRDA`
- Upstream repository: <https://github.com/lhoyer/HRDA>
- Local external snapshot checked: `c370b5b`
- Project-owned config: `configs/hrda/crack500_to_uav_hrda_mitb5_s0.py`
- Project-owned scripts:
  - `scripts/daformer/prepare_crack500_to_uav_mmseg.py`
  - `scripts/daformer/evaluate_daformer_threshold_sweep.py`
  - `scripts/experiments/run_hrda_crack500_to_uav.sh`

The third-party HRDA repository remains under ignored `external/`; only the adapter config, runner, and documentation are tracked here.

## Environment

Verified from the completed HRDA training log:

- Conda env: `daformer`
- Python: `3.10.20`
- PyTorch: `2.0.1+cu118`
- TorchVision: `0.15.2+cu118`
- CUDA visible in env: yes
- GPU verified: `NVIDIA GeForce RTX 4070 SUPER`
- MMCV: `1.4.0`
- OpenCV: `4.8.1`
- MMSegmentation: `0.16.0+5f9926c`

This is the same local compatibility stack used for the DAFormer reviewer baseline.

## Pretrained Weights

- Backbone weights path: `external/HRDA/pretrained/mit_b5.pth`
- Local setup: symlink to the converted DAFormer-compatible MiT-B5 ImageNet weights

## Dataset Adapter

The repository reuses the MMSeg-format adapter so HRDA sees the same split protocol as DAFormer.

Command:

```bash
python scripts/daformer/prepare_crack500_to_uav_mmseg.py \
  --output-root generated/hrda/crack500_to_uav \
  --force
```

Generated layout:

- Source train: `1896` `Crack500/train` samples with masks
- Target train: `189` `UAV/train` images, image-only during HRDA training
- Target val: `63` `UAV/val` samples with masks
- Target test: `63` `UAV/test` samples with masks

Important protocol detail:

- Target-train masks are intentionally not referenced by the HRDA training config.
- Target labels are used only for validation threshold selection and frozen-threshold test reporting.
- Mask labels are normalized to `background/crack = 0/1`, with `255` as ignore.
- Source-side `sample_class_stats.json` and `samples_with_class.json` are generated for rare-class sampling.

## Method Summary

HRDA extends DAFormer-style UDA with high-resolution domain adaptation. In this local binary crack setting it uses:

- a `360 x 360` outer crop aligned to the repository's frozen comparison protocol
- a low-resolution context branch with feature scale `0.5`
- a high-resolution detail crop branch with `hr_crop_size = 176 x 176`
- attention-based fusion over the dual-scale predictions

This is relevant for UAV crack segmentation because thin crack structures may benefit from local detail while cluttered UAV backgrounds still require broader scene context.

## Training Config

Config: `configs/hrda/crack500_to_uav_hrda_mitb5_s0.py`

### Model

- Architecture: `HRDAEncoderDecoder`
- Single-scale head family: `DAFormerHead`
- Decoder variant: `daformer_sepaspp_mitb5`
- Encoder: `MiT-B5`
- Classes: `background`, `crack`
- `num_classes = 2`
- Dual-scale settings:
  - `scales = [1, 0.5]`
  - `hr_crop_size = (176, 176)`
  - `feature_scale = 0.5`
  - `crop_coord_divisible = 8`
  - `hr_slide_inference = True`
  - `attention_classwise = True`
  - `hr_loss_weight = 0.1`
- Inference mode:
  - `test_cfg.mode = slide`
  - `batched_slide = True`
  - `stride = (176, 176)`
  - `crop_size = (360, 360)`

### Data And Augmentation

- Source supervised dataset: `Crack500/train`
- Target adaptation dataset: `UAV/train`
- Validation dataset: `UAV/val`
- Final test dataset: `UAV/test`
- Training/evaluation size: `360 x 360`
- Source and target resize ratio range: `(0.5, 2.0)`
- Random crop: `360 x 360`
- Random horizontal flip probability: `0.5`
- Batch size: `samples_per_gpu = 1`
- Workers: `workers_per_gpu = 1`

### UDA Parameters

- UDA family: `DACS`
- Mixing mode: `class`
- EMA alpha: `0.999`
- Pseudo-label threshold: `0.968`
- `blur = True`
- `color_jitter_strength = 0.2`
- `color_jitter_probability = 0.2`
- ImageNet feature distance:
  - `imnet_feature_dist_lambda = 0.005`
  - `imnet_feature_dist_classes = [1]`
  - `imnet_feature_dist_scale_min_ratio = 0.75`
- Rare-class sampling:
  - `min_pixels = 100`
  - `class_temp = 0.01`
  - `min_crop_ratio = 0.5`

### Optimization

- Optimizer: `AdamW`
- Learning rate: `6e-5`
- Betas: `(0.9, 0.999)`
- Weight decay: `0.01`
- Paramwise multipliers:
  - `head.lr_mult = 10.0`
  - `pos_block.decay_mult = 0.0`
  - `norm.decay_mult = 0.0`
- LR schedule: `poly10warm`
- Warmup:
  - `warmup = linear`
  - `warmup_iters = 1500`
  - `warmup_ratio = 1e-6`
- Runner: `IterBasedRunner`
- Max iterations: `40000`
- Checkpoint interval: `4000`
- Validation interval: `4000`
- Seed: `42`

## Run Commands

Smoke train:

```bash
HRDA_PYTHON=/home/jakeguan/miniconda3/envs/daformer/bin/python \
RUN_EVAL=0 \
MAX_ITERS=2 \
CHECKPOINT_INTERVAL=2 \
EVAL_INTERVAL=2 \
WORK_DIR=tmp/hrda_smoke_360 \
DATA_OUTPUT_ROOT=generated/hrda/crack500_to_uav \
bash scripts/experiments/run_hrda_crack500_to_uav.sh
```

Full run:

```bash
HRDA_PYTHON=/home/jakeguan/miniconda3/envs/daformer/bin/python \
bash scripts/experiments/run_hrda_crack500_to_uav.sh
```

Explicit resume:

```bash
HRDA_PYTHON=/home/jakeguan/miniconda3/envs/daformer/bin/python \
PREPARE_DATA=0 \
RUN_EVAL=0 \
WORK_DIR=work_dirs/hrda_crack500_to_uav_mitb5_s0_360 \
RESUME_FROM=work_dirs/hrda_crack500_to_uav_mitb5_s0_360/latest.pth \
bash scripts/experiments/run_hrda_crack500_to_uav.sh
```

## Evaluation Protocol

The paper-facing protocol matches the DAFormer and ADVENT reviewer baselines:

1. Sweep thresholds on `UAV val`.
2. Select the best threshold by validation IoU.
3. Evaluate `UAV test` once with that frozen threshold.
4. Log foreground crack `IoU`, `F1`, `precision`, and `recall`.

HRDA uses a 2-class softmax output, so the reported crack metrics are computed from the foreground probability rather than from MMSeg's background-plus-foreground mean IoU alone.

For smoke checks, use `LOG_TEST_RESULT=0` so the meaningless smoke metrics are not appended to `results/experiments.csv`.

## Smoke Verification

Completed before the full run:

- `python -m py_compile` passed for the HRDA config and threshold evaluator.
- `bash -n scripts/experiments/run_hrda_crack500_to_uav.sh` passed.
- Dataset adapter generated `generated/hrda/crack500_to_uav`.
- `2`-iteration smoke train completed and wrote `tmp/hrda_smoke_360/iter_2.pth`.
- Threshold-report smoke completed on `tmp/hrda_smoke_360/latest.pth` with `LOG_TEST_RESULT=0`.

## Full Run Result

Completed on 2026-05-27:

- Config: `configs/hrda/crack500_to_uav_hrda_mitb5_s0.py`
- Work dir: `work_dirs/hrda_crack500_to_uav_mitb5_s0_360`
- Final checkpoint: `work_dirs/hrda_crack500_to_uav_mitb5_s0_360/iter_40000.pth`
- Latest checkpoint link: `work_dirs/hrda_crack500_to_uav_mitb5_s0_360/latest.pth`
- Report assets: `results/report_assets/hrda_mitb5_threshold_sweep_360`
- Logged experiment row: `hrda_mitb5_crack500_to_uav_test_thrval_360`

Training-log validation snapshot at `40000` iterations:

- MMSeg crack-class validation IoU: `0.0832`
- MMSeg crack-class validation accuracy: `0.6788`

Validation threshold sweep:

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

Selected validation threshold:

- Threshold: `0.30`
- Validation IoU: `0.0840`
- Validation F1: `0.1550`
- Validation precision: `0.0863`
- Validation recall: `0.7576`

Frozen `UAV test` result:

- IoU: `0.1143`
- F1: `0.2052`
- Precision: `0.1191`
- Recall: `0.7390`

## Interpretation

- Compared with the same-protocol `DAFormer/MiT-B5` reviewer baseline (`IoU 0.1353`, `F1 0.2384`), HRDA is lower in this local binary crack setting.
- The selected threshold `0.30` and the final `precision 0.1191` indicate that the best validation operating point remains recall-heavy rather than precision-stable.
- This is therefore a completed negative reviewer baseline, but it is still useful evidence: adding HRDA's dual-scale high-resolution fusion did not recover the structured UAV false positives well enough to beat the simpler DAFormer baseline here.
