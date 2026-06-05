# DAFormer Reviewer Baseline Record

**Status updated:** 2026-05-25  
**Purpose:** Add a stronger recent UDA baseline for `Crack500 -> UAV_Crack_Segmentation_Kaggle`, addressing reviewer concern that ADVENT alone is not enough.

## Scope

This is a reviewer-facing DAFormer-style UDA baseline, not a new method in this repository.

- External code: `external/DAFormer`
- Upstream repository: <https://github.com/lhoyer/DAFormer>
- Local external snapshot checked: `e6f6e04`
- Project-owned config: `configs/daformer/crack500_to_uav_daformer_mitb5_s0.py`
- Project-owned scripts:
  - `scripts/daformer/prepare_crack500_to_uav_mmseg.py`
  - `scripts/daformer/convert_hf_mit_b5_to_daformer.py`
  - `scripts/daformer/evaluate_daformer_threshold_sweep.py`
  - `scripts/experiments/run_daformer_crack500_to_uav.sh`

The third-party DAFormer repository remains under ignored `external/`; only the adapter scripts, config, and documentation are tracked here.

## Environment

The official DAFormer README recommends Python `3.8.5`, PyTorch `1.7.1+cu110`, `mmcv-full==1.3.7`, and reports experiments on an RTX 2080 Ti. That stack is not a good fit for the local RTX 4070 SUPER because old CUDA 11.0 wheels are risky on Ada GPUs.

Local compatibility environment created instead:

- Conda env: `daformer`
- Python: `3.10.20`
- PyTorch: `2.0.1+cu118`
- TorchVision: `0.15.2+cu118`
- CUDA visible in env: yes
- GPU verified: `NVIDIA GeForce RTX 4070 SUPER`
- MMCV: `1.4.0` plain `mmcv`, not `mmcv-full`
- NumPy: `1.23.5`
- OpenCV: `4.8.1`
- `timm`: `0.6.13`
- `kornia`: `0.6.12`
- `yapf`: `0.31.0`

Notes:

- `mmcv==1.7.1` was rejected by DAFormer's bundled `mmseg` version guard, which requires `1.3.7 <= mmcv <= 1.4.0`.
- `mmcv-full==1.7.1` did not resolve a prebuilt wheel from the attempted OpenMMLab index and fell back to source build. Since the DAFormer code path used here does not call `mmcv.ops`, plain `mmcv==1.4.0` is sufficient for the smoke-tested training/evaluation path.
- `yapf` must stay at `0.31.0`; newer `yapf` versions remove the `verify` argument used by `mmcv.Config.pretty_text`.

## Pretrained Weights

The official DAFormer helper points to Google Drive MiT weights, but `gdown` could not retrieve the public link for `mit_b5.pth` in this environment. The reproducible fallback is:

```bash
conda run -n daformer python scripts/daformer/convert_hf_mit_b5_to_daformer.py \
  --output external/DAFormer/pretrained/mit_b5.pth
```

This downloads `nvidia/mit-b5` from Hugging Face and converts the Hugging Face key names to DAFormer/MMSeg MiT-B5 key names.

Verified conversion:

- Converted backbone tensors: `1052`
- DAFormer `mit_b5` model state keys: `1052`
- Skipped tensors: Hugging Face classification head only (`classifier.weight`, `classifier.bias`)
- Load check passed for `external/DAFormer/pretrained/mit_b5.pth`

## Dataset Adapter

The adapter converts this repository's paired split-file format into the `CustomDataset` layout expected by DAFormer/MMSeg.

Command:

```bash
python scripts/daformer/prepare_crack500_to_uav_mmseg.py \
  --output-root generated/daformer/crack500_to_uav \
  --force
```

Generated layout:

- Source train: `1896` `Crack500/train` samples with masks
- Target train: `189` `UAV/train` images, image-only in the DAFormer training config
- Target val: `63` `UAV/val` samples with masks
- Target test: `63` `UAV/test` samples with masks

Important protocol detail:

- Target-train masks are intentionally not converted or referenced by the DAFormer training config.
- Target labels are used only for validation threshold selection and frozen-threshold test reporting.
- Mask labels were normalized to foreground/background IDs `0/1`, with `255` reserved as ignore.
- Source-side `sample_class_stats.json` and `samples_with_class.json` are generated for DAFormer rare-class sampling.

## Training Config

Config: `configs/daformer/crack500_to_uav_daformer_mitb5_s0.py`

Main settings:

- Architecture: DAFormer `daformer_sepaspp_mitb5`
- Encoder: MiT-B5
- Classes: `background`, `crack`
- Training/evaluation size: `360 x 360`
- Source: labeled `Crack500/train`
- Target: unlabeled `UAV/train`
- Input-size rule: aligned to the repository's frozen `360 x 360` baseline protocol rather than DAFormer's original `512 x 512` urban-scene default
- UDA: DACS-style class mixing with DAFormer defaults adapted to binary cracks
- EMA alpha: `0.999`
- Pseudo-label threshold: `0.968`
- ImageNet feature distance: enabled on crack foreground class `[1]`
- Rare-class sampling: enabled from source mask statistics, `class_temp=0.01`, `min_pixels=100`
- Optimizer: AdamW, `lr=6e-5`, DAFormer paramwise LR multipliers
- Default schedule: `40000` iterations
- Default batch: `samples_per_gpu=2`

Run command:

```bash
DAFORMER_PYTHON=/home/jakeguan/miniconda3/envs/daformer/bin/python \
bash scripts/experiments/run_daformer_crack500_to_uav.sh
```

Useful overrides:

```bash
MAX_ITERS=40000 \
SAMPLES_PER_GPU=2 \
WORKERS_PER_GPU=2 \
CHECKPOINT_INTERVAL=4000 \
EVAL_INTERVAL=4000 \
DAFORMER_PYTHON=/home/jakeguan/miniconda3/envs/daformer/bin/python \
bash scripts/experiments/run_daformer_crack500_to_uav.sh
```

## Evaluation Protocol

DAFormer uses a 2-class softmax output, so paper-facing crack metrics are computed from the foreground class probability, not from MMSeg's background+foreground mean IoU alone.

Mandatory reporting protocol:

1. Sweep thresholds on `UAV val`.
2. Select the best threshold by validation IoU.
3. Evaluate `UAV test` once with that frozen threshold.
4. Log foreground crack `IoU`, `F1`, `precision`, and `recall`.

Script:

```bash
PYTHONPATH=external/DAFormer \
/home/jakeguan/miniconda3/envs/daformer/bin/python \
scripts/daformer/evaluate_daformer_threshold_sweep.py \
  --checkpoint work_dirs/daformer_crack500_to_uav_mitb5_s0_360/latest.pth \
  --output-dir results/report_assets/daformer_mitb5_threshold_sweep_360 \
  --num-workers 2 \
  --experiment-name daformer_mitb5_crack500_to_uav_test_thrval_360
```

This script appends the selected-threshold test row to `results/experiments.csv` unless `--no-log-test-result` is passed.

## Smoke Verification

Completed on 2026-05-24:

- `python -m py_compile` passed for all DAFormer adapter scripts.
- `bash -n scripts/experiments/run_daformer_crack500_to_uav.sh` passed.
- Dataset adapter regenerated `generated/daformer/crack500_to_uav`.
- Mask label check confirmed sampled source/val/test masks contain only `[0, 1]`.
- DAFormer config loaded and built:
  - source train: `1896`
  - target train: `189`
  - UDA dataset length: `358344`
  - target val: `63`
  - target test: `63`
  - classes: `('background', 'crack')`
- `mit_b5` converted weights loaded into DAFormer with `1052/1052` tensors.
- `2`-iteration smoke train completed with real MiT-B5 initialization and wrote `tmp/daformer_smoke/latest.pth`.
- Threshold-report smoke completed on `tmp/daformer_smoke/latest.pth`; the numeric result is not meaningful and must not be cited.

## Full Run Result

Completed on 2026-05-26:

- Config: `configs/daformer/crack500_to_uav_daformer_mitb5_s0.py`
- Work dir: `work_dirs/daformer_crack500_to_uav_mitb5_s0_360`
- Final checkpoint: `work_dirs/daformer_crack500_to_uav_mitb5_s0_360/iter_40000.pth`
- Latest checkpoint link: `work_dirs/daformer_crack500_to_uav_mitb5_s0_360/latest.pth`
- Report assets: `results/report_assets/daformer_mitb5_threshold_sweep_360`
- Logged experiment row: `daformer_mitb5_crack500_to_uav_test_thrval_360`

Validation threshold selection:

- Selected threshold: `0.60`
- Validation IoU: `0.0842`
- Validation F1: `0.1553`
- Validation precision: `0.0853`
- Validation recall: `0.8718`

Frozen `UAV test` result:

- IoU: `0.1353`
- F1: `0.2384`
- Precision: `0.1388`
- Recall: `0.8442`

This is the paper-facing DAFormer reviewer baseline under the repository's unified `360 x 360` input protocol. MMSeg validation `mIoU` from the training log should not be compared directly with the foreground crack metrics above.
