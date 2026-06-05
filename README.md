# Crack Detection: Cross-Domain Robustness from Ground Images to UAV Imagery

This repository tracks a controlled benchmark study on crack segmentation under domain shift. The core question is how much performance drops when models trained on ground-level crack images are transferred to UAV imagery, and which lightweight, non-architectural strategies can reduce that drop.

The current study centers on:
- `SegFormer-B2` as the main Transformer model
- `U-Net` as the CNN baseline
- `DeepLabV3+` as an additional ASPP-style CNN baseline
- `Crack500` as the source-domain dataset
- `UAV_Crack_Segmentation_Kaggle` as the primary UAV target benchmark
- `PaveCrack1300` as an auxiliary UAV benchmark-sensitivity target

See the current baseline record in [docs/baseline-results.md](docs/baseline-results.md).
See the DAFormer reviewer-baseline setup record in [docs/daformer-baseline.md](docs/daformer-baseline.md).
See the HRDA reviewer-baseline setup record in [docs/hrda-baseline.md](docs/hrda-baseline.md).
See [DATASETS.md](DATASETS.md) for dataset setup, redistribution boundaries, and public-release notes.

## Public Reproducibility Quick Start

The public-facing repo entrypoints are intentionally shallow:

- environment setup: [environment.yml](environment.yml) and [environment-daformer.yml](environment-daformer.yml)
- core Python package: [crack_detection](crack_detection)
- mirrored Kaggle UAV dataset with provenance: [kaggle_release](kaggle_release)
- fixed dataset splits: [split_manifests](split_manifests)
- `B1` bank presets and wrapper: [b1_bank_configs](b1_bank_configs)
- public evaluation entrypoints: [evaluation](evaluation), [evaluate_fixed_test.py](evaluate_fixed_test.py), [threshold_selection.py](threshold_selection.py)
- public figure-generation entrypoints: [figures_generation](figures_generation)

Typical commands:

```bash
python threshold_selection.py --help
python evaluate_fixed_test.py --help
python b1_mining_and_filtering.py --help
python figures_generation/generate_kaggle_progression_figure.py --help
```

## Reproducibility Package

This repository contains the code, configuration files, dataset split manifests, `B1` background-bank configuration files, evaluation scripts, figure-generation utilities, and lightweight result summaries used for the JSTARS manuscript.

`Crack500` and `PaveCrack1300` are not redistributed here. The Kaggle UAV crack dataset is redistributed under [kaggle_release](kaggle_release) because the Kaggle dataset page listed its license as `CC0: Public Domain` at the time of access. A license evidence file, access-time capture, source URL, file list, and `SHA-256` checksums are provided for provenance. Users should also consult the original Kaggle page when available, use the fixed split manifests under `split_manifests/`, and review privacy/publicity concerns before further redistribution.

Large generated files are intentionally excluded, including patchwise overlays, prediction masks, training work directories, and run logs.

The repository also includes a metadata-only `SAM799-CVAT` release stub under [sam799_release](sam799_release). It intentionally excludes the raw external-check images and masks until privacy and redistribution review is complete.

## Public Repository Scope

This repository is structured as a public code-and-results companion rather than a full mirror of every local experiment artifact.

- Included here:
  - code for training, evaluation, split generation, and figure generation
  - a mirrored Kaggle UAV dataset release with provenance files
  - public split manifests for the fixed Kaggle and PaveCrack1300 protocols
  - public wrapper entrypoints for `B1`, evaluation, and figure generation
  - reproducible environment files for the main stack and the DAFormer/HRDA compatibility stack
  - experiment logs and paper-facing summary assets
  - documentation and writing notes
- Not mirrored here:
  - `Crack500` and `PaveCrack1300` raw datasets
  - trained checkpoints
  - large mined-bank exports, full audit-card directories, and local scratch artifacts

The expected local dataset roots and preparation steps are documented in [DATASETS.md](DATASETS.md).

## Excluded Files

The following outputs are not included because of file size and/or data redistribution constraints:

- `results/external_sam799_cvat_patchwise/overlays/`
- `results/external_sam799_cvat_patchwise/predictions/`
- `results/external_sam799_cvat_patchwise_smoketest/`
- `results/external_sam799_cvat_patchwise_smoketest_profile/`
- `results/run_logs/`
- `work_dirs/`

## Research Scope

This project focuses on three evaluation settings:
- In-domain training and testing on the same dataset
- Cross-domain transfer from `Crack500` to a UAV dataset
- Robustness under photometric and resolution perturbations on UAV test data

The study does not propose a new backbone. Instead, it evaluates practical strategies that keep the backbone unchanged, including:
- photometric augmentation
- multi-scale or progressive-resolution training
- loss adjustments such as `BCE + Dice`
- foreground-aware sampling
- one-change-at-a-time ablations over resolution, sampling, augmentation, and loss

Reporting principle:

- Default baselines are always reported from raw model predictions.
- Postprocessed outputs are reported separately as deployment-oriented variants rather than replacing the raw baseline.

## Frozen Benchmark Snapshot

This README intentionally keeps only the public-facing frozen summary. Detailed run histories, historical reruns, threshold sweeps, and intermediate ablations are preserved in [docs/baseline-results.md](docs/baseline-results.md), [results/experiments.csv](results/experiments.csv), and the assets under [results/report_assets](results/report_assets).

Main fixed-hold-out `UAV_Crack_Segmentation_Kaggle/test` rows:

- Source-only raw:
  - `U-Net`: `IoU 0.1284`, `F1 0.2244`
  - `SegFormer-B2`: `IoU 0.1442`, `F1 0.2476`
  - `DeepLabV3+`: `IoU 0.1230`, `F1 0.2152`
- Reviewer-facing UDA:
  - ADVENT-style `DeepLabV3+`: `IoU 0.2022`, `F1 0.3076`
  - `DAFormer` and `HRDA` frozen rows are documented separately in [docs/daformer-baseline.md](docs/daformer-baseline.md) and [docs/hrda-baseline.md](docs/hrda-baseline.md)
- `B1` training-side mitigation:
  - promoted `SegFormer-B2` `TS-bank @ 0.6`: `IoU 0.3775`, `F1 0.5317`
  - promoted `DeepLabV3+` `area1200 @ 0.5`: `IoU 0.3860`, `F1 0.5470`
- `B2` few-shot recovery:
  - `SegFormer-B2 fs05 / fs10 / fs20`: `IoU 0.5074 / 0.5420 / 0.5686`
  - `DeepLabV3+ fs05 / fs10 / fs20`: `IoU 0.3599 / 0.4354 / 0.4760`
- Full-target reference:
  - `SegFormer-B2`: `IoU 0.5879`, `F1 0.7369`
  - `DeepLabV3+`: `IoU 0.5085`, `F1 0.6693`

Additional public support assets:

- `SAM799-CVAT` metadata-only release stub: [sam799_release](sam799_release)
- `SAM799-CVAT` patchwise external-check summaries: `results/external_sam799_cvat_patchwise*`
- Kaggle-source provenance mirror: [kaggle_release](kaggle_release)

## Experiment Log Notes

- `results/experiments.csv` preserves historical rows, exploratory runs, reruns, and threshold-selected reports in addition to the frozen paper-facing lines.
- Fixed-hold-out raw baselines, deployment-oriented postprocessed rows, and validation-selected threshold rows are distinct reports and should not be mixed.
- For repeated `(experiment_name, stage, split)` keys, use the latest `timestamp_utc`.
- The `crossdomain_all` exploratory diagnosis rows are not the same as the frozen `test`-split hold-out rows.

## Environment

- GPU: `NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)`
- OS: `Windows 11 + WSL2 Ubuntu`
- CUDA: `13.1`
- Python: `3.10` in conda environment `crackdet`
- Main libraries: `PyTorch`, `HuggingFace Transformers`, `segmentation-models-pytorch`, `albumentations`

## Repository Layout

```text
.
├── DATASETS.md             # dataset setup and redistribution notes
├── environment.yml         # main public environment
├── environment-daformer.yml  # DAFormer/HRDA compatibility environment
├── crack_detection/        # core Python package
├── dataset.py              # compatibility shim for crack_detection.dataset
├── model.py                # compatibility shim for crack_detection.model
├── loss.py                 # compatibility shim for crack_detection.loss
├── metrics.py              # compatibility shim for crack_detection.metrics
├── postprocess.py          # compatibility shim for crack_detection.postprocess
├── b1_bank_configs/        # public B1 presets and wrapper
├── evaluation/             # public evaluation entrypoints
├── figures_generation/     # public figure-generation entrypoints
├── split_manifests/        # tracked fixed split files
├── b1_mining_and_filtering.py  # top-level B1 wrapper
├── threshold_selection.py  # top-level threshold-selection wrapper
├── evaluate_fixed_test.py  # top-level fixed-test wrapper
├── train.py                # training entry point
├── test.py                 # test-set evaluation from saved checkpoint
├── scripts/
│   ├── data/               # dataset prep, splitting, and sanity-check utilities
│   ├── banks/              # hard-negative mining, export, and audit helpers
│   ├── reports/            # paper/report asset generators
│   └── experiments/        # reusable experiment shell entry points
├── notebooks/              # EDA and qualitative visualization notebooks
├── results/                # experiment logs and paper-facing report assets
├── generated/              # generated banks and diagnostics
└── docs/                   # baseline records and public notes
```
