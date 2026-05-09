# Dataset And Artifact Availability

This repository is intended to be released as a public code-and-results companion for the paper-facing benchmark study. It does **not** mirror third-party raw datasets, trained checkpoints, or large intermediate artifact exports.

## Included In The Public Repository

- training, evaluation, split-generation, and figure-generation code
- experiment logs such as [results/experiments.csv](results/experiments.csv)
- paper-facing summary assets under [results/report_assets](results/report_assets)
- documentation and writing notes under [docs](docs)

## Not Mirrored In The Public Repository

- third-party raw datasets:
  - `CRACK500/`
  - `UAV_Crack_Segmentation_Kaggle/`
  - `PaveCrack1300/`
  - `PaveCrack1300_raw/`
- trained model checkpoints under `checkpoints/`
- mined or curated crop banks under `generated/`
- full audit-card exports under `results/hard_negative_audit/`
- local scratch outputs under `tmp/`

These items are omitted to avoid redistributing third-party data, reduce repository size, and keep the public release focused on reproducible code and frozen paper-facing summaries.

## Dataset Roles

| Dataset | Paper role | Expected local root | Redistribution in this repo |
| --- | --- | --- | --- |
| `Crack500` | source-domain training dataset | `CRACK500/` | not mirrored |
| `UAV_Crack_Segmentation_Kaggle` | primary UAV target benchmark | `UAV_Crack_Segmentation_Kaggle/` | not mirrored |
| `PaveCrack1300` | auxiliary UAV benchmark-sensitivity target | `PaveCrack1300/` | not mirrored |

## Expected Local Layout

The codebase expects local dataset roots to live at the repository top level unless you pass an explicit override path.

```text
.
├── CRACK500/
├── UAV_Crack_Segmentation_Kaggle/
├── PaveCrack1300/
└── PaveCrack1300_raw/   # optional raw import source for prepare_pavecrack1300.py
```

## Preparing The Datasets Locally

### Crack500

Download `Crack500` from its original provider and place it under `CRACK500/` in the repository root. The current training and evaluation scripts assume the repository-native split-file layout already exists locally.

### UAV_Crack_Segmentation_Kaggle

Download the Kaggle UAV crack dataset locally and place it under `UAV_Crack_Segmentation_Kaggle/`. If you are starting from a single all-sample split file and want to regenerate the repository's reproducible `train / val / test` split while preserving the exploratory full list, run:

```bash
python split_uav_kaggle.py \
  --dataset-root UAV_Crack_Segmentation_Kaggle \
  --source-split test \
  --overwrite
```

This creates `train.txt`, `val.txt`, `test.txt`, `crossdomain_all.txt`, and `split_manifest.json` under the dataset root.

### PaveCrack1300

Download the public `PaveCrack1300` release locally, extract it to `PaveCrack1300_raw/`, and convert it into the repository-native split-file layout with:

```bash
python prepare_pavecrack1300.py \
  --raw-root PaveCrack1300_raw \
  --dataset-root PaveCrack1300
```

This creates a prepared dataset root containing `images/`, `masks/`, `train.txt`, `val.txt`, `test.txt`, `crossdomain_all.txt`, and `split_manifest.json`.

## Reproducibility Scope

The public repository is designed to support:

- inspection of the training and evaluation code paths
- recreation of dataset splits and figure assets
- verification of frozen paper-facing metrics from the experiment log

Before publishing the repository, review notebook outputs carefully. Jupyter notebooks can embed rendered dataset samples directly in cell outputs even when the raw dataset directories are excluded from version control.

The public repository is **not** intended to act as a full mirror for:

- raw third-party imagery
- all intermediate mined banks or review-card exports
- all trained checkpoints from every ablation

## Suggested Paper Data-Availability Wording

If you need a compact manuscript statement, the following wording matches this repository structure:

> The source datasets used in this study are publicly available from their respective providers, including Crack500, the UAV Crack Segmentation Kaggle dataset, and PaveCrack1300. The code, experiment logs, split-generation scripts, and paper-facing result assets supporting the findings of this study are publicly available in the project repository. Due to third-party dataset licensing and repository-size constraints, raw dataset files, trained checkpoints, and large intermediate artifact exports are not mirrored in the repository.
