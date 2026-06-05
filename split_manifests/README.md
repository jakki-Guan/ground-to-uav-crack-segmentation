# Public Split Manifests

This directory mirrors the frozen split files that define the public benchmark protocol without redistributing the raw datasets themselves.

- `kaggle_*.txt`: fixed `UAV_Crack_Segmentation_Kaggle` train, val, test, and full cross-domain lists.
- `pavecrack1300_*.txt`: fixed `PaveCrack1300` train, val, test, and full cross-domain lists.
- `*_split_manifest.json`: the corresponding split-generation metadata.

The relative paths inside these files are the same paths expected under the local dataset roots described in [DATASETS.md](../DATASETS.md).
