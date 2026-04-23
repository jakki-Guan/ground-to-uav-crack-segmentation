# Crack Detection: Cross-Domain Robustness from Ground Images to UAV Imagery

This repository tracks a controlled benchmark study on crack segmentation under domain shift. The core question is how much performance drops when models trained on ground-level crack images are transferred to UAV imagery, and which lightweight, non-architectural strategies can reduce that drop.

The current study centers on:
- `SegFormer-B2` as the main Transformer model
- `U-Net` as the CNN baseline
- `Crack500` as the source-domain dataset
- `DronePavSeg` or `UAV-PDD2023` as the UAV target-domain dataset

See the full translated plan in [docs/research-plan.md](docs/research-plan.md).
See the current baseline record in [docs/baseline-results.md](docs/baseline-results.md).

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

## Current Repository Status

As of `2026-04-22`, the workspace already includes:
- `dataset.py`: `CrackDataset` plus Albumentations-based transforms
- `model.py`: segmentation model factory for `Unet`, `FPN`, `Linknet`, and `PSPNet`
- `loss.py`: `BCE + Dice` loss
- `metrics.py`: `IoU` and `F1` metrics
- `check_dataset.py`: dataset sanity-check and visualization utilities
- `train.py`: end-to-end U-Net baseline training with validation checkpointing and early stopping
- `test.py`: final test-set evaluation from the saved best checkpoint
- `notebooks/03_test_visualization.ipynb`: qualitative review notebook for fixed test samples and failure cases
- `CRACK500/`: downloaded source-domain dataset with split files

Still pending:
- `SegFormer-B2` integration
- cross-domain evaluation scripts
- UAV target-domain dataset integration

## Environment

- GPU: `NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)`
- OS: `Windows 11 + WSL2 Ubuntu`
- CUDA: `13.1`
- Python: `3.10` in conda environment `crackdet`
- Main libraries: `PyTorch`, `HuggingFace Transformers`, `segmentation-models-pytorch`, `albumentations`

## Immediate Next Steps

1. Finalize the UAV target dataset choice.
2. Add and train `SegFormer-B2` on `Crack500`.
3. Compare `SegFormer-B2` against the recorded U-Net baseline.
4. Evaluate cross-domain transfer on UAV imagery.
5. Run perturbation tests for brightness, shadow, contrast, blur, and downsampling.

## Repository Layout

```text
.
├── CRACK500/               # source-domain dataset and split files
├── dataset.py              # dataset loader and transforms
├── model.py                # segmentation model factory
├── loss.py                 # training loss
├── metrics.py              # evaluation metrics
├── check_dataset.py        # dataset sanity checks and visualizations
├── train.py                # training entry point
├── test.py                 # test-set evaluation from saved checkpoint
├── notebooks/              # EDA and qualitative visualization notebooks
└── docs/                   # planning and baseline records
```
