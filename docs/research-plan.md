# Research Plan and Progress Summary

**Author:** Sizhe Guan  
**Original planning date:** 2026-04-20  
**Repository snapshot updated:** 2026-04-22

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

### Evaluation Metrics

- `mIoU` as the main metric
- `F1-score`
- `Precision`
- `Recall`
- performance curves under each perturbation condition

## 3. Hardware and Environment

| Item | Configuration |
| --- | --- |
| GPU | `NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)` |
| System | `Windows 11 + WSL2 Ubuntu` |
| CUDA | `13.1` |
| Python environment | conda `crackdet`, Python `3.10` |
| Main dependencies | `PyTorch`, `HuggingFace Transformers`, `segmentation-models-pytorch`, `albumentations` |

## 4. Current Progress

The table below reflects the current repository state as of `2026-04-22`, which is slightly ahead of the original draft in a few implementation items.

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
| Generic segmentation baseline scaffolding | Done | Training and test scripts are now in place |
| `U-Net` baseline on `Crack500` | Done | First CNN baseline has been trained, checkpointed, and tested |
| Baseline qualitative visualization | Done | Fixed-sample and failure-case review notebook is now in place |
| `SegFormer-B2` baseline training | Pending | Next major implementation step |
| Cross-domain experiments | Pending | After baseline training |
| Paper writing | Pending | After experimental results are collected |

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

1. Wait for the `DronePavSeg` reply. If there is no response within one week, switch to `UAV-PDD2023`.
2. Add the `SegFormer-B2` training pipeline to this repository.
3. Run the `SegFormer-B2` baseline on `Crack500` and compare it against the frozen U-Net baseline.
4. Continue reading one recent paper per day and organize the literature in `Zotero`.
