# Script Layout

- `data/`: dataset preparation, split generation, and sanity-check utilities
- `banks/`: hard-negative mining, export, calibration, and audit helpers
- `reports/`: paper-facing tables, figures, and summary asset generators
- `experiments/`: reusable shell entry points for multi-step experiment runs
- `experiments/run_advent_deeplabv3plus.sh`: source-initialized ADVENT-style UDA baseline for `Crack500 -> UAV_Crack_Segmentation_Kaggle`
- `experiments/run_daformer_crack500_to_uav.sh`: DAFormer/MiT-B5 UDA baseline entry point for `Crack500 -> UAV_Crack_Segmentation_Kaggle`
- `daformer/prepare_crack500_to_uav_mmseg.py`: convert repo split files into the MMSeg `CustomDataset` layout used by DAFormer
- `daformer/convert_hf_mit_b5_to_daformer.py`: convert Hugging Face `nvidia/mit-b5` ImageNet weights into DAFormer/MMSeg key format
- `daformer/evaluate_daformer_threshold_sweep.py`: DAFormer foreground-probability validation-threshold sweep followed by frozen-threshold test reporting
- `reports/make_kaggle_progression_figure.py`: export the fixed Kaggle UAV qualitative progression figure (`Input / GT / Source-only / ADVENT / B1 / B2 fs10`) for paper-facing review
- `experiments/run_segformer_seed_stability.sh` + `reports/make_seed_stability_report.py`: three-seed stability reruns and `mean ± std` aggregation for the four key `SegFormer-B2` settings
- `reports/run_threshold_sweep_report.py`: validation-threshold selection followed by frozen-threshold test reporting
