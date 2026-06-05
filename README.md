# Crack Detection: Cross-Domain Robustness from Ground Images to UAV Imagery

This repository tracks a controlled benchmark study on crack segmentation under domain shift. The core question is how much performance drops when models trained on ground-level crack images are transferred to UAV imagery, and which lightweight, non-architectural strategies can reduce that drop.

The current study centers on:
- `SegFormer-B2` as the main Transformer model
- `U-Net` as the CNN baseline
- `DeepLabV3+` as an additional ASPP-style CNN baseline
- `Crack500` as the source-domain dataset
- `UAV_Crack_Segmentation_Kaggle` as the primary UAV target benchmark
- `PaveCrack1300` as an auxiliary UAV benchmark-sensitivity target

See the full translated plan in [docs/research-plan.md](docs/research-plan.md).
See the paper-facing roadmap in [docs/paper-plan-v4.md](docs/paper-plan-v4.md).
See the `AIC`-oriented writing outline in [docs/paper-outline-detailed-v1.md](docs/paper-outline-detailed-v1.md).
See the first `AIC` draft in [docs/paper-draft-intro-problem-method-v1.md](docs/paper-draft-intro-problem-method-v1.md).
See the `Drones`-oriented fallback plan in [docs/paper-plan-drones-v1.md](docs/paper-plan-drones-v1.md).
See the current baseline record in [docs/baseline-results.md](docs/baseline-results.md).
See the DAFormer reviewer-baseline setup record in [docs/daformer-baseline.md](docs/daformer-baseline.md).
See the HRDA reviewer-baseline setup record in [docs/hrda-baseline.md](docs/hrda-baseline.md).
See [DATASETS.md](DATASETS.md) for dataset setup, redistribution boundaries, and public-release notes.

## Public Reproducibility Quick Start

The public-facing repo entrypoints are intentionally shallow:

- environment setup: [environment.yml](environment.yml) and [environment-daformer.yml](environment-daformer.yml)
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

## Current Repository Status

As of `2026-05-24`, the workspace already includes:
- `dataset.py`: `CrackDataset` plus Albumentations-based transforms
- `model.py`: segmentation model factory for `Unet`, `FPN`, `Linknet`, `PSPNet`, `DeepLabV3+`, and `SegFormer-B2`
- `loss.py`: configurable `BCE + Dice`, `Tversky`, and `Focal Tversky` losses
- `metrics.py`: `IoU` and `F1` metrics
- `postprocess.py`: optional connected-component filtering for deployment-oriented evaluation
- `scripts/data/check_dataset.py`: dataset sanity-check and visualization utilities
- `train.py`: configurable baseline training with validation checkpointing and early stopping
- `test.py`: configurable checkpoint evaluation for the saved model
- `experiment_logger.py`: experiment logging helpers for reproducible CSV records, including split/config/threshold/postprocess metadata plus `IoU`, `F1`, `precision`, and `recall`
- `scripts/reports/make_b1_report_assets.py`: report-asset generator for the calibrated `B1` comparison
- `scripts/reports/make_retention_report_assets.py`: report-asset generator for source-domain retention back-tests, `Kaggle test` bootstrap confidence intervals, and optional skeleton metrics
- `scripts/data/make_uav_fewshot_splits.py`: reproducible `5% / 10% / 20%` few-shot split generator for `B2`
- `scripts/data/split_uav_kaggle.py`: reproducible splitter for the primary `UAV_Crack_Segmentation_Kaggle` benchmark
- `scripts/data/prepare_pavecrack1300.py`: import the public `PaveCrack1300` release into the repository's standard split-file dataset format
- `scripts/banks/mine_uav_hard_negatives.py`: mine target-background banks for `B1`-style follow-up experiments
- `scripts/banks/export_curated_hard_negative_bank.py`: export audited hard-negative crops into trainable curated bank roots
- `scripts/reports/paper_figures.py`: paper-facing figure helper module for dataset overview, mechanism, and `B2` supervision-scaling assets
- `scripts/daformer/`: DAFormer reviewer-baseline adapters for MMSeg-format data preparation, MiT-B5 weight conversion, and validation-threshold test reporting
- `configs/daformer/crack500_to_uav_daformer_mitb5_s0.py`: project-specific DAFormer/MiT-B5 UDA config for `Crack500 -> UAV_Crack_Segmentation_Kaggle`
- `environment.yml`: public environment for the main training/evaluation/figure-generation stack
- `environment-daformer.yml`: compatibility environment for the DAFormer and HRDA reviewer baselines
- `split_manifests/`: tracked copies of the fixed Kaggle and PaveCrack1300 split files plus their split manifests
- `b1_bank_configs/`: YAML presets and a public wrapper for `B1` hard-negative mining and filtering
- `evaluation/`: public wrappers for threshold selection and fixed-test evaluation
- `figures_generation/`: public wrappers for paper-facing figure export
- `results/experiments.csv`: centralized experiment log
- `notebooks/03_test_visualization.ipynb`: qualitative review notebook for fixed test samples and failure cases
- `notebooks/06_round1_experiments.ipynb`: compact comparison dashboard for logged round-1 runs
- `notebooks/07_uav_crossdomain_diagnosis.ipynb`: UAV cross-domain diagnosis and threshold analysis
- `notebooks/10_segformer_audit_followup.ipynb`: compact SegFormer audit summary with keepable-share plots, curated-vs-raw comparison, and representative review-card browser
- `notebooks/11_paper_figures.ipynb`: notebook entry point for exported paper-facing figures
- `docs/paper-outline-detailed-v1.md`: detailed `AIC`-oriented writing outline with section logic, claim boundaries, and figure/table checklist
- `docs/paper-draft-intro-problem-method-v1.md`: first `AIC` prose draft for `Introduction`, `Problem Setting`, and `Method`
- `docs/paper-plan-drones-v1.md`: separate `Drones`-oriented paper plan built from the same frozen experiments

Public-release note:

- `Crack500` and `PaveCrack1300` raw datasets, trained checkpoints, large generated crop banks, and full audit-card exports are intentionally not mirrored in the repository.
- The primary Kaggle UAV dataset is mirrored separately under [kaggle_release](kaggle_release) with provenance files because its Kaggle page listed `CC0: Public Domain` at the time of access.
- Use [DATASETS.md](DATASETS.md) to reconstruct the expected local dataset layout before running the training or evaluation scripts.

Still pending:
- full `Related Work`, `Experiments`, and `Results` drafting
- venue-specific abstract and contribution polishing for `AIC` and `Drones`
- main-table / audit-taxonomy / few-shot figure polishing for the paper

## U-Net Ablation Outcome

The first single-variable `U-Net` ablation round on `Crack500` is now complete.

- Frozen reference: `unet_baseline_360`
  - validation `IoU`: `0.6545`
  - test `IoU`: `0.5680`
- `512` only, with baseline augmentation and `BCE + Dice`
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

Current interpretation:

- Higher resolution alone did not beat the frozen baseline on validation.
- Foreground-aware crop is helpful under the current setup.
- `Focal Tversky` is not helpful under the current setup.
- `mild` photometric augmentation helps more than `strong`.
- The current working best `U-Net` recipe is:
  - image size `360`
  - foreground-aware crop probability `0.5`
  - augmentation profile `mild`
  - loss `BCE + Dice`

## Experiment Log Caveat

- `results/experiments.csv` currently contains repeated experiment keys such as `unet_ablate_aug_strong_360_fgcrop` and `deeplabv3plus_plain_360`.
- Any notebook or summary script should take the latest `timestamp_utc` for repeated `(experiment_name, stage, split)` rows, otherwise an older result may be mixed into the comparison.
- `notebooks/06_round1_experiments.ipynb` now surfaces duplicate experiment keys explicitly, but any other analysis path should follow the same latest-row rule.

## Experiment Log Status

- The experiment-log schema was upgraded on `2026-04-26` so that new `train.py` and `test.py` runs now record:
  - `train_split`, `val_dataset_root`, and `val_split`
  - `aux_negative_*` settings for `B1`
  - `init_checkpoint_path` for `B2`
  - `eval_threshold` and postprocess parameters
  - `metric_precision` and `metric_recall` in addition to `loss`, `IoU`, and `F1`
- The CSV migration has already been applied in place, so future appends use one unified header.
- Older rows created before this schema upgrade may still have empty values in the newly added columns.
- When writing the paper, treat the curated summaries in `README.md`, `docs/baseline-results.md`, and `docs/research-plan.md` as the authoritative source for those earlier runs.

## SegFormer-B2 Outcome

The plain `SegFormer-B2` baseline and the first four single-variable transfer probes on `Crack500` are now recorded.

- `segformer_b2_plain_360`
  - validation `IoU`: `0.6666`
  - test `IoU`: `0.5854`
- `segformer_b2_mild_360`
  - validation `IoU`: `0.6644`
  - not promoted to test, because validation did not beat the plain baseline
- `segformer_b2_fgcrop_360`
  - validation `IoU`: `0.6703`
  - test `IoU`: `0.5816`
- `segformer_b2_ftversky_360`
  - validation `IoU`: `0.6460`
  - not promoted to test, because validation did not beat the plain baseline
- `segformer_b2_512_plain`
  - validation `IoU`: `0.6726`
  - test `IoU`: `0.5713`

Current interpretation:

- Plain `SegFormer-B2` is already a very strong baseline.
- It is essentially tied with the best tuned `U-Net` on validation and ahead of it on test.
- `mild` photometric augmentation did not improve `SegFormer-B2` under the current setup.
- Foreground-aware crop improved validation slightly, but did not improve test performance.
- `Focal Tversky` did not improve `SegFormer-B2` under the current setup.
- `512` resolution improved validation, but did not improve test performance.
- The current working best `SegFormer-B2` recipe is still the plain baseline:
  - image size `360`
  - baseline augmentation
  - no foreground-aware crop
  - loss `BCE + Dice`

## DeepLabV3+ Outcome

The first `DeepLabV3+` baseline package on `Crack500` and the first source-only UAV transfer checks are now recorded.

- `deeplabv3plus_plain_360`
  - earliest quick-stop run (`patience = 5`) peaked too early at `epoch 3`, so the stabilized reference uses `patience = 30`
  - stabilized validation `IoU`: `0.6443`
  - stabilized test `IoU`: `0.5648`
- `deeplabv3plus_mild_360`
  - validation `IoU`: `0.6514`
  - not promoted to test because `fgcrop_360` later exceeded it on validation and became the stronger confirmatory DeepLab run
- `deeplabv3plus_fgcrop_360`
  - validation `IoU`: `0.6613`
  - test `IoU`: `0.5709`
- `deeplabv3plus_ftversky_360`
  - validation `IoU`: `0.6263`
  - not promoted to test because validation did not beat the stabilized plain baseline
- `deeplabv3plus_512_plain`
  - validation `IoU`: `0.6282`
  - not promoted to test because validation did not beat the stabilized plain baseline

Current in-domain interpretation:

- `DeepLabV3+` needed a longer early-stopping window than the earliest quick-stop run.
- Foreground-aware crop helps `DeepLabV3+` in-domain, similar to `U-Net`.
- `mild` augmentation is promising on validation but is still below the current `fgcrop` winner.
- `Focal Tversky` and `512` resolution do not improve over the stabilized plain baseline under the current setup.
- The current working best `DeepLabV3+` recipe on `Crack500` is:
  - image size `360`
  - baseline augmentation
  - foreground-aware crop probability `0.5`
  - loss `BCE + Dice`
  - early-stopping patience `30`

Source-only `DeepLabV3+` on the fixed `UAV test` hold-out:

- `deeplabv3plus_plain_360_uav_holdout_raw`
  - `IoU = 0.1230`, `F1 = 0.2152`, `precision = 0.1303`, `recall = 0.6913`
- `deeplabv3plus_plain_360_uav_holdout_deploy`
  - `IoU = 0.1220`, `F1 = 0.2090`, `precision = 0.1447`, `recall = 0.4711`
- `deeplabv3plus_fgcrop_360_uav_holdout_raw`
  - `IoU = 0.0940`, `F1 = 0.1698`, `precision = 0.0987`, `recall = 0.6906`

Cross-domain interpretation:

- `DeepLabV3+` source-only transfer is weaker than the current `U-Net` and `SegFormer-B2` source-only references on the fixed UAV hold-out.
- The deploy-style threshold plus connected-component filter increases `precision` for `DeepLabV3+`, but unlike the `SegFormer-B2` source-only case it does not improve `IoU/F1`.
- The main `DeepLabV3+` transfer failure is false positives on UAV scenes.
- Foreground-aware crop helps `DeepLabV3+` in-domain but hurts cross-domain transfer almost entirely through a `precision` collapse at nearly unchanged `recall`.
- Therefore:
  - freeze `deeplabv3plus_fgcrop_360` as the current in-domain best `DeepLabV3+` recipe
  - freeze `deeplabv3plus_plain_360_uav_holdout_raw` as the `DeepLabV3+` source-only cross-domain reference
  - the current `DeepLabV3+ B1/B2` line should continue to start from `plain_360` rather than from `fgcrop_360`

DeepLab-specific transfer-prep status:

- A `DeepLabV3+` UAV hard-negative bank has been mined at `generated/uav_hard_negatives_deeplabv3plus_plain_360_train_thr080`.
- The bank contains `115` all-background crops from `80` source UAV training images and is ready for future `B1`-style mixed-training experiments.

DeepLab target-aware mitigation status:

- `deeplabv3plus_b1_negbank_uavval`
  - best validation checkpoint on `UAV val`: `epoch 9`, `IoU = 0.2704`, `F1 = 0.4009`, `precision = 0.5418`, `recall = 0.3750`
- `deeplabv3plus_b1_holdout_raw_thr050`
  - `IoU = 0.2830`, `F1 = 0.4143`, `precision = 0.5820`, `recall = 0.3767`
- `deeplabv3plus_b1_holdout_deploy`
  - `IoU = 0.1996`, `F1 = 0.3123`, `precision = 0.8021`, `recall = 0.2130`
- Temperature-scaled `B1` bank follow-up on `UAV val`
  - `deeplabv3plus_b1_tsbank_uavval`: `epoch 11`, `IoU = 0.3149`, `F1 = 0.4512`, `precision = 0.5672`, `recall = 0.4928`
  - `deeplabv3plus_b1_tsbank_thr080_mean080`: `epoch 13`, `IoU = 0.3068`, `F1 = 0.4544`, `precision = 0.4439`, `recall = 0.5218`
  - `deeplabv3plus_b1_tsbank_thr080_mean082`: `epoch 5`, `IoU = 0.2444`, `F1 = 0.3803`, `precision = 0.4056`, `recall = 0.4497`
- Audit-reviewed curated-bank follow-up on `UAV val`
  - manual audit completion: `223 / 223` layer-1 labels and `222 / 223` layer-2 labels filled
  - aggregate review distribution: `noise = 129`, `hard_fp = 66`, `ambiguous = 27`, `bad_crop = 1`
  - strongest reviewed keepable share among the audited `DeepLabV3+` banks: calibrated `TS-bank` `uav_hard_negatives_deeplabv3plus_plain_360_train_thr080_ts` with `23 / 43 = 53.5%` `hard_fp + ambiguous`
  - `deeplabv3plus_b1_tsbank_curated_hfpa_uavval`: `epoch 9`, `IoU = 0.2920`, `F1 = 0.4301`, `precision = 0.5727`, `recall = 0.4790`
  - `deeplabv3plus_b1_tsbank_curated_hardfp_uavval`: `epoch 10`, `IoU = 0.3099`, `F1 = 0.4567`, `precision = 0.5677`, `recall = 0.4570`
  - threshold sweep on the curated `hard_fp` checkpoint:
    - `thr = 0.35`: `IoU = 0.3112`, `F1 = 0.4600`
    - `thr = 0.40`: `IoU = 0.3122`, `F1 = 0.4605`
    - `thr = 0.45`: `IoU = 0.3114`, `F1 = 0.4589`
    - `thr = 0.50`: `IoU = 0.3099`, `F1 = 0.4567`
  - Audit-derived auto-filter follow-up on the same calibrated `TS-bank`
    - `scripts/banks/export_auto_filtered_hard_negative_bank.py` now exports manifest-filtered bank variants directly from the mined calibrated bank
    - `deeplabv3plus_b1_tsbank_autofilter_span035_uavval`: `span_ratio >= 0.35`, best explicit rerun `epoch 13`, `IoU = 0.2783`, `F1 = 0.4086`, `precision = 0.6296`, `recall = 0.3660`
    - `deeplabv3plus_b1_tsbank_autofilter_area1200_uavval`: `component_area >= 1200`, `epoch 25`, `IoU = 0.3564`, `F1 = 0.5136`, `precision = 0.5420`, `recall = 0.5329`
    - audited composition of the exported auto-filtered banks:
      - `span_ratio >= 0.35`: `23 / 43` kept = `15 hard_fp`, `3 ambiguous`, `5 noise`
      - `component_area >= 1200`: `28 / 43` kept = `16 hard_fp`, `4 ambiguous`, `8 noise`

Current `B1` interpretation:

- `DeepLabV3+ B1` now uses the promoted hold-out row `deeplabv3plus_b1_tsbank_autofilter_area1200_test`, which raises source-only raw `IoU` from `0.1230` to `0.3860` and `F1` from `0.2152` to `0.5470`.
- For the promoted `DeepLabV3+ B1` checkpoint, the default threshold `0.5` remains the best official raw operating point, so this line does not need the extra threshold retuning that benefited `SegFormer-B2`.
- The deployment-oriented switch is still clearly worse than the promoted raw `B1` row on `IoU / F1`, so it should remain comparison-only rather than becoming the preferred reporting line.
- A single matched random-background control on `UAV val` reaches `IoU 0.3023`, `F1 0.4368`, below the mined `component_area >= 1200` bank (`IoU 0.3564`, `F1 0.5136`). This is directionally consistent with useful mined nuisance signal for `DeepLabV3+`, but the evidence remains limited because only `3 / 28` random crops could be matched to the same source image as their reference crop.
- Temperature-scaled mining is still mixed for `DeepLabV3+` at the candidate level: `thr080_mean082` underperformed clearly, and the completed qualitative audit shows that the reviewed bank sample is still dominated by `noise`, so bank purity is a real concern rather than a cosmetic issue.
- Removing `ambiguous` crops helps materially relative to the curated `hard_fp + ambiguous` export (`0.2920 -> 0.3099` at the default threshold), which suggests that the ambiguous subset dilutes the training signal.
- The `span_ratio >= 0.35` auto-filter is too aggressive: it removes most audited `noise`, but it also trims useful nuisance structure and underperforms both the raw calibrated bank and the curated `hard_fp` export.
- By contrast, `component_area >= 1200` removes small junk while preserving all `16 / 16` audited `hard_fp`; that rule wins on validation (`IoU 0.3564`) and then confirms on the fixed hold-out split at `IoU 0.3860`, `F1 0.5470`.
- The revised reading is not “cleaner is always better,” but “keep the large target-domain nuisance structures while discarding tiny components.” Pure `hard_fp` curation was too restrictive, while the successful area rule preserved nuisance diversity without letting the smallest noise dominate.
- `deeplabv3plus_b1_tsbank_autofilter_area1200_test` is now the official `GT`-filtered `DeepLabV3+ B1` result, while the old raw `deeplabv3plus_b1_holdout_raw_thr050` row should be kept as historical comparison.

DeepLab few-shot `B2` status:

- `deeplabv3plus_b2_fs05_seed42`
  - default early-stopping run: validation `IoU = 0.3119`, best `epoch 25`
- `deeplabv3plus_b2_fs05_seed42_pat12`
  - longer-patience rerun: validation `IoU = 0.3239`, best `epoch 38`
  - fixed-hold-out test: `IoU = 0.3599`, `F1 = 0.5226`, `precision = 0.4967`, `recall = 0.5998`
- `deeplabv3plus_b2_fs10_seed42`
  - validation `IoU = 0.3410`
  - fixed-hold-out test: `IoU = 0.4354`, `F1 = 0.6005`, `precision = 0.5608`, `recall = 0.6620`
- `deeplabv3plus_b2_fs20_seed42`
  - validation `IoU = 0.4511`
  - fixed-hold-out test: `IoU = 0.4760`, `F1 = 0.6340`, `precision = 0.5903`, `recall = 0.7060`

Current `B2` interpretation:

- Limited target-domain supervision clearly helps `DeepLabV3+` beyond the `B1` setting.
- The smallest `DeepLabV3+` few-shot split is more sensitive to early stopping than the in-domain `Crack500` runs; increasing patience from `5` to `12` improved the `fs05` validation checkpoint.
- The fixed-hold-out `B2` test curve is now confirmed end to end: `fs05_pat12 -> fs10 -> fs20` improves `IoU` from `0.3599 -> 0.4354 -> 0.4760`.

DeepLab target-domain upper bound:

- `deeplabv3plus_uav_indomain_plain_360`
  - best validation `IoU = 0.5068`, `F1 = 0.6677`, best `epoch 37`
  - fixed-hold-out test: `IoU = 0.5085`, `F1 = 0.6693`, `precision = 0.6187`, `recall = 0.7295`

Current upper-bound interpretation:

- This run now gives `DeepLabV3+` a ceiling row under the same `UAV train / val / test`, `360`, `baseline`, `40`-epoch protocol already used for the `SegFormer-B2` upper bound.
- The remaining gap above `DeepLabV3+ fs20` is modest (`0.4760 -> 0.5085`), so limited target supervision already recovers a large fraction of the target-domain ceiling for this backbone too.

## Cross-Domain UAV Status

The first `Crack500 -> UAV_Crack_Segmentation_Kaggle` cross-domain evaluation round is now in place, and the dataset now has both an exploratory full split and a fixed official hold-out split.

Exploratory diagnosis on `crossdomain_all` (`315` images):

- `U-Net` raw prediction: `IoU = 0.1389`, `F1 = 0.2333`
- `SegFormer-B2` raw prediction: `IoU = 0.1781`, `F1 = 0.2829`
- These raw predictions remain the official cross-domain baselines because their role is to measure the model's own generalization under domain shift, not a model-plus-heuristic bundle.

Postprocessed deployment-oriented variant:

- An optional high-threshold plus connected-component filter is now available for deployment-oriented evaluation in cluttered UAV scenes.
- On the UAV test set, this improves deployable performance by suppressing false positives on non-crack objects:
  - `U-Net`: `IoU 0.1389 -> 0.1560`, `precision 0.1458 -> 0.1783`, `recall 0.7685 -> 0.5953`
  - `SegFormer-B2`: `IoU 0.1781 -> 0.2010`, `precision 0.1873 -> 0.2560`, `recall 0.7506 -> 0.4855`
- On `Crack500` in-domain, the same filter is not universally better:
  - `SegFormer-B2`: `IoU 0.5854 -> 0.5399`, `precision 0.6878 -> 0.7845`, `recall 0.8005 -> 0.6367`

Official hold-out on `test` (`63` images):

- Source-only `SegFormer-B2` raw:
  - `IoU = 0.1442`, `F1 = 0.2476`, `precision = 0.1495`, `recall = 0.7727`
- ADVENT-style `DeepLabV3+` UDA reviewer baseline:
  - source labels: `CRACK500/train`
  - unlabeled target images: `UAV_Crack_Segmentation_Kaggle/train`
  - validation-selected threshold: `0.9` on `UAV val`
  - `UAV test`: `IoU = 0.2022`, `F1 = 0.3076`, `precision = 0.2427`, `recall = 0.4845`
  - local implementation: `train_advent.py` plus `scripts/reports/run_threshold_sweep_report.py`
- HRDA-style `MiT-B5` UDA reviewer baseline:
  - source labels: `Crack500/train`
  - unlabeled target images: `UAV/train`
  - validation-selected threshold: `0.3` on `UAV val`
  - `UAV test`: `IoU = 0.1143`, `F1 = 0.2052`, `precision = 0.1191`, `recall = 0.7390`
  - local implementation: `external/HRDA` plus `scripts/experiments/run_hrda_crack500_to_uav.sh`
- Source-only `SegFormer-B2` deployment variant:
  - `IoU = 0.1784`, `F1 = 0.2900`, `precision = 0.2099`, `recall = 0.5081`
- `B1` hard-negative-mixed `SegFormer-B2` raw at the default threshold `0.5`:
  - `IoU = 0.3222`, `F1 = 0.4677`, `precision = 0.4660`, `recall = 0.5662`
- `B1` calibrated raw operating point at threshold `0.7`:
  - `IoU = 0.3325`, `F1 = 0.4727`, `precision = 0.5374`, `recall = 0.5133`
- `B1` high-precision raw operating point at threshold `0.9`:
  - `IoU = 0.3193`, `F1 = 0.4502`, `precision = 0.6497`, `recall = 0.4112`
- `B1` hard-negative-mixed `SegFormer-B2` deployment variant:
  - `IoU = 0.3166`, `F1 = 0.4469`, `precision = 0.6490`, `recall = 0.4071`
- `B2` few-shot fine-tuning from the source-domain `SegFormer-B2` checkpoint:
  - `fs05` (`9` labeled UAV train samples): `IoU = 0.5074`, `F1 = 0.6695`, `precision = 0.6232`, `recall = 0.7277`
  - `fs10` (`19` labeled UAV train samples): `IoU = 0.5420`, `F1 = 0.6988`, `precision = 0.6762`, `recall = 0.7251`
  - `fs20` (`38` labeled UAV train samples): `IoU = 0.5686`, `F1 = 0.7209`, `precision = 0.6724`, `recall = 0.7826`
- Target-domain upper bound with in-domain `SegFormer-B2` training:
  - `IoU = 0.5879`, `F1 = 0.7369`, `precision = 0.6970`, `recall = 0.7830`

Temperature-scaled `B1` bank follow-up on `UAV val`:

- `segformer_b1_tsbank_uavval`
  - `epoch 5`, `IoU = 0.3568`, `F1 = 0.5027`, `precision = 0.5499`, `recall = 0.5236`
- `segformer_b2_b1_tsbank_thr080_mean082`
  - `epoch 10`, `IoU = 0.3693`, `F1 = 0.5227`, `precision = 0.4996`, `recall = 0.6096`
- `segformer_b2_b1_tsbank_thr075_mean080`
  - `epoch 10`, `IoU = 0.2558`, `F1 = 0.3782`, `precision = 0.3280`, `recall = 0.6191`

Current interpretation:

- The default baseline should continue to use raw predictions.
- The postprocessed result should be reported separately as a deployment-oriented variant.
- Its gain is real for UAV cluttered scenes, but it comes from a clear `precision-recall` tradeoff rather than a universal model improvement.
- The ADVENT-style `DeepLabV3+` baseline improves over same-backbone source-only transfer, but its selected threshold of `0.9` shows that generic output-space alignment still needs a highly conservative operating point on UAV imagery.
- The first training-side mitigation result is now in place:
  - even the original `B1 raw` row was already well above the source-only deployment-oriented variant.
- The promoted `TS-bank` confirmatory hold-out update is now the strongest `GT`-filtered `SegFormer-B2 B1` result:
  - `segformer_b2_b1_tsbank_thr080_mean082_test_thr060` reaches `IoU 0.3775`, `F1 0.5317`, `precision 0.5257`, `recall 0.5779`.
  - This exceeds both the old `B1 raw @ 0.7` main-report row (`IoU 0.3325`) and the default-threshold TS-bank hold-out row (`IoU 0.3728`).
- The first single random-background control (`seed42`) looked much weaker than the mined bank (`IoU 0.2747 < 0.3693`), but a later paired `5-seed` rerun changed the picture:
  - matched random-background seeds `42 / 7 / 13 / 21 / 99`: `IoU 0.3458, 0.3268, 0.3629, 0.3101, 0.3551`, mean `0.3401 +- 0.0215`
  - mined `TS-bank` seeds `42 / 7 / 13 / 21 / 99`: `IoU 0.3143, 0.3284, 0.3275, 0.3455, 0.2815`, mean `0.3194 +- 0.0239`
- The current `SegFormer-B2` mechanism reading therefore needs to stay cautious: matched random background is at least competitive with the mined bank on `UAV val`, so a substantial share of the `B1` gain may come from generic target-background exposure rather than from mined negatives alone.
- A first `SegFormer-B2 TS-bank` audit package is now available at `results/hard_negative_audit/segformer_b2_tsbank_round1`, covering `thr080_mean082`, `thr080_mean080`, and `thr075_mean080` with `180` reviewed cards (`60` per bank).
- Audit-derived curated-bank follow-up on `UAV val` stayed below the raw mined winner `segformer_b2_b1_tsbank_thr080_mean082` (`IoU 0.3693`, `F1 0.5227`):
  - `thr080_mean080 hard_fp + ambiguous`: `IoU 0.3439`, `F1 0.4830`
  - `thr080_mean080 hard_fp`: `IoU 0.3420`, `F1 0.4836`
  - `thr080_mean082 hard_fp + ambiguous`: `IoU 0.3304`, `F1 0.4682`
  - `thr080_mean082 hard_fp`: `IoU 0.3535`, `F1 0.4993`
- The strongest curated export is therefore `thr080_mean082 hard_fp`, but it still remains a diagnostic follow-up rather than a promotable replacement because `0.3535 < 0.3693`.
- On `thr080_mean082`, removing `ambiguous` crops helps materially (`0.3304 -> 0.3535`), while on `thr080_mean080` the `hard_fp + ambiguous` and `hard_fp` exports are nearly tied. The current audit reading for `SegFormer-B2` is therefore explanatory rather than performance-changing.
- The old deployment switch is no longer an `IoU/F1` improvement by default after `B1`.
  - `B1 raw @ 0.9` slightly outperforms the deployment variant on `IoU/F1` while matching its high precision.
  - The promoted `TS-bank @ 0.6` row now serves as the main `GT`-filtered mitigation result, while `B1 raw @ 0.9` remains the high-precision comparison point.
- The target-domain upper bound is now close to `IoU 0.60`, which is much higher than both the source-only baseline and `B1`.
  - This strongly suggests that the main bottleneck is the domain gap itself rather than the intrinsic impossibility of the UAV task.
- `B2` now shows that the domain gap is highly correctable with limited target supervision.
  - Starting from the promoted `B1 TS-bank @ 0.6 = 0.3775`, few-shot target fine-tuning reaches `0.5074`, `0.5420`, and `0.5686` with only `5%`, `10%`, and `20%` labeled UAV data.
  - `fs20` already reaches about `96.7%` of the full in-domain upper bound.
- Temperature-scaled hard-negative mining therefore now has confirmatory fixed-hold-out support on `SegFormer-B2`.
  - `segformer_b2_b1_tsbank_thr080_mean082` improved validation (`0.3419 -> 0.3693`), survived threshold sweep, and then confirmed on test at `IoU 0.3775`.
  - The looser large-context alternative `thr075_mean080` underperforms badly, so bigger components are not automatically better.
- This creates a cleaner evaluation-study structure:
  - raw model performance = true cross-domain generalization
  - postprocessed deployment variant = deployable performance after adding a geometric prior to suppress false positives
  - training-side mitigation = raw performance improvement from target-style nuisance exposure rather than from post-hoc filtering
  - matched random-background control = mechanism check showing that generic target-background exposure explains a substantial share of the gain, while any extra mined-bank value remains model-dependent rather than universal
  - few-shot fine-tuning = supervision-efficiency curve for closing the remaining domain gap
  - target-domain upper bound = ceiling reference that quantifies the remaining gap after cross-domain mitigation

Source-domain retention and `Kaggle test` uncertainty follow-up on `2026-05-09`:

- Assets were generated with `scripts/reports/make_retention_report_assets.py` and saved under `results/report_assets/source_retention/`.
- The follow-up reused the frozen `SegFormer-B2` checkpoints for:
  - `source_only @ 0.5`
  - promoted `B1 TS-bank @ 0.6`
  - legacy `B1 raw @ 0.7`
  - `B2 fs05 / fs10 / fs20 @ 0.5`
- `Crack500 test` (`1124` images) was used as the source-domain retention back-test.
- `UAV_Crack_Segmentation_Kaggle test` (`63` images) was evaluated with `5000` image-level bootstrap resamples to estimate `95%` confidence intervals.
- The report also includes an optional lightweight morphological skeleton metric to check whether centerline quality follows the same trend as region metrics.
- Point estimates in this report are recomputed from per-image `tp / fp / fn` sums so they stay consistent with the bootstrap pipeline; they can therefore differ slightly from older batch-averaged rows in `results/experiments.csv`.

Key results:

- Source-only reference:
  - `Crack500 IoU = 0.5988`, `F1 = 0.7491`
  - `Kaggle IoU = 0.1394`, `F1 = 0.2447`, `95% CI [0.0982, 0.1929]`
- Promoted `B1 TS-bank @ 0.6`:
  - `Crack500 IoU = 0.5745`, which keeps `95.9%` of the source-only `IoU`
  - `Kaggle IoU = 0.3832`, gain `+0.2437` vs source-only, `95% CI [0.3035, 0.4592]`
- Legacy `B1 raw @ 0.7`:
  - `Crack500 IoU = 0.5792`, which keeps `96.7%` of the source-only `IoU`
  - `Kaggle IoU = 0.3539`, gain `+0.2145` vs source-only, `95% CI [0.2613, 0.4526]`
- `B2 fs05 / fs10 / fs20`:
  - `Crack500 IoU = 0.5458 / 0.5281 / 0.5173`
  - source-domain retention = `91.2% / 88.2% / 86.4%`
  - `Kaggle IoU = 0.5165 / 0.5506 / 0.5764`
  - `Kaggle IoU 95% CI = [0.4672, 0.5590] / [0.5007, 0.5935] / [0.5236, 0.6238]`
- Optional skeleton follow-up:
  - `Crack500 skeleton F1` stays roughly flat around `0.13`
  - `Kaggle skeleton F1` rises monotonically from `0.1838` to `0.3989`

Interpretation after the retention and CI follow-up:

- `B1` remains the best low-cost transfer step: it gives a large target-domain gain while only reducing `Crack500 IoU` by about `2% - 4%`.
- The promoted `TS-bank @ 0.6` checkpoint is slightly less source-preserving than the older `B1 raw @ 0.7`, but it is clearly stronger on the target domain, so it remains the correct official `B1` row.
- `B2` produces a smooth supervision-efficiency tradeoff rather than catastrophic forgetting: target-domain quality rises monotonically while source-domain `IoU` declines gradually from `91.2%` to `86.4%` retention.
- The `fs10` and `fs20` bootstrap intervals still overlap, so the direction of improvement is clear, but small differences inside the top end of the `B2` curve should be discussed with uncertainty attached rather than as absolute rankings.

## Environment

- GPU: `NVIDIA GeForce RTX 4070 SUPER (12GB VRAM)`
- OS: `Windows 11 + WSL2 Ubuntu`
- CUDA: `13.1`
- Python: `3.10` in conda environment `crackdet`
- Main libraries: `PyTorch`, `HuggingFace Transformers`, `segmentation-models-pytorch`, `albumentations`

## Immediate Next Steps

1. Freeze `360 + foreground crop + mild augmentation + BCE + Dice` as the current best `U-Net` recipe.
2. Use notebook summaries that deduplicate repeated experiment names by latest `timestamp_utc`.
3. Freeze `segformer_b2_plain_360` as the current `SegFormer-B2` baseline.
4. Freeze `deeplabv3plus_fgcrop_360` as the current in-domain `DeepLabV3+` recipe.
5. Freeze `deeplabv3plus_plain_360_uav_holdout_raw` as the `DeepLabV3+` source-only cross-domain reference and keep `deeplabv3plus_plain_360_uav_holdout_deploy` as a separate deployment-only comparison.
6. Freeze `deeplabv3plus_b1_tsbank_autofilter_area1200_test` as the current `DeepLabV3+` training-side mitigation row and keep `deeplabv3plus_b1_holdout_deploy` as a comparison-only deployment row.
7. Keep any further `DeepLabV3+ B2` work rooted in `plain_360`; the current frozen hold-out curve is `fs05_pat12` test `IoU 0.3599`, `fs10` test `IoU 0.4354`, and `fs20` test `IoU 0.4760`.
8. Keep the cross-domain baseline tables and plots on raw predictions rather than postprocessed outputs.
9. Report the UAV postprocessed results separately as deployment-oriented variants instead of replacing the baseline.
10. Freeze the current official `UAV test` reference rows:
   - source-only raw
   - source-only deployment
   - historical raw `B1` calibration comparisons (`0.5 / 0.7 / 0.9`) where relevant
   - promoted `SegFormer-B2 B1` `TS-bank thr080_mean082 @ 0.6`
   - promoted `DeepLabV3+ B1` `TS-bank area1200 @ 0.5`
   - `B1` deployment comparison rows
11. Freeze the in-domain `SegFormer-B2` upper bound on `UAV test` as the ceiling reference:
   - `IoU 0.5879`
   - `F1 0.7369`
   - `precision 0.6970`
   - `recall 0.7830`
12. Freeze `B2` few-shot results on the fixed `UAV test` split:
   - `fs05`: `IoU 0.5074`
   - `fs10`: `IoU 0.5420`
   - `fs20`: `IoU 0.5686`
13. Use the promoted `TS-bank` rows as the main training-side mitigation results:
   - `SegFormer-B2`: `TS-bank thr080_mean082 @ 0.6`
   - `DeepLabV3+`: `TS-bank area1200 @ 0.5`
14. Keep `SegFormer-B2 B1 raw @ 0.9` as the high-precision operating point and keep `B1` deployment only as comparison rows that show how little extra value the old postprocess switch adds after calibration.
15. Retire the old validation-only wording:
   - `segformer_b2_b1_tsbank_thr080_mean082` and `deeplabv3plus_b1_tsbank_autofilter_area1200_uavval` have now completed confirmatory hold-out promotion.
16. Treat the current `GT`-filtered `B1`, few-shot `B2`, and upper-bound rows as frozen paper-facing results unless a new ablation is explicitly needed.
17. Prepare the final paper-facing comparison assets:
   - source-only vs `B1` vs `B2` vs upper bound main table
   - `B2` supervision-scaling curve
   - representative qualitative figure set
18. Treat the current training, testing, split-generation, hard-negative-mining, and experiment-logging code paths as frozen unless a new paper-facing need appears.
19. Freeze `results/report_assets/source_retention/` as the paper-facing support package for source-retention and uncertainty claims:
   - `combined_summary.csv`
   - `summary.md`
   - per-stage `Crack500` retention and `Kaggle` bootstrap tables
20. When comparing close `UAV test` rows, cite the bootstrap intervals alongside point estimates, especially for late `B2` comparisons such as `fs10` vs `fs20`.

Important evaluation rule:

- Use the validation set to choose settings.
- Keep the test set confirmatory rather than using it for repeated tuning.
- When the same experiment is rerun with the same name, compare only the latest logged row for that `(experiment_name, stage, split)` key.

## Repository Layout

```text
.
├── DATASETS.md             # dataset setup and redistribution notes
├── environment.yml         # main public environment
├── environment-daformer.yml  # DAFormer/HRDA compatibility environment
├── dataset.py              # dataset loader and transforms
├── model.py                # segmentation model factory
├── loss.py                 # training loss
├── metrics.py              # evaluation metrics
├── postprocess.py          # optional deployment-oriented output filtering
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
└── docs/                   # planning and baseline records
```
