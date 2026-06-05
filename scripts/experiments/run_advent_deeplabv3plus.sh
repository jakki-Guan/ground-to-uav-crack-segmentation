#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="checkpoints/advent_deeplabv3plus_crack500_to_uav.pth"

# This first ADVENT run starts from the frozen source-only DeepLabV3+ checkpoint
# so it tests target adaptation without repeating source pretraining.
python train_advent.py \
  --dataset-root CRACK500 \
  --train-split train \
  --target-dataset-root UAV_Crack_Segmentation_Kaggle \
  --target-train-split train \
  --val-dataset-root UAV_Crack_Segmentation_Kaggle \
  --val-split val \
  --model-name DeepLabV3Plus \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --init-checkpoint-path checkpoints/deeplabv3plus_plain_360.pth \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --experiment-name advent_deeplabv3plus_crack500_to_uav_val \
  --epochs 30 \
  --batch-size 4 \
  --lr 1e-4 \
  --disc-lr 1e-4 \
  --lambda-adv-target 1e-3 \
  --img-size 360 \
  --early-stopping-patience 8 \
  --seed 42

python scripts/reports/run_threshold_sweep_report.py \
  --dataset-root UAV_Crack_Segmentation_Kaggle \
  --val-split val \
  --test-split test \
  --model-name DeepLabV3Plus \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --checkpoint-path "${CHECKPOINT_PATH}" \
  --experiment-name advent_deeplabv3plus_crack500_to_uav_test_thrval \
  --batch-size 8 \
  --img-size 360 \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 \
  --selection-metric iou \
  --output-dir results/report_assets/advent_deeplabv3plus_threshold_sweep
