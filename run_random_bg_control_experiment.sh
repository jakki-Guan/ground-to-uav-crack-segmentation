#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-segformer}"

case "$TARGET" in
  segformer)
    exec python train.py \
      --dataset-root CRACK500 \
      --val-dataset-root UAV_Crack_Segmentation_Kaggle \
      --val-split val \
      --model-name segformer-b2 \
      --checkpoint-path checkpoints/segformer_b2_b1_random_bg_thr080_mean082.pth \
      --batch-size 8 \
      --img-size 360 \
      --epochs 30 \
      --lr 1e-4 \
      --augmentation-profile baseline \
      --aux-negative-root generated/random_background_banks/segformer_b2_plain_360__thr080_mean082__random_bg \
      --aux-negative-split train \
      --aux-negative-repeat 2 \
      --experiment-name segformer_b2_b1_random_bg_thr080_mean082_uavval
    ;;
  deeplab)
    exec python train.py \
      --dataset-root CRACK500 \
      --val-dataset-root UAV_Crack_Segmentation_Kaggle \
      --val-split val \
      --model-name DeepLabV3+ \
      --encoder-name resnet34 \
      --encoder-weights imagenet \
      --checkpoint-path checkpoints/deeplabv3plus_b1_random_bg_area1200.pth \
      --batch-size 8 \
      --img-size 360 \
      --epochs 40 \
      --lr 1e-4 \
      --augmentation-profile baseline \
      --aux-negative-root generated/random_background_banks/deeplabv3plus_plain_360__area1200__random_bg \
      --aux-negative-split train \
      --aux-negative-repeat 3 \
      --early-stopping-patience 10 \
      --experiment-name deeplabv3plus_b1_random_bg_area1200_uavval
    ;;
  all)
    "$0" segformer
    "$0" deeplab
    ;;
  *)
    echo "Usage: $0 [segformer|deeplab|all]" >&2
    exit 1
    ;;
esac
