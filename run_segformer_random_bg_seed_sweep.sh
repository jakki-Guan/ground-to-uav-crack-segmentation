#!/usr/bin/env bash
set -euo pipefail

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  SEEDS=(7 13)
fi

REFERENCE_BANK_ROOT="tmp/ts_mining_sweep/segformer_b2_plain_360/thr080_mean082"
DATASET_ROOT="UAV_Crack_Segmentation_Kaggle"

for seed in "${SEEDS[@]}"; do
  tag=$(printf "seed%03d" "$seed")
  bank_root="generated/random_background_banks/segformer_b2_plain_360__thr080_mean082__random_bg_${tag}"
  checkpoint_path="checkpoints/segformer_b2_b1_random_bg_thr080_mean082_${tag}.pth"
  experiment_name="segformer_b2_b1_random_bg_thr080_mean082_${tag}_uavval"

  echo "=== Exporting random-background bank for ${tag} ==="
  python export_random_background_bank.py \
    --dataset-root "$DATASET_ROOT" \
    --split train \
    --reference-bank-root "$REFERENCE_BANK_ROOT" \
    --output-root "$bank_root" \
    --seed "$seed" \
    --overwrite

  echo "=== Training SegFormer-B2 random-background control for ${tag} ==="
  python train.py \
    --dataset-root CRACK500 \
    --val-dataset-root "$DATASET_ROOT" \
    --val-split val \
    --model-name segformer-b2 \
    --checkpoint-path "$checkpoint_path" \
    --batch-size 8 \
    --img-size 360 \
    --epochs 30 \
    --lr 1e-4 \
    --seed "$seed" \
    --augmentation-profile baseline \
    --aux-negative-root "$bank_root" \
    --aux-negative-split train \
    --aux-negative-repeat 2 \
    --experiment-name "$experiment_name"
done
