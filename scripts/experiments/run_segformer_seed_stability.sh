#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SEEDS=("$@")
if [ "$#" -eq 0 ]; then
  SEEDS=(7 13 42)
fi

DATASET_ROOT="UAV_Crack_Segmentation_Kaggle"
TS_BANK_ROOT="tmp/ts_mining_sweep/segformer_b2_plain_360/thr080_mean082"
RESULTS_CSV="results/experiments.csv"

run_train() {
  local checkpoint_path="$1"
  shift
  if [ -f "$checkpoint_path" ]; then
    echo "Skipping train; checkpoint already exists: $checkpoint_path"
    return 0
  fi
  python train.py "$@"
}

run_test() {
  python test.py "$@"
}

for seed in "${SEEDS[@]}"; do
  tag=$(printf "seed%03d" "$seed")

  source_checkpoint="checkpoints/segformer_b2_plain_360_${tag}.pth"
  source_train_exp="segformer_b2_plain_360_${tag}"
  source_test_exp="segformer_b2_plain_360_${tag}_uav_holdout_raw_test"

  b1_checkpoint="checkpoints/segformer_b2_b1_tsbank_thr080_mean082_${tag}.pth"
  b1_train_exp="segformer_b2_b1_tsbank_thr080_mean082_${tag}"
  b1_test_exp="segformer_b2_b1_tsbank_thr080_mean082_${tag}_test_thr060"

  fs05_checkpoint="checkpoints/segformer_b2_b2_fs05_datasplit042_${tag}.pth"
  fs05_train_exp="segformer_b2_b2_fs05_datasplit042_${tag}"
  fs05_test_exp="segformer_b2_b2_fs05_datasplit042_${tag}_test"

  fs10_checkpoint="checkpoints/segformer_b2_b2_fs10_datasplit042_${tag}.pth"
  fs10_train_exp="segformer_b2_b2_fs10_datasplit042_${tag}"
  fs10_test_exp="segformer_b2_b2_fs10_datasplit042_${tag}_test"

  fs20_checkpoint="checkpoints/segformer_b2_b2_fs20_datasplit042_${tag}.pth"
  fs20_train_exp="segformer_b2_b2_fs20_datasplit042_${tag}"
  fs20_test_exp="segformer_b2_b2_fs20_datasplit042_${tag}_test"

  echo "=== [${tag}] source-only train ==="
  run_train "$source_checkpoint" \
    --dataset-root CRACK500 \
    --train-split train \
    --val-split val \
    --model-name segformer-b2 \
    --checkpoint-path "$source_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --epochs 30 \
    --lr 1e-4 \
    --num-workers 0 \
    --seed "$seed" \
    --augmentation-profile baseline \
    --loss-name bce_dice \
    --experiment-name "$source_train_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] source-only test on UAV hold-out ==="
  run_test \
    --dataset-root "$DATASET_ROOT" \
    --split test \
    --model-name segformer-b2 \
    --checkpoint-path "$source_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --num-workers 0 \
    --loss-name bce_dice \
    --experiment-name "$source_test_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B1 promoted train ==="
  run_train "$b1_checkpoint" \
    --dataset-root CRACK500 \
    --train-split train \
    --val-dataset-root "$DATASET_ROOT" \
    --val-split val \
    --model-name segformer-b2 \
    --checkpoint-path "$b1_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --epochs 30 \
    --lr 1e-4 \
    --num-workers 0 \
    --seed "$seed" \
    --augmentation-profile baseline \
    --loss-name bce_dice \
    --aux-negative-root "$TS_BANK_ROOT" \
    --aux-negative-split train \
    --aux-negative-repeat 2 \
    --experiment-name "$b1_train_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B1 promoted test on UAV hold-out ==="
  run_test \
    --dataset-root "$DATASET_ROOT" \
    --split test \
    --model-name segformer-b2 \
    --checkpoint-path "$b1_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --num-workers 0 \
    --loss-name bce_dice \
    --eval-threshold 0.6 \
    --experiment-name "$b1_test_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B2 fs05 fine-tune ==="
  run_train "$fs05_checkpoint" \
    --dataset-root "$DATASET_ROOT" \
    --train-split train_fs05_seed42 \
    --val-split val \
    --model-name segformer-b2 \
    --init-checkpoint-path "$source_checkpoint" \
    --checkpoint-path "$fs05_checkpoint" \
    --batch-size 4 \
    --img-size 360 \
    --epochs 50 \
    --lr 5e-5 \
    --num-workers 0 \
    --seed "$seed" \
    --augmentation-profile baseline \
    --loss-name bce_dice \
    --experiment-name "$fs05_train_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B2 fs05 test on UAV hold-out ==="
  run_test \
    --dataset-root "$DATASET_ROOT" \
    --split test \
    --model-name segformer-b2 \
    --checkpoint-path "$fs05_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --num-workers 0 \
    --loss-name bce_dice \
    --experiment-name "$fs05_test_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B2 fs10 fine-tune ==="
  run_train "$fs10_checkpoint" \
    --dataset-root "$DATASET_ROOT" \
    --train-split train_fs10_seed42 \
    --val-split val \
    --model-name segformer-b2 \
    --init-checkpoint-path "$source_checkpoint" \
    --checkpoint-path "$fs10_checkpoint" \
    --batch-size 4 \
    --img-size 360 \
    --epochs 50 \
    --lr 5e-5 \
    --num-workers 0 \
    --seed "$seed" \
    --augmentation-profile baseline \
    --loss-name bce_dice \
    --experiment-name "$fs10_train_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B2 fs10 test on UAV hold-out ==="
  run_test \
    --dataset-root "$DATASET_ROOT" \
    --split test \
    --model-name segformer-b2 \
    --checkpoint-path "$fs10_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --num-workers 0 \
    --loss-name bce_dice \
    --experiment-name "$fs10_test_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B2 fs20 fine-tune ==="
  run_train "$fs20_checkpoint" \
    --dataset-root "$DATASET_ROOT" \
    --train-split train_fs20_seed42 \
    --val-split val \
    --model-name segformer-b2 \
    --init-checkpoint-path "$source_checkpoint" \
    --checkpoint-path "$fs20_checkpoint" \
    --batch-size 4 \
    --img-size 360 \
    --epochs 50 \
    --lr 5e-5 \
    --num-workers 0 \
    --seed "$seed" \
    --augmentation-profile baseline \
    --loss-name bce_dice \
    --experiment-name "$fs20_train_exp" \
    --results-csv "$RESULTS_CSV"

  echo "=== [${tag}] B2 fs20 test on UAV hold-out ==="
  run_test \
    --dataset-root "$DATASET_ROOT" \
    --split test \
    --model-name segformer-b2 \
    --checkpoint-path "$fs20_checkpoint" \
    --batch-size 8 \
    --img-size 360 \
    --num-workers 0 \
    --loss-name bce_dice \
    --experiment-name "$fs20_test_exp" \
    --results-csv "$RESULTS_CSV"
done

echo "=== Seed-stability runs complete ==="
echo "To aggregate the 15 hold-out rows, run:"
echo "python scripts/reports/make_seed_stability_report.py"
