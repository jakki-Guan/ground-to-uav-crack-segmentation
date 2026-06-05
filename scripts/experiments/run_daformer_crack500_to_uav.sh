#!/usr/bin/env bash
set -euo pipefail

DAFORMER_ROOT="${DAFORMER_ROOT:-external/DAFormer}"
DAFORMER_PYTHON="${DAFORMER_PYTHON:-python}"
CONFIG_PATH="${CONFIG_PATH:-configs/daformer/crack500_to_uav_daformer_mitb5_s0.py}"
DATA_OUTPUT_ROOT="${DATA_OUTPUT_ROOT:-generated/daformer/crack500_to_uav}"
WORK_DIR="${WORK_DIR:-work_dirs/daformer_crack500_to_uav_mitb5_s0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WORK_DIR}/latest.pth}"
REPORT_DIR="${REPORT_DIR:-results/report_assets/daformer_mitb5_threshold_sweep}"

PREPARE_DATA="${PREPARE_DATA:-1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

SEED="${SEED:-42}"
MAX_ITERS="${MAX_ITERS:-40000}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-4000}"

if [[ ! -d "${DAFORMER_ROOT}" ]]; then
  echo "DAFormer repository not found at ${DAFORMER_ROOT}."
  echo "Clone it with: git clone --depth 1 https://github.com/lhoyer/DAFormer.git ${DAFORMER_ROOT}"
  exit 1
fi

if [[ "${PREPARE_DATA}" == "1" ]]; then
  if [[ -d "${DATA_OUTPUT_ROOT}" && "${FORCE_PREPARE}" != "1" ]]; then
    echo "Using existing DAFormer dataset at ${DATA_OUTPUT_ROOT}. Set FORCE_PREPARE=1 to rebuild."
  else
    prepare_args=(
      scripts/daformer/prepare_crack500_to_uav_mmseg.py
      --output-root "${DATA_OUTPUT_ROOT}"
    )
    if [[ "${FORCE_PREPARE}" == "1" ]]; then
      prepare_args+=(--force)
    fi
    python "${prepare_args[@]}"
  fi
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  if [[ ! -f "${DAFORMER_ROOT}/pretrained/mit_b5.pth" ]]; then
    echo "Missing MiT-B5 ImageNet weights: ${DAFORMER_ROOT}/pretrained/mit_b5.pth"
    echo "Use the official DAFormer/SegFormer MiT-B5 download before training."
    echo "The official helper is: (cd ${DAFORMER_ROOT} && bash tools/download_checkpoints.sh)"
    echo "If the official Google Drive link is unavailable, run:"
    echo "  ${DAFORMER_PYTHON} scripts/daformer/convert_hf_mit_b5_to_daformer.py --output ${DAFORMER_ROOT}/pretrained/mit_b5.pth"
    exit 2
  fi

  PYTHONPATH="${DAFORMER_ROOT}:${PYTHONPATH:-}" "${DAFORMER_PYTHON}" "${DAFORMER_ROOT}/tools/train.py" \
    "${CONFIG_PATH}" \
    --work-dir "${WORK_DIR}" \
    --seed "${SEED}" \
    --options \
      runner.max_iters="${MAX_ITERS}" \
      data.samples_per_gpu="${SAMPLES_PER_GPU}" \
      data.workers_per_gpu="${WORKERS_PER_GPU}" \
      checkpoint_config.interval="${CHECKPOINT_INTERVAL}" \
      evaluation.interval="${EVAL_INTERVAL}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found for DAFormer evaluation: ${CHECKPOINT_PATH}"
    echo "Set CHECKPOINT_PATH to an existing .pth file, or finish training first."
    exit 3
  fi

  PYTHONPATH="${DAFORMER_ROOT}:${PYTHONPATH:-}" "${DAFORMER_PYTHON}" scripts/daformer/evaluate_daformer_threshold_sweep.py \
    --daformer-root "${DAFORMER_ROOT}" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output-dir "${REPORT_DIR}" \
    --num-workers "${WORKERS_PER_GPU}" \
    --experiment-name "daformer_mitb5_crack500_to_uav_test_thrval"
fi
