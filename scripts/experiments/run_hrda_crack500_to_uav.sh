#!/usr/bin/env bash
set -euo pipefail

HRDA_ROOT="${HRDA_ROOT:-external/HRDA}"
HRDA_PYTHON="${HRDA_PYTHON:-${DAFORMER_PYTHON:-python}}"
CONFIG_PATH="${CONFIG_PATH:-configs/hrda/crack500_to_uav_hrda_mitb5_s0.py}"
DATA_OUTPUT_ROOT="${DATA_OUTPUT_ROOT:-generated/hrda/crack500_to_uav}"
WORK_DIR="${WORK_DIR:-work_dirs/hrda_crack500_to_uav_mitb5_s0_360}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${WORK_DIR}/latest.pth}"
REPORT_DIR="${REPORT_DIR:-results/report_assets/hrda_mitb5_threshold_sweep_360}"
RESUME_FROM="${RESUME_FROM:-}"

PREPARE_DATA="${PREPARE_DATA:-1}"
FORCE_PREPARE="${FORCE_PREPARE:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
LOG_TEST_RESULT="${LOG_TEST_RESULT:-1}"

SEED="${SEED:-42}"
MAX_ITERS="${MAX_ITERS:-40000}"
SAMPLES_PER_GPU="${SAMPLES_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-4000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-4000}"

if [[ ! -d "${HRDA_ROOT}" ]]; then
  echo "HRDA repository not found at ${HRDA_ROOT}."
  echo "Clone it with: git clone --depth 1 https://github.com/lhoyer/HRDA.git ${HRDA_ROOT}"
  exit 1
fi

if [[ "${PREPARE_DATA}" == "1" ]]; then
  if [[ -d "${DATA_OUTPUT_ROOT}" && "${FORCE_PREPARE}" != "1" ]]; then
    echo "Using existing HRDA dataset at ${DATA_OUTPUT_ROOT}. Set FORCE_PREPARE=1 to rebuild."
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
  if [[ ! -f "${HRDA_ROOT}/pretrained/mit_b5.pth" ]]; then
    echo "Missing MiT-B5 ImageNet weights: ${HRDA_ROOT}/pretrained/mit_b5.pth"
    echo "If external/DAFormer already has converted weights, you can reuse them:"
    echo "  mkdir -p ${HRDA_ROOT}/pretrained && ln -s ../../DAFormer/pretrained/mit_b5.pth ${HRDA_ROOT}/pretrained/mit_b5.pth"
    echo "Otherwise run:"
    echo "  ${HRDA_PYTHON} scripts/daformer/convert_hf_mit_b5_to_daformer.py --output ${HRDA_ROOT}/pretrained/mit_b5.pth"
    exit 2
  fi

  train_args=(
    "${HRDA_ROOT}/tools/train.py"
    "${CONFIG_PATH}"
    --work-dir "${WORK_DIR}"
    --seed "${SEED}"
  )
  if [[ -n "${RESUME_FROM}" ]]; then
    if [[ ! -f "${RESUME_FROM}" ]]; then
      echo "Requested resume checkpoint not found: ${RESUME_FROM}"
      exit 4
    fi
    train_args+=(--resume-from "${RESUME_FROM}")
  fi
  train_args+=(
    --options
      runner.max_iters="${MAX_ITERS}"
      data.samples_per_gpu="${SAMPLES_PER_GPU}"
      data.workers_per_gpu="${WORKERS_PER_GPU}"
      checkpoint_config.interval="${CHECKPOINT_INTERVAL}"
      evaluation.interval="${EVAL_INTERVAL}"
  )

  PYTHONPATH="${HRDA_ROOT}:${PYTHONPATH:-}" "${HRDA_PYTHON}" "${train_args[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "Checkpoint not found for HRDA evaluation: ${CHECKPOINT_PATH}"
    echo "Set CHECKPOINT_PATH to an existing .pth file, or finish training first."
    exit 3
  fi

  eval_args=(
    scripts/daformer/evaluate_daformer_threshold_sweep.py
    --daformer-root "${HRDA_ROOT}" \
    --config "${CONFIG_PATH}" \
    --checkpoint "${CHECKPOINT_PATH}" \
    --output-dir "${REPORT_DIR}" \
    --num-workers "${WORKERS_PER_GPU}" \
    --experiment-name "hrda_mitb5_crack500_to_uav_test_thrval_360" \
    --report-title "HRDA Threshold Selection Report" \
    --model-name "HRDA" \
    --pretrained-model-name "${HRDA_ROOT}/pretrained/mit_b5.pth" \
    --dataset-root "${DATA_OUTPUT_ROOT}"
  )
  if [[ "${LOG_TEST_RESULT}" != "1" ]]; then
    eval_args+=(--no-log-test-result)
  fi

  PYTHONPATH="${HRDA_ROOT}:${PYTHONPATH:-}" "${HRDA_PYTHON}" "${eval_args[@]}"
fi
