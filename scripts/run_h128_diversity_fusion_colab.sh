#!/usr/bin/env bash
set -euo pipefail

# H128 diversity sweep for performance-first 3-model fusion.
#
# Order:
#   A. h128 + checkpoint selection by 4_macro_f1_plus_4_kappa
#   B. h128 + REM class-weight multiplier 1.2 / 1.5
#   C. h128 + N3 class-weight multiplier 1.2
#   D. h128 + label smoothing 0.03 / 0.05
#
# After training, each candidate is evaluated as the third model in:
#   original temporal + full w20 + candidate

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_FUSION="${RUN_FUSION:-1}"

path_for_seed() {
  local prefix="$1"
  local seed="$2"
  local suffix="$3"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${prefix}${suffix}"
  else
    echo "${OUTPUT_ROOT}/${prefix}_seed${seed}${suffix}"
  fi
}

out_dir_for_variant() {
  local variant="$1"
  local seed="$2"
  local base="${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_h128_${variant}"
  if [[ "${seed}" == "42" ]]; then
    echo "${base}"
  else
    echo "${base}_seed${seed}"
  fi
}

train_variant() {
  local variant="$1"
  local seed="$2"
  shift 2
  local npz_path
  local out_dir
  npz_path="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  out_dir="$(out_dir_for_variant "${variant}" "${seed}")"

  if [[ ! -f "${npz_path}" ]]; then
    echo "Missing full-w20 dataset NPZ: ${npz_path}" >&2
    exit 1
  fi

  echo "=== Train h128 diversity ${variant}, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
    --npz "${npz_path}" \
    --out-dir "${out_dir}" \
    --hidden-size 128 \
    --num-layers 1 \
    --dropout 0.4 \
    --class-weight-mode inverse \
    --seed "${seed}" \
    "$@"
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    train_variant "sel4combo" "${seed}" \
      --selection-metric 4_macro_f1_plus_4_kappa

    train_variant "rem12" "${seed}" \
      --selection-metric 5_macro_f1 \
      --rem-weight-multiplier 1.2

    train_variant "rem15" "${seed}" \
      --selection-metric 5_macro_f1 \
      --rem-weight-multiplier 1.5

    train_variant "n3_12" "${seed}" \
      --selection-metric 5_macro_f1 \
      --n3-weight-multiplier 1.2

    train_variant "ls003" "${seed}" \
      --selection-metric 5_macro_f1 \
      --label-smoothing 0.03

    train_variant "ls005" "${seed}" \
      --selection-metric 5_macro_f1 \
      --label-smoothing 0.05
  done

  summary_args=()
  for seed in "${SEEDS[@]}"; do
    baseline_metrics="$(path_for_seed "lstm_temporal_w20_context${CONTEXT_EPOCHS}_h64_inverse" "${seed}" "/lstm_metrics.json")"
    if [[ -f "${baseline_metrics}" ]]; then
      summary_args+=(--metrics "full_w20=${baseline_metrics}")
    fi
    for variant in sel4combo rem12 rem15 n3_12 ls003 ls005; do
      metrics_path="$(out_dir_for_variant "${variant}" "${seed}")/lstm_metrics.json"
      if [[ -f "${metrics_path}" ]]; then
        summary_args+=(--metrics "${variant}=${metrics_path}")
      fi
    done
  done

  if [[ "${#summary_args[@]}" -gt 0 ]]; then
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_lstm_metrics \
      "${summary_args[@]}" \
      --baseline-label full_w20 \
      --out-json "${OUTPUT_ROOT}/h128_diversity_context${CONTEXT_EPOCHS}_summary.json"
  fi
fi

if [[ "${RUN_FUSION}" == "1" ]]; then
  prefix_items=()
  for variant in sel4combo rem12 rem15 n3_12 ls003 ls005; do
    prefix_items+=("h128_${variant}=lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_h128_${variant}")
  done

  echo "=== Evaluate h128 diversity candidates as third-model fusion inputs ==="
  SEEDS="${SEEDS[*]}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
  HIDDEN_SIZE="${HIDDEN_SIZE}" \
  RUN_THIRD_VARIANTS=0 \
  THIRD_PREFIX_CANDIDATES="${prefix_items[*]}" \
  bash scripts/run_aggressive_fusion_colab.sh
fi

echo "=== H128 diversity fusion sweep complete ==="
