#!/usr/bin/env bash
set -euo pipefail

# Expand the current app-quality fusion candidates to 3 seeds.
#
# Candidates:
#   1. h128_sel4combo: h128 trained with checkpoint selection by 4_macro_f1_plus_4_kappa
#   2. capacity_h128: h128 trained with the original 5_macro_f1 checkpoint selection
#
# Fusion policy:
#   original temporal + full w20 + candidate
#   REM primary/full_w20 weight is fixed to 0.00, and REM secondary/candidate
#   weight is selected from 0.00..0.20 by validation. This preserves the
#   original-temporal-heavy REM behavior that was strongest in seed42.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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

train_h128_candidate() {
  local label="$1"
  local seed="$2"
  shift 2
  local npz_path
  local out_dir
  npz_path="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  out_dir="$(path_for_seed "lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_${label}" "${seed}" "")"

  if [[ ! -f "${npz_path}" ]]; then
    echo "Missing full-w20 dataset NPZ: ${npz_path}" >&2
    exit 1
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/lstm_best.pt" ]]; then
    echo "=== Skip existing ${label}, seed ${seed}: ${out_dir} ==="
    return 0
  fi

  echo "=== Train ${label}, seed ${seed} ==="
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
    train_h128_candidate "h128_sel4combo" "${seed}" \
      --selection-metric 4_macro_f1_plus_4_kappa

    train_h128_candidate "capacity_h128" "${seed}" \
      --selection-metric 5_macro_f1
  done
fi

if [[ "${RUN_FUSION}" == "1" ]]; then
  echo "=== Evaluate app-candidate constrained 3-model fusion ==="
  SEEDS="${SEEDS[*]}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
  HIDDEN_SIZE="${HIDDEN_SIZE}" \
  RUN_THIRD_VARIANTS=0 \
  THIRD_PREFIX_CANDIDATES="h128_sel4combo=lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_h128_sel4combo capacity_h128=lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_capacity_h128" \
  FUSION_REPORT_SUFFIX="_app_candidate" \
  THREE_NON_REM_PRIMARY_ALPHAS="0.80,0.85,0.90,0.95,1.00" \
  THREE_NON_REM_SECONDARY_ALPHAS="0,0.05,0.10,0.15,0.20" \
  THREE_REM_PRIMARY_ALPHAS="0" \
  THREE_REM_SECONDARY_ALPHAS="0,0.05,0.10,0.15,0.20" \
  bash scripts/run_aggressive_fusion_colab.sh
fi

echo "=== App-candidate 3-seed expansion complete ==="
