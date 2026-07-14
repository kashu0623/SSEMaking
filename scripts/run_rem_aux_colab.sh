#!/usr/bin/env bash
set -euo pipefail

# REM auxiliary-head and Deep+REM multi-task experiments for the current
# single-model full-w20 baseline.
#
# Defaults run seed42 for quick filtering. Expand promising candidates with:
#   SEEDS="42 7 123" VARIANTS="remaux_w02" bash scripts/run_rem_aux_colab.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
VARIANTS=(${VARIANTS:-remaux_w02 remaux_w05 deeprem_w02 deeprem_w05})
PYTHON_BIN="${PYTHON_BIN:-python}"
SELECTION_METRIC="${SELECTION_METRIC:-5_macro_f1}"

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

weight_name() {
  echo "$1" | tr -d "."
}

variant_args() {
  local variant="$1"
  case "${variant}" in
    remaux_w01)
      echo "--aux-head rem --aux-weight 0.1"
      ;;
    remaux_w02)
      echo "--aux-head rem --aux-weight 0.2"
      ;;
    remaux_w05)
      echo "--aux-head rem --aux-weight 0.5"
      ;;
    deeprem_w01)
      echo "--aux-head deep_rem --aux-weight 0.1"
      ;;
    deeprem_w02)
      echo "--aux-head deep_rem --aux-weight 0.2"
      ;;
    deeprem_w05)
      echo "--aux-head deep_rem --aux-weight 0.5"
      ;;
    *)
      echo "Unknown REM auxiliary variant: ${variant}" >&2
      return 1
      ;;
  esac
}

out_dir_for_variant() {
  local variant="$1"
  local seed="$2"
  local base="${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_${variant}"
  if [[ "${seed}" == "42" ]]; then
    echo "${base}"
  else
    echo "${base}_seed${seed}"
  fi
}

for seed in "${SEEDS[@]}"; do
  npz_path="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  if [[ ! -f "${npz_path}" ]]; then
    echo "Missing full-w20 dataset NPZ: ${npz_path}" >&2
    exit 1
  fi

  for variant in "${VARIANTS[@]}"; do
    out_dir="$(out_dir_for_variant "${variant}" "${seed}")"
    args=($(variant_args "${variant}"))
    echo "=== Train ${variant}, seed ${seed} ==="
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
      --npz "${npz_path}" \
      --out-dir "${out_dir}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --dropout "${DROPOUT}" \
      --class-weight-mode inverse \
      --aux-deep-pos-weight-mode balanced \
      --selection-metric "${SELECTION_METRIC}" \
      "${args[@]}" \
      --seed "${seed}"
  done
done

echo "=== REM auxiliary experiments complete ==="

summary_args=()
if [[ -f "${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json" ]]; then
  summary_args+=(--metrics "full_w20=${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json")
  summary_args+=(--baseline-label full_w20)
fi
for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" != "42" ]]; then
    baseline_metrics="${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}/lstm_metrics.json"
    if [[ -f "${baseline_metrics}" ]]; then
      summary_args+=(--metrics "full_w20=${baseline_metrics}")
    fi
  fi
  for variant in "${VARIANTS[@]}"; do
    metrics_path="$(out_dir_for_variant "${variant}" "${seed}")/lstm_metrics.json"
    if [[ -f "${metrics_path}" ]]; then
      summary_args+=(--metrics "${variant}=${metrics_path}")
    fi
  done
done

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_lstm_metrics "${summary_args[@]}"
fi
