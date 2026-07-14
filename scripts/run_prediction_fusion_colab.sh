#!/usr/bin/env bash
set -euo pipefail

# Evaluate original temporal + full w20 probability fusion.
# Alpha 0.0 means original temporal only; alpha 1.0 means full w20 only.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42})
PYTHON_BIN="${PYTHON_BIN:-python}"

prediction_path_for_seed() {
  local model_prefix="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${model_prefix}/lstm_predictions.npz"
  else
    echo "${OUTPUT_ROOT}/${model_prefix}_seed${seed}/lstm_predictions.npz"
  fi
}

for seed in "${SEEDS[@]}"; do
  original_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  original_predictions="$(prediction_path_for_seed "${original_prefix}" "${seed}")"
  w20_predictions="$(prediction_path_for_seed "${w20_prefix}" "${seed}")"

  if [[ ! -f "${original_predictions}" ]]; then
    echo "Missing original temporal predictions: ${original_predictions}" >&2
    exit 1
  fi
  if [[ ! -f "${w20_predictions}" ]]; then
    echo "Missing full w20 predictions: ${w20_predictions}" >&2
    exit 1
  fi

  if [[ "${seed}" == "42" ]]; then
    out_json="${OUTPUT_ROOT}/fusion_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}.json"
  else
    out_json="${OUTPUT_ROOT}/fusion_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_seed${seed}.json"
  fi

  echo "=== Evaluate original temporal + full w20 fusion, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_prediction_fusion \
    --base-predictions "${original_predictions}" \
    --candidate-predictions "${w20_predictions}" \
    --out-json "${out_json}" \
    --selection-metric 4_macro_f1_plus_4_kappa
done

echo "=== Prediction fusion evaluation complete ==="
