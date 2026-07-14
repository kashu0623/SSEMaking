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

npz_path_for_seed() {
  local dataset_prefix="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${dataset_prefix}_lstm_context${CONTEXT_EPOCHS}.npz"
  else
    echo "${OUTPUT_ROOT}/${dataset_prefix}_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
  fi
}

model_dir_for_seed() {
  local model_prefix="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${model_prefix}"
  else
    echo "${OUTPUT_ROOT}/${model_prefix}_seed${seed}"
  fi
}

prediction_has_probs() {
  local predictions="$1"
  "${PYTHON_BIN}" -c 'import numpy as np, sys
path = sys.argv[1]
with np.load(path, allow_pickle=True) as data:
    ok = all(key in data.files for key in ("val_probs", "test_probs"))
sys.exit(0 if ok else 1)' "${predictions}"
}

ensure_prediction_probs() {
  local predictions="$1"
  local npz_path="$2"
  local model_dir="$3"
  local checkpoint="${model_dir}/lstm_best.pt"

  if prediction_has_probs "${predictions}"; then
    return 0
  fi
  if [[ ! -f "${npz_path}" ]]; then
    echo "Missing dataset NPZ needed to export probabilities: ${npz_path}" >&2
    return 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint needed to export probabilities: ${checkpoint}" >&2
    return 1
  fi

  echo "=== Re-export prediction probabilities: ${model_dir} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.export_lstm_predictions \
    --npz "${npz_path}" \
    --checkpoint "${checkpoint}" \
    --out "${predictions}"
}

for seed in "${SEEDS[@]}"; do
  original_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  original_dataset_prefix="dreamt_100hz_temporal"
  w20_dataset_prefix="dreamt_100hz_temporal_w20"
  original_predictions="$(prediction_path_for_seed "${original_prefix}" "${seed}")"
  w20_predictions="$(prediction_path_for_seed "${w20_prefix}" "${seed}")"
  original_npz="$(npz_path_for_seed "${original_dataset_prefix}" "${seed}")"
  w20_npz="$(npz_path_for_seed "${w20_dataset_prefix}" "${seed}")"
  original_model_dir="$(model_dir_for_seed "${original_prefix}" "${seed}")"
  w20_model_dir="$(model_dir_for_seed "${w20_prefix}" "${seed}")"

  if [[ ! -f "${original_predictions}" ]]; then
    echo "Missing original temporal predictions: ${original_predictions}" >&2
    exit 1
  fi
  if [[ ! -f "${w20_predictions}" ]]; then
    echo "Missing full w20 predictions: ${w20_predictions}" >&2
    exit 1
  fi

  ensure_prediction_probs "${original_predictions}" "${original_npz}" "${original_model_dir}"
  ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "${w20_model_dir}"

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
