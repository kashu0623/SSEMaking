#!/usr/bin/env bash
set -euo pipefail

# Compact oracle/disagreement audit for the current four-model fusion best.
# This does not train models or search weights. It measures whether a dynamic
# gate can recover current fusion errors from the existing model pool.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"

CAPACITY_PREFIX="${CAPACITY_PREFIX:-lstm_temporal_w20_context20_inverse_capacity_h128}"
LS003_PREFIX="${LS003_PREFIX:-lstm_temporal_w20_context20_inverse_h128_ls003}"

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

  if [[ -f "${predictions}" ]] && prediction_has_probs "${predictions}"; then
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

  echo "=== Export prediction probabilities: ${model_dir} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.export_lstm_predictions \
    --npz "${npz_path}" \
    --checkpoint "${checkpoint}" \
    --out "${predictions}"
}

base_predictions=()
primary_predictions=()
secondary_predictions=()
tertiary_predictions=()
seed_labels=()

for seed in "${SEEDS[@]}"; do
  original_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  original_predictions="$(prediction_path_for_seed "${original_prefix}" "${seed}")"
  w20_predictions="$(prediction_path_for_seed "${w20_prefix}" "${seed}")"
  capacity_predictions="$(prediction_path_for_seed "${CAPACITY_PREFIX}" "${seed}")"
  ls003_predictions="$(prediction_path_for_seed "${LS003_PREFIX}" "${seed}")"
  original_npz="$(npz_path_for_seed "dreamt_100hz_temporal" "${seed}")"
  w20_npz="$(npz_path_for_seed "dreamt_100hz_temporal_w20" "${seed}")"

  ensure_prediction_probs "${original_predictions}" "${original_npz}" "$(model_dir_for_seed "${original_prefix}" "${seed}")"
  ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "$(model_dir_for_seed "${w20_prefix}" "${seed}")"
  ensure_prediction_probs "${capacity_predictions}" "${w20_npz}" "$(model_dir_for_seed "${CAPACITY_PREFIX}" "${seed}")"
  ensure_prediction_probs "${ls003_predictions}" "${w20_npz}" "$(model_dir_for_seed "${LS003_PREFIX}" "${seed}")"

  base_predictions+=("${original_predictions}")
  primary_predictions+=("${w20_predictions}")
  secondary_predictions+=("${capacity_predictions}")
  tertiary_predictions+=("${ls003_predictions}")
  seed_labels+=("${seed}")
done

summary_json="${OUTPUT_ROOT}/fusion4_current_best_oracle_audit_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"

PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_four_model_oracle_audit \
  --base-predictions "${base_predictions[@]}" \
  --primary-predictions "${primary_predictions[@]}" \
  --secondary-predictions "${secondary_predictions[@]}" \
  --tertiary-predictions "${tertiary_predictions[@]}" \
  --seed-labels "${seed_labels[@]}" \
  --splits val test \
  --out-json "${summary_json}"

echo "=== Four-model oracle audit complete: ${summary_json} ==="
