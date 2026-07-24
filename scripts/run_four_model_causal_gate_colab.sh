#!/usr/bin/env bash
set -euo pipefail

# Validation-trained causal temporal gate over the current four-model fusion.
# The gate uses only present/past model predictions. Its regularization is
# selected on held-out validation subjects; test labels are never used to fit it.

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

prediction_has_metadata() {
  local predictions="$1"
  "${PYTHON_BIN}" -c 'import numpy as np, sys
with np.load(sys.argv[1], allow_pickle=True) as data:
    keys = ("val_probs", "test_probs", "val_subject_ids", "test_subject_ids", "val_epoch_indices", "test_epoch_indices")
    ok = all(key in data.files for key in keys)
sys.exit(0 if ok else 1)' "${predictions}"
}

ensure_prediction_metadata() {
  local predictions="$1"
  local npz_path="$2"
  local model_dir="$3"
  local checkpoint="${model_dir}/lstm_best.pt"
  if [[ -f "${predictions}" ]] && prediction_has_metadata "${predictions}"; then
    return 0
  fi
  [[ -f "${npz_path}" ]] || { echo "Missing dataset NPZ: ${npz_path}" >&2; return 1; }
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 1; }
  echo "=== Re-export predictions with causal metadata: ${model_dir} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.export_lstm_predictions --npz "${npz_path}" --checkpoint "${checkpoint}" --out "${predictions}"
}

base_predictions=()
primary_predictions=()
secondary_predictions=()
tertiary_predictions=()
seed_labels=()
for seed in "${SEEDS[@]}"; do
  original_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  original="$(prediction_path_for_seed "${original_prefix}" "${seed}")"
  primary="$(prediction_path_for_seed "${w20_prefix}" "${seed}")"
  secondary="$(prediction_path_for_seed "${CAPACITY_PREFIX}" "${seed}")"
  tertiary="$(prediction_path_for_seed "${LS003_PREFIX}" "${seed}")"
  original_npz="$(npz_path_for_seed dreamt_100hz_temporal "${seed}")"
  w20_npz="$(npz_path_for_seed dreamt_100hz_temporal_w20 "${seed}")"
  ensure_prediction_metadata "${original}" "${original_npz}" "$(model_dir_for_seed "${original_prefix}" "${seed}")"
  ensure_prediction_metadata "${primary}" "${w20_npz}" "$(model_dir_for_seed "${w20_prefix}" "${seed}")"
  ensure_prediction_metadata "${secondary}" "${w20_npz}" "$(model_dir_for_seed "${CAPACITY_PREFIX}" "${seed}")"
  ensure_prediction_metadata "${tertiary}" "${w20_npz}" "$(model_dir_for_seed "${LS003_PREFIX}" "${seed}")"
  base_predictions+=("${original}")
  primary_predictions+=("${primary}")
  secondary_predictions+=("${secondary}")
  tertiary_predictions+=("${tertiary}")
  seed_labels+=("${seed}")
done

out_json="${OUTPUT_ROOT}/fusion4_current_best_causal_gate_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_four_model_causal_gate \
  --base-predictions "${base_predictions[@]}" \
  --primary-predictions "${primary_predictions[@]}" \
  --secondary-predictions "${secondary_predictions[@]}" \
  --tertiary-predictions "${tertiary_predictions[@]}" \
  --seed-labels "${seed_labels[@]}" \
  --out-json "${out_json}"

echo "=== Four-model causal gate complete: ${out_json} ==="
