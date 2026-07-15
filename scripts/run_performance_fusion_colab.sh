#!/usr/bin/env bash
set -euo pipefail

# Performance-first fusion search.
#
# 1) Dense original-temporal + full-w20 class-wise grid.
# 2) Optional 3-model fusion with a third model, defaulting to remaux_w05.
#
# This is meant for performance exploration when 2-model/3-model runtime cost is acceptable.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_THREE_MODEL="${RUN_THREE_MODEL:-1}"
THIRD_VARIANT="${THIRD_VARIANT:-remaux_w05}"
THIRD_MODEL_PREFIX="${THIRD_MODEL_PREFIX:-}"

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

two_model_reports=()
three_model_reports=()
for seed in "${SEEDS[@]}"; do
  original_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  if [[ -n "${THIRD_MODEL_PREFIX}" ]]; then
    third_prefix="${THIRD_MODEL_PREFIX}"
  else
    third_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_${THIRD_VARIANT}"
  fi
  original_predictions="$(prediction_path_for_seed "${original_prefix}" "${seed}")"
  w20_predictions="$(prediction_path_for_seed "${w20_prefix}" "${seed}")"
  third_predictions="$(prediction_path_for_seed "${third_prefix}" "${seed}")"
  original_npz="$(npz_path_for_seed "dreamt_100hz_temporal" "${seed}")"
  w20_npz="$(npz_path_for_seed "dreamt_100hz_temporal_w20" "${seed}")"
  original_model_dir="$(model_dir_for_seed "${original_prefix}" "${seed}")"
  w20_model_dir="$(model_dir_for_seed "${w20_prefix}" "${seed}")"
  third_model_dir="$(model_dir_for_seed "${third_prefix}" "${seed}")"

  ensure_prediction_probs "${original_predictions}" "${original_npz}" "${original_model_dir}"
  ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "${w20_model_dir}"

  if [[ "${seed}" == "42" ]]; then
    two_out_json="${OUTPUT_ROOT}/fusion_dense_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}.json"
    three_out_json="${OUTPUT_ROOT}/fusion3_original_full_w20_${THIRD_VARIANT}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}.json"
  else
    two_out_json="${OUTPUT_ROOT}/fusion_dense_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_seed${seed}.json"
    three_out_json="${OUTPUT_ROOT}/fusion3_original_full_w20_${THIRD_VARIANT}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_seed${seed}.json"
  fi

  echo "=== Dense 2-model fusion, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_prediction_fusion \
    --base-predictions "${original_predictions}" \
    --candidate-predictions "${w20_predictions}" \
    --out-json "${two_out_json}" \
    --classwise-non-rem-alphas "0.75,0.80,0.85,0.90,0.92,0.95,0.98,1.00" \
    --classwise-rem-alphas "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50" \
    --selection-metric 4_macro_f1_plus_4_kappa \
    --top 20
  two_model_reports+=("${two_out_json}")

  if [[ "${RUN_THREE_MODEL}" == "1" ]]; then
    if [[ ! -f "${third_model_dir}/lstm_best.pt" && ! -f "${third_predictions}" ]]; then
      echo "Skipping 3-model fusion; missing third model: ${third_model_dir}" >&2
      continue
    fi
    ensure_prediction_probs "${third_predictions}" "${w20_npz}" "${third_model_dir}"
    echo "=== 3-model fusion with ${THIRD_VARIANT}, seed ${seed} ==="
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_three_model_fusion \
      --base-predictions "${original_predictions}" \
      --primary-predictions "${w20_predictions}" \
      --secondary-predictions "${third_predictions}" \
      --out-json "${three_out_json}" \
      --selection-metric 4_macro_f1_plus_4_kappa \
      --top 20
    three_model_reports+=("${three_out_json}")
  fi
done

if [[ "${#two_model_reports[@]}" -gt 1 ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_prediction_fusion \
    --reports "${two_model_reports[@]}" \
    --out-json "${OUTPUT_ROOT}/fusion_dense_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
fi

if [[ "${#three_model_reports[@]}" -gt 1 ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_prediction_fusion \
    --reports "${three_model_reports[@]}" \
    --out-json "${OUTPUT_ROOT}/fusion3_original_full_w20_${THIRD_VARIANT}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
fi

echo "=== Performance fusion search complete ==="
