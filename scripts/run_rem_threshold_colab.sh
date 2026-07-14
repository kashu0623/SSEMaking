#!/usr/bin/env bash
set -euo pipefail

# Evaluate validation-selected REM threshold policies on saved prediction probabilities.
# This is a post-training policy/calibration diagnostic: no model training is run.
#
# Default target is the single full-w20 model. Use MODEL=fixed_fusion to evaluate
# the current 2-model fixed-fusion teacher probabilities.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42})
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-full_w20}"
NON_REM_ALPHA="${NON_REM_ALPHA:-0.9}"
REM_ALPHA="${REM_ALPHA:-0.2}"
THRESHOLDS="${THRESHOLDS:-0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70}"

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

prediction_has_train_probs() {
  local predictions="$1"
  "${PYTHON_BIN}" -c 'import numpy as np, sys
path = sys.argv[1]
with np.load(path, allow_pickle=True) as data:
    ok = all(key in data.files for key in ("train_probs", "val_probs", "test_probs"))
sys.exit(0 if ok else 1)' "${predictions}"
}

ensure_prediction_probs() {
  local predictions="$1"
  local npz_path="$2"
  local model_dir="$3"
  local require_train="${4:-no}"
  local checkpoint="${model_dir}/lstm_best.pt"

  if [[ -f "${predictions}" ]]; then
    if [[ "${require_train}" == "yes" ]] && prediction_has_train_probs "${predictions}"; then
      return 0
    fi
    if [[ "${require_train}" != "yes" ]] && prediction_has_probs "${predictions}"; then
      return 0
    fi
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

reports=()
for seed in "${SEEDS[@]}"; do
  w20_model_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_npz="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  w20_model_dir="$(model_dir_for_seed "${w20_model_prefix}" "${seed}")"
  w20_predictions="${w20_model_dir}/lstm_predictions.npz"

  if [[ "${MODEL}" == "full_w20" ]]; then
    ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "${w20_model_dir}" "no"
    predictions="${w20_predictions}"
    report_prefix="rem_threshold_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}"
  elif [[ "${MODEL}" == "fixed_fusion" ]]; then
    original_model_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
    original_npz="$(path_for_seed "dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
    original_model_dir="$(model_dir_for_seed "${original_model_prefix}" "${seed}")"
    original_predictions="${original_model_dir}/lstm_predictions.npz"
    ensure_prediction_probs "${original_predictions}" "${original_npz}" "${original_model_dir}" "yes"
    ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "${w20_model_dir}" "yes"

    teacher_prefix="fusion_teacher_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_nonrem090_rem020"
    predictions="$(path_for_seed "${teacher_prefix}" "${seed}" ".npz")"
    teacher_json="$(path_for_seed "${teacher_prefix}" "${seed}" ".json")"
    echo "=== Build fixed fusion probabilities, seed ${seed} ==="
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.build_fusion_teacher_probs \
      --base-predictions "${original_predictions}" \
      --candidate-predictions "${w20_predictions}" \
      --out-npz "${predictions}" \
      --out-json "${teacher_json}" \
      --non-rem-alpha "${NON_REM_ALPHA}" \
      --rem-alpha "${REM_ALPHA}"
    report_prefix="rem_threshold_fixed_fusion_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}"
  else
    echo "Unknown MODEL: ${MODEL}; expected full_w20 or fixed_fusion" >&2
    exit 1
  fi

  out_json="$(path_for_seed "${report_prefix}" "${seed}" ".json")"
  echo "=== Evaluate REM thresholds for ${MODEL}, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_rem_threshold \
    --predictions "${predictions}" \
    --out-json "${out_json}" \
    --thresholds "${THRESHOLDS}" \
    --selection-metric 4_macro_f1_plus_4_kappa
  reports+=("${out_json}")
done

if [[ "${#reports[@]}" -gt 1 ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_prediction_fusion \
    --reports "${reports[@]}" \
    --out-json "${OUTPUT_ROOT}/rem_threshold_${MODEL}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
fi

echo "=== REM threshold evaluation complete ==="
