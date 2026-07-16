#!/usr/bin/env bash
set -euo pipefail

# Four-model performance-first fusion:
#   original temporal + full_w20 + capacity_h128 + h128_ls003
#
# Wake, Light(N1/N2)/Deep(N3), and REM can use separate class-wise weights.
# If DEEP_* overrides are omitted, Deep keeps the same weights as Light.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4}"
FUSION_SELECTION_POLICY="${FUSION_SELECTION_POLICY:-standard}"
FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA:-0.0}"
FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE:-0.005}"
FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE:-0.005}"
FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE:-0.010}"
FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE:-0.020}"

CAPACITY_PREFIX="${CAPACITY_PREFIX:-lstm_temporal_w20_context20_inverse_capacity_h128}"
LS003_PREFIX="${LS003_PREFIX:-lstm_temporal_w20_context20_inverse_h128_ls003}"

WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.78,0.79}"
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.10,0.12,0.14}"
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0,0.03}"
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.72,0.74,0.76}"
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0,0.03}"
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.12,0.15,0.18}"
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-}"
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-}"
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-}"
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}"
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.25,0.30,0.32}"
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0,0.03}"

report_suffix=""
if [[ "${FUSION_SELECTION_POLICY}" != "standard" ]]; then
  report_suffix="_${FUSION_SELECTION_POLICY}"
fi
report_suffix="${report_suffix}${FUSION_REPORT_SUFFIX}"

deep_args=()
if [[ -n "${DEEP_PRIMARY_ALPHAS}" ]]; then
  deep_args+=(--deep-primary-alphas "${DEEP_PRIMARY_ALPHAS}")
fi
if [[ -n "${DEEP_SECONDARY_ALPHAS}" ]]; then
  deep_args+=(--deep-secondary-alphas "${DEEP_SECONDARY_ALPHAS}")
fi
if [[ -n "${DEEP_TERTIARY_ALPHAS}" ]]; then
  deep_args+=(--deep-tertiary-alphas "${DEEP_TERTIARY_ALPHAS}")
fi

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

fusion_reports=()
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

  if [[ "${seed}" == "42" ]]; then
    out_json="${OUTPUT_ROOT}/fusion4_original_full_w20_capacity_h128_ls003_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}${report_suffix}.json"
  else
    out_json="${OUTPUT_ROOT}/fusion4_original_full_w20_capacity_h128_ls003_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}${report_suffix}_seed${seed}.json"
  fi

  echo "=== 4-model flexible fusion, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_four_model_fusion \
    --base-predictions "${original_predictions}" \
    --primary-predictions "${w20_predictions}" \
    --secondary-predictions "${capacity_predictions}" \
    --tertiary-predictions "${ls003_predictions}" \
    --out-json "${out_json}" \
    --wake-primary-alphas "${WAKE_PRIMARY_ALPHAS}" \
    --wake-secondary-alphas "${WAKE_SECONDARY_ALPHAS}" \
    --wake-tertiary-alphas "${WAKE_TERTIARY_ALPHAS}" \
    --light-deep-primary-alphas "${LIGHT_DEEP_PRIMARY_ALPHAS}" \
    --light-deep-secondary-alphas "${LIGHT_DEEP_SECONDARY_ALPHAS}" \
    --light-deep-tertiary-alphas "${LIGHT_DEEP_TERTIARY_ALPHAS}" \
    "${deep_args[@]}" \
    --rem-primary-alphas "${REM_PRIMARY_ALPHAS}" \
    --rem-secondary-alphas "${REM_SECONDARY_ALPHAS}" \
    --rem-tertiary-alphas "${REM_TERTIARY_ALPHAS}" \
    --selection-metric 4_macro_f1_plus_4_kappa \
    --selection-policy "${FUSION_SELECTION_POLICY}" \
    --fixed-min-score-delta "${FIXED_MIN_SCORE_DELTA}" \
    --fixed-rem-tolerance "${FIXED_REM_TOLERANCE}" \
    --fixed-light-tolerance "${FIXED_LIGHT_TOLERANCE}" \
    --fixed-wake-tolerance "${FIXED_WAKE_TOLERANCE}" \
    --fixed-deep-tolerance "${FIXED_DEEP_TOLERANCE}" \
    --top 20
  fusion_reports+=("${out_json}")
done

if [[ "${#fusion_reports[@]}" -gt 1 ]]; then
  summary_json="${OUTPUT_ROOT}/fusion4_original_full_w20_capacity_h128_ls003_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}${report_suffix}_summary.json"
  rank_json="${OUTPUT_ROOT}/fusion4_original_full_w20_capacity_h128_ls003_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}${report_suffix}_rank.json"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_prediction_fusion \
    --reports "${fusion_reports[@]}" \
    --out-json "${summary_json}"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.rank_fusion_summaries \
    --summaries "${summary_json}" \
    --out-json "${rank_json}"
fi

echo "=== Four-model flexible fusion complete ==="
