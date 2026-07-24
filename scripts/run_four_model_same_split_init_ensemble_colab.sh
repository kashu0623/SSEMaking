#!/usr/bin/env bash
set -euo pipefail

# Performance-only same-split ensemble. For each existing outer subject split,
# train several new initialization replicas of all four current fusion roles,
# average each role's probabilities, then apply the current fixed best weights.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
INIT_SEEDS=(${INIT_SEEDS:-1001 2002 3003 4004 5005})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"

prediction_path_for_outer_seed() {
  local prefix="$1"
  local outer_seed="$2"
  if [[ "${outer_seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${prefix}/lstm_predictions.npz"
  else
    echo "${OUTPUT_ROOT}/${prefix}_seed${outer_seed}/lstm_predictions.npz"
  fi
}

npz_path_for_outer_seed() {
  local dataset_prefix="$1"
  local outer_seed="$2"
  if [[ "${outer_seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${dataset_prefix}_lstm_context${CONTEXT_EPOCHS}.npz"
  else
    echo "${OUTPUT_ROOT}/${dataset_prefix}_lstm_context${CONTEXT_EPOCHS}_seed${outer_seed}.npz"
  fi
}

replica_dir() {
  local role="$1"
  local outer_seed="$2"
  local init_seed="$3"
  echo "${OUTPUT_ROOT}/same_split_init_ensemble_${role}_outer${outer_seed}_init${init_seed}"
}

train_replica() {
  local role="$1"
  local npz_path="$2"
  local outer_seed="$3"
  local init_seed="$4"
  local out_dir
  out_dir="$(replica_dir "${role}" "${outer_seed}" "${init_seed}")"
  if [[ "${RUN_TRAIN}" != "1" || -f "${out_dir}/lstm_predictions.npz" ]]; then
    echo "=== Reuse ${role}, outer ${outer_seed}, init ${init_seed} ==="
    return 0
  fi
  echo "=== Train ${role}, outer ${outer_seed}, init ${init_seed} ==="
  case "${role}" in
    original)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" --out-dir "${out_dir}" --hidden-size 64 --dropout 0.4 \
        --class-weight-mode inverse --selection-metric 5_macro_f1 --seed "${init_seed}"
      ;;
    full_w20)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" --out-dir "${out_dir}" --hidden-size 64 --dropout 0.4 \
        --class-weight-mode inverse --selection-metric 5_macro_f1 --seed "${init_seed}"
      ;;
    capacity_h128)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" --out-dir "${out_dir}" --hidden-size 128 --dropout 0.4 \
        --class-weight-mode inverse --selection-metric 5_macro_f1 --seed "${init_seed}"
      ;;
    h128_ls003)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" --out-dir "${out_dir}" --hidden-size 128 --dropout 0.4 \
        --class-weight-mode inverse --label-smoothing 0.03 --selection-metric 5_macro_f1 --seed "${init_seed}"
      ;;
    *)
      echo "Unknown role: ${role}" >&2
      return 1
      ;;
  esac
}

fusion_reports=()
for outer_seed in "${OUTER_SEEDS[@]}"; do
  original_npz="$(npz_path_for_outer_seed dreamt_100hz_temporal "${outer_seed}")"
  w20_npz="$(npz_path_for_outer_seed dreamt_100hz_temporal_w20 "${outer_seed}")"
  [[ -f "${original_npz}" ]] || { echo "Missing NPZ: ${original_npz}" >&2; exit 1; }
  [[ -f "${w20_npz}" ]] || { echo "Missing NPZ: ${w20_npz}" >&2; exit 1; }

  for init_seed in "${INIT_SEEDS[@]}"; do
    train_replica original "${original_npz}" "${outer_seed}" "${init_seed}"
    train_replica full_w20 "${w20_npz}" "${outer_seed}" "${init_seed}"
    train_replica capacity_h128 "${w20_npz}" "${outer_seed}" "${init_seed}"
    train_replica h128_ls003 "${w20_npz}" "${outer_seed}" "${init_seed}"
  done

  original_members=("$(prediction_path_for_outer_seed "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse" "${outer_seed}")")
  full_members=("$(prediction_path_for_outer_seed "lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse" "${outer_seed}")")
  capacity_members=("$(prediction_path_for_outer_seed "lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_capacity_h128" "${outer_seed}")")
  ls003_members=("$(prediction_path_for_outer_seed "lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_h128_ls003" "${outer_seed}")")
  for init_seed in "${INIT_SEEDS[@]}"; do
    original_members+=("$(replica_dir original "${outer_seed}" "${init_seed}")/lstm_predictions.npz")
    full_members+=("$(replica_dir full_w20 "${outer_seed}" "${init_seed}")/lstm_predictions.npz")
    capacity_members+=("$(replica_dir capacity_h128 "${outer_seed}" "${init_seed}")/lstm_predictions.npz")
    ls003_members+=("$(replica_dir h128_ls003 "${outer_seed}" "${init_seed}")/lstm_predictions.npz")
  done
  for member in "${original_members[@]}" "${full_members[@]}" "${capacity_members[@]}" "${ls003_members[@]}"; do
    [[ -f "${member}" ]] || { echo "Missing prediction: ${member}" >&2; exit 1; }
  done

  ensemble_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  original_ensemble="${ensemble_dir}/original_predictions.npz"
  full_ensemble="${ensemble_dir}/full_w20_predictions.npz"
  capacity_ensemble="${ensemble_dir}/capacity_h128_predictions.npz"
  ls003_ensemble="${ensemble_dir}/h128_ls003_predictions.npz"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_prediction_ensemble --predictions "${original_members[@]}" --out "${original_ensemble}" --summary-out "${ensemble_dir}/original_summary.json"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_prediction_ensemble --predictions "${full_members[@]}" --out "${full_ensemble}" --summary-out "${ensemble_dir}/full_w20_summary.json"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_prediction_ensemble --predictions "${capacity_members[@]}" --out "${capacity_ensemble}" --summary-out "${ensemble_dir}/capacity_h128_summary.json"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_prediction_ensemble --predictions "${ls003_members[@]}" --out "${ls003_ensemble}" --summary-out "${ensemble_dir}/h128_ls003_summary.json"

  report="${OUTPUT_ROOT}/fusion4_same_split_init_ensemble_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_outer${outer_seed}.json"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_four_model_fusion \
    --base-predictions "${original_ensemble}" --primary-predictions "${full_ensemble}" \
    --secondary-predictions "${capacity_ensemble}" --tertiary-predictions "${ls003_ensemble}" \
    --out-json "${report}" \
    --wake-primary-alphas 0.72 --wake-secondary-alphas 0.06 --wake-tertiary-alphas 0.00 \
    --light-deep-primary-alphas 0.80 --light-deep-secondary-alphas 0.02 --light-deep-tertiary-alphas 0.15 \
    --deep-primary-alphas 0.82 --deep-secondary-alphas 0.00 --deep-tertiary-alphas 0.18 \
    --rem-primary-alphas 0.00 --rem-secondary-alphas 0.42 --rem-tertiary-alphas 0.13 \
    --selection-metric 4_macro_f1_plus_4_kappa --top 12
  fusion_reports+=("${report}")
done

summary_json="${OUTPUT_ROOT}/fusion4_same_split_init_ensemble_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
rank_json="${OUTPUT_ROOT}/fusion4_same_split_init_ensemble_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_rank.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_prediction_fusion --reports "${fusion_reports[@]}" --out-json "${summary_json}"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.rank_fusion_summaries --summaries "${summary_json}" --out-json "${rank_json}"
echo "=== Same-split initialization ensemble complete: ${summary_json} ==="
