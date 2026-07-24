#!/usr/bin/env bash
set -euo pipefail

# Train five direct4-original initialization replicas on each existing outer
# split, average them with the existing checkpoint, then recalibrate the hybrid.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
INIT_SEEDS=(${INIT_SEEDS:-1001 2002 3003 4004 5005})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"

original_npz_for_outer_seed() {
  local outer_seed="$1"
  if [[ "${outer_seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}.npz"
  else
    echo "${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${outer_seed}.npz"
  fi
}

replica_dir() {
  local outer_seed="$1"
  local init_seed="$2"
  echo "${OUTPUT_ROOT}/same_split_init_ensemble_direct4_original_outer${outer_seed}_init${init_seed}"
}

current_role_path() {
  local outer_seed="$1"
  local filename="$2"
  echo "${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}/${filename}"
}

original_paths=()
full_paths=()
capacity_paths=()
ls003_paths=()
direct4_ensemble_paths=()

for outer_seed in "${OUTER_SEEDS[@]}"; do
  original_npz="$(original_npz_for_outer_seed "${outer_seed}")"
  existing_direct4="${OUTPUT_ROOT}/direct4_original_outer${outer_seed}/lstm4_predictions.npz"
  [[ -f "${original_npz}" ]] || { echo "Missing NPZ: ${original_npz}" >&2; exit 1; }
  [[ -f "${existing_direct4}" ]] || { echo "Missing direct4 prediction: ${existing_direct4}" >&2; exit 1; }

  direct4_members=("${existing_direct4}")
  for init_seed in "${INIT_SEEDS[@]}"; do
    out_dir="$(replica_dir "${outer_seed}" "${init_seed}")"
    prediction="${out_dir}/lstm4_predictions.npz"
    if [[ "${RUN_TRAIN}" == "1" && ! -f "${prediction}" ]]; then
      echo "=== Train direct4 original, outer ${outer_seed}, init ${init_seed} ==="
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm_4class \
        --npz "${original_npz}" --out-dir "${out_dir}" \
        --hidden-size "${HIDDEN_SIZE}" --dropout 0.4 \
        --class-weight-mode inverse --seed "${init_seed}"
    else
      echo "=== Reuse direct4 original, outer ${outer_seed}, init ${init_seed} ==="
    fi
    [[ -f "${prediction}" ]] || { echo "Missing prediction: ${prediction}" >&2; exit 1; }
    direct4_members+=("${prediction}")
  done

  ensemble_dir="${OUTPUT_ROOT}/same_split_init_ensemble_direct4_original_outer${outer_seed}"
  direct4_ensemble="${ensemble_dir}/original_direct4_predictions.npz"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_prediction_ensemble_4class \
    --predictions "${direct4_members[@]}" \
    --out "${direct4_ensemble}" \
    --summary-out "${ensemble_dir}/original_direct4_summary.json"

  original_path="$(current_role_path "${outer_seed}" original_predictions.npz)"
  full_path="$(current_role_path "${outer_seed}" full_w20_predictions.npz)"
  capacity_path="$(current_role_path "${outer_seed}" capacity_h128_predictions.npz)"
  ls003_path="$(current_role_path "${outer_seed}" h128_ls003_predictions.npz)"
  for path in "${original_path}" "${full_path}" "${capacity_path}" "${ls003_path}"; do
    [[ -f "${path}" ]] || { echo "Missing current ensemble prediction: ${path}" >&2; exit 1; }
  done

  original_paths+=("${original_path}")
  full_paths+=("${full_path}")
  capacity_paths+=("${capacity_path}")
  ls003_paths+=("${ls003_path}")
  direct4_ensemble_paths+=("${direct4_ensemble}")
done

summary_json="${OUTPUT_ROOT}/fusion4_same_split_ensemble_plus_direct4_original_init_ensemble_hybrid_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_direct4_hybrid_deep_fusion \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-original-predictions "${direct4_ensemble_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --wake-alphas 0.15,0.225,0.30,0.3125,0.375,0.45 \
  --light-alphas 0.15,0.25,0.34,0.45,0.55 \
  --deep-alphas 0.50,0.70,0.85,1.00 \
  --rem-alphas 0.00 \
  --deep-gains 0.80,1.00,1.20,1.40,1.60 \
  --out-json "${summary_json}"

echo "=== Direct4 initialization ensemble hybrid complete: ${summary_json} ==="
