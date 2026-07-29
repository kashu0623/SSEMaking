#!/usr/bin/env bash
set -euo pipefail

# Train five original-h128-CE specialist replicas per outer split, compare
# every initialization, and evaluate their six-member probability ensemble.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
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
  echo "${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}_init${init_seed}"
}

ensemble_path() {
  local outer_seed="$1"
  echo "${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_ensemble6_outer${outer_seed}/light_deep_predictions.npz"
}

for outer_seed in "${OUTER_SEEDS[@]}"; do
  npz_path="$(original_npz_for_outer_seed "${outer_seed}")"
  existing="${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}/light_deep_predictions.npz"
  [[ -f "${npz_path}" ]] || { echo "Missing dataset: ${npz_path}" >&2; exit 1; }
  [[ -f "${existing}" ]] || { echo "Missing existing specialist: ${existing}" >&2; exit 1; }

  members=("${existing}")
  for init_seed in "${INIT_SEEDS[@]}"; do
    out_dir="$(replica_dir "${outer_seed}" "${init_seed}")"
    prediction="${out_dir}/light_deep_predictions.npz"
    if [[ "${RUN_TRAIN}" == "1" && ! -f "${prediction}" ]]; then
      echo "=== Train specialist outer ${outer_seed}, init ${init_seed} ==="
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_light_deep_specialist \
        --npz "${npz_path}" \
        --out-dir "${out_dir}" \
        --hidden-size 128 \
        --dropout 0.4 \
        --epochs 40 \
        --patience 8 \
        --class-weight-mode inverse \
        --loss-type cross_entropy \
        --seed "${init_seed}"
    fi
    [[ -f "${prediction}" ]] || { echo "Missing replica: ${prediction}" >&2; exit 1; }
    members+=("${prediction}")
  done

  ensemble="$(ensemble_path "${outer_seed}")"
  ensemble_dir="$(dirname "${ensemble}")"
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_light_deep_specialist_ensemble \
    --predictions "${members[@]}" \
    --out "${ensemble}" \
    --summary-out "${ensemble_dir}/ensemble_summary.json"
done

original_paths=()
full_paths=()
capacity_paths=()
ls003_paths=()
direct4_single_paths=()
direct4_ensemble_paths=()
for outer_seed in "${OUTER_SEEDS[@]}"; do
  current_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  original_paths+=("${current_dir}/original_predictions.npz")
  full_paths+=("${current_dir}/full_w20_predictions.npz")
  capacity_paths+=("${current_dir}/capacity_h128_predictions.npz")
  ls003_paths+=("${current_dir}/h128_ls003_predictions.npz")
  direct4_single_paths+=("${OUTPUT_ROOT}/direct4_original_outer${outer_seed}/lstm4_predictions.npz")
  direct4_ensemble_paths+=("${OUTPUT_ROOT}/same_split_init_ensemble_direct4_original_outer${outer_seed}/original_direct4_predictions.npz")
done

specialist_labels=(single)
for init_seed in "${INIT_SEEDS[@]}"; do
  specialist_labels+=("init${init_seed}")
done
specialist_labels+=(ensemble6)

# Config-major path order: all outer splits for one label, then the next label.
specialist_paths=()
for label in "${specialist_labels[@]}"; do
  for outer_seed in "${OUTER_SEEDS[@]}"; do
    if [[ "${label}" == "single" ]]; then
      prediction="${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}/light_deep_predictions.npz"
    elif [[ "${label}" == "ensemble6" ]]; then
      prediction="$(ensemble_path "${outer_seed}")"
    else
      init_seed="${label#init}"
      prediction="$(replica_dir "${outer_seed}" "${init_seed}")/light_deep_predictions.npz"
    fi
    [[ -f "${prediction}" ]] || { echo "Missing specialist source: ${prediction}" >&2; exit 1; }
    specialist_paths+=("${prediction}")
  done
done

summary_json="${OUTPUT_ROOT}/fusion4_light_deep_specialist_same_split_init_ensemble_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_light_deep_specialist_fusion \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-single-predictions "${direct4_single_paths[@]}" \
  --direct4-ensemble-predictions "${direct4_ensemble_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --specialist-labels "${specialist_labels[@]}" \
  --specialist-predictions "${specialist_paths[@]}" \
  --betas 0.50,0.65,0.80,0.90,0.95,1.00 \
  --scales 0.25,0.50,0.5375,0.75,1.00,1.25,1.50 \
  --biases=-1.00,-0.50,0.00,0.25,0.50,0.75,1.00 \
  --reference-specialist-label single \
  --reference-beta 1.00 \
  --reference-scale 0.5375 \
  --reference-bias 0.25 \
  --archive-top 100 \
  --out-json "${summary_json}"

echo "=== Light/Deep specialist same-split init ensemble complete: ${summary_json} ==="
