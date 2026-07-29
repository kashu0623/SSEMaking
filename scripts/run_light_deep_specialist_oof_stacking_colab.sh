#!/usr/bin/env bash
set -euo pipefail

# Learn validation-subject OOF logistic stacks over the six available
# Light-vs-Deep specialists, then recalibrate each stacked source in fusion.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
INIT_SEEDS=(${INIT_SEEDS:-1001 2002 3003 4004 5005})
C_VALUES=(${C_VALUES:-0.001 0.003 0.01 0.03 0.1 0.3 1.0 3.0 10.0})
CLASS_WEIGHTS=(${CLASS_WEIGHTS:-none balanced})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_STACK="${RUN_STACK:-1}"

replica_path() {
  local outer_seed="$1"
  local init_seed="$2"
  echo "${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}_init${init_seed}/light_deep_predictions.npz"
}

stack_label() {
  local class_weight="$1"
  local c_value="$2"
  echo "stack_${class_weight}_c${c_value}"
}

stack_path() {
  local outer_seed="$1"
  local class_weight="$2"
  local c_value="$3"
  local label
  label="$(stack_label "${class_weight}" "${c_value}")"
  echo "${OUTPUT_ROOT}/light_deep_specialist_oof_${label}_outer${outer_seed}/light_deep_predictions.npz"
}

for outer_seed in "${OUTER_SEEDS[@]}"; do
  members=(
    "${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}/light_deep_predictions.npz"
  )
  for init_seed in "${INIT_SEEDS[@]}"; do
    members+=("$(replica_path "${outer_seed}" "${init_seed}")")
  done
  for member in "${members[@]}"; do
    [[ -f "${member}" ]] || { echo "Missing specialist: ${member}" >&2; exit 1; }
  done

  for class_weight in "${CLASS_WEIGHTS[@]}"; do
    for c_value in "${C_VALUES[@]}"; do
      prediction="$(stack_path "${outer_seed}" "${class_weight}" "${c_value}")"
      out_dir="$(dirname "${prediction}")"
      if [[ "${RUN_STACK}" == "1" || ! -f "${prediction}" ]]; then
        echo "=== Stack outer ${outer_seed}, weight ${class_weight}, C ${c_value} ==="
        PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.stack_light_deep_specialists \
          --predictions "${members[@]}" \
          --out "${prediction}" \
          --summary-out "${out_dir}/stack_summary.json" \
          --c "${c_value}" \
          --class-weight "${class_weight}" \
          --folds 5 \
          --seed "${outer_seed}"
      fi
      [[ -f "${prediction}" ]] || { echo "Missing stack: ${prediction}" >&2; exit 1; }
    done
  done
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
specialist_paths=()
for outer_seed in "${OUTER_SEEDS[@]}"; do
  specialist_paths+=(
    "${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}/light_deep_predictions.npz"
  )
done
for class_weight in "${CLASS_WEIGHTS[@]}"; do
  for c_value in "${C_VALUES[@]}"; do
    label="$(stack_label "${class_weight}" "${c_value}")"
    specialist_labels+=("${label}")
    for outer_seed in "${OUTER_SEEDS[@]}"; do
      specialist_paths+=(
        "$(stack_path "${outer_seed}" "${class_weight}" "${c_value}")"
      )
    done
  done
done

summary_json="${OUTPUT_ROOT}/fusion4_light_deep_specialist_oof_stacking_context${CONTEXT_EPOCHS}_summary.json"
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
  --archive-top 120 \
  --out-json "${summary_json}"

echo "=== Light/Deep specialist OOF stacking complete: ${summary_json} ==="
