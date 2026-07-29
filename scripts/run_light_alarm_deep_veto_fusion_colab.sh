#!/usr/bin/env bash
set -euo pipefail

# Use the direct Light-vs-rest models as proposals and the current staging
# model's conditional Deep probability as an explicit veto.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
CURRENT_SUMMARY_JSON="${CURRENT_SUMMARY_JSON:-${OUTPUT_ROOT}/fusion4_light_deep_h128_h256_pair_blend_refine_context${CONTEXT_EPOCHS}_summary.json}"

CONFIGS=(
  direct_h128_deep1
  direct_h128_deep2
  direct_h128_deep4
  direct_h128_stagebalanced
  direct_h256_lstm2_deep2
  multitask_h128_deep1_aux025
  multitask_h128_deep2_aux025
  multitask_h128_deep4_aux025
  multitask_h256_lstm2_deep2_aux025
)

original_paths=()
full_paths=()
capacity_paths=()
ls003_paths=()
direct4_single_paths=()
direct4_ensemble_paths=()
specialist_paths=()
for outer_seed in "${OUTER_SEEDS[@]}"; do
  current_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  original_paths+=("${current_dir}/original_predictions.npz")
  full_paths+=("${current_dir}/full_w20_predictions.npz")
  capacity_paths+=("${current_dir}/capacity_h128_predictions.npz")
  ls003_paths+=("${current_dir}/h128_ls003_predictions.npz")
  direct4_single_paths+=("${OUTPUT_ROOT}/direct4_original_outer${outer_seed}/lstm4_predictions.npz")
  direct4_ensemble_paths+=("${OUTPUT_ROOT}/same_split_init_ensemble_direct4_original_outer${outer_seed}/original_direct4_predictions.npz")
  specialist_paths+=("${OUTPUT_ROOT}/light_deep_specialist_h128_h256_blend_a0600_outer${outer_seed}/light_deep_predictions.npz")
done

alarm_paths=()
for config in "${CONFIGS[@]}"; do
  for outer_seed in "${OUTER_SEEDS[@]}"; do
    prediction="${OUTPUT_ROOT}/light_alarm_${config}_outer${outer_seed}/light_alarm_predictions.npz"
    [[ -f "${prediction}" ]] || { echo "Missing alarm prediction: ${prediction}" >&2; exit 1; }
    alarm_paths+=("${prediction}")
  done
done

[[ -f "${CURRENT_SUMMARY_JSON}" ]] || {
  echo "Missing current summary: ${CURRENT_SUMMARY_JSON}" >&2
  exit 1
}

summary_json="${OUTPUT_ROOT}/fusion_light_alarm_deep_veto_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_light_alarm_deep_veto \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-single-predictions "${direct4_single_paths[@]}" \
  --direct4-ensemble-predictions "${direct4_ensemble_paths[@]}" \
  --current-specialist-predictions "${specialist_paths[@]}" \
  --current-beta 0.975 \
  --current-scale 0.75 \
  --current-bias 0.25 \
  --config-labels "${CONFIGS[@]}" \
  --alarm-prediction-paths "${alarm_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --alarm-alphas 0.00,0.25,0.50,0.75,1.00 \
  --veto-gammas 0.00,0.50,1.00,2.00,4.00 \
  --thresholds 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90 \
  --deep-leak-limits 0.10,0.20,0.30,0.40 \
  --current-summary-json "${CURRENT_SUMMARY_JSON}" \
  --archive-top 50 \
  --out-json "${summary_json}"

echo "=== Light alarm Deep-veto fusion complete: ${summary_json} ==="
