#!/usr/bin/env bash
set -euo pipefail

# Refine the original-h128-CE Light/Deep calibration ridge without retraining.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"

original_paths=()
full_paths=()
capacity_paths=()
ls003_paths=()
direct4_single_paths=()
direct4_ensemble_paths=()
specialist_paths=()

for outer_seed in "${OUTER_SEEDS[@]}"; do
  current_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  original="${current_dir}/original_predictions.npz"
  full="${current_dir}/full_w20_predictions.npz"
  capacity="${current_dir}/capacity_h128_predictions.npz"
  ls003="${current_dir}/h128_ls003_predictions.npz"
  direct4_single="${OUTPUT_ROOT}/direct4_original_outer${outer_seed}/lstm4_predictions.npz"
  direct4_ensemble="${OUTPUT_ROOT}/same_split_init_ensemble_direct4_original_outer${outer_seed}/original_direct4_predictions.npz"
  specialist="${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}/light_deep_predictions.npz"
  for path in \
    "${original}" "${full}" "${capacity}" "${ls003}" \
    "${direct4_single}" "${direct4_ensemble}" "${specialist}"; do
    [[ -f "${path}" ]] || { echo "Missing prediction: ${path}" >&2; exit 1; }
  done
  original_paths+=("${original}")
  full_paths+=("${full}")
  capacity_paths+=("${capacity}")
  ls003_paths+=("${ls003}")
  direct4_single_paths+=("${direct4_single}")
  direct4_ensemble_paths+=("${direct4_ensemble}")
  specialist_paths+=("${specialist}")
done

summary_json="${OUTPUT_ROOT}/fusion4_light_deep_specialist_fusion_refine_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_light_deep_specialist_fusion \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-single-predictions "${direct4_single_paths[@]}" \
  --direct4-ensemble-predictions "${direct4_ensemble_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --specialist-labels original_h128_ce \
  --specialist-predictions "${specialist_paths[@]}" \
  --betas 0.65,0.725,0.80,0.875,0.95,1.00 \
  --scales 0.25,0.35,0.40,0.45,0.50,0.55,0.625,0.75,0.875,1.00 \
  --biases=-0.75,-0.50,-0.25,0.00,0.25,0.50,0.75,1.00 \
  --reference-specialist-label original_h128_ce \
  --reference-beta 1.00 \
  --reference-scale 0.50 \
  --reference-bias 0.50 \
  --archive-top 80 \
  --out-json "${summary_json}"

echo "=== Light/Deep specialist fusion refinement complete: ${summary_json} ==="
