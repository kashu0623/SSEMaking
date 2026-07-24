#!/usr/bin/env bash
set -euo pipefail

# Reuse the existing single and six-checkpoint direct4 predictions. Blend the
# two sources by stage, retain promising sources, then recalibrate the hybrid.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"

original_paths=()
full_paths=()
capacity_paths=()
ls003_paths=()
direct4_single_paths=()
direct4_ensemble_paths=()

for outer_seed in "${OUTER_SEEDS[@]}"; do
  current_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  original="${current_dir}/original_predictions.npz"
  full="${current_dir}/full_w20_predictions.npz"
  capacity="${current_dir}/capacity_h128_predictions.npz"
  ls003="${current_dir}/h128_ls003_predictions.npz"
  direct4_single="${OUTPUT_ROOT}/direct4_original_outer${outer_seed}/lstm4_predictions.npz"
  direct4_ensemble="${OUTPUT_ROOT}/same_split_init_ensemble_direct4_original_outer${outer_seed}/original_direct4_predictions.npz"
  for path in \
    "${original}" "${full}" "${capacity}" "${ls003}" \
    "${direct4_single}" "${direct4_ensemble}"; do
    [[ -f "${path}" ]] || { echo "Missing prediction: ${path}" >&2; exit 1; }
  done
  original_paths+=("${original}")
  full_paths+=("${full}")
  capacity_paths+=("${capacity}")
  ls003_paths+=("${ls003}")
  direct4_single_paths+=("${direct4_single}")
  direct4_ensemble_paths+=("${direct4_ensemble}")
done

summary_json="${OUTPUT_ROOT}/fusion4_direct4_classwise_source_blend_hybrid_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_direct4_source_blend_hybrid \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-single-predictions "${direct4_single_paths[@]}" \
  --direct4-ensemble-predictions "${direct4_ensemble_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --source-wake-betas 0.00,0.50,1.00 \
  --source-light-betas 0.00,0.50,1.00 \
  --source-deep-betas 0.00,0.10,0.25,0.50,1.00 \
  --source-rem-betas 0.00,0.50,1.00 \
  --source-top-n 4 \
  --wake-alphas 0.10,0.15,0.225,0.30,0.3125 \
  --light-alphas 0.25,0.34,0.45,0.55 \
  --deep-alphas 0.70,0.85,1.00 \
  --rem-alphas 0.00 \
  --deep-gains 1.00,1.20,1.40,1.60 \
  --archive-top 50 \
  --out-json "${summary_json}"

echo "=== Direct4 classwise source blend hybrid complete: ${summary_json} ==="
