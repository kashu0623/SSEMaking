#!/usr/bin/env bash
set -euo pipefail

# Joint local refinement around the first classwise source-blend best. Reuse all
# prediction artifacts and preserve the adopted best as an exact reference.

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

summary_json="${OUTPUT_ROOT}/fusion4_direct4_classwise_source_blend_hybrid_round2_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_direct4_source_blend_hybrid \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-single-predictions "${direct4_single_paths[@]}" \
  --direct4-ensemble-predictions "${direct4_ensemble_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --source-wake-betas 0.00,0.10,0.20,1.00 \
  --source-light-betas 0.00,0.10,0.20,1.00 \
  --source-deep-betas 0.00,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,1.00 \
  --source-rem-betas 0.00,0.25,0.40,0.50,0.60,0.75,1.00 \
  --source-top-n 8 \
  --required-source-betas 0.00,0.00,0.25,0.50 \
  --wake-alphas 0.10,0.125,0.15,0.175,0.20,0.225,0.3125 \
  --light-alphas 0.34,0.45,0.50,0.55,0.60,0.65,0.70 \
  --deep-alphas 0.75,0.80,0.85,0.90,0.95 \
  --rem-alphas 0.00 \
  --deep-gains 1.10,1.15,1.20,1.25,1.30 \
  --reference-source-betas 0.00,0.00,0.25,0.50 \
  --reference-hybrid-alphas 0.15,0.55,0.85,0.00 \
  --reference-deep-gain 1.20 \
  --archive-top 60 \
  --out-json "${summary_json}"

echo "=== Direct4 classwise source blend hybrid round2 complete: ${summary_json} ==="
