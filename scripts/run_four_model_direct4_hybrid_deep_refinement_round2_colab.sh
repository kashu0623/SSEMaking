#!/usr/bin/env bash
set -euo pipefail

# Expand the winning Wake/Light edges and calibrate the direct4 Deep score.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
WAKE_ALPHAS="${WAKE_ALPHAS:-0.15,0.20,0.25,0.30,0.35,0.40,0.45}"
LIGHT_ALPHAS="${LIGHT_ALPHAS:-0.08,0.10,0.12,0.14,0.16,0.18,0.20,0.24}"
DEEP_ALPHAS="${DEEP_ALPHAS:-0.95,1.00}"
REM_ALPHAS="${REM_ALPHAS:-0.00}"
DEEP_GAINS="${DEEP_GAINS:-0.80,0.90,1.00,1.10,1.20,1.30,1.40,1.50,1.60}"
PYTHON_BIN="${PYTHON_BIN:-python}"

base_predictions=()
primary_predictions=()
secondary_predictions=()
tertiary_predictions=()
direct4_predictions=()
seed_labels=()

for outer_seed in "${OUTER_SEEDS[@]}"; do
  ensemble_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  base="${ensemble_dir}/original_predictions.npz"
  primary="${ensemble_dir}/full_w20_predictions.npz"
  secondary="${ensemble_dir}/capacity_h128_predictions.npz"
  tertiary="${ensemble_dir}/h128_ls003_predictions.npz"
  direct4="${OUTPUT_ROOT}/direct4_original_outer${outer_seed}/lstm4_predictions.npz"

  for predictions in "${base}" "${primary}" "${secondary}" "${tertiary}" "${direct4}"; do
    [[ -f "${predictions}" ]] || {
      echo "Missing prediction file: ${predictions}" >&2
      echo "Run the same-split ensemble and direct four-class experiments first." >&2
      exit 1
    }
  done

  base_predictions+=("${base}")
  primary_predictions+=("${primary}")
  secondary_predictions+=("${secondary}")
  tertiary_predictions+=("${tertiary}")
  direct4_predictions+=("${direct4}")
  seed_labels+=("${outer_seed}")
done

summary_json="${OUTPUT_ROOT}/fusion4_same_split_ensemble_plus_direct4_original_hybrid_deep_refine_round2_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"

PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_direct4_hybrid_deep_fusion \
  --original-temporal-predictions "${base_predictions[@]}" \
  --full-w20-predictions "${primary_predictions[@]}" \
  --capacity-h128-predictions "${secondary_predictions[@]}" \
  --h128-ls003-predictions "${tertiary_predictions[@]}" \
  --direct4-original-predictions "${direct4_predictions[@]}" \
  --seed-labels "${seed_labels[@]}" \
  --wake-alphas "${WAKE_ALPHAS}" \
  --light-alphas "${LIGHT_ALPHAS}" \
  --deep-alphas "${DEEP_ALPHAS}" \
  --rem-alphas "${REM_ALPHAS}" \
  --deep-gains "${DEEP_GAINS}" \
  --out-json "${summary_json}"

echo "=== Direct4 hybrid Deep refinement round2 complete: ${summary_json} ==="
