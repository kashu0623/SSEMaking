#!/usr/bin/env bash
set -euo pipefail

# Fine-refine the h128+h256 specialist ridge around the project-rule best
# (alpha 0.60, beta 0.975, scale 0.75, bias 0.25) and pure-score top
# (alpha 0.60, beta 1.00, scale 0.775, bias 0.25).

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"

BLENDS=(
  a0575=0.425,0.575
  a0580=0.420,0.580
  a0585=0.415,0.585
  a0590=0.410,0.590
  a0595=0.405,0.595
  a0600=0.400,0.600
  a0605=0.395,0.605
  a0610=0.390,0.610
  a0615=0.385,0.615
  a0620=0.380,0.620
  a0625=0.375,0.625
)

current_specialist_path() {
  local outer_seed="$1"
  echo "${OUTPUT_ROOT}/light_deep_specialist_original_h128_ce_outer${outer_seed}/light_deep_predictions.npz"
}

h256_specialist_path() {
  local outer_seed="$1"
  echo "${OUTPUT_ROOT}/light_deep_specialist_light_h256_inverse_lstm2_outer${outer_seed}/light_deep_predictions.npz"
}

blend_path() {
  local label="$1"
  local outer_seed="$2"
  echo "${OUTPUT_ROOT}/light_deep_specialist_h128_h256_blend_${label}_outer${outer_seed}/light_deep_predictions.npz"
}

for outer_seed in "${OUTER_SEEDS[@]}"; do
  current="$(current_specialist_path "${outer_seed}")"
  h256="$(h256_specialist_path "${outer_seed}")"
  [[ -f "${current}" ]] || { echo "Missing current specialist: ${current}" >&2; exit 1; }
  [[ -f "${h256}" ]] || { echo "Missing h256 specialist: ${h256}" >&2; exit 1; }
  for item in "${BLENDS[@]}"; do
    label="${item%%=*}"
    weights="${item#*=}"
    prediction="$(blend_path "${label}" "${outer_seed}")"
    out_dir="$(dirname "${prediction}")"
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.average_light_deep_specialist_ensemble \
      --predictions "${current}" "${h256}" \
      --weights "${weights}" \
      --out "${prediction}" \
      --summary-out "${out_dir}/blend_summary.json"
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

specialist_labels=()
specialist_paths=()
for item in "${BLENDS[@]}"; do
  label="${item%%=*}"
  specialist_labels+=("blend_${label}")
  for outer_seed in "${OUTER_SEEDS[@]}"; do
    specialist_paths+=("$(blend_path "${label}" "${outer_seed}")")
  done
done

summary_json="${OUTPUT_ROOT}/fusion4_light_deep_h128_h256_pair_blend_refine_round2_context${CONTEXT_EPOCHS}_summary.json"
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
  --betas 0.95,0.96,0.975,0.98,0.99,1.00 \
  --scales 0.72,0.73,0.74,0.75,0.76,0.77,0.775,0.79,0.80 \
  --biases 0.20,0.225,0.25,0.275,0.30 \
  --reference-specialist-label blend_a0600 \
  --reference-beta 0.975 \
  --reference-scale 0.75 \
  --reference-bias 0.25 \
  --archive-top 120 \
  --out-json "${summary_json}"

echo "=== Light/Deep h128+h256 pair blend refinement round2 complete: ${summary_json} ==="
