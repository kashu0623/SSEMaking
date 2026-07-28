#!/usr/bin/env bash
set -euo pipefail

# Train Light-vs-Deep specialists on two feature families and fuse only their
# conditional Deep probability into the current round5 best.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SPECIALISTS="${TRAIN_SPECIALISTS:-1}"

CONFIGS=(
  original_h64_ce
  original_h128_ce
  original_h128_focal1
  w20_h64_ce
  w20_h128_ce
  w20_h128_focal1
)

npz_for_seed() {
  local feature_family="$1"
  local outer_seed="$2"
  local prefix
  if [[ "${feature_family}" == "original" ]]; then
    prefix="dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}"
  elif [[ "${feature_family}" == "w20" ]]; then
    prefix="dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}"
  else
    echo "Unknown feature family: ${feature_family}" >&2
    return 1
  fi
  if [[ "${outer_seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${prefix}.npz"
  else
    echo "${OUTPUT_ROOT}/${prefix}_seed${outer_seed}.npz"
  fi
}

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

# Paths are config-major to match the evaluator CLI.
specialist_paths=()
for config in "${CONFIGS[@]}"; do
  feature_family="${config%%_*}"
  hidden_size=64
  loss_type="cross_entropy"
  focal_gamma=1.0
  [[ "${config}" == *"_h128_"* ]] && hidden_size=128
  [[ "${config}" == *"_focal1" ]] && loss_type="focal"

  for outer_seed in "${OUTER_SEEDS[@]}"; do
    npz_path="$(npz_for_seed "${feature_family}" "${outer_seed}")"
    out_dir="${OUTPUT_ROOT}/light_deep_specialist_${config}_outer${outer_seed}"
    predictions="${out_dir}/light_deep_predictions.npz"
    [[ -f "${npz_path}" ]] || { echo "Missing dataset: ${npz_path}" >&2; exit 1; }
    if [[ "${TRAIN_SPECIALISTS}" == "1" && ! -f "${predictions}" ]]; then
      echo "=== Train ${config}, outer seed ${outer_seed} ==="
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_light_deep_specialist \
        --npz "${npz_path}" \
        --out-dir "${out_dir}" \
        --hidden-size "${hidden_size}" \
        --dropout 0.4 \
        --epochs 40 \
        --patience 8 \
        --class-weight-mode inverse \
        --loss-type "${loss_type}" \
        --focal-gamma "${focal_gamma}" \
        --seed "${outer_seed}"
    fi
    [[ -f "${predictions}" ]] || {
      echo "Missing specialist prediction: ${predictions}" >&2
      exit 1
    }
    specialist_paths+=("${predictions}")
  done
done

summary_json="${OUTPUT_ROOT}/fusion4_light_deep_specialist_fusion_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_light_deep_specialist_fusion \
  --original-temporal-predictions "${original_paths[@]}" \
  --full-w20-predictions "${full_paths[@]}" \
  --capacity-h128-predictions "${capacity_paths[@]}" \
  --h128-ls003-predictions "${ls003_paths[@]}" \
  --direct4-single-predictions "${direct4_single_paths[@]}" \
  --direct4-ensemble-predictions "${direct4_ensemble_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --specialist-labels "${CONFIGS[@]}" \
  --specialist-predictions "${specialist_paths[@]}" \
  --betas 0.05,0.10,0.20,0.30,0.40,0.50,0.65,0.80,1.00 \
  --scales 0.50,0.75,1.00,1.25,1.50,2.00 \
  --biases=-2.00,-1.50,-1.00,-0.50,0.00,0.50 \
  --archive-top 80 \
  --out-json "${summary_json}"

echo "=== Light/Deep specialist fusion complete: ${summary_json} ==="
