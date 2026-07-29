#!/usr/bin/env bash
set -euo pipefail

# Train hard-boundary N2-vs-N3 specialists and compare them with the current
# all-Light-vs-Deep specialist under the same conditional fusion grid.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SPECIALISTS="${TRAIN_SPECIALISTS:-1}"

CONFIGS=(
  n2n3_h128_inverse_lstm
  n2n3_h128_sqrt_lstm
  n2n3_h128_none_lstm
  n2n3_h256_inverse_lstm
  n2n3_h128_inverse_lstm2
  n2n3_h256_inverse_lstm2
  n2n3_h128_inverse_gru
  n2n3_h128_inverse_ls003
  light_h256_inverse_lstm2
  light_h128_sqrt_lstm
)

npz_for_outer_seed() {
  local outer_seed="$1"
  if [[ "${outer_seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}.npz"
  else
    echo "${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${outer_seed}.npz"
  fi
}

train_config() {
  local config="$1"
  local outer_seed="$2"
  local npz_path="$3"
  local out_dir="$4"
  local negative_mode="n2_only"
  local hidden_size=128
  local num_layers=1
  local class_weight_mode="inverse"
  local model_type="lstm"
  local label_smoothing=0.0

  [[ "${config}" == light_* ]] && negative_mode="light"
  [[ "${config}" == *_h256_* ]] && hidden_size=256
  [[ "${config}" == *_lstm2 ]] && num_layers=2
  [[ "${config}" == *_sqrt_* ]] && class_weight_mode="sqrt"
  [[ "${config}" == *_none_* ]] && class_weight_mode="none"
  [[ "${config}" == *_gru ]] && model_type="gru"
  [[ "${config}" == *_ls003 ]] && label_smoothing=0.03

  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_light_deep_specialist \
    --npz "${npz_path}" \
    --out-dir "${out_dir}" \
    --hidden-size "${hidden_size}" \
    --num-layers "${num_layers}" \
    --dropout 0.4 \
    --epochs 50 \
    --patience 10 \
    --class-weight-mode "${class_weight_mode}" \
    --loss-type cross_entropy \
    --label-smoothing "${label_smoothing}" \
    --negative-mode "${negative_mode}" \
    --model-type "${model_type}" \
    --seed "${outer_seed}"
}

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

for config in "${CONFIGS[@]}"; do
  specialist_labels+=("${config}")
  for outer_seed in "${OUTER_SEEDS[@]}"; do
    npz_path="$(npz_for_outer_seed "${outer_seed}")"
    out_dir="${OUTPUT_ROOT}/light_deep_specialist_${config}_outer${outer_seed}"
    prediction="${out_dir}/light_deep_predictions.npz"
    [[ -f "${npz_path}" ]] || { echo "Missing dataset: ${npz_path}" >&2; exit 1; }
    if [[ "${TRAIN_SPECIALISTS}" == "1" && ! -f "${prediction}" ]]; then
      echo "=== Train ${config}, outer ${outer_seed} ==="
      train_config "${config}" "${outer_seed}" "${npz_path}" "${out_dir}"
    fi
    [[ -f "${prediction}" ]] || { echo "Missing specialist: ${prediction}" >&2; exit 1; }
    specialist_paths+=("${prediction}")
  done
done

summary_json="${OUTPUT_ROOT}/fusion4_light_deep_n2n3_specialist_context${CONTEXT_EPOCHS}_summary.json"
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

echo "=== Light/Deep N2-vs-N3 specialist complete: ${summary_json} ==="
