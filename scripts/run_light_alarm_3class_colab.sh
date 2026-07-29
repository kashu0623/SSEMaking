#!/usr/bin/env bash
set -euo pipefail

# Train task-aligned Other(Wake+REM)/Light/Deep models. Deep remains an
# explicit primary class while deployment scoring uses P(Light).

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
CURRENT_SUMMARY_JSON="${CURRENT_SUMMARY_JSON:-}"
if [[ -z "${CURRENT_SUMMARY_JSON}" ]]; then
  CURRENT_SUMMARY_JSON="${OUTPUT_ROOT}/fusion4_light_deep_h128_h256_pair_blend_refine_context${CONTEXT_EPOCHS}_summary.json"
fi

CONFIGS=(
  alarm3_h128_inv_ce_d050
  alarm3_h128_inv_ce_d100
  alarm3_h128_inv_ce_d200
  alarm3_h128_inv_ce_d400
  alarm3_h128_inv_focal_d050
  alarm3_h128_inv_focal_d100
  alarm3_h128_inv_focal_d200
  alarm3_h128_sqrt_focal_d200
  alarm3_h128_balanced_ce
  alarm3_h256_lstm2_inv_ce_d050
  alarm3_h256_lstm2_inv_ce_d100
  alarm3_h256_lstm2_inv_ce_d200
  alarm3_h256_lstm2_inv_focal_d100
  alarm3_h256_lstm2_inv_focal_d200
  alarm3_h256_lstm2_balanced_ce
)

npz_for_outer_seed() {
  local outer_seed="$1"
  if [[ "${outer_seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}.npz"
  else
    echo "${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${outer_seed}.npz"
  fi
}

out_dir() {
  local config="$1"
  local outer_seed="$2"
  echo "${OUTPUT_ROOT}/light_alarm_3class_${config}_outer${outer_seed}"
}

train_config() {
  local config="$1"
  local outer_seed="$2"
  local npz_path="$3"
  local destination="$4"
  local hidden_size=128
  local num_layers=1
  local class_weight_mode=inverse
  local deep_multiplier=1.0
  local train_sampler=none
  local loss_type=cross_entropy

  [[ "${config}" == *_h256_* ]] && hidden_size=256
  [[ "${config}" == *_lstm2_* ]] && num_layers=2
  [[ "${config}" == *_sqrt_* ]] && class_weight_mode=sqrt
  [[ "${config}" == *_d050 ]] && deep_multiplier=0.5
  [[ "${config}" == *_d200 ]] && deep_multiplier=2.0
  [[ "${config}" == *_d400 ]] && deep_multiplier=4.0
  [[ "${config}" == *_focal_* ]] && loss_type=focal
  if [[ "${config}" == *_balanced_* ]]; then
    class_weight_mode=none
    train_sampler=weighted
  fi

  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_light_alarm_3class \
    --npz "${npz_path}" \
    --out-dir "${destination}" \
    --hidden-size "${hidden_size}" \
    --num-layers "${num_layers}" \
    --dropout 0.4 \
    --batch-size 256 \
    --epochs 60 \
    --patience 12 \
    --class-weight-mode "${class_weight_mode}" \
    --deep-class-multiplier "${deep_multiplier}" \
    --train-sampler "${train_sampler}" \
    --loss-type "${loss_type}" \
    --focal-gamma 2.0 \
    --seed "${outer_seed}"
}

prediction_paths=()
for config in "${CONFIGS[@]}"; do
  for outer_seed in "${OUTER_SEEDS[@]}"; do
    npz_path="$(npz_for_outer_seed "${outer_seed}")"
    destination="$(out_dir "${config}" "${outer_seed}")"
    prediction="${destination}/light_alarm_3class_predictions.npz"
    [[ -f "${npz_path}" ]] || { echo "Missing dataset: ${npz_path}" >&2; exit 1; }
    if [[ "${RUN_TRAIN}" == "1" && ! -f "${prediction}" ]]; then
      echo "=== Train ${config}, outer ${outer_seed} ==="
      train_config "${config}" "${outer_seed}" "${npz_path}" "${destination}"
    fi
    [[ -f "${prediction}" ]] || { echo "Missing prediction: ${prediction}" >&2; exit 1; }
    prediction_paths+=("${prediction}")
  done
done

[[ -f "${CURRENT_SUMMARY_JSON}" ]] || {
  echo "Missing current summary: ${CURRENT_SUMMARY_JSON}" >&2
  exit 1
}

summary_json="${OUTPUT_ROOT}/fusion_light_alarm_3class_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_light_alarm \
  --config-labels "${CONFIGS[@]}" \
  --prediction-paths "${prediction_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --current-summary-json "${CURRENT_SUMMARY_JSON}" \
  --deep-leak-limits 0.10,0.20,0.30,0.40 \
  --archive-top 60 \
  --out-json "${summary_json}"

echo "=== Light alarm 3-class experiment complete: ${summary_json} ==="
