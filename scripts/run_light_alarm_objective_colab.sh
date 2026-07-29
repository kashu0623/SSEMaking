#!/usr/bin/env bash
set -euo pipefail

# Pause 4-class fusion refinement and train app-oriented Light-vs-rest models.
# Deep remains a hard negative; multitask variants retain a 4-class auxiliary head.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
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
  echo "${OUTPUT_ROOT}/light_alarm_${config}_outer${outer_seed}"
}

train_config() {
  local config="$1"
  local outer_seed="$2"
  local npz_path="$3"
  local destination="$4"
  local hidden_size=128
  local num_layers=1
  local deep_multiplier=1.0
  local train_sampler="none"
  local aux_weight=0.0

  [[ "${config}" == *_h256_* ]] && hidden_size=256
  [[ "${config}" == *_lstm2_* ]] && num_layers=2
  [[ "${config}" == *_deep2* ]] && deep_multiplier=2.0
  [[ "${config}" == *_deep4* ]] && deep_multiplier=4.0
  [[ "${config}" == *_stagebalanced ]] && train_sampler="stage_balanced"
  [[ "${config}" == multitask_* ]] && aux_weight=0.25

  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_light_alarm \
    --npz "${npz_path}" \
    --out-dir "${destination}" \
    --hidden-size "${hidden_size}" \
    --num-layers "${num_layers}" \
    --dropout 0.4 \
    --batch-size 256 \
    --epochs 60 \
    --patience 12 \
    --binary-weight-mode inverse \
    --deep-negative-multiplier "${deep_multiplier}" \
    --train-sampler "${train_sampler}" \
    --stage4-aux-weight "${aux_weight}" \
    --stage4-aux-class-weight-mode inverse \
    --seed "${outer_seed}"
}

prediction_paths=()
for config in "${CONFIGS[@]}"; do
  for outer_seed in "${OUTER_SEEDS[@]}"; do
    npz_path="$(npz_for_outer_seed "${outer_seed}")"
    destination="$(out_dir "${config}" "${outer_seed}")"
    prediction="${destination}/light_alarm_predictions.npz"
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

summary_json="${OUTPUT_ROOT}/fusion_light_alarm_objective_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_light_alarm \
  --config-labels "${CONFIGS[@]}" \
  --prediction-paths "${prediction_paths[@]}" \
  --seed-labels "${OUTER_SEEDS[@]}" \
  --current-summary-json "${CURRENT_SUMMARY_JSON}" \
  --deep-leak-limits 0.10,0.20,0.30,0.40 \
  --archive-top 40 \
  --out-json "${summary_json}"

echo "=== Light alarm objective experiment complete: ${summary_json} ==="
