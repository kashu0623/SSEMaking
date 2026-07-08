#!/usr/bin/env bash
set -euo pipefail

# Targeted long-window temporal feature experiments for the current default
# LSTM setup. The default variant keeps short temporal features for all core
# features, then adds 20-epoch delta/rolling features only to slow movement,
# cardio, and temperature cues.
#
# Defaults:
#   targeted_w20: acc_vm_activity, acc_vm_mean, hr_mean, hr_std,
#                 ibi_mean, ibi_std, temp_mean, temp_slope
#
# Optional narrower variants:
#   movement_only_w20: acc_vm_mean, acc_vm_activity
#   cardio_temp_w20: hr_mean, hr_std, ibi_mean, ibi_std, temp_mean, temp_slope
#
# Examples:
#   bash scripts/run_targeted_slow_w20_colab.sh
#   SEEDS="42 7 123" bash scripts/run_targeted_slow_w20_colab.sh
#   VARIANTS="targeted_w20 movement_only_w20 cardio_temp_w20" bash scripts/run_targeted_slow_w20_colab.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
RAW_INPUT_CSV="${RAW_INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features.csv}"
SHORT_TEMPORAL_CSV="${SHORT_TEMPORAL_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
VARIANTS=(${VARIANTS:-targeted_w20})

features_for_variant() {
  local variant="$1"
  case "${variant}" in
    targeted_w20)
      echo "acc_vm_activity,acc_vm_mean,hr_mean,hr_std,ibi_mean,ibi_std,temp_mean,temp_slope"
      ;;
    movement_only_w20)
      echo "acc_vm_mean,acc_vm_activity"
      ;;
    cardio_temp_w20)
      echo "hr_mean,hr_std,ibi_mean,ibi_std,temp_mean,temp_slope"
      ;;
    *)
      echo "Unknown targeted w20 variant: ${variant}" >&2
      return 1
      ;;
  esac
}

build_short_temporal_csv() {
  local summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_features_summary.json"
  if [[ ! -f "${SHORT_TEMPORAL_CSV}" ]]; then
    echo "=== Build short temporal feature CSV ==="
    PYTHONPATH=src python -m sse_sleep.add_temporal_features \
      --input-csv "${RAW_INPUT_CSV}" \
      --out-csv "${SHORT_TEMPORAL_CSV}" \
      --summary-out "${summary_path}"
  fi
}

build_variant_csv() {
  local variant="$1"
  local base_features
  base_features="$(features_for_variant "${variant}")"
  local out_csv="${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal_${variant}.csv"
  local summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_${variant}_features_summary.json"

  if [[ ! -f "${out_csv}" ]]; then
    echo "=== Add targeted 20-epoch temporal features: ${variant} ==="
    PYTHONPATH=src python -m sse_sleep.add_temporal_features \
      --input-csv "${SHORT_TEMPORAL_CSV}" \
      --out-csv "${out_csv}" \
      --summary-out "${summary_path}" \
      --base-features "${base_features}" \
      --delta-windows 20 \
      --rolling-window-list 20
  fi
}

metrics_path_for_run() {
  local variant="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/lstm_temporal_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json"
  else
    echo "${OUTPUT_ROOT}/lstm_temporal_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}/lstm_metrics.json"
  fi
}

train_variant_seed() {
  local variant="$1"
  local seed="$2"
  local npz_path
  local summary_path
  local out_dir

  if [[ "${seed}" == "42" ]]; then
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_${variant}_lstm_context${CONTEXT_EPOCHS}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_${variant}_lstm_context${CONTEXT_EPOCHS}_summary.json"
    out_dir="${OUTPUT_ROOT}/lstm_temporal_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  else
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_${variant}_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_${variant}_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"
    out_dir="${OUTPUT_ROOT}/lstm_temporal_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}"
  fi

  if [[ ! -f "${npz_path}" ]]; then
    echo "=== Build ${variant} context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
    PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
      --input-csv "${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal_${variant}.csv" \
      --out "${npz_path}" \
      --summary-out "${summary_path}" \
      --context-epochs "${CONTEXT_EPOCHS}" \
      --seed "${seed}"
  fi

  echo "=== Train ${variant}, seed ${seed} ==="
  PYTHONPATH=src python -m sse_sleep.train_lstm \
    --npz "${npz_path}" \
    --out-dir "${out_dir}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --class-weight-mode inverse \
    --seed "${seed}"
}

build_short_temporal_csv

for variant in "${VARIANTS[@]}"; do
  build_variant_csv "${variant}"
  for seed in "${SEEDS[@]}"; do
    train_variant_seed "${variant}" "${seed}"
  done
done

echo "=== Targeted slow w20 experiments complete ==="

summary_args=()
for variant in "${VARIANTS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    metrics_path="$(metrics_path_for_run "${variant}" "${seed}")"
    if [[ -f "${metrics_path}" ]]; then
      summary_args+=(--metrics "${variant}=${metrics_path}")
    fi
  done
done

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  if [[ -f "${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json" ]]; then
    summary_args+=(--metrics "full_w20=${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json")
    summary_args+=(--baseline-label full_w20)
  fi
  for seed in "${SEEDS[@]}"; do
    if [[ "${seed}" != "42" ]]; then
      baseline_metrics="${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}/lstm_metrics.json"
      if [[ -f "${baseline_metrics}" ]]; then
        summary_args+=(--metrics "full_w20=${baseline_metrics}")
      fi
    fi
  done
  PYTHONPATH=src python -m sse_sleep.summarize_lstm_metrics "${summary_args[@]}"
fi
