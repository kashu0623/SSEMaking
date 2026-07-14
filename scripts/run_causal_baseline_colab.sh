#!/usr/bin/env bash
set -euo pipefail

# Build and train causal per-night baseline feature experiments.
# Variants start from temporal feature CSVs and add prior-only expanding
# mean/std/z-score features.
#
# Variants:
#   temporal_baseline: acc_vm_activity, hr_mean, ibi_mean, temp_mean, bvp_std
#   cardio_baseline: hr_mean, ibi_mean
#   temporal_w20_baseline: full w20 CSV + temporal_baseline feature set
#
# Examples:
#   bash scripts/run_causal_baseline_colab.sh
#   VARIANTS="cardio_baseline" bash scripts/run_causal_baseline_colab.sh
#   SEEDS="42 7 123" VARIANTS="cardio_baseline" bash scripts/run_causal_baseline_colab.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
RAW_INPUT_CSV="${RAW_INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features.csv}"
SHORT_TEMPORAL_CSV="${SHORT_TEMPORAL_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
VARIANTS=(${VARIANTS:-temporal_baseline})

features_for_variant() {
  local variant="$1"
  case "${variant}" in
    temporal_baseline | temporal_w20_baseline)
      echo "acc_vm_activity,hr_mean,ibi_mean,temp_mean,bvp_std"
      ;;
    cardio_baseline)
      echo "hr_mean,ibi_mean"
      ;;
    *)
      echo "Unknown causal baseline variant: ${variant}" >&2
      return 1
      ;;
  esac
}

input_csv_for_variant() {
  local variant="$1"
  case "${variant}" in
    temporal_baseline | cardio_baseline)
      echo "${SHORT_TEMPORAL_CSV}"
      ;;
    temporal_w20_baseline)
      echo "${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal_w20.csv"
      ;;
    *)
      echo "Unknown causal baseline variant: ${variant}" >&2
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
  local input_csv
  base_features="$(features_for_variant "${variant}")"
  input_csv="$(input_csv_for_variant "${variant}")"
  local out_csv="${OUTPUT_ROOT}/dreamt_100hz_epoch_features_${variant}.csv"
  local summary_path="${OUTPUT_ROOT}/dreamt_100hz_${variant}_features_summary.json"

  if [[ ! -f "${input_csv}" ]]; then
    echo "Missing input CSV for ${variant}: ${input_csv}" >&2
    return 1
  fi

  if [[ ! -f "${out_csv}" ]]; then
    echo "=== Add causal baseline features: ${variant} ==="
    PYTHONPATH=src python -m sse_sleep.add_causal_baseline_features \
      --input-csv "${input_csv}" \
      --out-csv "${out_csv}" \
      --summary-out "${summary_path}" \
      --base-features "${base_features}"
  fi
}

metrics_path_for_run() {
  local variant="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/lstm_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json"
  else
    echo "${OUTPUT_ROOT}/lstm_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}/lstm_metrics.json"
  fi
}

train_variant_seed() {
  local variant="$1"
  local seed="$2"
  local npz_path
  local summary_path
  local out_dir

  if [[ "${seed}" == "42" ]]; then
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_${variant}_lstm_context${CONTEXT_EPOCHS}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_${variant}_lstm_context${CONTEXT_EPOCHS}_summary.json"
    out_dir="${OUTPUT_ROOT}/lstm_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  else
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_${variant}_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_${variant}_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"
    out_dir="${OUTPUT_ROOT}/lstm_${variant}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}"
  fi

  if [[ ! -f "${npz_path}" ]]; then
    echo "=== Build ${variant} context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
    PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
      --input-csv "${OUTPUT_ROOT}/dreamt_100hz_epoch_features_${variant}.csv" \
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

echo "=== Causal baseline experiments complete ==="

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
  if [[ -f "${OUTPUT_ROOT}/lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json" ]]; then
    summary_args+=(--metrics "original_temporal=${OUTPUT_ROOT}/lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse/lstm_metrics.json")
    summary_args+=(--baseline-label original_temporal)
  fi
  for seed in "${SEEDS[@]}"; do
    if [[ "${seed}" != "42" ]]; then
      baseline_metrics="${OUTPUT_ROOT}/lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}/lstm_metrics.json"
      if [[ -f "${baseline_metrics}" ]]; then
        summary_args+=(--metrics "original_temporal=${baseline_metrics}")
      fi
    fi
  done
  PYTHONPATH=src python -m sse_sleep.summarize_lstm_metrics "${summary_args[@]}"
fi
