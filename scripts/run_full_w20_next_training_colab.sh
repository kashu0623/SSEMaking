#!/usr/bin/env bash
set -euo pipefail

# Follow-up training candidates on the current app/4-class best input:
# temporal_w20 context20 h64 inverse.
#
# Default seed is 42 for quick filtering. Expand promising candidates with:
#   SEEDS="42 7 123" VARIANTS="ls005" bash scripts/run_full_w20_next_training_colab.sh
#
# Variants:
#   ls005: label smoothing 0.05
#   remx11: mild REM class-weight multiplier 1.1
#   longdrop_p10: train-time dropout for *_20 long-window features, p=0.10

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
INPUT_CSV="${INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal_w20.csv}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
VARIANTS=(${VARIANTS:-ls005 remx11 longdrop_p10})
PYTHON_BIN="${PYTHON_BIN:-python}"

ensure_npz() {
  local seed="$1"
  local npz_path="$2"
  local summary_path="$3"
  if [[ ! -f "${npz_path}" ]]; then
    echo "=== Build temporal w20 context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.build_npz_dataset \
      --input-csv "${INPUT_CSV}" \
      --out "${npz_path}" \
      --summary-out "${summary_path}" \
      --context-epochs "${CONTEXT_EPOCHS}" \
      --seed "${seed}"
  fi
}

out_dir_for_variant() {
  local variant="$1"
  local seed="$2"
  local base="${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_${variant}"
  if [[ "${seed}" == "42" ]]; then
    echo "${base}"
  else
    echo "${base}_seed${seed}"
  fi
}

train_variant() {
  local variant="$1"
  local npz_path="$2"
  local seed="$3"
  local out_dir
  out_dir="$(out_dir_for_variant "${variant}" "${seed}")"

  echo "=== Train full w20 ${variant}, seed ${seed} ==="
  case "${variant}" in
    ls005)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" \
        --out-dir "${out_dir}" \
        --hidden-size "${HIDDEN_SIZE}" \
        --dropout "${DROPOUT}" \
        --class-weight-mode inverse \
        --label-smoothing 0.05 \
        --seed "${seed}"
      ;;
    remx11)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" \
        --out-dir "${out_dir}" \
        --hidden-size "${HIDDEN_SIZE}" \
        --dropout "${DROPOUT}" \
        --class-weight-mode inverse \
        --rem-weight-multiplier 1.1 \
        --seed "${seed}"
      ;;
    longdrop_p10)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" \
        --out-dir "${out_dir}" \
        --hidden-size "${HIDDEN_SIZE}" \
        --dropout "${DROPOUT}" \
        --class-weight-mode inverse \
        --feature-dropout-pattern "_20" \
        --feature-dropout-prob 0.10 \
        --seed "${seed}"
      ;;
    *)
      echo "Unknown full w20 next-training variant: ${variant}" >&2
      return 1
      ;;
  esac
}

for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "42" ]]; then
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}_summary.json"
  else
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"
  fi
  ensure_npz "${seed}" "${npz_path}" "${summary_path}"

  for variant in "${VARIANTS[@]}"; do
    train_variant "${variant}" "${npz_path}" "${seed}"
  done
done

echo "=== Full w20 next-training experiments complete ==="

summary_args=()
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
  for variant in "${VARIANTS[@]}"; do
    metrics_path="$(out_dir_for_variant "${variant}" "${seed}")/lstm_metrics.json"
    if [[ -f "${metrics_path}" ]]; then
      summary_args+=(--metrics "${variant}=${metrics_path}")
    fi
  done
done

if [[ "${#summary_args[@]}" -gt 0 ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_lstm_metrics "${summary_args[@]}"
fi
