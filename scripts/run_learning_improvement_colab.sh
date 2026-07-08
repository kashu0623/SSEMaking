#!/usr/bin/env bash
set -euo pipefail

# Focused training-improvement experiments for the current default model:
# temporal context20, h64, dropout0.4, inverse class weighting.
#
# Default SEEDS is just 42 for quick candidate filtering. Set
# SEEDS="42 7 123" after one or two candidates look promising.
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
INPUT_CSV="${INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})

ensure_npz() {
  local seed="$1"
  local npz_path="$2"
  local summary_path="$3"
  if [[ ! -f "${npz_path}" ]]; then
    echo "=== Build temporal context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
    PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
      --input-csv "${INPUT_CSV}" \
      --out "${npz_path}" \
      --summary-out "${summary_path}" \
      --context-epochs "${CONTEXT_EPOCHS}" \
      --seed "${seed}"
  fi
}

train_candidate() {
  local npz_path="$1"
  local seed="$2"
  local name="$3"
  shift 3

  echo "=== Train ${name}, seed ${seed} ==="
  PYTHONPATH=src python -m sse_sleep.train_lstm \
    --npz "${npz_path}" \
    --out-dir "${OUTPUT_ROOT}/${name}_seed${seed}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --seed "${seed}" \
    "$@"
}

for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "42" ]]; then
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_summary.json"
  else
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"
  fi

  ensure_npz "${seed}" "${npz_path}" "${summary_path}"

  train_candidate "${npz_path}" "${seed}" "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_select4combo" \
    --class-weight-mode inverse \
    --selection-metric 4_macro_f1_plus_4_kappa

  train_candidate "${npz_path}" "${seed}" "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_focal_g1" \
    --class-weight-mode inverse \
    --loss-type focal \
    --focal-gamma 1.0

  train_candidate "${npz_path}" "${seed}" "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_focal_g2" \
    --class-weight-mode inverse \
    --loss-type focal \
    --focal-gamma 2.0

  train_candidate "${npz_path}" "${seed}" "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_ls005" \
    --class-weight-mode inverse \
    --label-smoothing 0.05

  train_candidate "${npz_path}" "${seed}" "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_weighted_sampler_none_weight" \
    --class-weight-mode none \
    --train-sampler weighted

  train_candidate "${npz_path}" "${seed}" "lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_weighted_sampler_sqrt_weight" \
    --class-weight-mode sqrt \
    --train-sampler weighted
done

echo "=== Learning-improvement candidate jobs complete ==="
