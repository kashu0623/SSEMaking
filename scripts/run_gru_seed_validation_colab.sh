#!/usr/bin/env bash
set -euo pipefail

# Compare GRU against the current LSTM baseline on the same temporal context20
# subject-wise splits. Seed 42 uses the original NPZ path; other seeds use the
# seed-suffixed NPZs created during repeated split validation.
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
INPUT_CSV="${INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-inverse}"
SEEDS=(${SEEDS:-42 7 123})

for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "42" ]]; then
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_summary.json"
  else
    npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
    summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"
  fi

  if [[ ! -f "${npz_path}" ]]; then
    echo "=== Build temporal context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
    PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
      --input-csv "${INPUT_CSV}" \
      --out "${npz_path}" \
      --summary-out "${summary_path}" \
      --context-epochs "${CONTEXT_EPOCHS}" \
      --seed "${seed}"
  fi

  echo "=== Train GRU baseline inverse, seed ${seed} ==="
  PYTHONPATH=src python -m sse_sleep.train_lstm \
    --npz "${npz_path}" \
    --out-dir "${OUTPUT_ROOT}/gru_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_${CLASS_WEIGHT_MODE}_seed${seed}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --class-weight-mode "${CLASS_WEIGHT_MODE}" \
    --model-type gru \
    --seed "${seed}"
done

echo "=== GRU seed validation jobs complete ==="
