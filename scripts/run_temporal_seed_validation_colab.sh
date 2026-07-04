#!/usr/bin/env bash
set -euo pipefail

# Colab/Drive paths used by the DreamT 100Hz temporal LSTM experiments.
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
INPUT_CSV="${INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
CLASS_WEIGHT_MODE="${CLASS_WEIGHT_MODE:-inverse}"

# Seed 42 has already been run in the current experiment log. These two add
# independent subject-wise splits and matching training initialization seeds.
SEEDS=(${SEEDS:-7 123})

for seed in "${SEEDS[@]}"; do
  npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
  summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"

  echo "=== Build temporal context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
  PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
    --input-csv "${INPUT_CSV}" \
    --out "${npz_path}" \
    --summary-out "${summary_path}" \
    --context-epochs "${CONTEXT_EPOCHS}" \
    --seed "${seed}"

  echo "=== Train baseline inverse, seed ${seed} ==="
  PYTHONPATH=src python -m sse_sleep.train_lstm \
    --npz "${npz_path}" \
    --out-dir "${OUTPUT_ROOT}/lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_${CLASS_WEIGHT_MODE}_seed${seed}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --class-weight-mode "${CLASS_WEIGHT_MODE}" \
    --seed "${seed}"

  echo "=== Train N3 x1.2 inverse, seed ${seed} ==="
  PYTHONPATH=src python -m sse_sleep.train_lstm \
    --npz "${npz_path}" \
    --out-dir "${OUTPUT_ROOT}/lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_${CLASS_WEIGHT_MODE}_n3x12_seed${seed}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --dropout "${DROPOUT}" \
    --class-weight-mode "${CLASS_WEIGHT_MODE}" \
    --n3-weight-multiplier 1.2 \
    --seed "${seed}"
done

echo "=== Seed validation jobs complete ==="
