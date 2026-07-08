#!/usr/bin/env bash
set -euo pipefail

# Deep/N3 auxiliary-head experiments for the current default model:
# temporal context20, h64, dropout0.4, inverse class weighting.
#
# Defaults run seed42 over a small aux-weight sweep. After a promising weight is
# found, run with e.g. SEEDS="42 7 123" AUX_WEIGHTS="0.5".
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
INPUT_CSV="${INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
AUX_WEIGHTS=(${AUX_WEIGHTS:-0.2 0.5 1.0})

weight_name() {
  echo "$1" | tr -d "."
}

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

  for aux_weight in "${AUX_WEIGHTS[@]}"; do
    aux_tag="$(weight_name "${aux_weight}")"
    echo "=== Train LSTM inverse deep aux ${aux_weight}, seed ${seed} ==="
    PYTHONPATH=src python -m sse_sleep.train_lstm \
      --npz "${npz_path}" \
      --out-dir "${OUTPUT_ROOT}/lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_deepaux_w${aux_tag}_seed${seed}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --dropout "${DROPOUT}" \
      --class-weight-mode inverse \
      --aux-head deep \
      --aux-weight "${aux_weight}" \
      --aux-deep-pos-weight-mode balanced \
      --seed "${seed}"
  done
done

echo "=== Deep auxiliary-head jobs complete ==="
