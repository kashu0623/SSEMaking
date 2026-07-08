#!/usr/bin/env bash
set -euo pipefail

# Build and train long-window temporal feature variants for the current default
# LSTM setup. Defaults compare isolated +10 and +20 epoch temporal windows:
#   w10: original 1/3 deltas + 3/5 rolling, plus 10
#   w20: original 1/3 deltas + 3/5 rolling, plus 20
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
INPUT_CSV="${INPUT_CSV:-${OUTPUT_ROOT}/dreamt_100hz_epoch_features.csv}"

CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
VARIANTS=(${VARIANTS:-10 20})

build_variant_csv() {
  local window="$1"
  local out_csv="${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal_w${window}.csv"
  local summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w${window}_features_summary.json"

  if [[ ! -f "${out_csv}" ]]; then
    echo "=== Add temporal features with extra ${window}-epoch window ==="
    PYTHONPATH=src python -m sse_sleep.add_temporal_features \
      --input-csv "${INPUT_CSV}" \
      --out-csv "${out_csv}" \
      --summary-out "${summary_path}" \
      --delta-windows 1 3 "${window}" \
      --rolling-window-list 3 5 "${window}"
  fi
}

for window in "${VARIANTS[@]}"; do
  build_variant_csv "${window}"

  for seed in "${SEEDS[@]}"; do
    if [[ "${seed}" == "42" ]]; then
      npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w${window}_lstm_context${CONTEXT_EPOCHS}.npz"
      summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w${window}_lstm_context${CONTEXT_EPOCHS}_summary.json"
      out_dir="${OUTPUT_ROOT}/lstm_temporal_w${window}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
    else
      npz_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w${window}_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
      summary_path="${OUTPUT_ROOT}/dreamt_100hz_temporal_w${window}_lstm_context${CONTEXT_EPOCHS}_seed${seed}_summary.json"
      out_dir="${OUTPUT_ROOT}/lstm_temporal_w${window}_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_seed${seed}"
    fi

    if [[ ! -f "${npz_path}" ]]; then
      echo "=== Build temporal w${window} context${CONTEXT_EPOCHS} NPZ for seed ${seed} ==="
      PYTHONPATH=src python -m sse_sleep.build_npz_dataset \
        --input-csv "${OUTPUT_ROOT}/dreamt_100hz_epoch_features_temporal_w${window}.csv" \
        --out "${npz_path}" \
        --summary-out "${summary_path}" \
        --context-epochs "${CONTEXT_EPOCHS}" \
        --seed "${seed}"
    fi

    echo "=== Train temporal w${window}, seed ${seed} ==="
    PYTHONPATH=src python -m sse_sleep.train_lstm \
      --npz "${npz_path}" \
      --out-dir "${out_dir}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --dropout "${DROPOUT}" \
      --class-weight-mode inverse \
      --seed "${seed}"
  done
done

echo "=== Long-window temporal experiments complete ==="
