#!/usr/bin/env bash
set -euo pipefail

# Train the four current model roles directly on Wake/Light/Deep/REM.
# N1/N2 are merged before loss computation. This is the new training baseline
# before expanding to same-split multi-initialization ensembles.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"

npz_for_seed() {
  local prefix="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${prefix}_lstm_context${CONTEXT_EPOCHS}.npz"
  else
    echo "${OUTPUT_ROOT}/${prefix}_lstm_context${CONTEXT_EPOCHS}_seed${seed}.npz"
  fi
}

out_dir() {
  local role="$1"
  local seed="$2"
  echo "${OUTPUT_ROOT}/direct4_${role}_outer${seed}"
}

train_role() {
  local role="$1"
  local npz_path="$2"
  local seed="$3"
  local destination
  destination="$(out_dir "${role}" "${seed}")"
  if [[ "${RUN_TRAIN}" != "1" || -f "${destination}/lstm4_predictions.npz" ]]; then
    echo "=== Reuse direct4 ${role}, outer ${seed} ==="
    return 0
  fi
  common=(--npz "${npz_path}" --out-dir "${destination}" --dropout 0.4 --class-weight-mode inverse --seed "${seed}")
  case "${role}" in
    original|full_w20)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm_4class "${common[@]}" --hidden-size 64
      ;;
    capacity_h128)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm_4class "${common[@]}" --hidden-size 128
      ;;
    h128_ls003)
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm_4class "${common[@]}" --hidden-size 128 --label-smoothing 0.03
      ;;
    *) echo "Unknown role: ${role}" >&2; return 1 ;;
  esac
}

original_predictions=()
full_predictions=()
capacity_predictions=()
ls003_predictions=()
seed_labels=()
for seed in "${OUTER_SEEDS[@]}"; do
  original_npz="$(npz_for_seed dreamt_100hz_temporal "${seed}")"
  w20_npz="$(npz_for_seed dreamt_100hz_temporal_w20 "${seed}")"
  [[ -f "${original_npz}" ]] || { echo "Missing NPZ: ${original_npz}" >&2; exit 1; }
  [[ -f "${w20_npz}" ]] || { echo "Missing NPZ: ${w20_npz}" >&2; exit 1; }
  train_role original "${original_npz}" "${seed}"
  train_role full_w20 "${w20_npz}" "${seed}"
  train_role capacity_h128 "${w20_npz}" "${seed}"
  train_role h128_ls003 "${w20_npz}" "${seed}"
  original_predictions+=("$(out_dir original "${seed}")/lstm4_predictions.npz")
  full_predictions+=("$(out_dir full_w20 "${seed}")/lstm4_predictions.npz")
  capacity_predictions+=("$(out_dir capacity_h128 "${seed}")/lstm4_predictions.npz")
  ls003_predictions+=("$(out_dir h128_ls003 "${seed}")/lstm4_predictions.npz")
  seed_labels+=("${seed}")
done

summary_json="${OUTPUT_ROOT}/fusion4_direct_4class_context${CONTEXT_EPOCHS}_summary.json"
PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_four_model_4class_fusion \
  --original-4class-predictions "${original_predictions[@]}" \
  --full-w20-4class-predictions "${full_predictions[@]}" \
  --capacity-h128-4class-predictions "${capacity_predictions[@]}" \
  --h128-ls003-4class-predictions "${ls003_predictions[@]}" \
  --seed-labels "${seed_labels[@]}" \
  --out-json "${summary_json}"
echo "=== Direct four-class experiment complete: ${summary_json} ==="
