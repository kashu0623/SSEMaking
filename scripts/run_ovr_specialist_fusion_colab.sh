#!/usr/bin/env bash
set -euo pipefail

# Train a full 5-stage one-vs-rest specialist bank and evaluate specialist fusion.
#
# Each specialist uses the current full-w20 feature/context input and predicts:
#   target stage vs all other stages
#
# Defaults run seed42 for quick filtering. Expand with:
#   SEEDS="42 7 123" bash scripts/run_ovr_specialist_fusion_colab.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
STAGES=(${STAGES:-Wake N1 N2 N3 REM})
PYTHON_BIN="${PYTHON_BIN:-python}"
SPECIALIST_SELECTION_METRIC="${SPECIALIST_SELECTION_METRIC:-positive_f1}"
SPECIALIST_CLASS_WEIGHT_MODE="${SPECIALIST_CLASS_WEIGHT_MODE:-inverse}"
SPECIALIST_TRAIN_SAMPLER="${SPECIALIST_TRAIN_SAMPLER:-none}"
FUSION_SELECTION_METRIC="${FUSION_SELECTION_METRIC:-4_macro_f1_plus_4_kappa}"
NON_REM_ALPHA="${NON_REM_ALPHA:-0.9}"
REM_ALPHA="${REM_ALPHA:-0.2}"

stage_slug() {
  local stage="$1"
  case "${stage}" in
    Wake) echo "wake" ;;
    N1) echo "n1" ;;
    N2) echo "n2" ;;
    N3) echo "n3" ;;
    REM) echo "rem" ;;
    *)
      echo "Unknown stage: ${stage}" >&2
      return 1
      ;;
  esac
}

path_for_seed() {
  local prefix="$1"
  local seed="$2"
  local suffix="$3"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${prefix}${suffix}"
  else
    echo "${OUTPUT_ROOT}/${prefix}_seed${seed}${suffix}"
  fi
}

model_dir_for_seed() {
  local model_prefix="$1"
  local seed="$2"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${model_prefix}"
  else
    echo "${OUTPUT_ROOT}/${model_prefix}_seed${seed}"
  fi
}

prediction_has_train_probs() {
  local predictions="$1"
  "${PYTHON_BIN}" -c 'import numpy as np, sys
path = sys.argv[1]
with np.load(path, allow_pickle=True) as data:
    ok = all(key in data.files for key in ("train_probs", "val_probs", "test_probs"))
sys.exit(0 if ok else 1)' "${predictions}"
}

ensure_prediction_probs() {
  local predictions="$1"
  local npz_path="$2"
  local model_dir="$3"
  local checkpoint="${model_dir}/lstm_best.pt"

  if [[ -f "${predictions}" ]] && prediction_has_train_probs "${predictions}"; then
    return 0
  fi
  if [[ ! -f "${npz_path}" ]]; then
    echo "Missing dataset NPZ needed to export probabilities: ${npz_path}" >&2
    return 1
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint needed to export probabilities: ${checkpoint}" >&2
    return 1
  fi

  echo "=== Export train/val/test prediction probabilities: ${model_dir} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.export_lstm_predictions \
    --npz "${npz_path}" \
    --checkpoint "${checkpoint}" \
    --out "${predictions}"
}

specialist_out_dir() {
  local stage="$1"
  local seed="$2"
  local slug
  slug="$(stage_slug "${stage}")"
  local base="${OUTPUT_ROOT}/lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse_ovr_${slug}"
  if [[ "${seed}" == "42" ]]; then
    echo "${base}"
  else
    echo "${base}_seed${seed}"
  fi
}

fusion_reports=()
for seed in "${SEEDS[@]}"; do
  w20_npz="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  if [[ ! -f "${w20_npz}" ]]; then
    echo "Missing full-w20 dataset NPZ: ${w20_npz}" >&2
    exit 1
  fi

  specialist_prediction_paths=()
  for stage in "${STAGES[@]}"; do
    out_dir="$(specialist_out_dir "${stage}" "${seed}")"
    echo "=== Train one-vs-rest specialist ${stage}, seed ${seed} ==="
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_binary_specialist \
      --npz "${w20_npz}" \
      --out-dir "${out_dir}" \
      --target-stage "${stage}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --dropout "${DROPOUT}" \
      --class-weight-mode "${SPECIALIST_CLASS_WEIGHT_MODE}" \
      --train-sampler "${SPECIALIST_TRAIN_SAMPLER}" \
      --selection-metric "${SPECIALIST_SELECTION_METRIC}" \
      --seed "${seed}"
    specialist_prediction_paths+=("${out_dir}/specialist_predictions.npz")
  done

  original_model_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_model_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  original_npz="$(path_for_seed "dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  original_predictions="$(model_dir_for_seed "${original_model_prefix}" "${seed}")/lstm_predictions.npz"
  w20_predictions="$(model_dir_for_seed "${w20_model_prefix}" "${seed}")/lstm_predictions.npz"
  original_model_dir="$(model_dir_for_seed "${original_model_prefix}" "${seed}")"
  w20_model_dir="$(model_dir_for_seed "${w20_model_prefix}" "${seed}")"

  ensure_prediction_probs "${original_predictions}" "${original_npz}" "${original_model_dir}"
  ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "${w20_model_dir}"

  teacher_prefix="fusion_teacher_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_nonrem090_rem020"
  teacher_npz="$(path_for_seed "${teacher_prefix}" "${seed}" ".npz")"
  teacher_json="$(path_for_seed "${teacher_prefix}" "${seed}" ".json")"
  echo "=== Build fixed fusion comparison probabilities, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.build_fusion_teacher_probs \
    --base-predictions "${original_predictions}" \
    --candidate-predictions "${w20_predictions}" \
    --out-npz "${teacher_npz}" \
    --out-json "${teacher_json}" \
    --non-rem-alpha "${NON_REM_ALPHA}" \
    --rem-alpha "${REM_ALPHA}"

  if [[ "${seed}" == "42" ]]; then
    fusion_json="${OUTPUT_ROOT}/ovr_specialist_fusion_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}.json"
  else
    fusion_json="${OUTPUT_ROOT}/ovr_specialist_fusion_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_seed${seed}.json"
  fi
  echo "=== Evaluate one-vs-rest specialist fusion, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_specialist_fusion \
    --specialist-predictions "${specialist_prediction_paths[@]}" \
    --base-predictions "original_temporal=${original_predictions}" \
    --base-predictions "full_w20=${w20_predictions}" \
    --base-predictions "fixed_fusion=${teacher_npz}" \
    --out-json "${fusion_json}" \
    --selection-metric "${FUSION_SELECTION_METRIC}" \
    --top 20
  fusion_reports+=("${fusion_json}")
done

if [[ "${#fusion_reports[@]}" -gt 1 ]]; then
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.summarize_prediction_fusion \
    --reports "${fusion_reports[@]}" \
    --out-json "${OUTPUT_ROOT}/ovr_specialist_fusion_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"
fi

echo "=== One-vs-rest specialist fusion experiments complete ==="
