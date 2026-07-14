#!/usr/bin/env bash
set -euo pipefail

# Distill the fixed class-wise fusion teacher into a single full-w20 student.
#
# Teacher:
#   original temporal + full w20 fixed class-wise fusion
#   Wake/N1/N2/N3: 90% full w20 + 10% original temporal
#   REM:           20% full w20 + 80% original temporal
#
# Default seed is 42 for quick filtering. Expand promising candidates with:
#   SEEDS="42 7 123" VARIANTS="distill_w05" bash scripts/run_distillation_colab.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.4}"
SEEDS=(${SEEDS:-42})
VARIANTS=(${VARIANTS:-distill_w02 distill_w05 distill_w10})
PYTHON_BIN="${PYTHON_BIN:-python}"
NON_REM_ALPHA="${NON_REM_ALPHA:-0.9}"
REM_ALPHA="${REM_ALPHA:-0.2}"

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

distill_weight_for_variant() {
  local variant="$1"
  case "${variant}" in
    distill_w02)
      echo "0.2"
      ;;
    distill_w05)
      echo "0.5"
      ;;
    distill_w10)
      echo "1.0"
      ;;
    *)
      echo "Unknown distillation variant: ${variant}" >&2
      return 1
      ;;
  esac
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

for seed in "${SEEDS[@]}"; do
  original_model_prefix="lstm_temporal_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  w20_model_prefix="lstm_temporal_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_inverse"
  original_npz="$(path_for_seed "dreamt_100hz_temporal_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  w20_npz="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
  original_predictions="$(model_dir_for_seed "${original_model_prefix}" "${seed}")/lstm_predictions.npz"
  w20_predictions="$(model_dir_for_seed "${w20_model_prefix}" "${seed}")/lstm_predictions.npz"
  original_model_dir="$(model_dir_for_seed "${original_model_prefix}" "${seed}")"
  w20_model_dir="$(model_dir_for_seed "${w20_model_prefix}" "${seed}")"

  ensure_prediction_probs "${original_predictions}" "${original_npz}" "${original_model_dir}"
  ensure_prediction_probs "${w20_predictions}" "${w20_npz}" "${w20_model_dir}"

  teacher_prefix="fusion_teacher_original_temporal_full_w20_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_nonrem090_rem020"
  teacher_npz="$(path_for_seed "${teacher_prefix}" "${seed}" ".npz")"
  teacher_json="$(path_for_seed "${teacher_prefix}" "${seed}" ".json")"
  echo "=== Build fixed fusion teacher probabilities, seed ${seed} ==="
  PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.build_fusion_teacher_probs \
    --base-predictions "${original_predictions}" \
    --candidate-predictions "${w20_predictions}" \
    --out-npz "${teacher_npz}" \
    --out-json "${teacher_json}" \
    --non-rem-alpha "${NON_REM_ALPHA}" \
    --rem-alpha "${REM_ALPHA}"

  for variant in "${VARIANTS[@]}"; do
    distill_weight="$(distill_weight_for_variant "${variant}")"
    out_dir="$(out_dir_for_variant "${variant}" "${seed}")"
    echo "=== Train distillation ${variant}, seed ${seed} ==="
    PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
      --npz "${w20_npz}" \
      --out-dir "${out_dir}" \
      --hidden-size "${HIDDEN_SIZE}" \
      --dropout "${DROPOUT}" \
      --class-weight-mode inverse \
      --teacher-probs-npz "${teacher_npz}" \
      --distill-weight "${distill_weight}" \
      --seed "${seed}"
  done
done

echo "=== Distillation experiments complete ==="

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
