#!/usr/bin/env bash
set -euo pipefail

# Performance-first fusion sweep.
#
# This keeps the current best 2-model fixed fusion as the baseline, then tries
# multiple third-model candidates in original + full-w20 + third-model fusion.
#
# It does not train models. Train candidate models first when needed, e.g.:
#   bash scripts/run_full_w20_capacity_colab.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS="${SEEDS:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_THIRD_VARIANTS="${RUN_THIRD_VARIANTS:-1}"
FUSION_SELECTION_POLICY="${FUSION_SELECTION_POLICY:-standard}"
FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA:-0.0}"
FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE:-0.005}"
FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE:-0.005}"
FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE:-0.010}"
FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE:-0.020}"

# Third variants follow the default full-w20 run naming:
#   lstm_temporal_w20_context20_h64_inverse_${THIRD_VARIANT}
THIRD_VARIANTS=(${THIRD_VARIANTS:-remaux_w05 remaux_w05_sel4combo})

# Prefix candidates are exact model directory prefixes. These are useful for
# capacity models or other runs that do not follow the default variant suffix.
THIRD_PREFIX_CANDIDATES=(${THIRD_PREFIX_CANDIDATES:-})

echo "=== Baseline dense 2-model fusion ==="
SEEDS="${SEEDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
HIDDEN_SIZE="${HIDDEN_SIZE}" \
RUN_TWO_MODEL=1 \
RUN_THREE_MODEL=0 \
FUSION_SELECTION_POLICY="${FUSION_SELECTION_POLICY}" \
FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA}" \
FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE}" \
FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE}" \
FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE}" \
FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE}" \
bash scripts/run_performance_fusion_colab.sh

if [[ "${RUN_THIRD_VARIANTS}" == "1" ]]; then
  for variant in "${THIRD_VARIANTS[@]}"; do
    echo "=== Aggressive 3-model fusion variant: ${variant} ==="
    SEEDS="${SEEDS}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
    HIDDEN_SIZE="${HIDDEN_SIZE}" \
    RUN_TWO_MODEL=0 \
    RUN_THREE_MODEL=1 \
    THIRD_VARIANT="${variant}" \
    FUSION_SELECTION_POLICY="${FUSION_SELECTION_POLICY}" \
    FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA}" \
    FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE}" \
    FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE}" \
    FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE}" \
    FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE}" \
    bash scripts/run_performance_fusion_colab.sh
  done
fi

for item in "${THIRD_PREFIX_CANDIDATES[@]}"; do
  # Accept either LABEL=PREFIX or just PREFIX. The label only controls report names.
  if [[ "${item}" == *"="* ]]; then
    label="${item%%=*}"
    prefix="${item#*=}"
  else
    prefix="${item}"
    label="$(basename "${prefix}")"
  fi
  echo "=== Aggressive 3-model fusion prefix: ${label} -> ${prefix} ==="
  SEEDS="${SEEDS}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
  HIDDEN_SIZE="${HIDDEN_SIZE}" \
  RUN_TWO_MODEL=0 \
  RUN_THREE_MODEL=1 \
  THIRD_VARIANT="${label}" \
  THIRD_MODEL_PREFIX="${prefix}" \
  FUSION_SELECTION_POLICY="${FUSION_SELECTION_POLICY}" \
  FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA}" \
  FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE}" \
  FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE}" \
  FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE}" \
  FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE}" \
  bash scripts/run_performance_fusion_colab.sh
done

echo "=== Aggressive fusion sweep complete ==="
