#!/usr/bin/env bash
set -euo pipefail

# Re-evaluate existing fusion candidates with the fixed-aware constrained selector.
# This does not train models. It assumes predictions/checkpoints already exist.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS="${SEEDS:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"

THIRD_PREFIX_CANDIDATES="${THIRD_PREFIX_CANDIDATES:-h128_sel4combo=lstm_temporal_w20_context20_inverse_h128_sel4combo h128_rem12=lstm_temporal_w20_context20_inverse_h128_rem12 h128_rem15=lstm_temporal_w20_context20_inverse_h128_rem15 h128_n3_12=lstm_temporal_w20_context20_inverse_h128_n3_12 h128_ls003=lstm_temporal_w20_context20_inverse_h128_ls003 h128_ls005=lstm_temporal_w20_context20_inverse_h128_ls005 capacity_h128=lstm_temporal_w20_context20_inverse_capacity_h128}"
THIRD_VARIANTS="${THIRD_VARIANTS:-remaux_w05 remaux_w05_sel4combo}"
FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA:--0.03}"
FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE:-0.005}"
FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE:-0.005}"
FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE:-0.010}"
FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE:-0.020}"

SEEDS="${SEEDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
HIDDEN_SIZE="${HIDDEN_SIZE}" \
RUN_THIRD_VARIANTS=1 \
THIRD_VARIANTS="${THIRD_VARIANTS}" \
THIRD_PREFIX_CANDIDATES="${THIRD_PREFIX_CANDIDATES}" \
FUSION_SELECTION_POLICY=fixed_aware_constrained \
FIXED_MIN_SCORE_DELTA="${FIXED_MIN_SCORE_DELTA}" \
FIXED_REM_TOLERANCE="${FIXED_REM_TOLERANCE}" \
FIXED_LIGHT_TOLERANCE="${FIXED_LIGHT_TOLERANCE}" \
FIXED_WAKE_TOLERANCE="${FIXED_WAKE_TOLERANCE}" \
FIXED_DEEP_TOLERANCE="${FIXED_DEEP_TOLERANCE}" \
bash scripts/run_aggressive_fusion_colab.sh

echo "=== Fixed-aware fusion re-evaluation complete ==="
