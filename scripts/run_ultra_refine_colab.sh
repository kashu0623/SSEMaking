#!/usr/bin/env bash
set -euo pipefail

# Ultra-narrow fixed-weight refinement around the current performance-only best:
#   capacity_h128 classwise3_nonrem_p0.78_s0.10_rem_p0.00_s0.25
#
# This does not train models. It compares nearby h128/capacity third-model
# candidates across the same fixed 3-seed fusion grid and writes a global rank.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS="${SEEDS:-42 7 123}"
PYTHON_BIN="${PYTHON_BIN:-python}"

THIRD_PREFIX_CANDIDATES="${THIRD_PREFIX_CANDIDATES:-capacity_h128=lstm_temporal_w20_context20_inverse_capacity_h128 h128_ls003=lstm_temporal_w20_context20_inverse_h128_ls003 h128_ls005=lstm_temporal_w20_context20_inverse_h128_ls005 h128_rem12=lstm_temporal_w20_context20_inverse_h128_rem12}"

SEEDS="${SEEDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
HIDDEN_SIZE="${HIDDEN_SIZE}" \
THIRD_PREFIX_CANDIDATES="${THIRD_PREFIX_CANDIDATES}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_ultra_refine}" \
THREE_NON_REM_PRIMARY_ALPHAS="${THREE_NON_REM_PRIMARY_ALPHAS:-0.74,0.75,0.76,0.77,0.78,0.79}" \
THREE_NON_REM_SECONDARY_ALPHAS="${THREE_NON_REM_SECONDARY_ALPHAS:-0.08,0.10,0.12,0.14,0.15}" \
THREE_REM_PRIMARY_ALPHAS="${THREE_REM_PRIMARY_ALPHAS:-0}" \
THREE_REM_SECONDARY_ALPHAS="${THREE_REM_SECONDARY_ALPHAS:-0.23,0.25,0.27,0.30}" \
bash scripts/run_performance_best_refinement_colab.sh

echo "=== Ultra refine complete ==="
