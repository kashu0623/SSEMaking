#!/usr/bin/env bash
set -euo pipefail

# Fine-grid refinement around the current performance-best 3-model fixed fusion.
#
# This script does not train models. It reuses existing prediction exports or
# checkpoints, evaluates a narrow high-performance grid across seeds, and writes
# per-candidate summaries for fixed-weight comparison.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS="${SEEDS:-42 7 123}"
PYTHON_BIN="${PYTHON_BIN:-python}"
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_best_refine}"

# Include the current best h128/capacity candidate plus nearby diversity models
# that may add useful Deep/REM error-pattern differences.
THIRD_PREFIX_CANDIDATES="${THIRD_PREFIX_CANDIDATES:-capacity_h128=lstm_temporal_w20_context20_inverse_capacity_h128 h128_sel4combo=lstm_temporal_w20_context20_inverse_h128_sel4combo h128_rem12=lstm_temporal_w20_context20_inverse_h128_rem12 h128_rem15=lstm_temporal_w20_context20_inverse_h128_rem15 h128_n3_12=lstm_temporal_w20_context20_inverse_h128_n3_12 h128_ls003=lstm_temporal_w20_context20_inverse_h128_ls003 h128_ls005=lstm_temporal_w20_context20_inverse_h128_ls005}"

SEEDS="${SEEDS}" \
PYTHON_BIN="${PYTHON_BIN}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
HIDDEN_SIZE="${HIDDEN_SIZE}" \
RUN_THIRD_VARIANTS=0 \
THIRD_PREFIX_CANDIDATES="${THIRD_PREFIX_CANDIDATES}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX}" \
THREE_NON_REM_PRIMARY_ALPHAS="${THREE_NON_REM_PRIMARY_ALPHAS:-0.75,0.78,0.80,0.82,0.85,0.88,0.90}" \
THREE_NON_REM_SECONDARY_ALPHAS="${THREE_NON_REM_SECONDARY_ALPHAS:-0.10,0.12,0.15,0.18,0.20}" \
THREE_REM_PRIMARY_ALPHAS="${THREE_REM_PRIMARY_ALPHAS:-0}" \
THREE_REM_SECONDARY_ALPHAS="${THREE_REM_SECONDARY_ALPHAS:-0.12,0.15,0.18,0.20,0.22,0.25}" \
bash scripts/run_aggressive_fusion_colab.sh

echo "=== Performance-best fine-grid refinement complete ==="
