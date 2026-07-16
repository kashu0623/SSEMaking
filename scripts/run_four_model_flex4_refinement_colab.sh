#!/usr/bin/env bash
set -euo pipefail

# Narrow second-pass refinement around the current 4-model grouped best:
#   classwise4_w_p0.78_c0.10_l0.00_ld_p0.74_c0.00_l0.15_rem_p0.00_c0.32_l0.03
#
# This reuses the four-model flexible fusion runner and only overrides the
# weight grid plus report suffix.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_refine}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.77,0.78,0.79}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.08,0.10,0.12}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0,0.02}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.74,0.75,0.76}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0,0.02,0.03}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.15,0.17,0.18}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.30,0.32,0.34}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.02,0.03,0.04}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 second-pass refinement complete ==="
