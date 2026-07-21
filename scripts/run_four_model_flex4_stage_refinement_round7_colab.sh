#!/usr/bin/env bash
set -euo pipefail

# Seventh stage-split refinement around the current performance-only best:
#   classwise4_w_p0.78_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.75_c0.01_l0.20_rem_p0.00_c0.42_l0.12
#
# Round6 selected a Wake+REM-strong candidate inside the top 4M+4K tie band.
# This covers that candidate and the pure 4M+4K top neighborhood together.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_stage_refine_round7}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.78,0.79}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.02,0.04,0.06}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.78,0.79}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0.02,0.04}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.13,0.15,0.17}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.74,0.75,0.76,0.77}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0,0.01,0.02}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.18,0.20,0.22}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.42,0.44}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.12,0.13}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 stage-split refinement round7 complete ==="
