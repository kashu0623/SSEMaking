#!/usr/bin/env bash
set -euo pipefail

# Fourth stage-split refinement around the current performance-only best:
#   classwise4_w_p0.77_c0.04_l0.00_li_p0.77_c0.02_l0.17_d_p0.76_c0.03_l0.20_rem_p0.00_c0.40_l0.08
#
# Round3 improved 4M+4K and Wake+REM, with Wake capacity low and REM weights
# still on the expanded grid edge. This searches one more local expansion.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_stage_refine_round4}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.77,0.78}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.02,0.04,0.06}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.77,0.78}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0.02,0.04}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.15,0.17,0.19}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.75,0.76,0.77}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0.02,0.03,0.04}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.18,0.20,0.22}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.38,0.40,0.42}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.08,0.09,0.10}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 stage-split refinement round4 complete ==="
