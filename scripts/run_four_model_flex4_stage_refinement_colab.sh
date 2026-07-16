#!/usr/bin/env bash
set -euo pipefail

# Split N1/N2 Light and N3 Deep weights around the current flex4_refine best:
#   classwise4_w_p0.77_c0.10_l0.00_ld_p0.76_c0.02_l0.17_rem_p0.00_c0.34_l0.04
#
# The base runner treats Light as N1/N2 and Deep as N3 when DEEP_* overrides
# are supplied. Without DEEP_* overrides it preserves the earlier grouped
# Light/Deep behavior.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_stage_refine}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.77,0.78}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.08,0.10}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.75,0.76,0.77}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0,0.02}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.15,0.17}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.74,0.76}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0,0.02}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.18,0.20}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.34,0.36}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.04,0.05}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 stage-split refinement complete ==="
