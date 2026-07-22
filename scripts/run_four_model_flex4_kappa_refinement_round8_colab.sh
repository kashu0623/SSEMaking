#!/usr/bin/env bash
set -euo pipefail

# Kappa-first round8 around the round7 pure top and selected overall best.
# Current overall best:
#   classwise4_w_p0.72_c0.06_l0.00_li_p0.80_c0.02_l0.15_d_p0.82_c0.00_l0.18_rem_p0.00_c0.42_l0.13
#
# Round7 preserved 4K > 0.2580 and improved Wake+REM. This grid keeps the
# search compact while checking the lower wake-primary edge and nearby REM.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_kappa_refine_round8}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.71,0.72,0.73}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.06,0.08}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.80,0.82,0.83}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0.02}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.13,0.15}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.81,0.82,0.83}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.18,0.20}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.40,0.42}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.13,0.14}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 kappa refinement round8 complete ==="
