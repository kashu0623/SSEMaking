#!/usr/bin/env bash
set -euo pipefail

# Kappa-first round6 around the round5 pure top and selected overall best.
# Current overall best:
#   classwise4_w_p0.74_c0.06_l0.00_li_p0.80_c0.02_l0.13_d_p0.81_c0.00_l0.18_rem_p0.00_c0.44_l0.12
#
# Round5 pushed 4K to 0.25784-0.25792. This compact grid checks whether the
# ridge extends further while keeping Deep near the recovered 0.126-0.128 band.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_kappa_refine_round6}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.73,0.74,0.75}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.04,0.06}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.80,0.81,0.82}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0.02,0.04}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.13,0.15}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.81,0.82}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.18,0.20}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.42,0.44,0.46}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.12,0.13}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 kappa refinement round6 complete ==="
