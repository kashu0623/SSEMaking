#!/usr/bin/env bash
set -euo pipefail

# Kappa-first grid refinement around the round6 high-4K neighborhood.
# Current overall best:
#   classwise4_w_p0.78_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.75_c0.01_l0.20_rem_p0.00_c0.42_l0.12
#
# Round6 already found candidates around 4K 0.2574. This grid widens the
# high-kappa axes to probe for 4K 0.2575-0.2580 while preserving full records
# so best overall and best-by-4K can be compared from the summary JSON.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_kappa_refine}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.77,0.78}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.01,0.02,0.03,0.04}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.78,0.79,0.80}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0.02,0.03,0.04}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.13,0.15,0.17}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.75,0.76,0.77,0.78}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0,0.01,0.02}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.14,0.16,0.18,0.20}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.42,0.44,0.46}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.11,0.12,0.13}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 kappa refinement complete ==="
