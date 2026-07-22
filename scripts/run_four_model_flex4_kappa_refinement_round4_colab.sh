#!/usr/bin/env bash
set -euo pipefail

# Kappa-first round4 around the round3 pure top and selected overall best.
# Current overall best:
#   classwise4_w_p0.76_c0.04_l0.00_li_p0.79_c0.02_l0.15_d_p0.79_c0.00_l0.16_rem_p0.00_c0.42_l0.12
#
# Round3 improved both the selection score and Wake+REM. This grid probes the
# nearby edge directions while keeping the candidate count small.

SEEDS="${SEEDS:-42 7 123}" \
FUSION_REPORT_SUFFIX="${FUSION_REPORT_SUFFIX:-_flex4_kappa_refine_round4}" \
WAKE_PRIMARY_ALPHAS="${WAKE_PRIMARY_ALPHAS:-0.75,0.76,0.77}" \
WAKE_SECONDARY_ALPHAS="${WAKE_SECONDARY_ALPHAS:-0.04,0.06}" \
WAKE_TERTIARY_ALPHAS="${WAKE_TERTIARY_ALPHAS:-0}" \
LIGHT_DEEP_PRIMARY_ALPHAS="${LIGHT_DEEP_PRIMARY_ALPHAS:-0.79,0.80}" \
LIGHT_DEEP_SECONDARY_ALPHAS="${LIGHT_DEEP_SECONDARY_ALPHAS:-0.02,0.04}" \
LIGHT_DEEP_TERTIARY_ALPHAS="${LIGHT_DEEP_TERTIARY_ALPHAS:-0.13,0.15}" \
DEEP_PRIMARY_ALPHAS="${DEEP_PRIMARY_ALPHAS:-0.77,0.79,0.80}" \
DEEP_SECONDARY_ALPHAS="${DEEP_SECONDARY_ALPHAS:-0}" \
DEEP_TERTIARY_ALPHAS="${DEEP_TERTIARY_ALPHAS:-0.16,0.18}" \
REM_PRIMARY_ALPHAS="${REM_PRIMARY_ALPHAS:-0}" \
REM_SECONDARY_ALPHAS="${REM_SECONDARY_ALPHAS:-0.42,0.44,0.46}" \
REM_TERTIARY_ALPHAS="${REM_TERTIARY_ALPHAS:-0.11,0.12}" \
bash scripts/run_four_model_flexible_fusion_colab.sh

echo "=== Four-model flex4 kappa refinement round4 complete ==="
