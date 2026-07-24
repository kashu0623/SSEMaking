#!/usr/bin/env bash
set -euo pipefail

# No-training causal temporal audit of current ensemble N3 probabilities.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
OUTER_SEEDS=(${OUTER_SEEDS:-42 7 123})
RECALL_TARGETS="${RECALL_TARGETS:-0.50,0.70,0.80,0.90}"
PYTHON_BIN="${PYTHON_BIN:-python}"

base_predictions=()
primary_predictions=()
secondary_predictions=()
tertiary_predictions=()
seed_labels=()

for outer_seed in "${OUTER_SEEDS[@]}"; do
  ensemble_dir="${OUTPUT_ROOT}/same_split_init_ensemble_outer${outer_seed}"
  base="${ensemble_dir}/original_predictions.npz"
  primary="${ensemble_dir}/full_w20_predictions.npz"
  secondary="${ensemble_dir}/capacity_h128_predictions.npz"
  tertiary="${ensemble_dir}/h128_ls003_predictions.npz"

  for predictions in "${base}" "${primary}" "${secondary}" "${tertiary}"; do
    [[ -f "${predictions}" ]] || {
      echo "Missing ensemble prediction file: ${predictions}" >&2
      echo "Run scripts/run_four_model_same_split_init_ensemble_colab.sh first." >&2
      exit 1
    }
  done

  base_predictions+=("${base}")
  primary_predictions+=("${primary}")
  secondary_predictions+=("${secondary}")
  tertiary_predictions+=("${tertiary}")
  seed_labels+=("${outer_seed}")
done

summary_json="${OUTPUT_ROOT}/fusion4_same_split_init_ensemble_deep_temporal_probability_audit_context${CONTEXT_EPOCHS}_h${HIDDEN_SIZE}_summary.json"

PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.evaluate_deep_temporal_probability_audit \
  --base-predictions "${base_predictions[@]}" \
  --primary-predictions "${primary_predictions[@]}" \
  --secondary-predictions "${secondary_predictions[@]}" \
  --tertiary-predictions "${tertiary_predictions[@]}" \
  --seed-labels "${seed_labels[@]}" \
  --recall-targets "${RECALL_TARGETS}" \
  --out-json "${summary_json}"

echo "=== Deep temporal probability audit complete: ${summary_json} ==="
