#!/usr/bin/env bash
set -euo pipefail

# Train the promising h128 diversity variants across 3 seeds, then run the
# performance-best fine-grid fusion refinement on those variants.
#
# This is the follow-up to the seed42 signal where h128_rem12/rem15/ls variants
# looked strong under the REM-primary=0 fixed-weight fusion family.

OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/SSE_outputs}"
CONTEXT_EPOCHS="${CONTEXT_EPOCHS:-20}"
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
SEEDS=(${SEEDS:-42 7 123})
VARIANTS=(${VARIANTS:-rem12 rem15 n3_12 ls003 ls005})
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_FUSION="${RUN_FUSION:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

path_for_seed() {
  local prefix="$1"
  local seed="$2"
  local suffix="$3"
  if [[ "${seed}" == "42" ]]; then
    echo "${OUTPUT_ROOT}/${prefix}${suffix}"
  else
    echo "${OUTPUT_ROOT}/${prefix}_seed${seed}${suffix}"
  fi
}

variant_args() {
  local variant="$1"
  case "${variant}" in
    rem12)
      echo "--selection-metric 5_macro_f1 --rem-weight-multiplier 1.2"
      ;;
    rem15)
      echo "--selection-metric 5_macro_f1 --rem-weight-multiplier 1.5"
      ;;
    n3_12)
      echo "--selection-metric 5_macro_f1 --n3-weight-multiplier 1.2"
      ;;
    ls003)
      echo "--selection-metric 5_macro_f1 --label-smoothing 0.03"
      ;;
    ls005)
      echo "--selection-metric 5_macro_f1 --label-smoothing 0.05"
      ;;
    *)
      echo "Unknown h128 diversity variant: ${variant}" >&2
      return 1
      ;;
  esac
}

out_dir_for_variant() {
  local variant="$1"
  local seed="$2"
  path_for_seed "lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_h128_${variant}" "${seed}" ""
}

if [[ "${RUN_TRAIN}" == "1" ]]; then
  for seed in "${SEEDS[@]}"; do
    npz_path="$(path_for_seed "dreamt_100hz_temporal_w20_lstm_context${CONTEXT_EPOCHS}" "${seed}" ".npz")"
    if [[ ! -f "${npz_path}" ]]; then
      echo "Missing full-w20 dataset NPZ: ${npz_path}" >&2
      exit 1
    fi

    for variant in "${VARIANTS[@]}"; do
      out_dir="$(out_dir_for_variant "${variant}" "${seed}")"
      if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/lstm_best.pt" ]]; then
        echo "=== Skip existing h128 ${variant}, seed ${seed}: ${out_dir} ==="
        continue
      fi
      args=($(variant_args "${variant}"))
      echo "=== Train h128 ${variant}, seed ${seed} ==="
      PYTHONPATH=src "${PYTHON_BIN}" -m sse_sleep.train_lstm \
        --npz "${npz_path}" \
        --out-dir "${out_dir}" \
        --hidden-size 128 \
        --num-layers 1 \
        --dropout 0.4 \
        --class-weight-mode inverse \
        --seed "${seed}" \
        "${args[@]}"
    done
  done
fi

if [[ "${RUN_FUSION}" == "1" ]]; then
  prefix_items=()
  for variant in "${VARIANTS[@]}"; do
    prefix_items+=("h128_${variant}=lstm_temporal_w20_context${CONTEXT_EPOCHS}_inverse_h128_${variant}")
  done

  echo "=== Refine h128 diversity variants as 3-model fixed-weight candidates ==="
  SEEDS="${SEEDS[*]}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  CONTEXT_EPOCHS="${CONTEXT_EPOCHS}" \
  HIDDEN_SIZE="${HIDDEN_SIZE}" \
  THIRD_PREFIX_CANDIDATES="${prefix_items[*]}" \
  bash scripts/run_performance_best_refinement_colab.sh
fi

echo "=== H128 diversity 3-seed refinement complete ==="
