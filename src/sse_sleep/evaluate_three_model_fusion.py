"""Evaluate class-wise probability fusion across three sleep-stage models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_prediction_fusion import (
    candidate_record,
    evaluate_probs,
    load_split,
    parse_float_list,
    validate_alignment,
)
from .labels import STAGE5_NAMES, STAGE5_TO_ID


def three_model_classwise_fusion(
    base_probs: np.ndarray,
    primary_probs: np.ndarray,
    secondary_probs: np.ndarray,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
) -> np.ndarray:
    base_alphas = 1.0 - primary_alphas - secondary_alphas
    if np.any(base_alphas < -1e-6):
        raise ValueError("Class-wise fusion weights must sum to <= 1.0")
    base_alphas = np.maximum(base_alphas, 0.0)
    fused = (
        base_alphas.reshape(1, -1) * base_probs
        + primary_alphas.reshape(1, -1) * primary_probs
        + secondary_alphas.reshape(1, -1) * secondary_probs
    )
    row_sums = fused.sum(axis=1, keepdims=True)
    return np.divide(fused, row_sums, out=np.zeros_like(fused), where=row_sums > 0)


def build_class_weights(
    non_rem_primary: float,
    non_rem_secondary: float,
    rem_primary: float,
    rem_secondary: float,
) -> tuple[np.ndarray, np.ndarray]:
    primary_alphas = np.full(len(STAGE5_NAMES), float(non_rem_primary), dtype=np.float32)
    secondary_alphas = np.full(len(STAGE5_NAMES), float(non_rem_secondary), dtype=np.float32)
    rem_index = STAGE5_TO_ID["REM"]
    primary_alphas[rem_index] = float(rem_primary)
    secondary_alphas[rem_index] = float(rem_secondary)
    if np.any(primary_alphas + secondary_alphas > 1.0 + 1e-6):
        raise ValueError("primary + secondary weights must be <= 1.0 for every class")
    return primary_alphas, secondary_alphas


def evaluate_three_model_fusion(
    base_predictions: Path,
    primary_predictions: Path,
    secondary_predictions: Path,
    out_json: Path,
    non_rem_primary_alphas: Sequence[float],
    non_rem_secondary_alphas: Sequence[float],
    rem_primary_alphas: Sequence[float],
    rem_secondary_alphas: Sequence[float],
    selection_metric: str,
) -> dict[str, Any]:
    base_val = load_split(base_predictions, "val")
    base_test = load_split(base_predictions, "test")
    primary_val = load_split(primary_predictions, "val")
    primary_test = load_split(primary_predictions, "test")
    secondary_val = load_split(secondary_predictions, "val")
    secondary_test = load_split(secondary_predictions, "test")
    validate_alignment(base_val, primary_val, "val")
    validate_alignment(base_test, primary_test, "test")
    validate_alignment(base_val, secondary_val, "val")
    validate_alignment(base_test, secondary_test, "test")

    records: list[dict[str, Any]] = []
    for name, split_val, split_test in (
        ("base_original_temporal", base_val, base_test),
        ("primary_full_w20", primary_val, primary_test),
        ("secondary_model", secondary_val, secondary_test),
    ):
        records.append(
            candidate_record(
                name=name,
                kind="baseline",
                params={},
                val_metrics=evaluate_probs(split_val["y_true"], split_val["probs"]),
                test_metrics=evaluate_probs(split_test["y_true"], split_test["probs"]),
                selection_metric=selection_metric,
            )
        )

    for non_rem_primary in non_rem_primary_alphas:
        for non_rem_secondary in non_rem_secondary_alphas:
            if non_rem_primary + non_rem_secondary > 1.0 + 1e-6:
                continue
            for rem_primary in rem_primary_alphas:
                for rem_secondary in rem_secondary_alphas:
                    if rem_primary + rem_secondary > 1.0 + 1e-6:
                        continue
                    primary_alphas, secondary_alphas = build_class_weights(
                        non_rem_primary=non_rem_primary,
                        non_rem_secondary=non_rem_secondary,
                        rem_primary=rem_primary,
                        rem_secondary=rem_secondary,
                    )
                    fused_val = three_model_classwise_fusion(
                        base_val["probs"],
                        primary_val["probs"],
                        secondary_val["probs"],
                        primary_alphas,
                        secondary_alphas,
                    )
                    fused_test = three_model_classwise_fusion(
                        base_test["probs"],
                        primary_test["probs"],
                        secondary_test["probs"],
                        primary_alphas,
                        secondary_alphas,
                    )
                    records.append(
                        candidate_record(
                            name=(
                                f"classwise3_nonrem_p{non_rem_primary:.2f}_s{non_rem_secondary:.2f}"
                                f"_rem_p{rem_primary:.2f}_s{rem_secondary:.2f}"
                            ),
                            kind="classwise3_nonrem_rem",
                            params={
                                "class_primary_alphas": {
                                    name: float(primary_alphas[idx]) for idx, name in enumerate(STAGE5_NAMES)
                                },
                                "class_secondary_alphas": {
                                    name: float(secondary_alphas[idx]) for idx, name in enumerate(STAGE5_NAMES)
                                },
                            },
                            val_metrics=evaluate_probs(base_val["y_true"], fused_val),
                            test_metrics=evaluate_probs(base_test["y_true"], fused_test),
                            selection_metric=selection_metric,
                        )
                    )

    best = max(records, key=lambda item: item["selection_score"])
    report = {
        "base_predictions": str(base_predictions),
        "primary_predictions": str(primary_predictions),
        "secondary_predictions": str(secondary_predictions),
        "base_model_role": "original_temporal",
        "primary_model_role": "full_w20",
        "secondary_model_role": "extra_model",
        "selection_metric": selection_metric,
        "stage5_names": list(STAGE5_NAMES),
        "record_count": len(records),
        "best_by_validation": best,
        "records": records,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def print_top(report: dict[str, Any], limit: int) -> None:
    records = sorted(report["records"], key=lambda item: item["selection_score"], reverse=True)
    print("| rank | name | val score | 4 Macro | 4 Kappa | Wake | Light | Deep | REM |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for rank, record in enumerate(records[:limit], start=1):
        summary = record["test"]["summary"]
        print(
            f"| {rank} | {record['name']} | {record['selection_score']:.4f} | "
            f"{summary['4_macro_f1']:.4f} | {summary['4_kappa']:.4f} | "
            f"{summary['wake_f1']:.4f} | {summary['light_f1']:.4f} | "
            f"{summary['deep_f1']:.4f} | {summary['rem_f1']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate original + full-w20 + third-model fusion.")
    parser.add_argument("--base-predictions", type=Path, required=True, help="Prediction NPZ for original temporal.")
    parser.add_argument("--primary-predictions", type=Path, required=True, help="Prediction NPZ for full w20.")
    parser.add_argument("--secondary-predictions", type=Path, required=True, help="Prediction NPZ for an extra model.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--non-rem-primary-alphas", default="0.75,0.80,0.85,0.90,0.95,1.00")
    parser.add_argument("--non-rem-secondary-alphas", default="0,0.05,0.10,0.15,0.20")
    parser.add_argument("--rem-primary-alphas", default="0,0.10,0.20,0.30,0.40")
    parser.add_argument("--rem-secondary-alphas", default="0,0.05,0.10,0.15,0.20")
    parser.add_argument(
        "--selection-metric",
        choices=(
            "5_macro_f1",
            "4_macro_f1",
            "5_kappa",
            "4_kappa",
            "5_macro_f1_plus_4_kappa",
            "4_macro_f1_plus_4_kappa",
        ),
        default="4_macro_f1_plus_4_kappa",
    )
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    report = evaluate_three_model_fusion(
        base_predictions=args.base_predictions,
        primary_predictions=args.primary_predictions,
        secondary_predictions=args.secondary_predictions,
        out_json=args.out_json,
        non_rem_primary_alphas=parse_float_list(args.non_rem_primary_alphas, default=[]),
        non_rem_secondary_alphas=parse_float_list(args.non_rem_secondary_alphas, default=[]),
        rem_primary_alphas=parse_float_list(args.rem_primary_alphas, default=[]),
        rem_secondary_alphas=parse_float_list(args.rem_secondary_alphas, default=[]),
        selection_metric=args.selection_metric,
    )
    print_top(report, limit=args.top)


if __name__ == "__main__":
    main()
