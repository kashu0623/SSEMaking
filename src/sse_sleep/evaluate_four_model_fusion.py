"""Evaluate grouped class-wise probability fusion across four sleep-stage models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_prediction_fusion import (
    candidate_record,
    classwise_fusion,
    evaluate_probs,
    load_split,
    parse_float_list,
    select_best_record,
    validate_alignment,
)
from .evaluate_three_model_fusion import three_model_classwise_fusion
from .labels import STAGE5_NAMES, STAGE5_TO_ID


def four_model_classwise_fusion(
    base_probs: np.ndarray,
    primary_probs: np.ndarray,
    secondary_probs: np.ndarray,
    tertiary_probs: np.ndarray,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> np.ndarray:
    base_alphas = 1.0 - primary_alphas - secondary_alphas - tertiary_alphas
    if np.any(base_alphas < -1e-6):
        raise ValueError("Class-wise fusion weights must sum to <= 1.0")
    base_alphas = np.maximum(base_alphas, 0.0)
    fused = (
        base_alphas.reshape(1, -1) * base_probs
        + primary_alphas.reshape(1, -1) * primary_probs
        + secondary_alphas.reshape(1, -1) * secondary_probs
        + tertiary_alphas.reshape(1, -1) * tertiary_probs
    )
    row_sums = fused.sum(axis=1, keepdims=True)
    return np.divide(fused, row_sums, out=np.zeros_like(fused), where=row_sums > 0)


def build_grouped_class_weights(
    wake_primary: float,
    wake_secondary: float,
    wake_tertiary: float,
    light_deep_primary: float,
    light_deep_secondary: float,
    light_deep_tertiary: float,
    deep_primary: float,
    deep_secondary: float,
    deep_tertiary: float,
    rem_primary: float,
    rem_secondary: float,
    rem_tertiary: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary_alphas = np.full(len(STAGE5_NAMES), float(light_deep_primary), dtype=np.float32)
    secondary_alphas = np.full(len(STAGE5_NAMES), float(light_deep_secondary), dtype=np.float32)
    tertiary_alphas = np.full(len(STAGE5_NAMES), float(light_deep_tertiary), dtype=np.float32)

    wake_index = STAGE5_TO_ID["Wake"]
    deep_index = STAGE5_TO_ID["N3"]
    rem_index = STAGE5_TO_ID["REM"]
    primary_alphas[wake_index] = float(wake_primary)
    secondary_alphas[wake_index] = float(wake_secondary)
    tertiary_alphas[wake_index] = float(wake_tertiary)
    primary_alphas[deep_index] = float(deep_primary)
    secondary_alphas[deep_index] = float(deep_secondary)
    tertiary_alphas[deep_index] = float(deep_tertiary)
    primary_alphas[rem_index] = float(rem_primary)
    secondary_alphas[rem_index] = float(rem_secondary)
    tertiary_alphas[rem_index] = float(rem_tertiary)

    if np.any(primary_alphas + secondary_alphas + tertiary_alphas > 1.0 + 1e-6):
        raise ValueError("primary + secondary + tertiary weights must be <= 1.0 for every class")
    return primary_alphas, secondary_alphas, tertiary_alphas


def reference_record(
    name: str,
    kind: str,
    params: dict[str, Any],
    base_val: dict[str, np.ndarray],
    base_test: dict[str, np.ndarray],
    fused_val: np.ndarray,
    fused_test: np.ndarray,
    selection_metric: str,
) -> dict[str, Any]:
    return candidate_record(
        name=name,
        kind=kind,
        params=params,
        val_metrics=evaluate_probs(base_val["y_true"], fused_val),
        test_metrics=evaluate_probs(base_test["y_true"], fused_test),
        selection_metric=selection_metric,
    )


def add_reference_records(
    records: list[dict[str, Any]],
    base_val: dict[str, np.ndarray],
    base_test: dict[str, np.ndarray],
    primary_val: dict[str, np.ndarray],
    primary_test: dict[str, np.ndarray],
    secondary_val: dict[str, np.ndarray],
    secondary_test: dict[str, np.ndarray],
    tertiary_val: dict[str, np.ndarray],
    tertiary_test: dict[str, np.ndarray],
    selection_metric: str,
) -> dict[str, Any]:
    fixed_class_alphas = np.full(len(STAGE5_NAMES), 0.90, dtype=np.float32)
    fixed_class_alphas[STAGE5_TO_ID["REM"]] = 0.20
    fixed_reference = reference_record(
        name="fixed_classwise_nonrem0.90_rem0.20",
        kind="fixed_reference",
        params={
            "class_alphas": {
                name: float(fixed_class_alphas[idx]) for idx, name in enumerate(STAGE5_NAMES)
            }
        },
        base_val=base_val,
        base_test=base_test,
        fused_val=classwise_fusion(base_val["probs"], primary_val["probs"], fixed_class_alphas),
        fused_test=classwise_fusion(base_test["probs"], primary_test["probs"], fixed_class_alphas),
        selection_metric=selection_metric,
    )
    records.append(fixed_reference)

    current_primary, current_secondary = build_three_model_weights(
        non_rem_primary=0.78,
        non_rem_secondary=0.12,
        rem_primary=0.00,
        rem_secondary=0.30,
    )
    records.append(
        reference_record(
            name="current_best_3model_capacity_h128_p0.78_s0.12_rem0.00_s0.30",
            kind="reference_3model",
            params={
                "secondary_model": "capacity_h128",
                "class_primary_alphas": class_alpha_dict(current_primary),
                "class_secondary_alphas": class_alpha_dict(current_secondary),
            },
            base_val=base_val,
            base_test=base_test,
            fused_val=three_model_classwise_fusion(
                base_val["probs"],
                primary_val["probs"],
                secondary_val["probs"],
                current_primary,
                current_secondary,
            ),
            fused_test=three_model_classwise_fusion(
                base_test["probs"],
                primary_test["probs"],
                secondary_test["probs"],
                current_primary,
                current_secondary,
            ),
            selection_metric=selection_metric,
        )
    )

    ls003_primary, ls003_secondary = build_three_model_weights(
        non_rem_primary=0.74,
        non_rem_secondary=0.15,
        rem_primary=0.00,
        rem_secondary=0.30,
    )
    records.append(
        reference_record(
            name="pure_top_3model_h128_ls003_p0.74_s0.15_rem0.00_s0.30",
            kind="reference_3model",
            params={
                "secondary_model": "h128_ls003",
                "class_primary_alphas": class_alpha_dict(ls003_primary),
                "class_secondary_alphas": class_alpha_dict(ls003_secondary),
            },
            base_val=base_val,
            base_test=base_test,
            fused_val=three_model_classwise_fusion(
                base_val["probs"],
                primary_val["probs"],
                tertiary_val["probs"],
                ls003_primary,
                ls003_secondary,
            ),
            fused_test=three_model_classwise_fusion(
                base_test["probs"],
                primary_test["probs"],
                tertiary_test["probs"],
                ls003_primary,
                ls003_secondary,
            ),
            selection_metric=selection_metric,
        )
    )
    return fixed_reference


def build_three_model_weights(
    non_rem_primary: float,
    non_rem_secondary: float,
    rem_primary: float,
    rem_secondary: float,
) -> tuple[np.ndarray, np.ndarray]:
    primary_alphas = np.full(len(STAGE5_NAMES), float(non_rem_primary), dtype=np.float32)
    secondary_alphas = np.full(len(STAGE5_NAMES), float(non_rem_secondary), dtype=np.float32)
    primary_alphas[STAGE5_TO_ID["REM"]] = float(rem_primary)
    secondary_alphas[STAGE5_TO_ID["REM"]] = float(rem_secondary)
    return primary_alphas, secondary_alphas


def class_alpha_dict(alphas: np.ndarray) -> dict[str, float]:
    return {name: float(alphas[idx]) for idx, name in enumerate(STAGE5_NAMES)}


def evaluate_four_model_fusion(
    base_predictions: Path,
    primary_predictions: Path,
    secondary_predictions: Path,
    tertiary_predictions: Path,
    out_json: Path,
    wake_primary_alphas: Sequence[float],
    wake_secondary_alphas: Sequence[float],
    wake_tertiary_alphas: Sequence[float],
    light_deep_primary_alphas: Sequence[float],
    light_deep_secondary_alphas: Sequence[float],
    light_deep_tertiary_alphas: Sequence[float],
    deep_primary_alphas: Sequence[float] | None,
    deep_secondary_alphas: Sequence[float] | None,
    deep_tertiary_alphas: Sequence[float] | None,
    rem_primary_alphas: Sequence[float],
    rem_secondary_alphas: Sequence[float],
    rem_tertiary_alphas: Sequence[float],
    selection_metric: str,
    selection_policy: str,
    fixed_min_score_delta: float,
    fixed_rem_tolerance: float,
    fixed_light_tolerance: float,
    fixed_wake_tolerance: float,
    fixed_deep_tolerance: float,
) -> dict[str, Any]:
    base_val = load_split(base_predictions, "val")
    base_test = load_split(base_predictions, "test")
    primary_val = load_split(primary_predictions, "val")
    primary_test = load_split(primary_predictions, "test")
    secondary_val = load_split(secondary_predictions, "val")
    secondary_test = load_split(secondary_predictions, "test")
    tertiary_val = load_split(tertiary_predictions, "val")
    tertiary_test = load_split(tertiary_predictions, "test")
    for candidate_val, candidate_test in (
        (primary_val, primary_test),
        (secondary_val, secondary_test),
        (tertiary_val, tertiary_test),
    ):
        validate_alignment(base_val, candidate_val, "val")
        validate_alignment(base_test, candidate_test, "test")

    records: list[dict[str, Any]] = []
    for name, split_val, split_test in (
        ("base_original_temporal", base_val, base_test),
        ("primary_full_w20", primary_val, primary_test),
        ("secondary_capacity_h128", secondary_val, secondary_test),
        ("tertiary_h128_ls003", tertiary_val, tertiary_test),
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

    fixed_reference = add_reference_records(
        records=records,
        base_val=base_val,
        base_test=base_test,
        primary_val=primary_val,
        primary_test=primary_test,
        secondary_val=secondary_val,
        secondary_test=secondary_test,
        tertiary_val=tertiary_val,
        tertiary_test=tertiary_test,
        selection_metric=selection_metric,
    )

    for wake_primary in wake_primary_alphas:
        for wake_secondary in wake_secondary_alphas:
            for wake_tertiary in wake_tertiary_alphas:
                if wake_primary + wake_secondary + wake_tertiary > 1.0 + 1e-6:
                    continue
                for light_deep_primary in light_deep_primary_alphas:
                    for light_deep_secondary in light_deep_secondary_alphas:
                        for light_deep_tertiary in light_deep_tertiary_alphas:
                            if light_deep_primary + light_deep_secondary + light_deep_tertiary > 1.0 + 1e-6:
                                continue
                            deep_primary_values = (
                                [light_deep_primary] if deep_primary_alphas is None else deep_primary_alphas
                            )
                            deep_secondary_values = (
                                [light_deep_secondary] if deep_secondary_alphas is None else deep_secondary_alphas
                            )
                            deep_tertiary_values = (
                                [light_deep_tertiary] if deep_tertiary_alphas is None else deep_tertiary_alphas
                            )
                            for deep_primary in deep_primary_values:
                                for deep_secondary in deep_secondary_values:
                                    for deep_tertiary in deep_tertiary_values:
                                        if deep_primary + deep_secondary + deep_tertiary > 1.0 + 1e-6:
                                            continue
                                        for rem_primary in rem_primary_alphas:
                                            for rem_secondary in rem_secondary_alphas:
                                                for rem_tertiary in rem_tertiary_alphas:
                                                    if rem_primary + rem_secondary + rem_tertiary > 1.0 + 1e-6:
                                                        continue
                                                    primary_alphas, secondary_alphas, tertiary_alphas = (
                                                        build_grouped_class_weights(
                                                            wake_primary=wake_primary,
                                                            wake_secondary=wake_secondary,
                                                            wake_tertiary=wake_tertiary,
                                                            light_deep_primary=light_deep_primary,
                                                            light_deep_secondary=light_deep_secondary,
                                                            light_deep_tertiary=light_deep_tertiary,
                                                            deep_primary=deep_primary,
                                                            deep_secondary=deep_secondary,
                                                            deep_tertiary=deep_tertiary,
                                                            rem_primary=rem_primary,
                                                            rem_secondary=rem_secondary,
                                                            rem_tertiary=rem_tertiary,
                                                        )
                                                    )
                                                    fused_val = four_model_classwise_fusion(
                                                        base_val["probs"],
                                                        primary_val["probs"],
                                                        secondary_val["probs"],
                                                        tertiary_val["probs"],
                                                        primary_alphas,
                                                        secondary_alphas,
                                                        tertiary_alphas,
                                                    )
                                                    fused_test = four_model_classwise_fusion(
                                                        base_test["probs"],
                                                        primary_test["probs"],
                                                        secondary_test["probs"],
                                                        tertiary_test["probs"],
                                                        primary_alphas,
                                                        secondary_alphas,
                                                        tertiary_alphas,
                                                    )
                                                    records.append(
                                                        candidate_record(
                                                            name=(
                                                                f"classwise4_w_p{wake_primary:.2f}"
                                                                f"_c{wake_secondary:.2f}_l{wake_tertiary:.2f}"
                                                                f"_li_p{light_deep_primary:.2f}"
                                                                f"_c{light_deep_secondary:.2f}"
                                                                f"_l{light_deep_tertiary:.2f}"
                                                                f"_d_p{deep_primary:.2f}_c{deep_secondary:.2f}"
                                                                f"_l{deep_tertiary:.2f}"
                                                                f"_rem_p{rem_primary:.2f}_c{rem_secondary:.2f}"
                                                                f"_l{rem_tertiary:.2f}"
                                                            ),
                                                            kind="classwise4_grouped",
                                                            params={
                                                                "model_roles": {
                                                                    "base": "original_temporal",
                                                                    "primary": "full_w20",
                                                                    "secondary": "capacity_h128",
                                                                    "tertiary": "h128_ls003",
                                                                },
                                                                "class_primary_alphas": class_alpha_dict(primary_alphas),
                                                                "class_secondary_alphas": class_alpha_dict(
                                                                    secondary_alphas
                                                                ),
                                                                "class_tertiary_alphas": class_alpha_dict(
                                                                    tertiary_alphas
                                                                ),
                                                            },
                                                            val_metrics=evaluate_probs(base_val["y_true"], fused_val),
                                                            test_metrics=evaluate_probs(base_test["y_true"], fused_test),
                                                            selection_metric=selection_metric,
                                                        )
                                                    )

    best, selection_details = select_best_record(
        records=records,
        selection_policy=selection_policy,
        fixed_reference=fixed_reference,
        min_score_delta=fixed_min_score_delta,
        rem_tolerance=fixed_rem_tolerance,
        light_tolerance=fixed_light_tolerance,
        wake_tolerance=fixed_wake_tolerance,
        deep_tolerance=fixed_deep_tolerance,
    )
    report = {
        "base_predictions": str(base_predictions),
        "primary_predictions": str(primary_predictions),
        "secondary_predictions": str(secondary_predictions),
        "tertiary_predictions": str(tertiary_predictions),
        "base_model_role": "original_temporal",
        "primary_model_role": "full_w20",
        "secondary_model_role": "capacity_h128",
        "tertiary_model_role": "h128_ls003",
        "selection_metric": selection_metric,
        "selection_policy": selection_policy,
        "selection_details": selection_details,
        "fixed_reference": fixed_reference,
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
    parser = argparse.ArgumentParser(description="Evaluate grouped class-wise fusion across four models.")
    parser.add_argument("--base-predictions", type=Path, required=True, help="Prediction NPZ for original temporal.")
    parser.add_argument("--primary-predictions", type=Path, required=True, help="Prediction NPZ for full w20.")
    parser.add_argument("--secondary-predictions", type=Path, required=True, help="Prediction NPZ for capacity_h128.")
    parser.add_argument("--tertiary-predictions", type=Path, required=True, help="Prediction NPZ for h128_ls003.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--wake-primary-alphas", default="0.78,0.79")
    parser.add_argument("--wake-secondary-alphas", default="0.10,0.12,0.14")
    parser.add_argument("--wake-tertiary-alphas", default="0,0.03")
    parser.add_argument("--light-deep-primary-alphas", default="0.72,0.74,0.76")
    parser.add_argument("--light-deep-secondary-alphas", default="0,0.03")
    parser.add_argument("--light-deep-tertiary-alphas", default="0.12,0.15,0.18")
    parser.add_argument("--deep-primary-alphas", default=None)
    parser.add_argument("--deep-secondary-alphas", default=None)
    parser.add_argument("--deep-tertiary-alphas", default=None)
    parser.add_argument("--rem-primary-alphas", default="0")
    parser.add_argument("--rem-secondary-alphas", default="0.25,0.30,0.32")
    parser.add_argument("--rem-tertiary-alphas", default="0,0.03")
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
    parser.add_argument(
        "--selection-policy",
        choices=("standard", "fixed_aware_constrained"),
        default="standard",
    )
    parser.add_argument("--fixed-min-score-delta", type=float, default=0.0)
    parser.add_argument("--fixed-rem-tolerance", type=float, default=0.005)
    parser.add_argument("--fixed-light-tolerance", type=float, default=0.005)
    parser.add_argument("--fixed-wake-tolerance", type=float, default=0.010)
    parser.add_argument("--fixed-deep-tolerance", type=float, default=0.020)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    report = evaluate_four_model_fusion(
        base_predictions=args.base_predictions,
        primary_predictions=args.primary_predictions,
        secondary_predictions=args.secondary_predictions,
        tertiary_predictions=args.tertiary_predictions,
        out_json=args.out_json,
        wake_primary_alphas=parse_float_list(args.wake_primary_alphas, default=[]),
        wake_secondary_alphas=parse_float_list(args.wake_secondary_alphas, default=[]),
        wake_tertiary_alphas=parse_float_list(args.wake_tertiary_alphas, default=[]),
        light_deep_primary_alphas=parse_float_list(args.light_deep_primary_alphas, default=[]),
        light_deep_secondary_alphas=parse_float_list(args.light_deep_secondary_alphas, default=[]),
        light_deep_tertiary_alphas=parse_float_list(args.light_deep_tertiary_alphas, default=[]),
        deep_primary_alphas=(
            None if args.deep_primary_alphas is None else parse_float_list(args.deep_primary_alphas, default=[])
        ),
        deep_secondary_alphas=(
            None if args.deep_secondary_alphas is None else parse_float_list(args.deep_secondary_alphas, default=[])
        ),
        deep_tertiary_alphas=(
            None if args.deep_tertiary_alphas is None else parse_float_list(args.deep_tertiary_alphas, default=[])
        ),
        rem_primary_alphas=parse_float_list(args.rem_primary_alphas, default=[]),
        rem_secondary_alphas=parse_float_list(args.rem_secondary_alphas, default=[]),
        rem_tertiary_alphas=parse_float_list(args.rem_tertiary_alphas, default=[]),
        selection_metric=args.selection_metric,
        selection_policy=args.selection_policy,
        fixed_min_score_delta=args.fixed_min_score_delta,
        fixed_rem_tolerance=args.fixed_rem_tolerance,
        fixed_light_tolerance=args.fixed_light_tolerance,
        fixed_wake_tolerance=args.fixed_wake_tolerance,
        fixed_deep_tolerance=args.fixed_deep_tolerance,
    )
    print_top(report, limit=args.top)


if __name__ == "__main__":
    main()
