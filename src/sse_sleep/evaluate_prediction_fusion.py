"""Evaluate probability fusion between two saved sleep-stage prediction files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .labels import STAGE5_NAMES, STAGE5_TO_ID
from .metrics import evaluate_5_and_4


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def parse_float_list(text: str | None, default: Sequence[float]) -> list[float]:
    if text is None:
        return list(default)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def load_split(path: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        required = [f"{split}_y_true", f"{split}_probs"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing required arrays in {path}: {', '.join(missing)}")
        arrays = {
            "y_true": data[f"{split}_y_true"].astype(np.int64),
            "probs": data[f"{split}_probs"].astype(np.float32),
        }
        for optional_key in (f"{split}_subject_ids", f"{split}_epoch_indices"):
            if optional_key in data.files:
                arrays[optional_key.removeprefix(f"{split}_")] = data[optional_key]
        return arrays


def validate_alignment(base: dict[str, np.ndarray], candidate: dict[str, np.ndarray], split: str) -> None:
    if base["y_true"].shape != candidate["y_true"].shape:
        raise ValueError(f"{split} y_true shapes differ: {base['y_true'].shape} vs {candidate['y_true'].shape}")
    if not np.array_equal(base["y_true"], candidate["y_true"]):
        raise ValueError(f"{split} y_true arrays differ; cannot fuse unaligned prediction files")
    if base["probs"].shape != candidate["probs"].shape:
        raise ValueError(f"{split} probability shapes differ: {base['probs'].shape} vs {candidate['probs'].shape}")
    for key in ("subject_ids", "epoch_indices"):
        if key in base and key in candidate and not np.array_equal(base[key], candidate[key]):
            raise ValueError(f"{split} {key} arrays differ; cannot fuse unaligned prediction files")


def metric_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name == "5_macro_f1":
        return float(metrics["5_class"]["macro_f1"])
    if metric_name == "4_macro_f1":
        return float(metrics["4_class"]["macro_f1"])
    if metric_name == "5_kappa":
        return float(metrics["5_class"]["cohen_kappa"])
    if metric_name == "4_kappa":
        return float(metrics["4_class"]["cohen_kappa"])
    if metric_name == "5_macro_f1_plus_4_kappa":
        return float(metrics["5_class"]["macro_f1"]) + float(metrics["4_class"]["cohen_kappa"])
    if metric_name == "4_macro_f1_plus_4_kappa":
        return float(metrics["4_class"]["macro_f1"]) + float(metrics["4_class"]["cohen_kappa"])
    raise ValueError(f"Unknown selection metric: {metric_name}")


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    y_pred = probs.argmax(axis=1).astype(np.int64)
    return json_ready(evaluate_5_and_4(y_true.tolist(), y_pred.tolist()))


def scalar_fusion(base_probs: np.ndarray, candidate_probs: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * candidate_probs + (1.0 - alpha) * base_probs


def classwise_fusion(base_probs: np.ndarray, candidate_probs: np.ndarray, class_alphas: np.ndarray) -> np.ndarray:
    fused = class_alphas.reshape(1, -1) * candidate_probs + (1.0 - class_alphas.reshape(1, -1)) * base_probs
    row_sums = fused.sum(axis=1, keepdims=True)
    return np.divide(fused, row_sums, out=np.zeros_like(fused), where=row_sums > 0)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "5_macro_f1": float(metrics["5_class"]["macro_f1"]),
        "5_kappa": float(metrics["5_class"]["cohen_kappa"]),
        "4_macro_f1": float(metrics["4_class"]["macro_f1"]),
        "4_kappa": float(metrics["4_class"]["cohen_kappa"]),
        "wake_f1": float(metrics["4_class"]["class_wise"]["Wake"]["f1"]),
        "light_f1": float(metrics["4_class"]["class_wise"]["Light"]["f1"]),
        "deep_f1": float(metrics["4_class"]["class_wise"]["Deep"]["f1"]),
        "n3_f1": float(metrics["4_class"]["class_wise"]["Deep"]["f1"]),
        "rem_f1": float(metrics["4_class"]["class_wise"]["REM"]["f1"]),
    }


def candidate_record(
    name: str,
    kind: str,
    params: dict[str, Any],
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    selection_metric: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "params": params,
        "selection_metric": selection_metric,
        "selection_score": metric_value(val_metrics, selection_metric),
        "val": {
            "metrics": val_metrics,
            "summary": summarize_metrics(val_metrics),
        },
        "test": {
            "metrics": test_metrics,
            "summary": summarize_metrics(test_metrics),
        },
    }


def fixed_aware_passes(
    record: dict[str, Any],
    fixed_reference: dict[str, Any],
    min_score_delta: float,
    rem_tolerance: float,
    light_tolerance: float,
    wake_tolerance: float,
    deep_tolerance: float,
) -> bool:
    summary = record["val"]["summary"]
    fixed_summary = fixed_reference["val"]["summary"]
    return (
        record["selection_score"] >= fixed_reference["selection_score"] + min_score_delta
        and summary["rem_f1"] >= fixed_summary["rem_f1"] - rem_tolerance
        and summary["light_f1"] >= fixed_summary["light_f1"] - light_tolerance
        and summary["wake_f1"] >= fixed_summary["wake_f1"] - wake_tolerance
        and summary["deep_f1"] >= fixed_summary["deep_f1"] - deep_tolerance
    )


def select_best_record(
    records: Sequence[dict[str, Any]],
    selection_policy: str,
    fixed_reference: dict[str, Any] | None,
    min_score_delta: float,
    rem_tolerance: float,
    light_tolerance: float,
    wake_tolerance: float,
    deep_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if selection_policy == "standard":
        best = max(records, key=lambda item: item["selection_score"])
        return best, {
            "policy": selection_policy,
            "eligible_count": len(records),
            "fallback_to_fixed": False,
        }

    if fixed_reference is None:
        best = max(records, key=lambda item: item["selection_score"])
        return best, {
            "policy": selection_policy,
            "eligible_count": len(records),
            "fallback_to_fixed": False,
            "warning": "fixed_reference_missing; fell back to standard selection",
        }

    eligible = [
        record
        for record in records
        if record["name"] != fixed_reference["name"]
        if fixed_aware_passes(
            record=record,
            fixed_reference=fixed_reference,
            min_score_delta=min_score_delta,
            rem_tolerance=rem_tolerance,
            light_tolerance=light_tolerance,
            wake_tolerance=wake_tolerance,
            deep_tolerance=deep_tolerance,
        )
    ]
    if not eligible:
        return fixed_reference, {
            "policy": selection_policy,
            "eligible_count": 0,
            "fallback_to_fixed": True,
        }
    best = max(eligible, key=lambda item: item["selection_score"])
    return best, {
        "policy": selection_policy,
        "eligible_count": len(eligible),
        "fallback_to_fixed": best["name"] == fixed_reference["name"],
    }


def evaluate_prediction_fusion(
    base_predictions: Path,
    candidate_predictions: Path,
    out_json: Path,
    scalar_alphas: Sequence[float],
    classwise_non_rem_alphas: Sequence[float],
    classwise_rem_alphas: Sequence[float],
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
    candidate_val = load_split(candidate_predictions, "val")
    candidate_test = load_split(candidate_predictions, "test")
    validate_alignment(base_val, candidate_val, "val")
    validate_alignment(base_test, candidate_test, "test")

    records: list[dict[str, Any]] = []
    base_val_metrics = evaluate_probs(base_val["y_true"], base_val["probs"])
    base_test_metrics = evaluate_probs(base_test["y_true"], base_test["probs"])
    candidate_val_metrics = evaluate_probs(candidate_val["y_true"], candidate_val["probs"])
    candidate_test_metrics = evaluate_probs(candidate_test["y_true"], candidate_test["probs"])
    records.append(candidate_record("base_alpha0.00", "baseline", {"alpha": 0.0}, base_val_metrics, base_test_metrics, selection_metric))
    records.append(candidate_record("candidate_alpha1.00", "baseline", {"alpha": 1.0}, candidate_val_metrics, candidate_test_metrics, selection_metric))

    for alpha in scalar_alphas:
        fused_val = scalar_fusion(base_val["probs"], candidate_val["probs"], alpha)
        fused_test = scalar_fusion(base_test["probs"], candidate_test["probs"], alpha)
        records.append(
            candidate_record(
                name=f"scalar_alpha{alpha:.2f}",
                kind="scalar",
                params={"alpha": alpha},
                val_metrics=evaluate_probs(base_val["y_true"], fused_val),
                test_metrics=evaluate_probs(base_test["y_true"], fused_test),
                selection_metric=selection_metric,
            )
        )

    rem_index = STAGE5_TO_ID["REM"]
    for non_rem_alpha in classwise_non_rem_alphas:
        for rem_alpha in classwise_rem_alphas:
            class_alphas = np.full(len(STAGE5_NAMES), float(non_rem_alpha), dtype=np.float32)
            class_alphas[rem_index] = float(rem_alpha)
            fused_val = classwise_fusion(base_val["probs"], candidate_val["probs"], class_alphas)
            fused_test = classwise_fusion(base_test["probs"], candidate_test["probs"], class_alphas)
            records.append(
                candidate_record(
                    name=f"classwise_nonrem{non_rem_alpha:.2f}_rem{rem_alpha:.2f}",
                    kind="classwise_nonrem_rem",
                    params={
                        "class_alphas": {
                            name: float(class_alphas[idx])
                            for idx, name in enumerate(STAGE5_NAMES)
                        }
                    },
                    val_metrics=evaluate_probs(base_val["y_true"], fused_val),
                    test_metrics=evaluate_probs(base_test["y_true"], fused_test),
                    selection_metric=selection_metric,
                )
            )

    fixed_reference = next((record for record in records if record["name"] == "classwise_nonrem0.90_rem0.20"), None)
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
        "candidate_predictions": str(candidate_predictions),
        "base_model_role": "alpha_0_original_temporal",
        "candidate_model_role": "alpha_1_full_w20",
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
    if report.get("selection_policy") != "standard":
        best = report["best_by_validation"]
        details = report.get("selection_details", {})
        print(
            f"\nselected_by_{report['selection_policy']}: {best['name']} "
            f"(eligible={details.get('eligible_count')}, fallback_to_fixed={details.get('fallback_to_fixed')})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate original-temporal + full-w20 prediction fusion.")
    parser.add_argument("--base-predictions", type=Path, required=True, help="Prediction NPZ for original temporal model.")
    parser.add_argument("--candidate-predictions", type=Path, required=True, help="Prediction NPZ for full w20 model.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--scalar-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--classwise-non-rem-alphas", default="0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--classwise-rem-alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
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
        help="standard=max validation score. fixed_aware_constrained filters candidates against fixed class-wise fusion on validation.",
    )
    parser.add_argument("--fixed-min-score-delta", type=float, default=0.0)
    parser.add_argument("--fixed-rem-tolerance", type=float, default=0.005)
    parser.add_argument("--fixed-light-tolerance", type=float, default=0.005)
    parser.add_argument("--fixed-wake-tolerance", type=float, default=0.010)
    parser.add_argument("--fixed-deep-tolerance", type=float, default=0.020)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    report = evaluate_prediction_fusion(
        base_predictions=args.base_predictions,
        candidate_predictions=args.candidate_predictions,
        out_json=args.out_json,
        scalar_alphas=parse_float_list(args.scalar_alphas, default=[]),
        classwise_non_rem_alphas=parse_float_list(args.classwise_non_rem_alphas, default=[]),
        classwise_rem_alphas=parse_float_list(args.classwise_rem_alphas, default=[]),
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
