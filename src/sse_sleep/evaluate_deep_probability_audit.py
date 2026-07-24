"""Audit N3 probability ranking and validation-selected alarm veto thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from .evaluate_four_model_fusion import build_grouped_class_weights, four_model_classwise_fusion
from .evaluate_prediction_fusion import load_split, validate_alignment
from .labels import STAGE5_TO_ID


MODEL_ROLES = ("original_temporal", "full_w20", "capacity_h128", "h128_ls003")
FUSION_ROLE = "current_best_fusion"
SOURCES = (*MODEL_ROLES, FUSION_ROLE)
METRIC_FIELDS = (
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "f1",
    "balanced_accuracy",
    "predicted_positive_rate",
)


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def binary_targets(y_true_5: np.ndarray) -> np.ndarray:
    return (y_true_5.astype(np.int64) == STAGE5_TO_ID["N3"]).astype(np.int64)


def binary_metrics_from_counts(
    threshold: float,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
) -> dict[str, Any]:
    support = tp + fp + tn + fn
    positive_support = tp + fn
    negative_support = tn + fp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    return {
        "threshold": float(threshold),
        "support": support,
        "positive_support": positive_support,
        "negative_support": negative_support,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1.0 - specificity,
        "f1": safe_divide(2.0 * precision * recall, precision + recall),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "prevalence": safe_divide(positive_support, support),
        "predicted_positive_rate": safe_divide(tp + fp, support),
    }


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = y_score >= threshold
    positive = y_true == 1
    negative = ~positive
    return binary_metrics_from_counts(
        threshold=threshold,
        tp=int((y_pred & positive).sum()),
        fp=int((y_pred & negative).sum()),
        tn=int((~y_pred & negative).sum()),
        fn=int((~y_pred & positive).sum()),
    )


def binary_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return binary_metrics(y_true, y_pred.astype(np.float32), threshold=0.5)


def threshold_curve(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    order = np.argsort(y_score, kind="mergesort")[::-1]
    sorted_score = y_score[order]
    sorted_true = y_true[order]
    distinct_indices = np.flatnonzero(np.diff(sorted_score))
    threshold_indices = np.concatenate(
        [distinct_indices, np.asarray([sorted_true.shape[0] - 1], dtype=np.int64)]
    )
    true_positives = np.cumsum(sorted_true, dtype=np.int64)[threshold_indices]
    false_positives = 1 + threshold_indices - true_positives
    positive_count = int(true_positives[-1])
    negative_count = int(false_positives[-1])
    if not positive_count or not negative_count:
        raise ValueError("Deep probability audit requires both N3 and non-N3 samples")
    return {
        "thresholds": sorted_score[threshold_indices],
        "true_positives": true_positives,
        "false_positives": false_positives,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    curve = threshold_curve(y_true, y_score)
    true_positives = curve["true_positives"]
    false_positives = curve["false_positives"]
    positive_count = int(curve["positive_count"])
    negative_count = int(curve["negative_count"])
    true_positive_rate = np.concatenate(
        [np.asarray([0.0]), true_positives / positive_count]
    )
    false_positive_rate = np.concatenate(
        [np.asarray([0.0]), false_positives / negative_count]
    )
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count
    average_precision = np.sum(np.diff(np.concatenate([np.asarray([0.0]), recall])) * precision)
    roc_auc = np.sum(
        np.diff(false_positive_rate)
        * (true_positive_rate[1:] + true_positive_rate[:-1])
        / 2.0
    )
    return {
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
    }


def score_distribution(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    quantiles = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    result: dict[str, Any] = {}
    for label, mask in (("non_deep", y_true == 0), ("deep", y_true == 1)):
        values = y_score[mask]
        result[label] = {
            "count": int(values.shape[0]),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "quantiles": {
                f"{quantile:.2f}": float(np.quantile(values, quantile))
                for quantile in quantiles
            },
        }
    return result


def threshold_metric_candidates(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> list[dict[str, Any]]:
    curve = threshold_curve(y_true, y_score)
    positive_count = int(curve["positive_count"])
    negative_count = int(curve["negative_count"])
    return [
        binary_metrics_from_counts(
            threshold=float(threshold),
            tp=int(tp),
            fp=int(fp),
            tn=negative_count - int(fp),
            fn=positive_count - int(tp),
        )
        for threshold, tp, fp in zip(
            curve["thresholds"],
            curve["true_positives"],
            curve["false_positives"],
            strict=True,
        )
    ]


def select_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_targets: Sequence[float],
) -> dict[str, dict[str, Any]]:
    candidates = threshold_metric_candidates(y_true, y_score)
    best_f1 = max(
        candidates,
        key=lambda item: (item["f1"], item["precision"], item["specificity"], item["threshold"]),
    )
    selected = {
        "max_f1": {
            "selection": "maximum_validation_f1",
            "target_recall": None,
            "threshold": best_f1["threshold"],
            "val_metrics": best_f1,
        }
    }
    for target in recall_targets:
        if not 0.0 < target <= 1.0:
            raise ValueError(f"Recall target must be in (0, 1]: {target}")
        eligible = [item for item in candidates if item["recall"] + 1e-12 >= target]
        best = max(
            eligible,
            key=lambda item: (item["precision"], item["specificity"], item["threshold"]),
        )
        selected[f"recall_{int(round(target * 100)):02d}"] = {
            "selection": "maximum_validation_precision_at_recall_floor",
            "target_recall": float(target),
            "threshold": best["threshold"],
            "val_metrics": best,
        }
    return selected


def load_source_probabilities(
    prediction_paths: dict[str, Path],
    split: str,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    loaded = {role: load_split(path, split) for role, path in prediction_paths.items()}
    base = loaded[MODEL_ROLES[0]]
    for role in MODEL_ROLES[1:]:
        validate_alignment(base, loaded[role], split)
    source_probs = {role: loaded[role]["probs"] for role in MODEL_ROLES}
    source_probs[FUSION_ROLE] = four_model_classwise_fusion(
        loaded["original_temporal"]["probs"],
        loaded["full_w20"]["probs"],
        loaded["capacity_h128"]["probs"],
        loaded["h128_ls003"]["probs"],
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    return base["y_true"], source_probs


def audit_seed(
    seed_label: str,
    prediction_paths: dict[str, Path],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
    recall_targets: Sequence[float],
) -> dict[str, Any]:
    val_y_5, val_probs = load_source_probabilities(
        prediction_paths,
        "val",
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    test_y_5, test_probs = load_source_probabilities(
        prediction_paths,
        "test",
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    val_y = binary_targets(val_y_5)
    test_y = binary_targets(test_y_5)
    deep_index = STAGE5_TO_ID["N3"]
    source_reports: dict[str, Any] = {}
    for source in SOURCES:
        val_score = val_probs[source][:, deep_index]
        test_score = test_probs[source][:, deep_index]
        selected = select_thresholds(val_y, val_score, recall_targets)
        for policy in selected.values():
            policy["test_metrics"] = binary_metrics(test_y, test_score, policy["threshold"])
        source_reports[source] = {
            "val": {
                "ranking": ranking_metrics(val_y, val_score),
                "argmax": binary_prediction_metrics(
                    val_y,
                    val_probs[source].argmax(axis=1) == deep_index,
                ),
                "score_distribution": score_distribution(val_y, val_score),
            },
            "test": {
                "ranking": ranking_metrics(test_y, test_score),
                "argmax": binary_prediction_metrics(
                    test_y,
                    test_probs[source].argmax(axis=1) == deep_index,
                ),
                "score_distribution": score_distribution(test_y, test_score),
            },
            "selected_thresholds": selected,
        }
    return {
        "seed": seed_label,
        "prediction_paths": {role: str(path) for role, path in prediction_paths.items()},
        "sources": source_reports,
    }


def mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(float(value) for value in values),
        "std": pstdev(float(value) for value in values),
    }


def aggregate_reports(seed_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for source in SOURCES:
        source_aggregate: dict[str, Any] = {
            "ranking": {},
            "argmax": {},
            "selected_thresholds": {},
        }
        for split in ("val", "test"):
            source_aggregate["ranking"][split] = {
                field: mean_std(
                    [report["sources"][source][split]["ranking"][field] for report in seed_reports]
                )
                for field in ("roc_auc", "average_precision")
            }
            source_aggregate["argmax"][split] = {
                field: mean_std(
                    [report["sources"][source][split]["argmax"][field] for report in seed_reports]
                )
                for field in METRIC_FIELDS
            }
        policies = seed_reports[0]["sources"][source]["selected_thresholds"]
        for policy_name in policies:
            source_aggregate["selected_thresholds"][policy_name] = {
                "target_recall": policies[policy_name]["target_recall"],
                "threshold": mean_std(
                    [
                        report["sources"][source]["selected_thresholds"][policy_name]["threshold"]
                        for report in seed_reports
                    ]
                ),
                "val": {
                    field: mean_std(
                        [
                            report["sources"][source]["selected_thresholds"][policy_name]["val_metrics"][field]
                            for report in seed_reports
                        ]
                    )
                    for field in METRIC_FIELDS
                },
                "test": {
                    field: mean_std(
                        [
                            report["sources"][source]["selected_thresholds"][policy_name]["test_metrics"][field]
                            for report in seed_reports
                        ]
                    )
                    for field in METRIC_FIELDS
                },
            }
        aggregate[source] = source_aggregate
    return aggregate


def evaluate_deep_probability_audit(
    prediction_sets: Sequence[dict[str, Path]],
    seed_labels: Sequence[str],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
    recall_targets: Sequence[float],
) -> dict[str, Any]:
    seed_reports = [
        audit_seed(
            seed_label,
            prediction_paths,
            primary_alphas,
            secondary_alphas,
            tertiary_alphas,
            recall_targets,
        )
        for seed_label, prediction_paths in zip(seed_labels, prediction_sets, strict=True)
    ]
    return {
        "audit": "deep_probability_threshold",
        "positive_class": "N3",
        "threshold_selection_split": "val",
        "threshold_evaluation_split": "test",
        "seed_count": len(seed_reports),
        "model_roles": list(MODEL_ROLES),
        "fusion_role": FUSION_ROLE,
        "recall_targets": list(recall_targets),
        "fusion_weights": {
            "primary_full_w20": primary_alphas.tolist(),
            "secondary_capacity_h128": secondary_alphas.tolist(),
            "tertiary_h128_ls003": tertiary_alphas.tolist(),
            "base_original_temporal": (
                1.0 - primary_alphas - secondary_alphas - tertiary_alphas
            ).tolist(),
        },
        "seed_reports": seed_reports,
        "aggregate": aggregate_reports(seed_reports),
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"=== Deep probability audit ({report['seed_count']} seeds) ===")
    print("| source | test ROC-AUC | test AP | argmax P | argmax R | argmax F1 |")
    print("|---|---:|---:|---:|---:|---:|")
    for source in SOURCES:
        summary = report["aggregate"][source]
        ranking = summary["ranking"]["test"]
        argmax = summary["argmax"]["test"]
        print(
            f"| {source} | {ranking['roc_auc']['mean']:.4f} | "
            f"{ranking['average_precision']['mean']:.4f} | "
            f"{argmax['precision']['mean']:.4f} | {argmax['recall']['mean']:.4f} | "
            f"{argmax['f1']['mean']:.4f} |"
        )
    print("\n=== Validation-selected thresholds: current_best_fusion test ===")
    policies = report["aggregate"][FUSION_ROLE]["selected_thresholds"]
    print("| policy | threshold | precision | recall | specificity | F1 | predicted positive |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, policy in policies.items():
        test = policy["test"]
        print(
            f"| {name} | {policy['threshold']['mean']:.4f} | "
            f"{test['precision']['mean']:.4f} | {test['recall']['mean']:.4f} | "
            f"{test['specificity']['mean']:.4f} | {test['f1']['mean']:.4f} | "
            f"{test['predicted_positive_rate']['mean']:.4f} |"
        )


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit N3 probability ranking and validation-selected veto thresholds."
    )
    parser.add_argument("--base-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--secondary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--tertiary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", default=None)
    parser.add_argument("--recall-targets", default="0.50,0.70,0.80,0.90")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--wake-primary", type=float, default=0.72)
    parser.add_argument("--wake-secondary", type=float, default=0.06)
    parser.add_argument("--wake-tertiary", type=float, default=0.00)
    parser.add_argument("--light-primary", type=float, default=0.80)
    parser.add_argument("--light-secondary", type=float, default=0.02)
    parser.add_argument("--light-tertiary", type=float, default=0.15)
    parser.add_argument("--deep-primary", type=float, default=0.82)
    parser.add_argument("--deep-secondary", type=float, default=0.00)
    parser.add_argument("--deep-tertiary", type=float, default=0.18)
    parser.add_argument("--rem-primary", type=float, default=0.00)
    parser.add_argument("--rem-secondary", type=float, default=0.42)
    parser.add_argument("--rem-tertiary", type=float, default=0.13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_lists = (
        args.base_predictions,
        args.primary_predictions,
        args.secondary_predictions,
        args.tertiary_predictions,
    )
    prediction_count = len(args.base_predictions)
    if any(len(paths) != prediction_count for paths in path_lists):
        raise ValueError("All prediction path lists must have the same length")
    seed_labels = args.seed_labels or [str(index + 1) for index in range(prediction_count)]
    if len(seed_labels) != prediction_count:
        raise ValueError("--seed-labels must match the number of prediction sets")
    recall_targets = parse_float_list(args.recall_targets)
    prediction_sets = [
        {
            "original_temporal": args.base_predictions[index],
            "full_w20": args.primary_predictions[index],
            "capacity_h128": args.secondary_predictions[index],
            "h128_ls003": args.tertiary_predictions[index],
        }
        for index in range(prediction_count)
    ]
    primary_alphas, secondary_alphas, tertiary_alphas = build_grouped_class_weights(
        wake_primary=args.wake_primary,
        wake_secondary=args.wake_secondary,
        wake_tertiary=args.wake_tertiary,
        light_deep_primary=args.light_primary,
        light_deep_secondary=args.light_secondary,
        light_deep_tertiary=args.light_tertiary,
        deep_primary=args.deep_primary,
        deep_secondary=args.deep_secondary,
        deep_tertiary=args.deep_tertiary,
        rem_primary=args.rem_primary,
        rem_secondary=args.rem_secondary,
        rem_tertiary=args.rem_tertiary,
    )
    report = evaluate_deep_probability_audit(
        prediction_sets,
        seed_labels,
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
        recall_targets,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(report)


if __name__ == "__main__":
    main()
