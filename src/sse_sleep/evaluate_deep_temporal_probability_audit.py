"""Audit causal temporal transforms of the current fusion's N3 probability."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_deep_probability_audit import (
    METRIC_FIELDS,
    binary_metrics,
    binary_prediction_metrics,
    binary_targets,
    mean_std,
    ranking_metrics,
    select_thresholds,
)
from .evaluate_four_model_fusion import build_grouped_class_weights, four_model_classwise_fusion
from .evaluate_prediction_fusion import load_split, validate_alignment
from .labels import STAGE5_TO_ID


MODEL_ROLES = ("original_temporal", "full_w20", "capacity_h128", "h128_ls003")
VARIANTS = (
    "raw",
    "causal_mean_3",
    "causal_mean_5",
    "causal_mean_10",
    "causal_ema_0.20",
    "causal_ema_0.40",
    "causal_ema_0.60",
    "causal_ema_0.80",
)
SEQUENCE_METRIC_FIELDS = (
    "run_detection_rate",
    "onset_epoch_recall",
    "early_2_epoch_recall",
    "mean_first_detection_delay_detected",
)


def contiguous_groups(subject_ids: np.ndarray, epoch_indices: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for subject_id in sorted(set(subject_ids.tolist()), key=str):
        subject_rows = np.flatnonzero(subject_ids == subject_id)
        ordered = subject_rows[np.argsort(epoch_indices[subject_rows], kind="stable")]
        start = 0
        for position in range(1, ordered.shape[0] + 1):
            is_end = position == ordered.shape[0]
            has_gap = (
                not is_end
                and epoch_indices[ordered[position]] != epoch_indices[ordered[position - 1]] + 1
            )
            if is_end or has_gap:
                groups.append(ordered[start:position])
                start = position
    return groups


def causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        raise ValueError("Moving-average window must be positive")
    cumulative = np.cumsum(values, dtype=np.float64)
    result = np.empty(values.shape[0], dtype=np.float32)
    for index in range(values.shape[0]):
        start = max(0, index - window + 1)
        total = cumulative[index] - (cumulative[start - 1] if start else 0.0)
        result[index] = total / (index - start + 1)
    return result


def causal_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EMA alpha must be in (0, 1]")
    result = np.empty(values.shape[0], dtype=np.float32)
    if not values.size:
        return result
    result[0] = values[0]
    for index in range(1, values.shape[0]):
        result[index] = alpha * values[index] + (1.0 - alpha) * result[index - 1]
    return result


def transform_scores(
    raw_scores: np.ndarray,
    subject_ids: np.ndarray,
    epoch_indices: np.ndarray,
    variants: Sequence[str],
) -> dict[str, np.ndarray]:
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown temporal variants: {', '.join(unknown)}")
    transformed = {
        variant: np.empty(raw_scores.shape[0], dtype=np.float32)
        for variant in variants
    }
    for group in contiguous_groups(subject_ids, epoch_indices):
        values = raw_scores[group]
        for variant in variants:
            if variant == "raw":
                result = values
            elif variant.startswith("causal_mean_"):
                result = causal_moving_average(values, int(variant.rsplit("_", 1)[1]))
            elif variant.startswith("causal_ema_"):
                result = causal_ema(values, float(variant.rsplit("_", 1)[1]))
            else:
                raise ValueError(f"Unknown temporal variant: {variant}")
            transformed[variant][group] = result
    return transformed


def deep_run_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    groups: Sequence[np.ndarray],
) -> dict[str, Any]:
    predicted_deep = y_score >= threshold
    run_count = 0
    detected_count = 0
    onset_count = 0
    onset_detected = 0
    early_count = 0
    early_detected = 0
    detection_delays: list[int] = []
    for group in groups:
        group_true = y_true[group]
        start = 0
        while start < group.shape[0]:
            if group_true[start] != 1:
                start += 1
                continue
            end = start + 1
            while end < group.shape[0] and group_true[end] == 1:
                end += 1
            run_rows = group[start:end]
            run_predictions = predicted_deep[run_rows]
            run_count += 1
            onset_count += 1
            onset_detected += int(run_predictions[0])
            early = run_predictions[:2]
            early_count += int(early.shape[0])
            early_detected += int(early.sum())
            detected_offsets = np.flatnonzero(run_predictions)
            if detected_offsets.size:
                detected_count += 1
                detection_delays.append(int(detected_offsets[0]))
            start = end
    return {
        "run_count": run_count,
        "detected_run_count": detected_count,
        "undetected_run_count": run_count - detected_count,
        "run_detection_rate": detected_count / run_count if run_count else 0.0,
        "onset_epoch_recall": onset_detected / onset_count if onset_count else 0.0,
        "early_2_epoch_recall": early_detected / early_count if early_count else 0.0,
        "mean_first_detection_delay_detected": (
            float(np.mean(detection_delays)) if detection_delays else 0.0
        ),
        "max_first_detection_delay_detected": max(detection_delays, default=0),
    }


def load_fusion_split(
    prediction_paths: dict[str, Path],
    split: str,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> dict[str, np.ndarray]:
    loaded = {role: load_split(path, split) for role, path in prediction_paths.items()}
    base = loaded[MODEL_ROLES[0]]
    for role in MODEL_ROLES[1:]:
        validate_alignment(base, loaded[role], split)
    missing = [key for key in ("subject_ids", "epoch_indices") if key not in base]
    if missing:
        raise ValueError(
            f"{split} ensemble predictions need temporal metadata: {', '.join(missing)}"
        )
    fused_probs = four_model_classwise_fusion(
        loaded["original_temporal"]["probs"],
        loaded["full_w20"]["probs"],
        loaded["capacity_h128"]["probs"],
        loaded["h128_ls003"]["probs"],
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    return {
        "y_true": base["y_true"],
        "raw_scores": fused_probs[:, STAGE5_TO_ID["N3"]],
        "argmax_deep": fused_probs.argmax(axis=1) == STAGE5_TO_ID["N3"],
        "subject_ids": base["subject_ids"],
        "epoch_indices": base["epoch_indices"],
    }


def best_variant_for_policy(
    variant_reports: dict[str, Any],
    policy_name: str,
) -> tuple[str, dict[str, Any]]:
    variant, report = max(
        (
            (variant, details["selected_thresholds"][policy_name])
            for variant, details in variant_reports.items()
        ),
        key=lambda item: (
            item[1]["val_metrics"]["precision"],
            item[1]["val_metrics"]["specificity"],
            item[1]["val_metrics"]["f1"],
        ),
    )
    return variant, report


def audit_seed(
    seed_label: str,
    prediction_paths: dict[str, Path],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
    variants: Sequence[str],
    recall_targets: Sequence[float],
) -> dict[str, Any]:
    val = load_fusion_split(
        prediction_paths,
        "val",
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    test = load_fusion_split(
        prediction_paths,
        "test",
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    val_y = binary_targets(val["y_true"])
    test_y = binary_targets(test["y_true"])
    val_scores = transform_scores(
        val["raw_scores"],
        val["subject_ids"],
        val["epoch_indices"],
        variants,
    )
    test_scores = transform_scores(
        test["raw_scores"],
        test["subject_ids"],
        test["epoch_indices"],
        variants,
    )
    val_groups = contiguous_groups(val["subject_ids"], val["epoch_indices"])
    test_groups = contiguous_groups(test["subject_ids"], test["epoch_indices"])
    variant_reports: dict[str, Any] = {}
    for variant in variants:
        selected = select_thresholds(val_y, val_scores[variant], recall_targets)
        for policy in selected.values():
            policy["val_sequence_metrics"] = deep_run_metrics(
                val_y,
                val_scores[variant],
                policy["threshold"],
                val_groups,
            )
            policy["test_metrics"] = binary_metrics(
                test_y,
                test_scores[variant],
                policy["threshold"],
            )
            policy["test_sequence_metrics"] = deep_run_metrics(
                test_y,
                test_scores[variant],
                policy["threshold"],
                test_groups,
            )
        variant_reports[variant] = {
            "val_ranking": ranking_metrics(val_y, val_scores[variant]),
            "test_ranking": ranking_metrics(test_y, test_scores[variant]),
            "selected_thresholds": selected,
        }
    policy_names = list(next(iter(variant_reports.values()))["selected_thresholds"])
    selected_variants = {}
    for policy_name in policy_names:
        variant, policy = best_variant_for_policy(variant_reports, policy_name)
        selected_variants[policy_name] = {
            "variant": variant,
            **policy,
        }
    return {
        "seed": seed_label,
        "prediction_paths": {role: str(path) for role, path in prediction_paths.items()},
        "val_contiguous_group_count": len(val_groups),
        "test_contiguous_group_count": len(test_groups),
        "test_prevalence": float(test_y.mean()),
        "test_raw_argmax": binary_prediction_metrics(test_y, test["argmax_deep"]),
        "variants": variant_reports,
        "validation_selected_variants": selected_variants,
    }


def aggregate_reports(
    seed_reports: Sequence[dict[str, Any]],
    variants: Sequence[str],
) -> dict[str, Any]:
    variant_aggregate: dict[str, Any] = {}
    for variant in variants:
        policies = seed_reports[0]["variants"][variant]["selected_thresholds"]
        variant_aggregate[variant] = {
            "ranking": {
                split: {
                    field: mean_std(
                        [
                            report["variants"][variant][f"{split}_ranking"][field]
                            for report in seed_reports
                        ]
                    )
                    for field in ("roc_auc", "average_precision")
                }
                for split in ("val", "test")
            },
            "selected_thresholds": {
                policy_name: {
                    "target_recall": policy["target_recall"],
                    "threshold": mean_std(
                        [
                            report["variants"][variant]["selected_thresholds"][policy_name][
                                "threshold"
                            ]
                            for report in seed_reports
                        ]
                    ),
                    "val": {
                        field: mean_std(
                            [
                                report["variants"][variant]["selected_thresholds"][
                                    policy_name
                                ]["val_metrics"][field]
                                for report in seed_reports
                            ]
                        )
                        for field in METRIC_FIELDS
                    },
                    "test": {
                        field: mean_std(
                            [
                                report["variants"][variant]["selected_thresholds"][
                                    policy_name
                                ]["test_metrics"][field]
                                for report in seed_reports
                            ]
                        )
                        for field in METRIC_FIELDS
                    },
                    "test_sequence": {
                        field: mean_std(
                            [
                                report["variants"][variant]["selected_thresholds"][
                                    policy_name
                                ]["test_sequence_metrics"][field]
                                for report in seed_reports
                            ]
                        )
                        for field in SEQUENCE_METRIC_FIELDS
                    },
                }
                for policy_name, policy in policies.items()
            },
        }

    selected_policy_aggregate: dict[str, Any] = {}
    selected_policies = seed_reports[0]["validation_selected_variants"]
    for policy_name, first_policy in selected_policies.items():
        selected = [
            report["validation_selected_variants"][policy_name]
            for report in seed_reports
        ]
        selected_policy_aggregate[policy_name] = {
            "target_recall": first_policy["target_recall"],
            "variant_counts": dict(Counter(item["variant"] for item in selected)),
            "threshold": mean_std([item["threshold"] for item in selected]),
            "val": {
                field: mean_std([item["val_metrics"][field] for item in selected])
                for field in METRIC_FIELDS
            },
            "test": {
                field: mean_std([item["test_metrics"][field] for item in selected])
                for field in METRIC_FIELDS
            },
            "test_sequence": {
                field: mean_std([item["test_sequence_metrics"][field] for item in selected])
                for field in SEQUENCE_METRIC_FIELDS
            },
        }
    return {
        "test_prevalence": mean_std([report["test_prevalence"] for report in seed_reports]),
        "test_raw_argmax": {
            field: mean_std([report["test_raw_argmax"][field] for report in seed_reports])
            for field in METRIC_FIELDS
        },
        "variants": variant_aggregate,
        "validation_selected_variants": selected_policy_aggregate,
    }


def evaluate_temporal_audit(
    prediction_sets: Sequence[dict[str, Path]],
    seed_labels: Sequence[str],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
    variants: Sequence[str],
    recall_targets: Sequence[float],
) -> dict[str, Any]:
    seed_reports = [
        audit_seed(
            seed_label,
            prediction_paths,
            primary_alphas,
            secondary_alphas,
            tertiary_alphas,
            variants,
            recall_targets,
        )
        for seed_label, prediction_paths in zip(seed_labels, prediction_sets, strict=True)
    ]
    return {
        "audit": "deep_causal_temporal_probability",
        "positive_class": "N3",
        "causal": True,
        "subject_boundaries_reset_history": True,
        "epoch_gaps_reset_history": True,
        "threshold_selection_split": "val",
        "threshold_evaluation_split": "test",
        "seed_count": len(seed_reports),
        "variants": list(variants),
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
        "aggregate": aggregate_reports(seed_reports, variants),
    }


def print_summary(report: dict[str, Any]) -> None:
    print(f"=== Deep causal temporal probability audit ({report['seed_count']} seeds) ===")
    print("| variant | test ROC-AUC | test AP | R70 P/R/Sp/F1 | R90 P/R/Sp/F1 | R90 onset/run |")
    print("|---|---:|---:|---|---|---|")
    for variant, summary in report["aggregate"]["variants"].items():
        ranking = summary["ranking"]["test"]
        recall_70 = summary["selected_thresholds"].get("recall_70", {}).get("test")
        recall_90 = summary["selected_thresholds"].get("recall_90", {}).get("test")
        recall_90_sequence = summary["selected_thresholds"].get("recall_90", {}).get(
            "test_sequence"
        )

        def format_policy(policy: dict[str, Any] | None) -> str:
            if policy is None:
                return "n/a"
            return (
                f"{policy['precision']['mean']:.3f}/"
                f"{policy['recall']['mean']:.3f}/"
                f"{policy['specificity']['mean']:.3f}/"
                f"{policy['f1']['mean']:.3f}"
            )

        print(
            f"| {variant} | {ranking['roc_auc']['mean']:.4f} | "
            f"{ranking['average_precision']['mean']:.4f} | "
            f"{format_policy(recall_70)} | {format_policy(recall_90)} | "
            f"{recall_90_sequence['onset_epoch_recall']['mean']:.3f}/"
            f"{recall_90_sequence['run_detection_rate']['mean']:.3f} |"
        )
    print("\n=== Per-seed validation-selected temporal variants ===")
    for policy_name, policy in report["aggregate"]["validation_selected_variants"].items():
        test = policy["test"]
        print(
            f"{policy_name}: variants={policy['variant_counts']} / "
            f"test P {test['precision']['mean']:.3f} R {test['recall']['mean']:.3f} "
            f"Sp {test['specificity']['mean']:.3f} F1 {test['f1']['mean']:.3f}"
        )


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_string_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit causal temporal transforms of current fusion N3 probabilities."
    )
    parser.add_argument("--base-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--secondary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--tertiary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", default=None)
    parser.add_argument("--variants", default=",".join(VARIANTS))
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
    report = evaluate_temporal_audit(
        prediction_sets,
        seed_labels,
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
        parse_string_list(args.variants),
        parse_float_list(args.recall_targets),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(report)


if __name__ == "__main__":
    main()
