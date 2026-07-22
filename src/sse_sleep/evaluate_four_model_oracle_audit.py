"""Measure oracle headroom and error diversity for the current four-model fusion."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from .evaluate_four_model_fusion import build_grouped_class_weights, four_model_classwise_fusion
from .evaluate_prediction_fusion import load_split, validate_alignment
from .labels import STAGE4_NAMES, STAGE5_NAMES, merge_many_5_to_4
from .metrics import evaluate


MODEL_ROLES = ("original_temporal", "full_w20", "capacity_h128", "h128_ls003")
FUSION_ROLE = "current_best_fusion"


def safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def stage4_ids(labels_5: np.ndarray) -> np.ndarray:
    return np.asarray(merge_many_5_to_4(labels_5.tolist()), dtype=np.int64)


def metrics4(y_true_4: np.ndarray, y_pred_4: np.ndarray) -> dict[str, Any]:
    result = evaluate(y_true_4.tolist(), y_pred_4.tolist(), STAGE4_NAMES)
    return asdict(result)


def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "kappa": float(metrics["cohen_kappa"]),
        "macro_f1_plus_kappa": float(metrics["macro_f1"] + metrics["cohen_kappa"]),
        "class_f1": {
            stage: float(metrics["class_wise"][stage]["f1"])
            for stage in STAGE4_NAMES
        },
    }


def oracle_prediction(
    y_true_4: np.ndarray,
    fallback_pred_4: np.ndarray,
    candidate_pred_4: Sequence[np.ndarray],
) -> np.ndarray:
    oracle = fallback_pred_4.copy()
    unresolved = oracle != y_true_4
    for prediction in candidate_pred_4:
        rescue = unresolved & (prediction == y_true_4)
        oracle[rescue] = prediction[rescue]
        unresolved[rescue] = False
    return oracle


def model_rescue_stats(
    y_true_4: np.ndarray,
    fusion_pred_4: np.ndarray,
    model_pred_4: dict[str, np.ndarray],
    model_probs: dict[str, np.ndarray],
    fusion_probs: np.ndarray,
) -> dict[str, Any]:
    fusion_error = fusion_pred_4 != y_true_4
    model_correct = {
        role: prediction == y_true_4
        for role, prediction in model_pred_4.items()
    }
    stats: dict[str, Any] = {}
    for role in MODEL_ROLES:
        rescue = fusion_error & model_correct[role]
        other_correct = np.zeros(y_true_4.shape[0], dtype=bool)
        for other_role in MODEL_ROLES:
            if other_role != role:
                other_correct |= model_correct[other_role]
        exclusive = rescue & ~other_correct
        role_confidence = model_probs[role].max(axis=1)
        fusion_confidence = fusion_probs.max(axis=1)
        stats[role] = {
            "rescue_count": int(rescue.sum()),
            "rescue_rate_of_fusion_errors": safe_divide(rescue.sum(), fusion_error.sum()),
            "exclusive_rescue_count": int(exclusive.sum()),
            "exclusive_rescue_rate_of_fusion_errors": safe_divide(exclusive.sum(), fusion_error.sum()),
            "mean_role_confidence_on_rescues": float(role_confidence[rescue].mean()) if rescue.any() else 0.0,
            "mean_fusion_confidence_on_rescues": (
                float(fusion_confidence[rescue].mean()) if rescue.any() else 0.0
            ),
        }
    return stats


def pairwise_error_stats(
    y_true_4: np.ndarray,
    model_pred_4: dict[str, np.ndarray],
) -> dict[str, Any]:
    pairs: dict[str, Any] = {}
    for left_index, left_role in enumerate(MODEL_ROLES):
        for right_role in MODEL_ROLES[left_index + 1 :]:
            left_pred = model_pred_4[left_role]
            right_pred = model_pred_4[right_role]
            left_error = left_pred != y_true_4
            right_error = right_pred != y_true_4
            union_error = left_error | right_error
            pair_name = f"{left_role}__{right_role}"
            pairs[pair_name] = {
                "prediction_disagreement_rate": float((left_pred != right_pred).mean()),
                "joint_error_rate": float((left_error & right_error).mean()),
                "error_jaccard": safe_divide((left_error & right_error).sum(), union_error.sum()),
                "left_only_correct_rate": float((~left_error & right_error).mean()),
                "right_only_correct_rate": float((left_error & ~right_error).mean()),
            }
    return pairs


def stage_headroom(
    y_true_4: np.ndarray,
    fusion_pred_4: np.ndarray,
    model_pred_4: dict[str, np.ndarray],
) -> dict[str, Any]:
    any_model_correct = np.logical_or.reduce(
        [prediction == y_true_4 for prediction in model_pred_4.values()]
    )
    result: dict[str, Any] = {}
    for stage_id, stage in enumerate(STAGE4_NAMES):
        mask = y_true_4 == stage_id
        fusion_correct = mask & (fusion_pred_4 == y_true_4)
        fusion_error = mask & (fusion_pred_4 != y_true_4)
        recoverable = fusion_error & any_model_correct
        role_rescues = {
            role: int((fusion_error & (prediction == y_true_4)).sum())
            for role, prediction in model_pred_4.items()
        }
        support = int(mask.sum())
        result[stage] = {
            "support": support,
            "fusion_correct_count": int(fusion_correct.sum()),
            "fusion_recall": safe_divide(fusion_correct.sum(), support),
            "fusion_error_count": int(fusion_error.sum()),
            "recoverable_by_any_model_count": int(recoverable.sum()),
            "recoverable_rate_of_fusion_errors": safe_divide(recoverable.sum(), fusion_error.sum()),
            "oracle_recall": safe_divide(fusion_correct.sum() + recoverable.sum(), support),
            "oracle_recall_headroom": safe_divide(recoverable.sum(), support),
            "rescue_count_by_model": role_rescues,
        }
    return result


def audit_split(
    prediction_paths: dict[str, Path],
    split: str,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> dict[str, Any]:
    loaded = {role: load_split(path, split) for role, path in prediction_paths.items()}
    base = loaded[MODEL_ROLES[0]]
    for role in MODEL_ROLES[1:]:
        validate_alignment(base, loaded[role], split)

    fused_probs = four_model_classwise_fusion(
        loaded["original_temporal"]["probs"],
        loaded["full_w20"]["probs"],
        loaded["capacity_h128"]["probs"],
        loaded["h128_ls003"]["probs"],
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    y_true_4 = stage4_ids(base["y_true"])
    model_probs = {role: loaded[role]["probs"] for role in MODEL_ROLES}
    model_pred_4 = {
        role: stage4_ids(probabilities.argmax(axis=1).astype(np.int64))
        for role, probabilities in model_probs.items()
    }
    fusion_pred_4 = stage4_ids(fused_probs.argmax(axis=1).astype(np.int64))
    all_pred_4 = {**model_pred_4, FUSION_ROLE: fusion_pred_4}

    oracle_models_pred = oracle_prediction(
        y_true_4,
        fallback_pred_4=fusion_pred_4,
        candidate_pred_4=list(model_pred_4.values()),
    )
    oracle_all_pred = oracle_prediction(
        y_true_4,
        fallback_pred_4=fusion_pred_4,
        candidate_pred_4=list(all_pred_4.values()),
    )
    metric_summaries = {
        role: metric_summary(metrics4(y_true_4, prediction))
        for role, prediction in all_pred_4.items()
    }
    metric_summaries["oracle_models_with_fusion_fallback"] = metric_summary(
        metrics4(y_true_4, oracle_models_pred)
    )
    metric_summaries["oracle_models_plus_fusion"] = metric_summary(
        metrics4(y_true_4, oracle_all_pred)
    )

    base_matrix = np.stack([model_pred_4[role] for role in MODEL_ROLES], axis=1)
    model_disagreement = np.any(base_matrix != base_matrix[:, :1], axis=1)
    fusion_correct = fusion_pred_4 == y_true_4
    any_model_correct = np.any(base_matrix == y_true_4.reshape(-1, 1), axis=1)
    recoverable = ~fusion_correct & any_model_correct

    return {
        "samples": int(y_true_4.shape[0]),
        "metrics": metric_summaries,
        "headline": {
            "fusion_accuracy": float(fusion_correct.mean()),
            "fusion_error_count": int((~fusion_correct).sum()),
            "fusion_errors_recoverable_by_any_model_count": int(recoverable.sum()),
            "fusion_errors_recoverable_by_any_model_rate": safe_divide(recoverable.sum(), (~fusion_correct).sum()),
            "oracle_accuracy": float((fusion_correct | any_model_correct).mean()),
            "oracle_accuracy_headroom": float(recoverable.mean()),
            "model_disagreement_count": int(model_disagreement.sum()),
            "model_disagreement_rate": float(model_disagreement.mean()),
            "fusion_accuracy_on_model_agreement": (
                float(fusion_correct[~model_disagreement].mean()) if (~model_disagreement).any() else 0.0
            ),
            "fusion_accuracy_on_model_disagreement": (
                float(fusion_correct[model_disagreement].mean()) if model_disagreement.any() else 0.0
            ),
        },
        "stage_headroom": stage_headroom(y_true_4, fusion_pred_4, model_pred_4),
        "model_rescues": model_rescue_stats(
            y_true_4,
            fusion_pred_4,
            model_pred_4,
            model_probs,
            fused_probs,
        ),
        "pairwise": pairwise_error_stats(y_true_4, model_pred_4),
    }


def scalar_mean_std(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": mean(float(value) for value in values),
        "std": pstdev(float(value) for value in values),
    }


def aggregate_split(seed_reports: Sequence[dict[str, Any]], split: str) -> dict[str, Any]:
    reports = [report["splits"][split] for report in seed_reports]
    metric_roles = (*MODEL_ROLES, FUSION_ROLE, "oracle_models_plus_fusion")
    metrics: dict[str, Any] = {}
    for role in metric_roles:
        metrics[role] = {
            field: scalar_mean_std([report["metrics"][role][field] for report in reports])
            for field in ("accuracy", "macro_f1", "kappa", "macro_f1_plus_kappa")
        }
        metrics[role]["class_f1"] = {
            stage: scalar_mean_std(
                [report["metrics"][role]["class_f1"][stage] for report in reports]
            )
            for stage in STAGE4_NAMES
        }

    fusion_score = metrics[FUSION_ROLE]["macro_f1_plus_kappa"]["mean"]
    oracle_score = metrics["oracle_models_plus_fusion"]["macro_f1_plus_kappa"]["mean"]
    headline_fields = (
        "fusion_accuracy",
        "fusion_errors_recoverable_by_any_model_rate",
        "oracle_accuracy",
        "oracle_accuracy_headroom",
        "model_disagreement_rate",
        "fusion_accuracy_on_model_agreement",
        "fusion_accuracy_on_model_disagreement",
    )
    headline = {
        field: scalar_mean_std([report["headline"][field] for report in reports])
        for field in headline_fields
    }
    headline["oracle_4m_plus_4k_headroom"] = oracle_score - fusion_score

    stage_summary: dict[str, Any] = {}
    for stage in STAGE4_NAMES:
        stage_summary[stage] = {
            field: scalar_mean_std(
                [report["stage_headroom"][stage][field] for report in reports]
            )
            for field in (
                "fusion_recall",
                "recoverable_rate_of_fusion_errors",
                "oracle_recall",
                "oracle_recall_headroom",
            )
        }
        stage_summary[stage]["rescue_count_by_model"] = {
            role: {
                "total": sum(
                    int(report["stage_headroom"][stage]["rescue_count_by_model"][role])
                    for report in reports
                )
            }
            for role in MODEL_ROLES
        }

    model_rescues = {
        role: {
            field: scalar_mean_std([report["model_rescues"][role][field] for report in reports])
            for field in (
                "rescue_rate_of_fusion_errors",
                "exclusive_rescue_rate_of_fusion_errors",
                "mean_role_confidence_on_rescues",
                "mean_fusion_confidence_on_rescues",
            )
        }
        for role in MODEL_ROLES
    }
    pairwise = {
        pair_name: {
            field: scalar_mean_std([report["pairwise"][pair_name][field] for report in reports])
            for field in (
                "prediction_disagreement_rate",
                "joint_error_rate",
                "error_jaccard",
                "left_only_correct_rate",
                "right_only_correct_rate",
            )
        }
        for pair_name in reports[0]["pairwise"]
    }
    return {
        "seed_count": len(reports),
        "metrics": metrics,
        "headline": headline,
        "stage_headroom": stage_summary,
        "model_rescues": model_rescues,
        "pairwise": pairwise,
    }


def evaluate_oracle_audit(
    prediction_sets: Sequence[dict[str, Path]],
    seed_labels: Sequence[str],
    splits: Sequence[str],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> dict[str, Any]:
    seed_reports = []
    for seed_label, prediction_paths in zip(seed_labels, prediction_sets, strict=True):
        seed_reports.append(
            {
                "seed": seed_label,
                "prediction_paths": {role: str(path) for role, path in prediction_paths.items()},
                "splits": {
                    split: audit_split(
                        prediction_paths,
                        split,
                        primary_alphas,
                        secondary_alphas,
                        tertiary_alphas,
                    )
                    for split in splits
                },
            }
        )
    return {
        "audit": "four_model_oracle_headroom",
        "stage5_names": list(STAGE5_NAMES),
        "stage4_names": list(STAGE4_NAMES),
        "model_roles": list(MODEL_ROLES),
        "fusion_role": FUSION_ROLE,
        "fusion_weights": {
            "primary_full_w20": primary_alphas.tolist(),
            "secondary_capacity_h128": secondary_alphas.tolist(),
            "tertiary_h128_ls003": tertiary_alphas.tolist(),
            "base_original_temporal": (1.0 - primary_alphas - secondary_alphas - tertiary_alphas).tolist(),
        },
        "seed_reports": seed_reports,
        "aggregate": {split: aggregate_split(seed_reports, split) for split in splits},
    }


def print_summary(report: dict[str, Any]) -> None:
    for split, summary in report["aggregate"].items():
        fusion = summary["metrics"][FUSION_ROLE]
        oracle = summary["metrics"]["oracle_models_plus_fusion"]
        print(f"=== {split} oracle audit ({summary['seed_count']} seeds) ===")
        print(
            f"fusion 4M {fusion['macro_f1']['mean']:.4f} / 4K {fusion['kappa']['mean']:.4f} / "
            f"4M+4K {fusion['macro_f1_plus_kappa']['mean']:.4f}"
        )
        print(
            f"oracle 4M {oracle['macro_f1']['mean']:.4f} / 4K {oracle['kappa']['mean']:.4f} / "
            f"4M+4K {oracle['macro_f1_plus_kappa']['mean']:.4f} / "
            f"headroom {summary['headline']['oracle_4m_plus_4k_headroom']:+.4f}"
        )
        recoverable = summary["headline"]["fusion_errors_recoverable_by_any_model_rate"]["mean"]
        disagreement = summary["headline"]["model_disagreement_rate"]["mean"]
        print(f"recoverable fusion errors {recoverable:.1%} / model disagreement {disagreement:.1%}")
        for stage in STAGE4_NAMES:
            stage_report = summary["stage_headroom"][stage]
            print(
                f"{stage}: fusion recall {stage_report['fusion_recall']['mean']:.3f} -> "
                f"oracle {stage_report['oracle_recall']['mean']:.3f} "
                f"(headroom {stage_report['oracle_recall_headroom']['mean']:+.3f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit oracle headroom in the current four-model fusion.")
    parser.add_argument("--base-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--secondary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--tertiary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=("val", "test"))
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
    report = evaluate_oracle_audit(
        prediction_sets=prediction_sets,
        seed_labels=seed_labels,
        splits=args.splits,
        primary_alphas=primary_alphas,
        secondary_alphas=secondary_alphas,
        tertiary_alphas=tertiary_alphas,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
