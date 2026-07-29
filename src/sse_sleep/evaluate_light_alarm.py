"""Evaluate Light-vs-rest alarm models with validation-selected thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .labels import STAGE4_NAMES, STAGE4_TO_ID
from .metrics import evaluate


BINARY_NAMES = ("Other", "Light")
SCALAR_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "binary_kappa",
    "light_objective",
    "light_precision",
    "light_recall",
    "light_f1",
    "other_recall",
    "wake_to_light_rate",
    "deep_to_light_rate",
    "rem_to_light_rate",
)
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.arange(0.10, 0.901, 0.025))
DEFAULT_DEEP_LIMITS = (0.10, 0.20, 0.30, 0.40)


def parse_float_list(raw: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if raw is None:
        return tuple(float(value) for value in default)
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one float")
    return values


def load_prediction(path: Path) -> dict[str, dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {key: data[key] for key in data.files}
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in ("val", "test"):
        label_key = f"{split}_y_true4"
        probability_key = f"{split}_light_probs"
        if label_key not in arrays or probability_key not in arrays:
            raise ValueError(f"Missing {label_key}/{probability_key} in {path}")
        labels = arrays[label_key].astype(np.int64)
        probabilities = arrays[probability_key].astype(np.float64)
        if labels.ndim != 1 or probabilities.ndim != 1:
            raise ValueError(f"Expected one-dimensional labels/probabilities in {path}")
        if labels.shape != probabilities.shape:
            raise ValueError(f"Label/probability shape mismatch in {path}")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError(f"Probabilities outside [0, 1] in {path}")
        result[split] = {
            "y_true4": labels,
            "light_probs": probabilities,
        }
        for suffix in ("subject_ids", "epoch_indices"):
            key = f"{split}_{suffix}"
            if key in arrays:
                result[split][suffix] = arrays[key]
    return result


def validate_alignment(
    reference: dict[str, dict[str, np.ndarray]],
    candidate: dict[str, dict[str, np.ndarray]],
    path: Path,
) -> None:
    for split in ("val", "test"):
        if not np.array_equal(
            reference[split]["y_true4"],
            candidate[split]["y_true4"],
        ):
            raise ValueError(f"{path} {split} labels do not align")
        for suffix in ("subject_ids", "epoch_indices"):
            if (
                suffix in reference[split]
                and suffix in candidate[split]
                and not np.array_equal(
                    reference[split][suffix],
                    candidate[split][suffix],
                )
            ):
                raise ValueError(f"{path} {split} {suffix} do not align")


def positive_rate(
    labels_4: np.ndarray,
    predictions: np.ndarray,
    stage_name: str,
) -> float:
    mask = labels_4 == STAGE4_TO_ID[stage_name]
    return float(predictions[mask].mean()) if np.any(mask) else 0.0


def metrics_at_threshold(
    labels_4: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    true_binary = (labels_4 == STAGE4_TO_ID["Light"]).astype(np.int64)
    predictions = (probabilities >= threshold).astype(np.int64)
    result = evaluate(true_binary.tolist(), predictions.tolist(), BINARY_NAMES)
    light = result.class_wise["Light"]
    other = result.class_wise["Other"]
    return {
        "accuracy": float(result.accuracy),
        "balanced_accuracy": float((light.recall + other.recall) / 2.0),
        "macro_f1": float(result.macro_f1),
        "binary_kappa": float(result.cohen_kappa),
        "light_objective": float(light.f1 + result.cohen_kappa),
        "light_precision": float(light.precision),
        "light_recall": float(light.recall),
        "light_f1": float(light.f1),
        "other_recall": float(other.recall),
        "wake_to_light_rate": positive_rate(labels_4, predictions, "Wake"),
        "deep_to_light_rate": positive_rate(labels_4, predictions, "Deep"),
        "rem_to_light_rate": positive_rate(labels_4, predictions, "REM"),
        "confusion_matrix": result.confusion_matrix,
    }


def aggregate_split(
    seed_splits: Sequence[dict[str, np.ndarray]],
    threshold: float,
) -> dict[str, Any]:
    seed_metrics = [
        metrics_at_threshold(
            split["y_true4"],
            split["light_probs"],
            threshold,
        )
        for split in seed_splits
    ]
    labels = np.concatenate([split["y_true4"] for split in seed_splits])
    probabilities = np.concatenate([split["light_probs"] for split in seed_splits])
    pooled = metrics_at_threshold(labels, probabilities, threshold)
    return {
        metric: {
            "mean": float(np.mean([record[metric] for record in seed_metrics])),
            "std": float(np.std([record[metric] for record in seed_metrics])),
        }
        for metric in SCALAR_METRICS
    } | {
        "pooled": pooled,
    }


def candidate_record(
    label: str,
    seeds: Sequence[dict[str, dict[str, np.ndarray]]],
    threshold: float,
) -> dict[str, Any]:
    return {
        "name": f"{label}__threshold{threshold:.3f}",
        "config": label,
        "threshold": float(threshold),
        "val": aggregate_split([seed["val"] for seed in seeds], threshold),
        "test": aggregate_split([seed["test"] for seed in seeds], threshold),
    }


def select_records(
    candidates: Sequence[dict[str, Any]],
    deep_limits: Sequence[float],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    best_objective = max(
        candidates,
        key=lambda item: (
            item["val"]["light_objective"]["mean"],
            -item["val"]["deep_to_light_rate"]["mean"],
            item["val"]["light_precision"]["mean"],
        ),
    )
    best_light_f1 = max(
        candidates,
        key=lambda item: (
            item["val"]["light_f1"]["mean"],
            -item["val"]["deep_to_light_rate"]["mean"],
            item["val"]["binary_kappa"]["mean"],
        ),
    )
    precision_pool = [
        item
        for item in candidates
        if item["val"]["light_recall"]["mean"] >= 0.50
    ]
    best_precision = (
        max(
            precision_pool,
            key=lambda item: (
                item["val"]["light_precision"]["mean"],
                -item["val"]["deep_to_light_rate"]["mean"],
                item["val"]["light_recall"]["mean"],
            ),
        )
        if precision_pool
        else None
    )
    safe_profiles: dict[str, Any] = {}
    for limit in deep_limits:
        eligible = [
            item
            for item in candidates
            if item["val"]["deep_to_light_rate"]["mean"] <= limit
        ]
        safe_profiles[f"deep_leak_at_most_{limit:.2f}"] = (
            max(
                eligible,
                key=lambda item: (
                    item["val"]["light_recall"]["mean"],
                    item["val"]["light_precision"]["mean"],
                    item["val"]["binary_kappa"]["mean"],
                ),
            )
            if eligible
            else None
        )
    return {
        "selection_uses": "validation 3-seed mean only; test is reporting only",
        "best_light_objective": best_objective,
        "best_light_f1": best_light_f1,
        "best_light_precision_with_recall_at_least_0.50": best_precision,
        "safe_profiles": safe_profiles,
    }


def binary_metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    if confusion.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got {confusion.shape}")
    tn, fp = (float(value) for value in confusion[0])
    fn, tp = (float(value) for value in confusion[1])
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / max(total, 1.0)
    light_precision = tp / max(tp + fp, 1.0)
    light_recall = tp / max(tp + fn, 1.0)
    light_f1 = (
        2.0 * light_precision * light_recall / (light_precision + light_recall)
        if light_precision + light_recall
        else 0.0
    )
    other_precision = tn / max(tn + fn, 1.0)
    other_recall = tn / max(tn + fp, 1.0)
    other_f1 = (
        2.0 * other_precision * other_recall / (other_precision + other_recall)
        if other_precision + other_recall
        else 0.0
    )
    row_totals = confusion.sum(axis=1)
    column_totals = confusion.sum(axis=0)
    expected = float(np.dot(row_totals, column_totals) / max(total * total, 1.0))
    kappa = (
        (accuracy - expected) / (1.0 - expected)
        if expected < 1.0
        else 0.0
    )
    return {
        "accuracy": accuracy,
        "balanced_accuracy": (light_recall + other_recall) / 2.0,
        "macro_f1": (light_f1 + other_f1) / 2.0,
        "binary_kappa": kappa,
        "light_objective": light_f1 + kappa,
        "light_precision": light_precision,
        "light_recall": light_recall,
        "light_f1": light_f1,
        "other_recall": other_recall,
    }


def collapse_stage4_confusion(confusion: np.ndarray) -> np.ndarray:
    if confusion.shape != (4, 4):
        raise ValueError(f"Expected 4x4 confusion matrix, got {confusion.shape}")
    light = STAGE4_TO_ID["Light"]
    tp = int(confusion[light, light])
    fn = int(confusion[light, :].sum() - tp)
    fp = int(confusion[:, light].sum() - tp)
    tn = int(confusion.sum() - tp - fn - fp)
    return np.asarray(((tn, fp), (fn, tp)), dtype=np.int64)


def current_baseline(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source = summary.get("selections", {}).get("selected_by_project_rule")
    if source is None:
        source = summary.get("current_best_reference")
    if source is None:
        raise ValueError(f"Cannot find current selected candidate in {summary_path}")
    result: dict[str, Any] = {
        "source_path": str(summary_path),
        "source_name": source.get("name"),
        "method": "Collapse selected 4-class argmax confusion into Light vs Other",
    }
    for split in ("val", "test"):
        confusion_4 = np.asarray(
            source[split]["pooled_confusion_matrix"],
            dtype=np.int64,
        )
        confusion_2 = collapse_stage4_confusion(confusion_4)
        metrics = binary_metrics_from_confusion(confusion_2)
        stage_totals = confusion_4.sum(axis=1)
        metrics.update(
            {
                "wake_to_light_rate": float(
                    confusion_4[STAGE4_TO_ID["Wake"], STAGE4_TO_ID["Light"]]
                    / max(stage_totals[STAGE4_TO_ID["Wake"]], 1)
                ),
                "deep_to_light_rate": float(
                    confusion_4[STAGE4_TO_ID["Deep"], STAGE4_TO_ID["Light"]]
                    / max(stage_totals[STAGE4_TO_ID["Deep"]], 1)
                ),
                "rem_to_light_rate": float(
                    confusion_4[STAGE4_TO_ID["REM"], STAGE4_TO_ID["Light"]]
                    / max(stage_totals[STAGE4_TO_ID["REM"]], 1)
                ),
                "binary_confusion_matrix": confusion_2.tolist(),
                "source_stage4_confusion_matrix": confusion_4.tolist(),
            }
        )
        result[split] = metrics
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Light-vs-rest alarm models with validation threshold selection."
    )
    parser.add_argument("--config-labels", nargs="+", required=True)
    parser.add_argument(
        "--prediction-paths",
        type=Path,
        nargs="+",
        required=True,
        help="Config-major paths: all seeds for config1, then config2.",
    )
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--deep-leak-limits", default=None)
    parser.add_argument("--current-summary-json", type=Path, required=True)
    parser.add_argument("--archive-top", type=int, default=40)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_count = len(args.seed_labels)
    expected = len(args.config_labels) * seed_count
    if len(args.prediction_paths) != expected:
        raise ValueError(f"Expected {expected} prediction paths, got {len(args.prediction_paths)}")
    thresholds = parse_float_list(args.thresholds, DEFAULT_THRESHOLDS)
    deep_limits = parse_float_list(args.deep_leak_limits, DEFAULT_DEEP_LIMITS)
    if any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise ValueError("Thresholds must be in (0, 1)")
    if any(not 0.0 <= limit <= 1.0 for limit in deep_limits):
        raise ValueError("Deep leak limits must be in [0, 1]")

    configs: dict[str, list[dict[str, dict[str, np.ndarray]]]] = {}
    reference_by_seed: list[dict[str, dict[str, np.ndarray]] | None] = [
        None for _ in range(seed_count)
    ]
    for config_index, label in enumerate(args.config_labels):
        seeds = []
        for seed_index in range(seed_count):
            path = args.prediction_paths[config_index * seed_count + seed_index]
            loaded = load_prediction(path)
            reference = reference_by_seed[seed_index]
            if reference is None:
                reference_by_seed[seed_index] = loaded
            else:
                validate_alignment(reference, loaded, path)
            seeds.append(loaded)
        configs[label] = seeds

    candidates = [
        candidate_record(label, seeds, threshold)
        for label, seeds in configs.items()
        for threshold in thresholds
    ]
    global_selections = select_records(candidates, deep_limits)
    source_selections = {
        label: select_records(
            [candidate for candidate in candidates if candidate["config"] == label],
            deep_limits,
        )
        for label in args.config_labels
    }
    archived_candidates = [
        *sorted(
            candidates,
            key=lambda item: item["val"]["light_objective"]["mean"],
            reverse=True,
        )[: args.archive_top],
        *sorted(
            candidates,
            key=lambda item: item["val"]["light_f1"]["mean"],
            reverse=True,
        )[: args.archive_top],
        *sorted(
            candidates,
            key=lambda item: item["val"]["deep_to_light_rate"]["mean"],
        )[: args.archive_top],
    ]
    for selection in (global_selections, *source_selections.values()):
        for key, value in selection.items():
            if isinstance(value, dict) and "name" in value:
                archived_candidates.append(value)
            if key == "safe_profiles":
                archived_candidates.extend(
                    item for item in value.values() if item is not None
                )
    archived = {item["name"]: item for item in archived_candidates}
    report = {
        "experiment": "light_alarm_objective_audit",
        "binary_names": list(BINARY_NAMES),
        "stage4_names": list(STAGE4_NAMES),
        "selection": {
            "primary": "maximum validation 3-seed mean Light F1 + binary Kappa",
            "tie_break": "lower validation Deep-to-Light rate, then higher Light precision",
            "test_usage": "reporting only",
        },
        "config_labels": args.config_labels,
        "seed_labels": args.seed_labels,
        "thresholds": [float(value) for value in thresholds],
        "deep_leak_limits": [float(value) for value in deep_limits],
        "candidate_count": len(candidates),
        "current_best_argmax_baseline": current_baseline(args.current_summary_json),
        "global_selections": global_selections,
        "source_selections": source_selections,
        "archived_candidates": list(archived.values()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selected = global_selections["best_light_objective"]
    print(
        f"candidates {len(candidates)} / selected {selected['name']} / "
        f"val objective {selected['val']['light_objective']['mean']:.6f} / "
        f"test Light F1 {selected['test']['light_f1']['mean']:.6f} / "
        f"test binary Kappa {selected['test']['binary_kappa']['mean']:.6f} / "
        f"test Deep->Light {selected['test']['deep_to_light_rate']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
