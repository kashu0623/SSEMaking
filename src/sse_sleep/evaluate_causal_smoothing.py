"""Evaluate causal post-processing for sleep-stage prediction sequences."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .labels import STAGE5_NAMES, STAGE5_TO_ID
from .metrics import evaluate_5_and_4


METHODS = (
    "raw",
    "majority_vote_3",
    "majority_vote_5",
    "probability_ma_3",
    "probability_ma_5",
    "transition_guard_n3_rem_2",
)

SUMMARY_METRICS = {
    "accuracy_5": ("5_class", "accuracy"),
    "macro_f1_5": ("5_class", "macro_f1"),
    "kappa_5": ("5_class", "cohen_kappa"),
    "accuracy_4": ("4_class", "accuracy"),
    "macro_f1_4": ("4_class", "macro_f1"),
    "kappa_4": ("4_class", "cohen_kappa"),
}


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


def ordered_contiguous_runs(subject_ids: Sequence[str], epoch_indices: Sequence[int]) -> Iterable[list[int]]:
    """Yield index runs sorted by subject and epoch, resetting across gaps."""
    if len(subject_ids) != len(epoch_indices):
        raise ValueError("subject_ids and epoch_indices must have the same length")

    order = sorted(range(len(subject_ids)), key=lambda idx: (str(subject_ids[idx]), int(epoch_indices[idx])))
    run: list[int] = []
    previous_subject: str | None = None
    previous_epoch: int | None = None

    for idx in order:
        subject_id = str(subject_ids[idx])
        epoch_index = int(epoch_indices[idx])
        contiguous = (
            run
            and previous_subject == subject_id
            and previous_epoch is not None
            and epoch_index == previous_epoch + 1
        )
        if not run or contiguous:
            run.append(idx)
        else:
            yield run
            run = [idx]
        previous_subject = subject_id
        previous_epoch = epoch_index

    if run:
        yield run


def majority_vote(predictions: np.ndarray, window: int, num_classes: int) -> np.ndarray:
    """Causal majority vote with ties resolved by the most recent tied class."""
    smoothed = np.empty_like(predictions)
    for idx in range(predictions.shape[0]):
        start = max(0, idx - window + 1)
        history = predictions[start : idx + 1]
        counts = np.bincount(history, minlength=num_classes)
        best_count = counts.max()
        for candidate in history[::-1]:
            if counts[int(candidate)] == best_count:
                smoothed[idx] = candidate
                break
    return smoothed


def probability_moving_average(probabilities: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average over class probabilities."""
    smoothed = np.empty(probabilities.shape[0], dtype=np.int64)
    for idx in range(probabilities.shape[0]):
        start = max(0, idx - window + 1)
        smoothed[idx] = int(np.mean(probabilities[start : idx + 1], axis=0).argmax())
    return smoothed


def transition_guard(
    predictions: np.ndarray,
    guarded_classes: set[int],
    min_epochs: int,
) -> np.ndarray:
    """Delay entry into guarded classes until the raw prediction persists."""
    if min_epochs <= 0:
        raise ValueError("min_epochs must be positive")
    if predictions.size == 0:
        return predictions.copy()

    smoothed = np.empty_like(predictions)
    accepted = int(predictions[0])
    candidate: int | None = None
    candidate_count = 0
    smoothed[0] = accepted

    for idx in range(1, predictions.shape[0]):
        raw_prediction = int(predictions[idx])
        if raw_prediction == accepted:
            candidate = None
            candidate_count = 0
        elif raw_prediction in guarded_classes:
            if raw_prediction == candidate:
                candidate_count += 1
            else:
                candidate = raw_prediction
                candidate_count = 1
            if candidate_count >= min_epochs:
                accepted = raw_prediction
                candidate = None
                candidate_count = 0
        else:
            accepted = raw_prediction
            candidate = None
            candidate_count = 0
        smoothed[idx] = accepted

    return smoothed


def apply_causal_method(
    method: str,
    raw_predictions: np.ndarray,
    probabilities: np.ndarray | None,
    subject_ids: np.ndarray,
    epoch_indices: np.ndarray,
    transition_guard_min_epochs: int,
) -> np.ndarray:
    smoothed = np.empty_like(raw_predictions)
    guarded_classes = {STAGE5_TO_ID["N3"], STAGE5_TO_ID["REM"]}

    for run in ordered_contiguous_runs(subject_ids.astype(str).tolist(), epoch_indices.astype(int).tolist()):
        run_raw = raw_predictions[run]
        if method == "raw":
            run_smoothed = run_raw
        elif method == "majority_vote_3":
            run_smoothed = majority_vote(run_raw, window=3, num_classes=len(STAGE5_NAMES))
        elif method == "majority_vote_5":
            run_smoothed = majority_vote(run_raw, window=5, num_classes=len(STAGE5_NAMES))
        elif method == "probability_ma_3":
            if probabilities is None:
                raise ValueError("probability_ma_3 requires probabilities in the predictions NPZ")
            run_smoothed = probability_moving_average(probabilities[run], window=3)
        elif method == "probability_ma_5":
            if probabilities is None:
                raise ValueError("probability_ma_5 requires probabilities in the predictions NPZ")
            run_smoothed = probability_moving_average(probabilities[run], window=5)
        elif method == "transition_guard_n3_rem_2":
            run_smoothed = transition_guard(
                run_raw,
                guarded_classes=guarded_classes,
                min_epochs=transition_guard_min_epochs,
            )
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        smoothed[run] = run_smoothed

    return smoothed


def load_split_arrays(
    data: np.lib.npyio.NpzFile,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    required_keys = [f"{split}_y_true", f"{split}_y_pred", f"{split}_subject_ids", f"{split}_epoch_indices"]
    missing = [key for key in required_keys if key not in data.files]
    if missing:
        raise ValueError(f"Missing required arrays for causal smoothing: {', '.join(missing)}")

    probabilities = data[f"{split}_probs"] if f"{split}_probs" in data.files else None
    return (
        data[f"{split}_y_true"].astype(np.int64),
        data[f"{split}_y_pred"].astype(np.int64),
        None if probabilities is None else probabilities.astype(np.float32),
        data[f"{split}_subject_ids"].astype(str),
        data[f"{split}_epoch_indices"].astype(np.int64),
    )


def evaluate_prediction_file(
    path: Path,
    splits: Sequence[str],
    methods: Sequence[str],
    transition_guard_min_epochs: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        split_reports: dict[str, Any] = {}
        for split in splits:
            y_true, raw_predictions, probabilities, subject_ids, epoch_indices = load_split_arrays(data, split)
            method_reports: dict[str, Any] = {}
            for method in methods:
                smoothed_predictions = apply_causal_method(
                    method=method,
                    raw_predictions=raw_predictions,
                    probabilities=probabilities,
                    subject_ids=subject_ids,
                    epoch_indices=epoch_indices,
                    transition_guard_min_epochs=transition_guard_min_epochs,
                )
                method_reports[method] = {
                    "metrics": json_ready(evaluate_5_and_4(y_true.tolist(), smoothed_predictions.tolist())),
                    "prediction_changes_vs_raw": int(np.sum(smoothed_predictions != raw_predictions)),
                    "prediction_change_rate_vs_raw": float(np.mean(smoothed_predictions != raw_predictions)),
                }
            contiguous_run_count = sum(
                1 for _ in ordered_contiguous_runs(subject_ids.tolist(), epoch_indices.tolist())
            )
            split_reports[split] = {
                "sample_count": int(y_true.shape[0]),
                "subject_count": int(len(set(subject_ids.tolist()))),
                "contiguous_run_count": int(contiguous_run_count),
                "methods": method_reports,
            }

    return {
        "predictions_path": str(path),
        "splits": split_reports,
    }


def metric_value(metrics: dict[str, Any], metric_path: tuple[str, str]) -> float:
    group, name = metric_path
    return float(metrics[group][name])


def aggregate_reports(file_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    class_f1_values: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file_report in file_reports:
        for split, split_report in file_report["splits"].items():
            for method, method_report in split_report["methods"].items():
                metrics = method_report["metrics"]
                for output_name, path in SUMMARY_METRICS.items():
                    values[split][method][output_name].append(metric_value(metrics, path))
                for class_name in STAGE5_NAMES:
                    class_f1_values[split][method][class_name].append(
                        float(metrics["5_class"]["class_wise"][class_name]["f1"])
                    )

    aggregate: dict[str, Any] = {}
    for split, split_values in values.items():
        aggregate[split] = {}
        for method, method_values in split_values.items():
            aggregate[split][method] = {
                metric_name: {
                    "mean": float(np.mean(metric_values)),
                    "std": float(np.std(metric_values)),
                    "min": float(np.min(metric_values)),
                    "max": float(np.max(metric_values)),
                }
                for metric_name, metric_values in method_values.items()
            }
            aggregate[split][method]["class_f1_5"] = {
                class_name: {
                    "mean": float(np.mean(class_values)),
                    "std": float(np.std(class_values)),
                    "min": float(np.min(class_values)),
                    "max": float(np.max(class_values)),
                }
                for class_name, class_values in class_f1_values[split][method].items()
            }

    return aggregate


def evaluate_causal_smoothing(
    prediction_paths: Sequence[Path],
    out_json: Path,
    splits: Sequence[str],
    methods: Sequence[str],
    transition_guard_min_epochs: int,
) -> dict[str, Any]:
    invalid_methods = sorted(set(methods) - set(METHODS))
    if invalid_methods:
        raise ValueError(f"Unknown smoothing methods: {', '.join(invalid_methods)}")

    file_reports = [
        evaluate_prediction_file(
            path=path,
            splits=splits,
            methods=methods,
            transition_guard_min_epochs=transition_guard_min_epochs,
        )
        for path in prediction_paths
    ]
    report = {
        "prediction_paths": [str(path) for path in prediction_paths],
        "splits": list(splits),
        "methods": list(methods),
        "design": {
            "causal": True,
            "subject_boundaries_reset_history": True,
            "epoch_gaps_reset_history": True,
            "majority_vote_tie_break": "most_recent_tied_prediction",
            "probability_moving_average": "mean of current and previous epochs within each run",
            "transition_guard": {
                "guarded_classes": ["N3", "REM"],
                "min_epochs": transition_guard_min_epochs,
                "behavior": "delay entry into guarded classes until raw predictions persist",
            },
        },
        "per_file": file_reports,
        "aggregate": aggregate_reports(file_reports),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate causal smoothing over saved LSTM/GRU predictions."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more lstm_predictions.npz files.",
    )
    parser.add_argument("--out-json", type=Path, required=True, help="Output smoothing evaluation JSON.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("val", "test"),
        default=["test"],
        help="Prediction splits to evaluate.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=list(METHODS),
        help="Smoothing methods to evaluate.",
    )
    parser.add_argument(
        "--transition-guard-min-epochs",
        type=int,
        default=2,
        help="Minimum consecutive raw epochs before accepting guarded N3/REM transitions.",
    )
    args = parser.parse_args()

    report = evaluate_causal_smoothing(
        prediction_paths=args.predictions,
        out_json=args.out_json,
        splits=args.splits,
        methods=args.methods,
        transition_guard_min_epochs=args.transition_guard_min_epochs,
    )
    print(
        json.dumps(
            {"out_json": str(args.out_json), "aggregate": report["aggregate"]},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
