"""Evaluate one-vs-rest specialist probability fusion for sleep staging."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate_prediction_fusion import load_split, metric_value, summarize_metrics, validate_alignment
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


def parse_labeled_path(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Label cannot be empty")
    return label, Path(raw_path)


def load_specialist_split(path: Path, split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        required = [f"{split}_y_true", f"{split}_probs", f"{split}_logits", "target_stage"]
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"Missing required arrays in {path}: {', '.join(missing)}")
        target_stage = str(data["target_stage"].item())
        logits = data[f"{split}_logits"].astype(np.float32)
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(f"Expected {split}_logits with shape (N, 2) in {path}, got {logits.shape}")
        arrays: dict[str, Any] = {
            "target_stage": target_stage,
            "target_stage_id": STAGE5_TO_ID[target_stage],
            "y_true": data[f"{split}_y_true"].astype(np.int64),
            "probs": data[f"{split}_probs"].astype(np.float32),
            "positive_logits": (logits[:, 1] - logits[:, 0]).astype(np.float32),
        }
        for optional_key in (f"{split}_subject_ids", f"{split}_epoch_indices"):
            if optional_key in data.files:
                arrays[optional_key.removeprefix(f"{split}_")] = data[optional_key]
        return arrays


def validate_same_labels(reference: dict[str, Any], candidate: dict[str, Any], split: str, label: str) -> None:
    if reference["y_true"].shape != candidate["y_true"].shape:
        raise ValueError(f"{split} y_true shape mismatch for {label}")
    if not np.array_equal(reference["y_true"], candidate["y_true"]):
        raise ValueError(f"{split} y_true mismatch for {label}")
    for key in ("subject_ids", "epoch_indices"):
        if key in reference and key in candidate and not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{split} {key} mismatch for {label}")


def load_specialist_bank(paths: Sequence[Path], split: str) -> dict[str, Any]:
    by_stage: dict[str, dict[str, Any]] = {}
    reference: dict[str, Any] | None = None
    for path in paths:
        split_data = load_specialist_split(path, split)
        stage = split_data["target_stage"]
        if stage in by_stage:
            raise ValueError(f"Duplicate specialist for stage {stage}")
        if reference is None:
            reference = split_data
        else:
            validate_same_labels(reference, split_data, split, stage)
        by_stage[stage] = split_data
    missing = [stage for stage in STAGE5_NAMES if stage not in by_stage]
    if missing:
        raise ValueError(f"Missing specialists for stages: {', '.join(missing)}")
    assert reference is not None
    probs = np.stack([by_stage[stage]["probs"] for stage in STAGE5_NAMES], axis=1).astype(np.float32)
    logits = np.stack([by_stage[stage]["positive_logits"] for stage in STAGE5_NAMES], axis=1).astype(np.float32)
    return {
        "y_true": reference["y_true"],
        "probs": probs,
        "logits": logits,
        "subject_ids": reference.get("subject_ids"),
        "epoch_indices": reference.get("epoch_indices"),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return json_ready(evaluate_5_and_4(y_true.astype(int).tolist(), y_pred.astype(int).tolist()))


def candidate_record(
    name: str,
    kind: str,
    params: dict[str, Any],
    val_y_true: np.ndarray,
    val_pred: np.ndarray,
    test_y_true: np.ndarray,
    test_pred: np.ndarray,
    selection_metric: str,
) -> dict[str, Any]:
    val_metrics = evaluate_predictions(val_y_true, val_pred)
    test_metrics = evaluate_predictions(test_y_true, test_pred)
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


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    row_sums = scores.sum(axis=1, keepdims=True)
    return np.divide(scores, row_sums, out=np.zeros_like(scores), where=row_sums > 0)


def fit_platt_calibrators(val_logits: np.ndarray, y_val: np.ndarray) -> list[Any]:
    calibrators = []
    for idx, _stage in enumerate(STAGE5_NAMES):
        y_binary = (y_val == idx).astype(np.int64)
        if len(np.unique(y_binary)) < 2:
            calibrators.append(None)
            continue
        model = LogisticRegression(max_iter=1000, solver="lbfgs")
        model.fit(val_logits[:, [idx]], y_binary)
        calibrators.append(model)
    return calibrators


def apply_platt_calibrators(calibrators: Sequence[Any], logits: np.ndarray) -> np.ndarray:
    columns: list[np.ndarray] = []
    for idx, model in enumerate(calibrators):
        if model is None:
            columns.append(1.0 / (1.0 + np.exp(-logits[:, idx])))
        else:
            columns.append(model.predict_proba(logits[:, [idx]])[:, 1])
    return np.stack(columns, axis=1).astype(np.float32)


def build_meta_features(
    specialist_probs: np.ndarray,
    specialist_logits: np.ndarray,
    base_splits: Sequence[tuple[str, dict[str, np.ndarray]]],
) -> np.ndarray:
    features = [specialist_logits, specialist_probs, safe_logit(specialist_probs)]
    for _label, split in base_splits:
        features.append(split["probs"].astype(np.float32))
        features.append(safe_logit(split["probs"].astype(np.float32)))
    return np.concatenate(features, axis=1).astype(np.float32)


def load_base_prediction_splits(
    base_predictions: Sequence[tuple[str, Path]],
    val_reference: dict[str, Any],
    test_reference: dict[str, Any],
) -> tuple[list[tuple[str, dict[str, np.ndarray]]], list[tuple[str, dict[str, np.ndarray]]]]:
    val_splits: list[tuple[str, dict[str, np.ndarray]]] = []
    test_splits: list[tuple[str, dict[str, np.ndarray]]] = []
    for label, path in base_predictions:
        val_split = load_split(path, "val")
        test_split = load_split(path, "test")
        validate_alignment(
            {"y_true": val_reference["y_true"], "probs": np.zeros((val_reference["y_true"].shape[0], len(STAGE5_NAMES)))},
            val_split,
            "val",
        )
        validate_alignment(
            {"y_true": test_reference["y_true"], "probs": np.zeros((test_reference["y_true"].shape[0], len(STAGE5_NAMES)))},
            test_split,
            "test",
        )
        val_splits.append((label, val_split))
        test_splits.append((label, test_split))
    return val_splits, test_splits


def fit_meta_model(x_val: np.ndarray, y_val: np.ndarray, class_weight: str | None) -> Any:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            multi_class="auto",
            class_weight=class_weight,
        ),
    )
    model.fit(x_val, y_val)
    return model


def evaluate_specialist_fusion(
    specialist_predictions: Sequence[Path],
    out_json: Path,
    base_predictions: Sequence[tuple[str, Path]],
    selection_metric: str,
) -> dict[str, Any]:
    val_bank = load_specialist_bank(specialist_predictions, "val")
    test_bank = load_specialist_bank(specialist_predictions, "test")
    val_base_splits, test_base_splits = load_base_prediction_splits(base_predictions, val_bank, test_bank)

    records: list[dict[str, Any]] = []
    y_val = val_bank["y_true"]
    y_test = test_bank["y_true"]

    raw_prob_val_pred = val_bank["probs"].argmax(axis=1).astype(np.int64)
    raw_prob_test_pred = test_bank["probs"].argmax(axis=1).astype(np.int64)
    records.append(
        candidate_record(
            name="specialist_raw_prob_argmax",
            kind="specialist_argmax",
            params={"score": "positive_probability"},
            val_y_true=y_val,
            val_pred=raw_prob_val_pred,
            test_y_true=y_test,
            test_pred=raw_prob_test_pred,
            selection_metric=selection_metric,
        )
    )

    raw_logit_val_pred = val_bank["logits"].argmax(axis=1).astype(np.int64)
    raw_logit_test_pred = test_bank["logits"].argmax(axis=1).astype(np.int64)
    records.append(
        candidate_record(
            name="specialist_raw_logit_argmax",
            kind="specialist_argmax",
            params={"score": "positive_logit_margin"},
            val_y_true=y_val,
            val_pred=raw_logit_val_pred,
            test_y_true=y_test,
            test_pred=raw_logit_test_pred,
            selection_metric=selection_metric,
        )
    )

    calibrators = fit_platt_calibrators(val_bank["logits"], y_val)
    val_calibrated = normalize_rows(apply_platt_calibrators(calibrators, val_bank["logits"]))
    test_calibrated = normalize_rows(apply_platt_calibrators(calibrators, test_bank["logits"]))
    records.append(
        candidate_record(
            name="specialist_platt_prob_argmax",
            kind="specialist_calibrated_argmax",
            params={"calibration": "per_stage_platt_on_validation"},
            val_y_true=y_val,
            val_pred=val_calibrated.argmax(axis=1).astype(np.int64),
            test_y_true=y_test,
            test_pred=test_calibrated.argmax(axis=1).astype(np.int64),
            selection_metric=selection_metric,
        )
    )

    for label, val_split in val_base_splits:
        matching_test = dict(test_base_splits)[label]
        matching_path = dict(base_predictions)[label]
        records.append(
            candidate_record(
                name=f"base_{label}",
                kind="baseline",
                params={"prediction_file": str(matching_path)},
                val_y_true=y_val,
                val_pred=val_split["probs"].argmax(axis=1).astype(np.int64),
                test_y_true=y_test,
                test_pred=matching_test["probs"].argmax(axis=1).astype(np.int64),
                selection_metric=selection_metric,
            )
        )

    specialist_only_val_features = build_meta_features(val_bank["probs"], val_bank["logits"], [])
    specialist_only_test_features = build_meta_features(test_bank["probs"], test_bank["logits"], [])
    for class_weight_label, class_weight in (("none", None), ("balanced", "balanced")):
        model = fit_meta_model(specialist_only_val_features, y_val, class_weight=class_weight)
        records.append(
            candidate_record(
                name=f"meta_specialists_lr_{class_weight_label}",
                kind="meta_logistic_regression",
                params={
                    "features": "specialist_logits_probs_logit_probs",
                    "fit_split": "validation",
                    "class_weight": class_weight_label,
                },
                val_y_true=y_val,
                val_pred=model.predict(specialist_only_val_features).astype(np.int64),
                test_y_true=y_test,
                test_pred=model.predict(specialist_only_test_features).astype(np.int64),
                selection_metric=selection_metric,
            )
        )

    if val_base_splits:
        with_base_val_features = build_meta_features(val_bank["probs"], val_bank["logits"], val_base_splits)
        with_base_test_features = build_meta_features(test_bank["probs"], test_bank["logits"], test_base_splits)
        base_labels = [label for label, _split in val_base_splits]
        for class_weight_label, class_weight in (("none", None), ("balanced", "balanced")):
            model = fit_meta_model(with_base_val_features, y_val, class_weight=class_weight)
            records.append(
                candidate_record(
                    name=f"meta_specialists_plus_base_lr_{class_weight_label}",
                    kind="meta_logistic_regression",
                    params={
                        "features": "specialist_features_plus_base_prediction_features",
                        "base_predictions": base_labels,
                        "fit_split": "validation",
                        "class_weight": class_weight_label,
                    },
                    val_y_true=y_val,
                    val_pred=model.predict(with_base_val_features).astype(np.int64),
                    test_y_true=y_test,
                    test_pred=model.predict(with_base_test_features).astype(np.int64),
                    selection_metric=selection_metric,
                )
            )

    best = max(records, key=lambda item: item["selection_score"])
    report = {
        "specialist_predictions": [str(path) for path in specialist_predictions],
        "base_predictions": {label: str(path) for label, path in base_predictions},
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
    print("| rank | name | val score | 4 Macro | 4 Kappa | Wake | N3 | REM |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|")
    for rank, record in enumerate(records[:limit], start=1):
        summary = record["test"]["summary"]
        print(
            f"| {rank} | {record['name']} | {record['selection_score']:.4f} | "
            f"{summary['4_macro_f1']:.4f} | {summary['4_kappa']:.4f} | "
            f"{summary['wake_f1']:.4f} | {summary['n3_f1']:.4f} | {summary['rem_f1']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full one-vs-rest specialist fusion.")
    parser.add_argument("--specialist-predictions", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--base-predictions",
        action="append",
        type=parse_labeled_path,
        default=[],
        metavar="LABEL=PATH",
        help="Optional 5-class prediction NPZ to include as a baseline and meta-fusion feature.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
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
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    labeled_base_predictions = [
        (label if label is not None else f"base{idx + 1}", path)
        for idx, (label, path) in enumerate(args.base_predictions)
    ]
    report = evaluate_specialist_fusion(
        specialist_predictions=args.specialist_predictions,
        out_json=args.out_json,
        base_predictions=labeled_base_predictions,
        selection_metric=args.selection_metric,
    )
    print_top(report, limit=args.top)


if __name__ == "__main__":
    main()
