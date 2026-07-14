"""Evaluate validation-selected REM threshold policies for saved sleep-stage probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_prediction_fusion import (
    json_ready,
    load_split,
    metric_value,
    parse_float_list,
    summarize_metrics,
)
from .labels import STAGE5_NAMES, STAGE5_TO_ID
from .metrics import evaluate_5_and_4


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    return json_ready(evaluate_5_and_4(y_true.astype(int).tolist(), y_pred.astype(int).tolist()))


def rem_threshold_predictions(probs: np.ndarray, threshold: float) -> np.ndarray:
    y_pred = probs.argmax(axis=1).astype(np.int64)
    rem_index = STAGE5_TO_ID["REM"]
    y_pred[probs[:, rem_index] >= threshold] = rem_index
    return y_pred


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


def evaluate_rem_threshold(
    predictions: Path,
    out_json: Path,
    thresholds: Sequence[float],
    selection_metric: str,
) -> dict[str, Any]:
    val = load_split(predictions, "val")
    test = load_split(predictions, "test")

    records: list[dict[str, Any]] = []
    val_argmax = val["probs"].argmax(axis=1).astype(np.int64)
    test_argmax = test["probs"].argmax(axis=1).astype(np.int64)
    records.append(
        candidate_record(
            name="argmax_baseline",
            kind="baseline",
            params={},
            val_metrics=evaluate_predictions(val["y_true"], val_argmax),
            test_metrics=evaluate_predictions(test["y_true"], test_argmax),
            selection_metric=selection_metric,
        )
    )

    for threshold in thresholds:
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1]: {threshold}")
        records.append(
            candidate_record(
                name=f"rem_threshold{threshold:.2f}",
                kind="rem_threshold",
                params={"rem_threshold": threshold},
                val_metrics=evaluate_predictions(val["y_true"], rem_threshold_predictions(val["probs"], threshold)),
                test_metrics=evaluate_predictions(test["y_true"], rem_threshold_predictions(test["probs"], threshold)),
                selection_metric=selection_metric,
            )
        )

    best = max(records, key=lambda item: item["selection_score"])
    report = {
        "predictions": str(predictions),
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
    parser = argparse.ArgumentParser(description="Evaluate REM threshold policies selected on validation.")
    parser.add_argument("--predictions", type=Path, required=True, help="Prediction NPZ with val_probs/test_probs.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
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

    report = evaluate_rem_threshold(
        predictions=args.predictions,
        out_json=args.out_json,
        thresholds=parse_float_list(args.thresholds, default=[]),
        selection_metric=args.selection_metric,
    )
    print_top(report, limit=args.top)


if __name__ == "__main__":
    main()
