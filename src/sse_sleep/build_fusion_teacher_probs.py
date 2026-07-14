"""Build fixed class-wise fusion teacher probabilities for distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_prediction_fusion import classwise_fusion, evaluate_probs, load_split, summarize_metrics, validate_alignment
from .labels import STAGE5_NAMES, STAGE5_TO_ID


def split_available(path: Path, split: str) -> bool:
    with np.load(path, allow_pickle=True) as data:
        return f"{split}_y_true" in data.files and f"{split}_probs" in data.files


def build_class_alphas(non_rem_alpha: float, rem_alpha: float) -> np.ndarray:
    class_alphas = np.full(len(STAGE5_NAMES), float(non_rem_alpha), dtype=np.float32)
    class_alphas[STAGE5_TO_ID["REM"]] = float(rem_alpha)
    return class_alphas


def build_fusion_teacher_probs(
    base_predictions: Path,
    candidate_predictions: Path,
    out_npz: Path,
    out_json: Path,
    non_rem_alpha: float,
    rem_alpha: float,
    splits: Sequence[str],
) -> dict[str, Any]:
    class_alphas = build_class_alphas(non_rem_alpha=non_rem_alpha, rem_alpha=rem_alpha)
    output_arrays: dict[str, np.ndarray] = {
        "stage5_names": np.asarray(STAGE5_NAMES),
        "class_alphas": class_alphas,
    }
    split_reports: dict[str, Any] = {}

    for split in splits:
        if not split_available(base_predictions, split) or not split_available(candidate_predictions, split):
            continue
        base = load_split(base_predictions, split)
        candidate = load_split(candidate_predictions, split)
        validate_alignment(base, candidate, split)
        fused_probs = classwise_fusion(base["probs"], candidate["probs"], class_alphas)
        y_pred = fused_probs.argmax(axis=1).astype(np.int64)

        output_arrays[f"{split}_y_true"] = base["y_true"]
        output_arrays[f"{split}_probs"] = fused_probs.astype(np.float32)
        output_arrays[f"{split}_y_pred"] = y_pred
        for key in ("subject_ids", "epoch_indices"):
            if key in base:
                output_arrays[f"{split}_{key}"] = base[key]

        metrics = evaluate_probs(base["y_true"], fused_probs)
        split_reports[split] = {
            "sample_count": int(base["y_true"].shape[0]),
            "metrics": metrics,
            "summary": summarize_metrics(metrics),
        }

    if "train_probs" not in output_arrays:
        raise ValueError("Teacher probabilities require train split probabilities in both prediction files")

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **output_arrays)

    report = {
        "base_predictions": str(base_predictions),
        "candidate_predictions": str(candidate_predictions),
        "out_npz": str(out_npz),
        "non_rem_alpha": non_rem_alpha,
        "rem_alpha": rem_alpha,
        "class_alphas": {name: float(class_alphas[idx]) for idx, name in enumerate(STAGE5_NAMES)},
        "splits": split_reports,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed class-wise fusion teacher probabilities.")
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--out-npz", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--non-rem-alpha", type=float, default=0.9)
    parser.add_argument("--rem-alpha", type=float, default=0.2)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    args = parser.parse_args()

    report = build_fusion_teacher_probs(
        base_predictions=args.base_predictions,
        candidate_predictions=args.candidate_predictions,
        out_npz=args.out_npz,
        out_json=args.out_json,
        non_rem_alpha=args.non_rem_alpha,
        rem_alpha=args.rem_alpha,
        splits=args.splits,
    )
    print(json.dumps({split: data["summary"] for split, data in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
