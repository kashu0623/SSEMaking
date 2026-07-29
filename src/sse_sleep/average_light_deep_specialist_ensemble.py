"""Average or weight-blend aligned Light-vs-Deep specialist predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_prediction_fusion import load_split


def validate(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    label: str,
) -> None:
    if not np.array_equal(reference["y_true"], candidate["y_true"]):
        raise ValueError(f"Unaligned labels for {label}")
    if reference["probs"].shape != candidate["probs"].shape:
        raise ValueError(f"Unaligned probabilities for {label}")
    for key in ("subject_ids", "epoch_indices"):
        if (
            key in reference
            and key in candidate
            and not np.array_equal(reference[key], candidate[key])
        ):
            raise ValueError(f"Unaligned {key} for {label}")


def normalize_weights(
    weights: Sequence[float] | None,
    member_count: int,
) -> np.ndarray:
    if member_count < 1:
        raise ValueError("At least one prediction path is required")
    if weights is None:
        return np.full(member_count, 1.0 / member_count, dtype=np.float64)
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (member_count,):
        raise ValueError(
            f"Expected {member_count} weights, got shape {values.shape}"
        )
    if np.any(values < 0.0) or not np.isfinite(values).all():
        raise ValueError("Weights must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("At least one weight must be positive")
    return values / total


def average_split(
    paths: Sequence[Path],
    split: str,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("At least one prediction path is required")
    loaded = [load_split(path, split) for path in paths]
    reference = loaded[0]
    if reference["probs"].ndim != 2 or reference["probs"].shape[1] != 2:
        raise ValueError(
            f"Expected Light/Deep probabilities, got {reference['probs'].shape}"
        )
    for index, candidate in enumerate(loaded[1:], start=2):
        validate(reference, candidate, f"{split} member {index}")
    probabilities = np.average(
        np.stack([item["probs"] for item in loaded], axis=0),
        axis=0,
        weights=weights,
    ).astype(np.float32)
    result = {
        "y_true": reference["y_true"],
        "probs": probabilities.astype(np.float32),
        "y_pred": probabilities.argmax(axis=1).astype(np.int64),
    }
    for key in ("subject_ids", "epoch_indices"):
        if key in reference:
            result[key] = reference[key]
    return result


def average_predictions(
    paths: Sequence[Path],
    out_path: Path,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    normalized_weights = normalize_weights(weights, len(paths))
    splits = {
        split: average_split(paths, split, normalized_weights)
        for split in ("val", "test")
    }
    arrays = {
        f"{split}_{key}": value
        for split, data in splits.items()
        for key, value in data.items()
    }
    arrays["specialist_names"] = np.asarray(("Light", "Deep"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {
        "members": [str(path) for path in paths],
        "weights": normalized_weights.tolist(),
        "member_count": len(paths),
        "out_path": str(out_path),
        "val_rows": int(splits["val"]["y_true"].shape[0]),
        "test_rows": int(splits["test"]["y_true"].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average or weight-blend Light-vs-Deep specialist probabilities."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional comma-separated member weights.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()
    weights = (
        None
        if args.weights is None
        else [
            float(part.strip())
            for part in args.weights.split(",")
            if part.strip()
        ]
    )
    report = average_predictions(args.predictions, args.out, weights)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "members"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
