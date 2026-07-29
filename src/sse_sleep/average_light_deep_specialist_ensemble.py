"""Average aligned Light-vs-Deep specialist prediction files."""

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


def average_split(paths: Sequence[Path], split: str) -> dict[str, np.ndarray]:
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
    probabilities = np.mean(
        [item["probs"] for item in loaded],
        axis=0,
        dtype=np.float32,
    )
    result = {
        "y_true": reference["y_true"],
        "probs": probabilities.astype(np.float32),
        "y_pred": probabilities.argmax(axis=1).astype(np.int64),
    }
    for key in ("subject_ids", "epoch_indices"):
        if key in reference:
            result[key] = reference[key]
    return result


def average_predictions(paths: Sequence[Path], out_path: Path) -> dict[str, Any]:
    splits = {
        split: average_split(paths, split)
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
        "member_count": len(paths),
        "out_path": str(out_path),
        "val_rows": int(splits["val"]["y_true"].shape[0]),
        "test_rows": int(splits["test"]["y_true"].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average aligned Light-vs-Deep specialist probabilities."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    args = parser.parse_args()
    report = average_predictions(args.predictions, args.out)
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
