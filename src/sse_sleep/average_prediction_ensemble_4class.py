"""Average aligned direct four-class prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_four_model_4class_fusion import load_split, summarize, validate


def average_split(paths: Sequence[Path], split: str) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("At least one prediction path is required")
    loaded = [load_split(path, split) for path in paths]
    reference = loaded[0]
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
    splits = {split: average_split(paths, split) for split in ("val", "test")}
    arrays = {
        f"{split}_{key}": value
        for split, data in splits.items()
        for key, value in data.items()
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {
        "members": [str(path) for path in paths],
        "member_count": len(paths),
        "out_path": str(out_path),
        "val_metrics": summarize(splits["val"]["y_true"], splits["val"]["probs"]),
        "test_metrics": summarize(splits["test"]["y_true"], splits["test"]["probs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average aligned direct four-class LSTM predictions."
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
