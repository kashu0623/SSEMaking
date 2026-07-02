"""Build subject-wise train/val/test NPZ datasets from DreamT epoch feature CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .labels import STAGE5_NAMES


METADATA_COLUMNS = {
    "subject_id",
    "source_file",
    "start_row",
    "end_row",
    "aligned_epoch_index",
    "raw_label",
    "label_5",
    "class_id_5",
}


@dataclass(frozen=True)
class EpochRow:
    subject_id: str
    aligned_epoch_index: int
    class_id_5: int
    features: list[float]


@dataclass(frozen=True)
class SplitSummary:
    name: str
    subjects: list[str]
    epochs: int
    samples: int
    label_counts: dict[str, int]


def parse_float_or_nan(value: str) -> float:
    text = value.strip()
    if not text:
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def read_epoch_feature_csv(path: Path) -> tuple[list[str], dict[str, list[EpochRow]]]:
    """Read epoch feature CSV grouped by subject."""
    by_subject: dict[str, list[EpochRow]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        feature_names = [name for name in reader.fieldnames if name not in METADATA_COLUMNS]
        for row in reader:
            subject_id = row["subject_id"]
            by_subject[subject_id].append(
                EpochRow(
                    subject_id=subject_id,
                    aligned_epoch_index=int(row["aligned_epoch_index"]),
                    class_id_5=int(row["class_id_5"]),
                    features=[parse_float_or_nan(row[name]) for name in feature_names],
                )
            )

    for rows in by_subject.values():
        rows.sort(key=lambda item: item.aligned_epoch_index)
    return feature_names, dict(by_subject)


def split_subjects(
    subjects: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Deterministically split subjects without subject leakage."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    shuffled = list(subjects)
    random.Random(seed).shuffle(shuffled)
    train_end = int(round(len(shuffled) * train_ratio))
    val_end = train_end + int(round(len(shuffled) * val_ratio))
    return {
        "train": sorted(shuffled[:train_end]),
        "val": sorted(shuffled[train_end:val_end]),
        "test": sorted(shuffled[val_end:]),
    }


def build_sequences_for_subject(
    rows: Sequence[EpochRow],
    context_epochs: int,
    require_contiguous: bool,
) -> tuple[list[list[list[float]]], list[int], list[int]]:
    """Build causal context windows for one subject."""
    if context_epochs <= 0:
        raise ValueError("context_epochs must be positive")

    features: list[list[list[float]]] = []
    labels: list[int] = []
    target_epoch_indices: list[int] = []
    for end_idx in range(context_epochs - 1, len(rows)):
        window = rows[end_idx - context_epochs + 1 : end_idx + 1]
        if require_contiguous:
            expected = list(range(window[0].aligned_epoch_index, window[0].aligned_epoch_index + context_epochs))
            actual = [row.aligned_epoch_index for row in window]
            if actual != expected:
                continue
        features.append([row.features for row in window])
        labels.append(rows[end_idx].class_id_5)
        target_epoch_indices.append(rows[end_idx].aligned_epoch_index)
    return features, labels, target_epoch_indices


def build_split_arrays(
    by_subject: dict[str, list[EpochRow]],
    subjects: Sequence[str],
    context_epochs: int,
    require_contiguous: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_features: list[list[list[float]]] = []
    all_labels: list[int] = []
    all_subject_ids: list[str] = []
    all_epoch_indices: list[int] = []

    for subject_id in subjects:
        features, labels, epoch_indices = build_sequences_for_subject(
            by_subject[subject_id],
            context_epochs=context_epochs,
            require_contiguous=require_contiguous,
        )
        all_features.extend(features)
        all_labels.extend(labels)
        all_subject_ids.extend([subject_id] * len(labels))
        all_epoch_indices.extend(epoch_indices)

    if all_features:
        x = np.asarray(all_features, dtype=np.float32)
    else:
        feature_count = len(next(iter(by_subject.values()))[0].features)
        x = np.empty((0, context_epochs, feature_count), dtype=np.float32)
    return (
        x,
        np.asarray(all_labels, dtype=np.int64),
        np.asarray(all_subject_ids),
        np.asarray(all_epoch_indices, dtype=np.int64),
    )


def normalization_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_train.size == 0:
        raise ValueError("Cannot compute normalization stats from an empty training set")
    flat = x_train.reshape(-1, x_train.shape[-1])
    mean = np.nanmean(flat, axis=0)
    std = np.nanstd(flat, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_with_train_stats(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    filled = np.where(np.isnan(x), mean.reshape(1, 1, -1), x)
    return ((filled - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)).astype(np.float32)


def label_count_dict(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(label) for label in labels.tolist())
    return {STAGE5_NAMES[label_id]: counts.get(label_id, 0) for label_id in range(len(STAGE5_NAMES))}


def split_summary(name: str, subjects: list[str], by_subject: dict[str, list[EpochRow]], labels: np.ndarray) -> SplitSummary:
    return SplitSummary(
        name=name,
        subjects=subjects,
        epochs=sum(len(by_subject[subject]) for subject in subjects),
        samples=int(labels.shape[0]),
        label_counts=label_count_dict(labels),
    )


def build_npz_dataset(
    input_csv: Path,
    out_path: Path,
    summary_path: Path,
    context_epochs: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    require_contiguous: bool,
) -> dict[str, object]:
    feature_names, by_subject = read_epoch_feature_csv(input_csv)
    split = split_subjects(sorted(by_subject), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for split_name, subjects in split.items():
        arrays[split_name] = build_split_arrays(
            by_subject=by_subject,
            subjects=subjects,
            context_epochs=context_epochs,
            require_contiguous=require_contiguous,
        )

    x_train, y_train, train_subject_ids, train_epoch_indices = arrays["train"]
    mean, std = normalization_stats(x_train)
    x_train = normalize_with_train_stats(x_train, mean, std)
    x_val, y_val, val_subject_ids, val_epoch_indices = arrays["val"]
    x_test, y_test, test_subject_ids, test_epoch_indices = arrays["test"]
    x_val = normalize_with_train_stats(x_val, mean, std)
    x_test = normalize_with_train_stats(x_test, mean, std)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_train=x_train,
        y_train=y_train,
        train_subject_ids=train_subject_ids,
        train_epoch_indices=train_epoch_indices,
        X_val=x_val,
        y_val=y_val,
        val_subject_ids=val_subject_ids,
        val_epoch_indices=val_epoch_indices,
        X_test=x_test,
        y_test=y_test,
        test_subject_ids=test_subject_ids,
        test_epoch_indices=test_epoch_indices,
        feature_names=np.asarray(feature_names),
        stage5_names=np.asarray(STAGE5_NAMES),
        train_feature_mean=mean,
        train_feature_std=std,
        context_epochs=np.asarray(context_epochs),
        require_contiguous=np.asarray(require_contiguous),
    )

    summary = {
        "input_csv": str(input_csv),
        "out_path": str(out_path),
        "context_epochs": context_epochs,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": 1.0 - train_ratio - val_ratio,
        "seed": seed,
        "require_contiguous": require_contiguous,
        "subject_count": len(by_subject),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "splits": [
            asdict(split_summary("train", split["train"], by_subject, y_train)),
            asdict(split_summary("val", split["val"], by_subject, y_val)),
            asdict(split_summary("test", split["test"], by_subject, y_test)),
        ],
        "array_shapes": {
            "X_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "X_val": list(x_val.shape),
            "y_val": list(y_val.shape),
            "X_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/test NPZ from DreamT epoch feature CSV.")
    parser.add_argument("--input-csv", type=Path, required=True, help="dreamt_100hz_epoch_features.csv path.")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz path.")
    parser.add_argument("--summary-out", type=Path, required=True, help="Output summary JSON path.")
    parser.add_argument("--context-epochs", type=int, default=10, help="Causal epoch context length.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Subject-wise train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Subject-wise validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic subject split seed.")
    parser.add_argument(
        "--allow-noncontiguous",
        action="store_true",
        help="Allow context windows across missing aligned_epoch_index gaps.",
    )
    args = parser.parse_args()

    summary = build_npz_dataset(
        input_csv=args.input_csv,
        out_path=args.out,
        summary_path=args.summary_out,
        context_epochs=args.context_epochs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        require_contiguous=not args.allow_noncontiguous,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

