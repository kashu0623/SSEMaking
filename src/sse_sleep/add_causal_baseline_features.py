"""Add causal subject/session-local baseline features to epoch feature CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_BASE_FEATURES = (
    "acc_vm_activity",
    "hr_mean",
    "ibi_mean",
    "temp_mean",
    "bvp_std",
)

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


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float | None:
        if self.count < 2:
            return None
        return (self.m2 / self.count) ** 0.5


def parse_feature_list(text: str | None) -> tuple[str, ...]:
    if text is None:
        return DEFAULT_BASE_FEATURES
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_float_or_nan(value: str) -> float:
    text = value.strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def optional_float_text(value: float | None) -> str:
    if value is None or value != value:
        return ""
    return f"{value:.10g}"


def is_contiguous(previous_epoch: int | None, current_epoch: int) -> bool:
    return previous_epoch is not None and current_epoch == previous_epoch + 1


def added_feature_names(base_features: Sequence[str]) -> list[str]:
    names: list[str] = []
    for feature in base_features:
        names.append(f"{feature}_expanding_mean")
        names.append(f"{feature}_expanding_std")
        names.append(f"{feature}_causal_zscore")
    return names


def add_causal_baseline_features_to_rows(
    rows: list[dict[str, str]],
    base_features: Sequence[str],
    min_std: float,
) -> list[dict[str, str]]:
    stats = {feature: RunningStats() for feature in base_features}
    previous_epoch: int | None = None
    output_rows: list[dict[str, str]] = []

    for row in sorted(rows, key=lambda item: int(item["aligned_epoch_index"])):
        epoch_index = int(row["aligned_epoch_index"])
        if not is_contiguous(previous_epoch, epoch_index):
            stats = {feature: RunningStats() for feature in base_features}

        out_row = dict(row)
        values = {feature: parse_float_or_nan(row.get(feature, "")) for feature in base_features}

        for feature in base_features:
            current_stats = stats[feature]
            current_value = values[feature]
            if current_stats.count > 0:
                out_row[f"{feature}_expanding_mean"] = optional_float_text(current_stats.mean)
            else:
                out_row[f"{feature}_expanding_mean"] = ""

            feature_std = current_stats.std
            out_row[f"{feature}_expanding_std"] = optional_float_text(feature_std)
            if (
                feature_std is not None
                and feature_std > min_std
                and current_value == current_value
            ):
                out_row[f"{feature}_causal_zscore"] = optional_float_text((current_value - current_stats.mean) / feature_std)
            else:
                out_row[f"{feature}_causal_zscore"] = ""

        for feature, value in values.items():
            if value == value:
                stats[feature].update(value)

        previous_epoch = epoch_index
        output_rows.append(out_row)

    return output_rows


def read_grouped_rows(path: Path) -> tuple[list[str], dict[tuple[str, str], list[dict[str, str]]]]:
    by_session: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            by_session[(row["subject_id"], row.get("source_file", ""))].append(row)
    return fieldnames, dict(by_session)


def add_causal_baseline_features(
    input_csv: Path,
    out_csv: Path,
    summary_out: Path,
    base_features: Sequence[str],
    min_std: float,
) -> dict[str, object]:
    if min_std <= 0:
        raise ValueError("min_std must be positive")

    fieldnames, by_session = read_grouped_rows(input_csv)
    missing = [feature for feature in base_features if feature not in fieldnames]
    if missing:
        raise ValueError(f"Missing base features in input CSV: {missing}")

    added_names = added_feature_names(base_features)
    out_fieldnames = list(fieldnames) + added_names
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fieldnames)
        writer.writeheader()
        for session_key in sorted(by_session):
            out_rows = add_causal_baseline_features_to_rows(
                rows=by_session[session_key],
                base_features=base_features,
                min_std=min_std,
            )
            writer.writerows(out_rows)
            rows_written += len(out_rows)

    original_feature_count = len([name for name in fieldnames if name not in METADATA_COLUMNS])
    summary = {
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "session_count": len(by_session),
        "rows_written": rows_written,
        "base_features": list(base_features),
        "history_mode": "prior_only",
        "min_std": min_std,
        "added_feature_count": len(added_names),
        "added_features": added_names,
        "original_feature_count": original_feature_count,
        "new_feature_count": original_feature_count + len(added_names),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add causal per-night baseline features to epoch feature CSV.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--base-features", default=None, help="Comma-separated feature names.")
    parser.add_argument("--min-std", type=float, default=1e-8, help="Minimum prior std for z-score output.")
    args = parser.parse_args()

    summary = add_causal_baseline_features(
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        summary_out=args.summary_out,
        base_features=parse_feature_list(args.base_features),
        min_std=args.min_std,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
