"""Add subject-local rolling and delta features to DreamT epoch feature CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Sequence


DEFAULT_BASE_FEATURES = (
    "bvp_mean",
    "bvp_std",
    "acc_vm_mean",
    "acc_vm_activity",
    "temp_mean",
    "temp_slope",
    "hr_mean",
    "hr_std",
    "ibi_mean",
    "ibi_std",
)

DEFAULT_DELTA_LAGS = (1, 3)
DEFAULT_ROLLING_WINDOWS = (3, 5)
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


def format_suffix(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def parse_int_list(values: Sequence[str] | str | None) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raw_parts = values.split(",")
    else:
        raw_parts = []
        for value in values:
            raw_parts.extend(value.split(","))
    parsed = tuple(int(part.strip()) for part in raw_parts if part.strip())
    if any(value <= 0 for value in parsed):
        raise ValueError("All window/lag values must be positive")
    return parsed


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


def is_contiguous(previous_epoch: int | None, current_epoch: int) -> bool:
    return previous_epoch is not None and current_epoch == previous_epoch + 1


def mean(values: Sequence[float]) -> float | None:
    clean = [value for value in values if value == value]
    if not clean:
        return None
    return sum(clean) / len(clean)


def std(values: Sequence[float]) -> float | None:
    clean = [value for value in values if value == value]
    if len(clean) < 2:
        return 0.0 if len(clean) == 1 else None
    avg = sum(clean) / len(clean)
    return (sum((value - avg) ** 2 for value in clean) / len(clean)) ** 0.5


def optional_float_text(value: float | None) -> str:
    if value is None or value != value:
        return ""
    return f"{value:.10g}"


def added_feature_names(
    base_features: Sequence[str],
    delta_lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> list[str]:
    names: list[str] = []
    for feature in base_features:
        names.extend(f"{feature}_delta_{lag}" for lag in delta_lags)
        for window in rolling_windows:
            names.append(f"{feature}_roll_mean_{window}")
            names.append(f"{feature}_roll_std_{window}")
    return names


def add_temporal_features_to_rows(
    rows: list[dict[str, str]],
    fieldnames: Sequence[str],
    base_features: Sequence[str],
    delta_lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> list[dict[str, str]]:
    max_history = max((*delta_lags, *rolling_windows), default=0)
    history: deque[dict[str, float]] = deque(maxlen=max_history)
    previous_epoch: int | None = None
    output_rows: list[dict[str, str]] = []

    for row in sorted(rows, key=lambda item: int(item["aligned_epoch_index"])):
        epoch_index = int(row["aligned_epoch_index"])
        if not is_contiguous(previous_epoch, epoch_index):
            history.clear()
        values = {feature: parse_float_or_nan(row.get(feature, "")) for feature in base_features}
        out_row = dict(row)

        for feature in base_features:
            current = values[feature]
            for lag in delta_lags:
                if len(history) >= lag:
                    previous = history[-lag].get(feature)
                    if previous is not None and previous == previous and current == current:
                        out_row[f"{feature}_delta_{lag}"] = optional_float_text(current - previous)
                    else:
                        out_row[f"{feature}_delta_{lag}"] = ""
                else:
                    out_row[f"{feature}_delta_{lag}"] = ""

            for window in rolling_windows:
                if len(history) >= window:
                    window_values = [history[-idx].get(feature, float("nan")) for idx in range(1, window + 1)]
                    out_row[f"{feature}_roll_mean_{window}"] = optional_float_text(mean(window_values))
                    out_row[f"{feature}_roll_std_{window}"] = optional_float_text(std(window_values))
                else:
                    out_row[f"{feature}_roll_mean_{window}"] = ""
                    out_row[f"{feature}_roll_std_{window}"] = ""

        history.append(values)
        previous_epoch = epoch_index
        output_rows.append(out_row)

    return output_rows


def read_grouped_rows(path: Path) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    by_subject: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        for row in reader:
            by_subject[row["subject_id"]].append(row)
    return fieldnames, dict(by_subject)


def add_temporal_features(
    input_csv: Path,
    out_csv: Path,
    summary_out: Path,
    base_features: Sequence[str],
    delta_lags: Sequence[int],
    rolling_windows: Sequence[int],
) -> dict[str, object]:
    fieldnames, by_subject = read_grouped_rows(input_csv)
    missing = [feature for feature in base_features if feature not in fieldnames]
    if missing:
        raise ValueError(f"Missing base features in input CSV: {missing}")

    added_names = added_feature_names(base_features, delta_lags, rolling_windows)
    out_fieldnames = list(fieldnames) + added_names
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fieldnames)
        writer.writeheader()
        for subject_id in sorted(by_subject):
            out_rows = add_temporal_features_to_rows(
                rows=by_subject[subject_id],
                fieldnames=fieldnames,
                base_features=base_features,
                delta_lags=delta_lags,
                rolling_windows=rolling_windows,
            )
            writer.writerows(out_rows)
            rows_written += len(out_rows)

    original_feature_count = len([name for name in fieldnames if name not in METADATA_COLUMNS])
    summary = {
        "input_csv": str(input_csv),
        "out_csv": str(out_csv),
        "subject_count": len(by_subject),
        "rows_written": rows_written,
        "base_features": list(base_features),
        "delta_lags": list(delta_lags),
        "rolling_windows": list(rolling_windows),
        "added_feature_count": len(added_names),
        "added_features": added_names,
        "original_feature_count": original_feature_count,
        "new_feature_count": original_feature_count + len(added_names),
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Add rolling/delta temporal features to epoch feature CSV.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--base-features", default=None, help="Comma-separated feature names. Defaults to core physiological features.")
    parser.add_argument("--delta-lags", default=format_suffix(DEFAULT_DELTA_LAGS), help="Comma-separated epoch lags.")
    parser.add_argument("--rolling-windows", default=format_suffix(DEFAULT_ROLLING_WINDOWS), help="Comma-separated rolling windows.")
    parser.add_argument(
        "--delta-windows",
        nargs="+",
        default=None,
        help="Alias for --delta-lags. Accepts space-separated or comma-separated epoch lags.",
    )
    parser.add_argument(
        "--rolling-window-list",
        nargs="+",
        default=None,
        help="Alias for --rolling-windows. Accepts space-separated or comma-separated rolling windows.",
    )
    args = parser.parse_args()
    delta_values = args.delta_windows if args.delta_windows is not None else args.delta_lags
    rolling_values = args.rolling_window_list if args.rolling_window_list is not None else args.rolling_windows

    summary = add_temporal_features(
        input_csv=args.input_csv,
        out_csv=args.out_csv,
        summary_out=args.summary_out,
        base_features=parse_feature_list(args.base_features),
        delta_lags=parse_int_list(delta_values),
        rolling_windows=parse_int_list(rolling_values),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
