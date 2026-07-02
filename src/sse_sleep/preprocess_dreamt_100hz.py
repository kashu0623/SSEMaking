"""Stream DreamT 100Hz CSV files into 30-second epoch feature rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .dreamt_100hz import (
    LABEL_COLUMN,
    ROWS_PER_EPOCH,
    TIMESTAMP_COLUMN,
    subject_id_from_path,
)
from .features import acc_epoch_features, basic_stats, quality_features, temp_epoch_features
from .labels import STAGE5_NAMES, map_label_5


BASE_COLUMNS = (
    "BVP",
    "ACC_X",
    "ACC_Y",
    "ACC_Z",
    "TEMP",
    "HR",
    "IBI",
)


@dataclass
class AlignmentSummary:
    offset: int
    transition_offset_counts: dict[int, int]
    transitions_considered: int


@dataclass
class FilePreprocessSummary:
    path: str
    subject_id: str
    alignment_offset: int
    rows_read: int
    epochs_written: int
    skipped_ignored_label: int
    skipped_mixed_label: int
    skipped_partial: int
    label_counts: dict[str, int]
    transition_offset_counts: dict[int, int]


def parse_float(value: str) -> float | None:
    text = value.strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric != numeric:
        return None
    return numeric


def require_columns(header: Sequence[str], columns: Sequence[str]) -> dict[str, int]:
    missing = [column for column in columns if column not in header]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return {column: header.index(column) for column in columns}


def detect_alignment_offset(
    path: Path,
    rows_per_epoch: int,
    max_rows: int | None = None,
) -> AlignmentSummary:
    """Find the dominant 30-second boundary offset from non-ignored stage transitions."""
    offset_counts: Counter[int] = Counter()
    previous_value: str | None = None
    rows_read = 0

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        indices = require_columns(header, (LABEL_COLUMN,))
        stage_index = indices[LABEL_COLUMN]
        for row_index, row in enumerate(reader):
            if max_rows is not None and rows_read >= max_rows:
                break
            value = row[stage_index].strip() if stage_index < len(row) else ""
            if previous_value is not None and value != previous_value:
                previous_mapped = map_label_5(previous_value)
                current_mapped = map_label_5(value)
                if not previous_mapped.ignored and not current_mapped.ignored:
                    offset_counts[row_index % rows_per_epoch] += 1
            previous_value = value
            rows_read += 1

    if offset_counts:
        offset = offset_counts.most_common(1)[0][0]
    else:
        offset = 0
    return AlignmentSummary(
        offset=offset,
        transition_offset_counts=dict(offset_counts.most_common()),
        transitions_considered=sum(offset_counts.values()),
    )


def empty_epoch_buffer(columns: Sequence[str]) -> dict[str, list[float | None]]:
    return {column: [] for column in columns}


def append_epoch_row(
    buffer: dict[str, list[float | None]],
    row: Sequence[str],
    indices: dict[str, int],
    columns: Sequence[str],
) -> None:
    for column in columns:
        column_index = indices[column]
        value = row[column_index] if column_index < len(row) else ""
        buffer[column].append(parse_float(value))


def extract_dreamt_epoch_features(
    buffer: dict[str, list[float | None]],
    include_sao2: bool,
) -> dict[str, float | None]:
    features: dict[str, float | None] = {}
    features.update(basic_stats(buffer["BVP"], "bvp"))
    features.update(quality_features(buffer["BVP"], "bvp"))
    features.update(acc_epoch_features(buffer["ACC_X"], buffer["ACC_Y"], buffer["ACC_Z"]))
    features.update(temp_epoch_features(buffer["TEMP"]))
    features.update(basic_stats(buffer["HR"], "hr"))
    features.update(quality_features(buffer["HR"], "hr"))
    features.update(basic_stats(buffer["IBI"], "ibi"))
    features.update(quality_features(buffer["IBI"], "ibi"))
    if include_sao2:
        features.update(basic_stats(buffer["SAO2"], "sao2"))
        features.update(quality_features(buffer["SAO2"], "sao2"))
    return features


def feature_fieldnames(include_sao2: bool) -> list[str]:
    columns = list(BASE_COLUMNS)
    if include_sao2:
        columns.append("SAO2")
    dummy = {column: [None] for column in columns}
    return list(extract_dreamt_epoch_features(dummy, include_sao2).keys())


def stage_window_label(stage_values: Sequence[str]) -> tuple[str | None, str | None, int | None, str]:
    counts = Counter(stage_values)
    non_empty_values = {value for value in counts if value}
    if len(non_empty_values) != 1:
        return None, None, None, "mixed"
    raw_label = next(iter(non_empty_values))
    mapped = map_label_5(raw_label)
    if mapped.ignored or mapped.canonical is None or mapped.class_id_5 is None:
        return raw_label, None, None, "ignored"
    return raw_label, mapped.canonical, mapped.class_id_5, "ok"


def iter_subject_files(root: Path, pattern: str, limit_files: int | None) -> list[Path]:
    paths = sorted(root.glob(pattern))
    if limit_files is not None:
        paths = paths[:limit_files]
    return paths


def preprocess_file(
    path: Path,
    writer: csv.DictWriter,
    include_sao2: bool,
    rows_per_epoch: int,
    max_rows: int | None,
    progress_rows: int,
) -> FilePreprocessSummary:
    subject_id = subject_id_from_path(path)
    alignment = detect_alignment_offset(path, rows_per_epoch=rows_per_epoch, max_rows=max_rows)
    input_columns = list(BASE_COLUMNS)
    if include_sao2:
        input_columns.append("SAO2")

    rows_read = 0
    epochs_written = 0
    skipped_ignored = 0
    skipped_mixed = 0
    label_counts: Counter[str] = Counter()
    stage_values: list[str] = []
    buffer = empty_epoch_buffer(input_columns)
    window_start_row: int | None = None

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        required_columns = [TIMESTAMP_COLUMN, LABEL_COLUMN, *input_columns]
        indices = require_columns(header, required_columns)
        stage_index = indices[LABEL_COLUMN]

        for row_index, row in enumerate(reader):
            if max_rows is not None and rows_read >= max_rows:
                break
            if window_start_row is None:
                if row_index % rows_per_epoch != alignment.offset:
                    rows_read += 1
                    continue
                window_start_row = row_index
                stage_values = []
                buffer = empty_epoch_buffer(input_columns)

            stage_value = row[stage_index].strip() if stage_index < len(row) else ""
            stage_values.append(stage_value)
            append_epoch_row(buffer, row, indices, input_columns)
            rows_read += 1

            if len(stage_values) == rows_per_epoch:
                raw_label, canonical_label, class_id_5, status = stage_window_label(stage_values)
                if status == "ok" and raw_label is not None and canonical_label is not None and class_id_5 is not None:
                    features = extract_dreamt_epoch_features(buffer, include_sao2)
                    label_counts[canonical_label] += 1
                    row_out: dict[str, object] = {
                        "subject_id": subject_id,
                        "source_file": path.name,
                        "start_row": window_start_row,
                        "end_row": window_start_row + rows_per_epoch,
                        "aligned_epoch_index": (window_start_row - alignment.offset) // rows_per_epoch,
                        "raw_label": raw_label,
                        "label_5": canonical_label,
                        "class_id_5": class_id_5,
                    }
                    row_out.update(features)
                    writer.writerow(row_out)
                    epochs_written += 1
                elif status == "ignored":
                    skipped_ignored += 1
                else:
                    skipped_mixed += 1

                if progress_rows and rows_read % progress_rows < rows_per_epoch:
                    print(f"{path.name}: rows_read={rows_read}, epochs_written={epochs_written}", flush=True)
                window_start_row = None

    skipped_partial = 1 if window_start_row is not None and stage_values else 0
    return FilePreprocessSummary(
        path=str(path),
        subject_id=subject_id,
        alignment_offset=alignment.offset,
        rows_read=rows_read,
        epochs_written=epochs_written,
        skipped_ignored_label=skipped_ignored,
        skipped_mixed_label=skipped_mixed,
        skipped_partial=skipped_partial,
        label_counts=dict(label_counts),
        transition_offset_counts=alignment.transition_offset_counts,
    )


def preprocess_root(
    root: Path,
    out_dir: Path,
    pattern: str,
    limit_files: int | None,
    max_rows: int | None,
    include_sao2: bool,
    rows_per_epoch: int,
    progress_rows: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / "dreamt_100hz_epoch_features.csv"
    summary_path = out_dir / "dreamt_100hz_preprocess_summary.json"

    metadata_columns = [
        "subject_id",
        "source_file",
        "start_row",
        "end_row",
        "aligned_epoch_index",
        "raw_label",
        "label_5",
        "class_id_5",
    ]
    fieldnames = metadata_columns + feature_fieldnames(include_sao2)

    paths = iter_subject_files(root, pattern, limit_files)
    summaries: list[FilePreprocessSummary] = []
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            print(f"Preprocessing {path.name}", flush=True)
            summaries.append(
                preprocess_file(
                    path=path,
                    writer=writer,
                    include_sao2=include_sao2,
                    rows_per_epoch=rows_per_epoch,
                    max_rows=max_rows,
                    progress_rows=progress_rows,
                )
            )

    total_label_counts: Counter[str] = Counter()
    for summary in summaries:
        total_label_counts.update(summary.label_counts)

    report = {
        "root": str(root),
        "out_dir": str(out_dir),
        "feature_path": str(feature_path),
        "pattern": pattern,
        "limit_files": limit_files,
        "max_rows": max_rows,
        "include_sao2": include_sao2,
        "rows_per_epoch": rows_per_epoch,
        "input_columns": list(BASE_COLUMNS) + (["SAO2"] if include_sao2 else []),
        "stage5_names": STAGE5_NAMES,
        "files_processed": len(summaries),
        "total_epochs_written": sum(summary.epochs_written for summary in summaries),
        "total_label_counts": dict(total_label_counts),
        "files": [asdict(summary) for summary in summaries],
    }
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess DreamT 100Hz CSV files into epoch features.")
    parser.add_argument("--root", type=Path, required=True, help="DreamT data_100Hz root directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for feature CSV and summary JSON.")
    parser.add_argument("--pattern", default="S*_PSG_df_updated.csv", help="CSV glob pattern.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional number of files to process.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row limit per file for smoke tests.")
    parser.add_argument("--include-sao2", action="store_true", help="Include SAO2 optional features.")
    parser.add_argument("--rows-per-epoch", type=int, default=ROWS_PER_EPOCH, help="Rows in one 30-second epoch.")
    parser.add_argument("--progress-rows", type=int, default=300000, help="Progress print interval.")
    args = parser.parse_args()

    preprocess_root(
        root=args.root,
        out_dir=args.out_dir,
        pattern=args.pattern,
        limit_files=args.limit_files,
        max_rows=args.max_rows,
        include_sao2=args.include_sao2,
        rows_per_epoch=args.rows_per_epoch,
        progress_rows=args.progress_rows,
    )


if __name__ == "__main__":
    main()
