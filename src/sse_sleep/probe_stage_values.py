"""Probe raw Sleep_Stage values in large DreamT 100Hz CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .dreamt_100hz import LABEL_COLUMN, ROWS_PER_EPOCH


@dataclass
class TransitionSample:
    row_index: int
    epoch_index: int
    previous_value: str
    current_value: str


@dataclass
class StageValueSummary:
    path: str
    rows_checked: int
    non_empty_rows: int
    counts: dict[str, int]
    first_seen_row: dict[str, int]
    last_seen_row: dict[str, int]
    transitions: list[TransitionSample]


def summarize_stage_values(
    path: Path,
    stage_column: str,
    max_rows: int | None,
    max_transitions: int,
) -> StageValueSummary:
    counts: Counter[str] = Counter()
    first_seen_row: dict[str, int] = {}
    last_seen_row: dict[str, int] = {}
    transitions: list[TransitionSample] = []
    rows_checked = 0
    non_empty_rows = 0
    previous_value: str | None = None

    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            header = []
        if stage_column not in header:
            raise ValueError(f"{stage_column!r} not found in {path}")
        stage_index = header.index(stage_column)

        for row in reader:
            if max_rows is not None and rows_checked >= max_rows:
                break
            value = row[stage_index].strip() if stage_index < len(row) else ""
            if value:
                counts[value] += 1
                non_empty_rows += 1
                first_seen_row.setdefault(value, rows_checked)
                last_seen_row[value] = rows_checked
            if previous_value is not None and value != previous_value and len(transitions) < max_transitions:
                transitions.append(
                    TransitionSample(
                        row_index=rows_checked,
                        epoch_index=rows_checked // ROWS_PER_EPOCH,
                        previous_value=previous_value,
                        current_value=value,
                    )
                )
            previous_value = value
            rows_checked += 1

    return StageValueSummary(
        path=str(path),
        rows_checked=rows_checked,
        non_empty_rows=non_empty_rows,
        counts=dict(counts.most_common()),
        first_seen_row=first_seen_row,
        last_seen_row=last_seen_row,
        transitions=transitions,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Sleep_Stage values in DreamT 100Hz CSV files.")
    parser.add_argument("--root", type=Path, required=True, help="DreamT data_100Hz root directory.")
    parser.add_argument("--glob", default="S*_PSG_df_updated.csv", help="CSV glob pattern under root.")
    parser.add_argument("--limit-files", type=int, default=3, help="Number of files to scan.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row scan limit per file.")
    parser.add_argument("--stage-column", default=LABEL_COLUMN, help="Stage column name.")
    parser.add_argument("--max-transitions", type=int, default=20, help="Transition samples to keep per file.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    paths = sorted(args.root.glob(args.glob))[: args.limit_files]
    summaries = [
        summarize_stage_values(
            path=path,
            stage_column=args.stage_column,
            max_rows=args.max_rows,
            max_transitions=args.max_transitions,
        )
        for path in paths
    ]
    report = {
        "root": str(args.root),
        "glob": args.glob,
        "limit_files": args.limit_files,
        "max_rows": args.max_rows,
        "stage_column": args.stage_column,
        "files": [asdict(summary) for summary in summaries],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

