"""Summarize JSON output from sse_sleep.probe_stage_values."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .dreamt_100hz import ROWS_PER_EPOCH, SAMPLING_RATE_HZ


def row_to_time(row_index: int) -> str:
    """Format a 100Hz row index as approximate recording time."""
    seconds = row_index / SAMPLING_RATE_HZ
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def stage_counts_text(counts: dict[str, int], top_n: int) -> str:
    if not counts:
        return "(none)"
    counter = Counter(counts)
    return ", ".join(f"{stage}:{count}" for stage, count in counter.most_common(top_n))


def summarize_file(file_summary: dict[str, Any], top_n: int) -> list[str]:
    path = Path(file_summary["path"])
    rows_checked = int(file_summary.get("rows_checked") or 0)
    non_empty_rows = int(file_summary.get("non_empty_rows") or 0)
    counts = file_summary.get("counts") or {}
    first_seen = file_summary.get("first_seen_row") or {}
    transitions = file_summary.get("transitions") or []

    lines = [
        f"FILE {path.name}",
        f"  rows_checked={rows_checked} ({row_to_time(rows_checked)}), non_empty_rows={non_empty_rows}",
        f"  stage_counts={stage_counts_text(counts, top_n)}",
    ]

    if first_seen:
        first_seen_parts = []
        for stage, row in sorted(first_seen.items(), key=lambda item: int(item[1])):
            row_int = int(row)
            first_seen_parts.append(f"{stage}@row{row_int}/epoch{row_int // ROWS_PER_EPOCH}/{row_to_time(row_int)}")
        lines.append("  first_seen=" + ", ".join(first_seen_parts[:top_n]))
    else:
        lines.append("  first_seen=(none)")

    if transitions:
        lines.append("  transitions:")
        for transition in transitions[:top_n]:
            row = int(transition["row_index"])
            lines.append(
                "    "
                f"row={row}, epoch={transition['epoch_index']}, time={row_to_time(row)}, "
                f"{transition['previous_value']} -> {transition['current_value']}"
            )
    else:
        lines.append("  transitions=(none)")

    return lines


def summarize_probe(report: dict[str, Any], top_n: int) -> str:
    files = report.get("files") or []
    total_counts: Counter[str] = Counter()
    for file_summary in files:
        total_counts.update(file_summary.get("counts") or {})

    lines = [
        "DreamT stage probe summary",
        f"root={report.get('root')}",
        f"stage_column={report.get('stage_column')}",
        f"files_scanned={len(files)}",
        f"max_rows={report.get('max_rows')}",
        f"total_stage_counts={stage_counts_text(dict(total_counts), top_n)}",
        "",
    ]

    for file_summary in files:
        lines.extend(summarize_file(file_summary, top_n))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize dreamt_stage_probe.json.")
    parser.add_argument("--input", type=Path, required=True, help="JSON file from sse_sleep.probe_stage_values.")
    parser.add_argument("--out", type=Path, default=None, help="Optional text summary output path.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of counts/transitions to display.")
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    text = summarize_probe(report, args.top_n)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

