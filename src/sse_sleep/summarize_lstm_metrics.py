"""Summarize one or more LSTM metric JSON files by experiment label."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRIC_KEYS = (
    ("5_macro_f1", ("final_test", "metrics", "5_class", "macro_f1")),
    ("5_kappa", ("final_test", "metrics", "5_class", "cohen_kappa")),
    ("4_macro_f1", ("final_test", "metrics", "4_class", "macro_f1")),
    ("4_kappa", ("final_test", "metrics", "4_class", "cohen_kappa")),
    ("wake_f1", ("final_test", "metrics", "5_class", "class_wise", "Wake", "f1")),
    ("n3_f1", ("final_test", "metrics", "5_class", "class_wise", "N3", "f1")),
    ("rem_f1", ("final_test", "metrics", "5_class", "class_wise", "REM", "f1")),
)


def nested_get(data: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = data
    for key in path:
        value = value[key]
    return float(value)


def parse_metric_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Metric label cannot be empty")
    path = Path(raw_path).expanduser()
    if path.is_dir():
        path = path / "lstm_metrics.json"
    return label, path


def read_metric_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"runs": len(records), "metrics": {}}
    for metric_name, _ in METRIC_KEYS:
        values = [float(record[metric_name]) for record in records]
        metric_summary = {"mean": statistics.fmean(values)}
        metric_summary["std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        metric_summary["values"] = values
        summary["metrics"][metric_name] = metric_summary
    return summary


def format_value(value: float) -> str:
    return f"{value:.4f}"


def format_delta(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:+.4f}"


def print_markdown_table(summaries: dict[str, dict[str, Any]], baseline_label: str | None) -> None:
    baseline_metrics = summaries.get(baseline_label, {}).get("metrics", {}) if baseline_label else {}
    headers = [
        "label",
        "runs",
        "5 Macro F1",
        "5 Kappa",
        "4 Macro F1",
        "4 Kappa",
        "Wake F1",
        "N3 F1",
        "REM F1",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for label, summary in summaries.items():
        cells = [label, str(summary["runs"])]
        for metric_name, _ in METRIC_KEYS:
            mean_value = summary["metrics"][metric_name]["mean"]
            baseline_value = baseline_metrics.get(metric_name, {}).get("mean")
            delta = mean_value - baseline_value if baseline_value is not None and label != baseline_label else None
            cell = format_value(mean_value)
            if delta is not None:
                cell = f"{cell} ({format_delta(delta)})"
            cells.append(cell)
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LSTM metric JSON files by label.")
    parser.add_argument(
        "--metrics",
        action="append",
        type=parse_metric_arg,
        required=True,
        metavar="LABEL=PATH",
        help="Metric JSON path, or a run directory containing lstm_metrics.json. Repeat for multiple seeds.",
    )
    parser.add_argument("--baseline-label", default=None, help="Optional label to show mean deltas against.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path for machine-readable summary.")
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label, path in args.metrics:
        metric_data = read_metric_file(path)
        record = {metric_name: nested_get(metric_data, metric_path) for metric_name, metric_path in METRIC_KEYS}
        record["seed"] = metric_data.get("seed")
        record["path"] = str(path)
        grouped[label].append(record)

    summaries = {label: summarize(records) for label, records in grouped.items()}
    output = {"baseline_label": args.baseline_label, "summaries": summaries, "records": grouped}

    print_markdown_table(summaries, args.baseline_label)

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
