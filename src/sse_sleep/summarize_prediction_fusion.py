"""Summarize prediction-fusion reports across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence


SUMMARY_FIELDS = ("4_macro_f1", "4_kappa", "wake_f1", "light_f1", "deep_f1", "rem_f1")


def summary_value(record: dict[str, Any], field: str) -> float:
    summary = record["test"]["summary"]
    if field in summary:
        return float(summary[field])
    if field == "deep_f1" and "n3_f1" in summary:
        return float(summary["n3_f1"])
    raise KeyError(field)


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_records(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        report = load_report(path)
        for record in report["records"]:
            grouped.setdefault(record["name"], []).append(record)
    return grouped


def summarize_records(paths: Sequence[Path]) -> dict[str, Any]:
    grouped = collect_records(paths)
    expected_count = len(paths)
    summaries: dict[str, Any] = {}
    for name, records in grouped.items():
        if len(records) != expected_count:
            continue
        test_values = {
            field: [summary_value(record, field) for record in records]
            for field in SUMMARY_FIELDS
        }
        summaries[name] = {
            "count": len(records),
            "kind": records[0]["kind"],
            "params": records[0]["params"],
            "selection_score_mean": mean(float(record["selection_score"]) for record in records),
            "selection_score_std": pstdev(float(record["selection_score"]) for record in records),
            "test_mean": {field: mean(values) for field, values in test_values.items()},
            "test_std": {field: pstdev(values) for field, values in test_values.items()},
        }
    return {
        "reports": [str(path) for path in paths],
        "report_count": len(paths),
        "summaries": summaries,
    }


def sort_key(item: tuple[str, dict[str, Any]]) -> tuple[float, float, float]:
    summary = item[1]
    test_mean = summary["test_mean"]
    return (
        float(summary["selection_score_mean"]),
        float(test_mean["4_macro_f1"]),
        float(test_mean["4_kappa"]),
    )


def print_table(report: dict[str, Any], top: int) -> None:
    rows = sorted(report["summaries"].items(), key=sort_key, reverse=True)
    print("| rank | name | val score | 4 Macro | 4 Kappa | Wake | Light | Deep | REM |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for rank, (name, summary) in enumerate(rows[:top], start=1):
        test_mean = summary["test_mean"]
        print(
            f"| {rank} | {name} | {summary['selection_score_mean']:.4f} | "
            f"{test_mean['4_macro_f1']:.4f} | {test_mean['4_kappa']:.4f} | "
            f"{test_mean['wake_f1']:.4f} | {test_mean['light_f1']:.4f} | "
            f"{test_mean['deep_f1']:.4f} | {test_mean['rem_f1']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prediction-fusion JSON reports across seeds.")
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    report = summarize_records(args.reports)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(report, top=args.top)


if __name__ == "__main__":
    main()
