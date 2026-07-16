"""Rank fixed-weight fusion candidates across summary reports."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence


SUMMARY_FIELDS = ("4_macro_f1", "4_kappa", "wake_f1", "light_f1", "deep_f1", "rem_f1")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_source_label(path: Path) -> str:
    stem = path.stem
    match = re.match(r"fusion3_original_full_w20_(.+?)_context\d+_h\d+", stem)
    if match:
        return match.group(1)
    match = re.match(r"fusion4_original_full_w20_(.+?)_context\d+_h\d+", stem)
    if match:
        return match.group(1)
    return stem


def combined_score(row: dict[str, Any]) -> float:
    test_mean = row["test_mean"]
    return float(test_mean["4_macro_f1"]) + float(test_mean["4_kappa"])


def wake_rem_score(row: dict[str, Any]) -> float:
    test_mean = row["test_mean"]
    return float(test_mean["wake_f1"]) + float(test_mean["rem_f1"])


def row_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
    test_mean = row["test_mean"]
    return (
        combined_score(row),
        wake_rem_score(row),
        float(test_mean["deep_f1"]),
        float(row["selection_score_mean"]),
    )


def collect_rows(paths: Sequence[Path], kinds: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        report = load_json(path)
        source = infer_source_label(path)
        for name, summary in report["summaries"].items():
            if kinds and summary.get("kind") not in kinds:
                continue
            test_mean = summary["test_mean"]
            rows.append(
                {
                    "source": source,
                    "name": name,
                    "kind": summary.get("kind"),
                    "selection_score_mean": float(summary["selection_score_mean"]),
                    "selection_score_std": float(summary["selection_score_std"]),
                    "test_mean": {field: float(test_mean[field]) for field in SUMMARY_FIELDS},
                    "test_std": {
                        field: float(summary["test_std"][field]) for field in SUMMARY_FIELDS
                    },
                    "params": summary.get("params", {}),
                    "summary_path": str(path),
                }
            )
    return rows


def rank_rows(rows: Sequence[dict[str, Any]], tie_tolerance: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=row_score, reverse=True)
    best_combined = combined_score(sorted_rows[0])
    best_band = [row for row in sorted_rows if combined_score(row) >= best_combined - tie_tolerance]
    rest = [row for row in sorted_rows if combined_score(row) < best_combined - tie_tolerance]
    best_band = sorted(
        best_band,
        key=lambda row: (
            wake_rem_score(row),
            combined_score(row),
            row["test_mean"]["deep_f1"],
            row["selection_score_mean"],
        ),
        reverse=True,
    )
    return best_band + rest


def print_table(rows: Sequence[dict[str, Any]], top: int) -> None:
    print("| rank | source | name | 4M+4K | 4 Macro | 4 Kappa | Wake | Light | Deep | REM |")
    print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for rank, row in enumerate(rows[:top], start=1):
        test_mean = row["test_mean"]
        combined = test_mean["4_macro_f1"] + test_mean["4_kappa"]
        print(
            f"| {rank} | {row['source']} | {row['name']} | {combined:.4f} | "
            f"{test_mean['4_macro_f1']:.4f} | {test_mean['4_kappa']:.4f} | "
            f"{test_mean['wake_f1']:.4f} | {test_mean['light_f1']:.4f} | "
            f"{test_mean['deep_f1']:.4f} | {test_mean['rem_f1']:.4f} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank fixed-weight fusion summaries.")
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=("classwise3_nonrem_rem", "classwise4_grouped"),
        help="Record kinds to include. Pass no values after --kinds to include all kinds.",
    )
    parser.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.0005,
        help="For the best band, prefer Wake+REM when 4M+4K is within this tolerance.",
    )
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows = rank_rows(
        collect_rows(args.summaries, kinds=set(args.kinds)),
        tie_tolerance=args.tie_tolerance,
    )
    report = {
        "summaries": [str(path) for path in args.summaries],
        "sort": (
            "best band: within tie_tolerance of max test 4_macro_f1 + test 4_kappa, "
            "then Wake+REM; remaining rows: 4M+4K, Wake+REM, Deep, validation score"
        ),
        "tie_tolerance": args.tie_tolerance,
        "rows": rows,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_table(rows, top=args.top)


if __name__ == "__main__":
    main()
