"""Compare Green PPG to DreamT-like BVP transform candidates by calibration audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.potch.audit_potch_feature_calibration_colab import audit_schema
from scripts.potch.run_current_best_outer42_inference_colab import (
    FEATURE_PROFILES,
    HR_IBI_FEATURE_SOURCES,
    HR_IBI_SLOPE_SOURCES,
    TEMP_FEATURE_SOURCES,
    find_existing,
)
from sse_sleep.potch_raw import PPG_AC_SCALE, PPG_TRANSFORMS, QualityFilter, build_potch_epoch_features


TARGET_FEATURES = (
    "bvp_mean",
    "bvp_mean_delta_1",
    "bvp_mean_delta_3",
    "bvp_mean_delta_20",
    "bvp_mean_roll_mean_3",
    "bvp_mean_roll_mean_5",
    "bvp_mean_roll_mean_20",
    "bvp_std",
    "bvp_iqr",
    "bvp_min",
    "bvp_max",
    "bvp_slope",
    "hr_slope",
    "ibi_slope",
    "hr_std",
    "ibi_std",
)


def safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def feature_lookup(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["feature"]: item for item in audit["features"]}


def compact_feature(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "abs_z_p95": item["z"]["abs_p95"],
        "rate_abs_gt_5": item["z"]["rate_abs_gt_5"],
        "raw_median": item["raw"]["median"],
        "raw_p05": item["raw"]["p05"],
        "raw_p95": item["raw"]["p95"],
        "train_mean": item["train_mean"],
        "train_std": item["train_std"],
        "imputed_rate": item["imputed_rate"],
    }


def target_summary(audit: dict[str, Any]) -> dict[str, Any]:
    by_feature = feature_lookup(audit)
    return {
        feature: compact_feature(by_feature.get(feature))
        for feature in TARGET_FEATURES
        if feature in by_feature
    }


def family_summary(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        family: {
            "max_abs_p95": values["max_abs_p95"],
            "max_rate_abs_gt_5": values["max_rate_abs_gt_5"],
            "mean_imputed_rate": values["mean_imputed_rate"],
        }
        for family, values in audit["family_summary"].items()
        if family in {"bvp", "hr", "ibi", "acc_axis", "acc_vm", "temp"}
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Potch PPG transform candidates with DreamT audit stats.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--raw-zip", type=Path)
    input_group.add_argument("--raw-bin", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/content/drive/MyDrive/SSE_outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("/content/drive/MyDrive/SSE_outputs/potch_bvp_transform_compare"))
    parser.add_argument("--original-npz", type=Path)
    parser.add_argument("--w20-npz", type=Path)
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILES, default="current")
    parser.add_argument("--hr-ibi-slope-source", choices=HR_IBI_SLOPE_SOURCES, default="current")
    parser.add_argument("--hr-ibi-feature-source", choices=HR_IBI_FEATURE_SOURCES, default="current")
    parser.add_argument("--temp-feature-source", choices=TEMP_FEATURE_SOURCES, default="none")
    parser.add_argument("--ppg-transforms", choices=PPG_TRANSFORMS, nargs="+", default=list(PPG_TRANSFORMS))
    parser.add_argument("--ppg-scale", type=float, default=PPG_AC_SCALE)
    parser.add_argument("--session-gap-ms", type=int, default=30_000)
    parser.add_argument("--packet-count-min", type=int, default=216)
    parser.add_argument("--duration-ms-min", type=int, default=25_000)
    parser.add_argument("--seq-gap-count-required", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.raw_zip or args.raw_bin
    if raw_path is None:
        raise ValueError("Expected --raw-zip or --raw-bin")

    original_npz = args.original_npz or find_existing(
        [
            args.output_root / "dreamt_100hz_temporal_lstm_context20.npz",
            args.output_root / "dreamt_100hz_temporal_lstm_context20_seed42.npz",
        ],
        "original temporal context20 NPZ",
    )
    w20_npz = args.w20_npz or find_existing(
        [
            args.output_root / "dreamt_100hz_temporal_w20_lstm_context20.npz",
            args.output_root / "dreamt_100hz_temporal_w20_lstm_context20_seed42.npz",
        ],
        "full-w20 temporal context20 NPZ",
    )

    quality = QualityFilter(
        packet_count_min=args.packet_count_min,
        duration_ms_min=args.duration_ms_min,
        seq_gap_count_required=args.seq_gap_count_required,
    )
    candidates: dict[str, Any] = {}
    for transform in args.ppg_transforms:
        candidate_dir = args.out_dir / safe_name(transform)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        clean_csv = candidate_dir / "potch_clean_epoch_features.csv"
        raw_report = build_potch_epoch_features(
            raw_path=raw_path,
            out_csv=candidate_dir / "potch_epoch_features.csv",
            clean_csv=clean_csv,
            summary_json=candidate_dir / "potch_epoch_feature_summary.json",
            clean_npz=candidate_dir / "potch_clean_epoch_features.npz",
            session_gap_ms=args.session_gap_ms,
            quality=quality,
            ppg_transform=transform,
            ppg_scale=args.ppg_scale,
        )
        original = audit_schema(
            clean_csv,
            original_npz,
            args.feature_profile,
            args.hr_ibi_slope_source,
            args.hr_ibi_feature_source,
            args.temp_feature_source,
        )
        w20 = audit_schema(
            clean_csv,
            w20_npz,
            args.feature_profile,
            args.hr_ibi_slope_source,
            args.hr_ibi_feature_source,
            args.temp_feature_source,
        )
        candidates[transform] = {
            "clean_csv": str(clean_csv),
            "raw_feature_report": raw_report,
            "original_family_summary": family_summary(original),
            "w20_family_summary": family_summary(w20),
            "original_targets": target_summary(original),
            "w20_targets": target_summary(w20),
        }

    report = {
        "raw_input": str(raw_path),
        "output_root": str(args.output_root),
        "feature_profile": args.feature_profile,
        "hr_ibi_feature_source": args.hr_ibi_feature_source,
        "hr_ibi_slope_source": args.hr_ibi_slope_source,
        "temp_feature_source": args.temp_feature_source,
        "ppg_scale": args.ppg_scale,
        "transforms": list(args.ppg_transforms),
        "candidates": candidates,
    }
    out_json = args.out_dir / "potch_bvp_transform_comparison.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "out_json": str(out_json),
        "feature_profile": args.feature_profile,
        "hr_ibi_feature_source": args.hr_ibi_feature_source,
        "hr_ibi_slope_source": args.hr_ibi_slope_source,
        "temp_feature_source": args.temp_feature_source,
        "ppg_scale": args.ppg_scale,
        "candidate_scores": {
            transform: {
                "original_bvp_max_abs_p95": data["original_family_summary"]["bvp"]["max_abs_p95"],
                "w20_bvp_max_abs_p95": data["w20_family_summary"]["bvp"]["max_abs_p95"],
                "original_hr_max_abs_p95": data["original_family_summary"]["hr"]["max_abs_p95"],
                "original_ibi_max_abs_p95": data["original_family_summary"]["ibi"]["max_abs_p95"],
                "bvp_mean_abs_p95": data["original_targets"].get("bvp_mean", {}).get("abs_z_p95"),
                "bvp_std_abs_p95": data["original_targets"].get("bvp_std", {}).get("abs_z_p95"),
                "hr_slope_abs_p95": data["original_targets"].get("hr_slope", {}).get("abs_z_p95"),
                "ibi_slope_abs_p95": data["original_targets"].get("ibi_slope", {}).get("abs_z_p95"),
            }
            for transform, data in candidates.items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
