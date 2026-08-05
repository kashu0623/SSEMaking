"""Audit Potch serving feature values against DreamT training normalization stats."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.potch.run_current_best_outer42_inference_colab import (
    BASE_FEATURE_MAP,
    FEATURE_PROFILES,
    find_existing,
    load_train_schema,
    sorted_clean_df,
    values_for_training_features,
)
from sse_sleep.potch_raw import PPG_AC_SCALE, PPG_TRANSFORMS, QualityFilter, build_potch_epoch_features


def quantile(values: np.ndarray, q: float) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.quantile(finite, q))


def feature_family(name: str) -> str:
    if name.startswith("bvp_"):
        return "bvp"
    if name.startswith("acc_vm_"):
        return "acc_vm"
    if name.startswith("acc_"):
        return "acc_axis"
    if name.startswith("hr_"):
        return "hr"
    if name.startswith("ibi_"):
        return "ibi"
    if name.startswith("temp_"):
        return "temp"
    if "_delta_" in name or "_roll_" in name:
        base = name.split("_delta_", 1)[0].split("_roll_", 1)[0]
        return f"temporal:{feature_family(base)}"
    return "other"


def summarize_feature(
    name: str,
    raw_values: np.ndarray,
    normalized_values: np.ndarray,
    train_mean: float,
    train_std: float,
    imputed_mask: np.ndarray,
) -> dict[str, Any]:
    finite_raw = raw_values[np.isfinite(raw_values)]
    finite_z = normalized_values[np.isfinite(normalized_values)]
    abs_z = np.abs(finite_z)
    return {
        "feature": name,
        "family": feature_family(name),
        "train_mean": train_mean,
        "train_std": train_std,
        "observed_count": int(finite_raw.size),
        "imputed_count": int(np.sum(imputed_mask)),
        "imputed_rate": float(np.mean(imputed_mask)) if imputed_mask.size else 0.0,
        "raw": {
            "mean": float(np.mean(finite_raw)) if finite_raw.size else None,
            "std": float(np.std(finite_raw)) if finite_raw.size else None,
            "p05": quantile(raw_values, 0.05),
            "median": quantile(raw_values, 0.50),
            "p95": quantile(raw_values, 0.95),
            "min": float(np.min(finite_raw)) if finite_raw.size else None,
            "max": float(np.max(finite_raw)) if finite_raw.size else None,
        },
        "z": {
            "mean": float(np.mean(finite_z)) if finite_z.size else None,
            "std": float(np.std(finite_z)) if finite_z.size else None,
            "p05": quantile(normalized_values, 0.05),
            "median": quantile(normalized_values, 0.50),
            "p95": quantile(normalized_values, 0.95),
            "min": float(np.min(finite_z)) if finite_z.size else None,
            "max": float(np.max(finite_z)) if finite_z.size else None,
            "abs_p95": quantile(abs_z, 0.95) if abs_z.size else None,
            "abs_max": float(np.max(abs_z)) if abs_z.size else None,
            "rate_abs_gt_3": float(np.mean(abs_z > 3.0)) if abs_z.size else None,
            "rate_abs_gt_5": float(np.mean(abs_z > 5.0)) if abs_z.size else None,
        },
    }


def audit_schema(clean_csv: Path, train_npz: Path, feature_profile: str) -> dict[str, Any]:
    df = sorted_clean_df(clean_csv)
    schema = load_train_schema(train_npz)
    feature_names = schema["feature_names"].astype(str)
    context_epochs = int(schema["context_epochs"])
    raw_values, feature_report = values_for_training_features(
        df,
        feature_names,
        BASE_FEATURE_MAP,
        feature_profile,
    )

    windows: list[np.ndarray] = []
    meta_rows = []
    for _, group in df.groupby("session_id", sort=True):
        positions = group.index.to_numpy()
        epoch_indices = group["epoch_index"].to_numpy()
        for local_end in range(context_epochs - 1, len(positions)):
            span = epoch_indices[local_end - context_epochs + 1 : local_end + 1]
            if not np.array_equal(span, np.arange(span[0], span[0] + context_epochs)):
                continue
            selected_positions = positions[local_end - context_epochs + 1 : local_end + 1]
            windows.append(raw_values[selected_positions])
            meta_rows.append(df.loc[positions[local_end]])
    if not windows:
        raise ValueError(f"No contiguous context{context_epochs} windows could be built from {clean_csv}")
    raw_windows = np.stack(windows).astype(np.float32)
    meta_first = meta_rows[0]
    meta_last = meta_rows[-1]

    mean = schema["mean"].reshape(1, 1, -1)
    std = schema["std"].reshape(1, 1, -1)
    filled = np.where(np.isfinite(raw_windows), raw_windows, mean)
    normalized = ((filled - mean) / std).astype(np.float32)
    imputed_mask = ~np.isfinite(raw_windows)

    flattened_raw = raw_windows.reshape(-1, raw_windows.shape[-1])
    flattened_z = normalized.reshape(-1, normalized.shape[-1])
    flattened_imputed = imputed_mask.reshape(-1, imputed_mask.shape[-1])
    features = [
        summarize_feature(
            name=str(name),
            raw_values=flattened_raw[:, index],
            normalized_values=flattened_z[:, index],
            train_mean=float(schema["mean"][index]),
            train_std=float(schema["std"][index]),
            imputed_mask=flattened_imputed[:, index],
        )
        for index, name in enumerate(feature_names)
    ]
    features_by_abs_z = sorted(
        features,
        key=lambda item: (
            -1.0 if item["z"]["abs_p95"] is None else -float(item["z"]["abs_p95"]),
            item["feature"],
        ),
    )
    features_by_impute = sorted(
        features,
        key=lambda item: (-float(item["imputed_rate"]), item["feature"]),
    )
    family_summary: dict[str, dict[str, Any]] = {}
    for item in features:
        family = item["family"]
        family_items = family_summary.setdefault(
            family,
            {
                "feature_count": 0,
                "mean_imputed_rate": 0.0,
                "max_abs_p95": 0.0,
                "max_rate_abs_gt_5": 0.0,
            },
        )
        family_items["feature_count"] += 1
        family_items["mean_imputed_rate"] += item["imputed_rate"]
        abs_p95 = item["z"]["abs_p95"]
        rate_gt_5 = item["z"]["rate_abs_gt_5"]
        if abs_p95 is not None:
            family_items["max_abs_p95"] = max(family_items["max_abs_p95"], float(abs_p95))
        if rate_gt_5 is not None:
            family_items["max_rate_abs_gt_5"] = max(family_items["max_rate_abs_gt_5"], float(rate_gt_5))
    for item in family_summary.values():
        item["mean_imputed_rate"] /= max(item["feature_count"], 1)

    return {
        "train_npz": str(train_npz),
        "context_epochs": context_epochs,
        "window_count": int(raw_windows.shape[0]),
        "feature_count": int(raw_windows.shape[-1]),
        "feature_report": {
            "train_npz": str(train_npz),
            "context_epochs": context_epochs,
            "training_feature_count": int(len(feature_names)),
            "input_epoch_count": int(len(df)),
            "window_count": int(raw_windows.shape[0]),
            "feature_profile": feature_profile,
            **feature_report,
        },
        "meta": {
            "first_session_id": int(meta_first["session_id"]),
            "first_epoch_index": int(meta_first["epoch_index"]),
            "last_epoch_index": int(meta_last["epoch_index"]),
        },
        "family_summary": family_summary,
        "top_abs_z_p95": features_by_abs_z[:40],
        "top_imputed": features_by_impute[:40],
        "features": features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Potch features against DreamT schema stats.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--clean-csv", type=Path)
    input_group.add_argument("--raw-zip", type=Path)
    input_group.add_argument("--raw-bin", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/content/drive/MyDrive/SSE_outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("/content/drive/MyDrive/SSE_outputs/potch_feature_audit"))
    parser.add_argument("--original-npz", type=Path)
    parser.add_argument("--w20-npz", type=Path)
    parser.add_argument("--feature-profile", choices=FEATURE_PROFILES, default="current")
    parser.add_argument("--session-gap-ms", type=int, default=30_000)
    parser.add_argument("--ppg-transform", choices=PPG_TRANSFORMS, default="epoch-median")
    parser.add_argument("--ppg-scale", type=float, default=PPG_AC_SCALE)
    parser.add_argument("--packet-count-min", type=int, default=216)
    parser.add_argument("--duration-ms-min", type=int, default=25_000)
    parser.add_argument("--seq-gap-count-required", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_csv = args.clean_csv
    raw_path = args.raw_zip or args.raw_bin
    raw_report = None
    if raw_path is not None:
        clean_csv = args.out_dir / "potch_clean_epoch_features.csv"
        raw_report = build_potch_epoch_features(
            raw_path=raw_path,
            out_csv=args.out_dir / "potch_epoch_features.csv",
            clean_csv=clean_csv,
            summary_json=args.out_dir / "potch_epoch_feature_summary.json",
            clean_npz=args.out_dir / "potch_clean_epoch_features.npz",
            session_gap_ms=args.session_gap_ms,
            ppg_transform=args.ppg_transform,
            ppg_scale=args.ppg_scale,
            quality=QualityFilter(
                packet_count_min=args.packet_count_min,
                duration_ms_min=args.duration_ms_min,
                seq_gap_count_required=args.seq_gap_count_required,
            ),
        )
    if clean_csv is None:
        raise ValueError("Expected --clean-csv, --raw-zip, or --raw-bin")

    output_root = args.output_root
    original_npz = args.original_npz or find_existing(
        [
            output_root / "dreamt_100hz_temporal_lstm_context20.npz",
            output_root / "dreamt_100hz_temporal_lstm_context20_seed42.npz",
        ],
        "original temporal context20 NPZ",
    )
    w20_npz = args.w20_npz or find_existing(
        [
            output_root / "dreamt_100hz_temporal_w20_lstm_context20.npz",
            output_root / "dreamt_100hz_temporal_w20_lstm_context20_seed42.npz",
        ],
        "full-w20 temporal context20 NPZ",
    )
    report = {
        "clean_csv": str(clean_csv),
        "raw_input": str(raw_path) if raw_path is not None else None,
        "feature_profile": args.feature_profile,
        "ppg_transform": args.ppg_transform,
        "ppg_scale": args.ppg_scale,
        "raw_feature_report": raw_report,
        "original": audit_schema(clean_csv, original_npz, args.feature_profile),
        "w20": audit_schema(clean_csv, w20_npz, args.feature_profile),
    }
    out_json = args.out_dir / "potch_feature_calibration_audit.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {
        "out_json": str(out_json),
        "original_family_summary": report["original"]["family_summary"],
        "w20_family_summary": report["w20"]["family_summary"],
        "original_top_abs_z_p95": report["original"]["top_abs_z_p95"][:12],
        "w20_top_abs_z_p95": report["w20"]["top_abs_z_p95"][:12],
        "original_top_imputed": report["original"]["top_imputed"][:12],
        "w20_top_imputed": report["w20"]["top_imputed"][:12],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2)[:20000])


if __name__ == "__main__":
    main()
