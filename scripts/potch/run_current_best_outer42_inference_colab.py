"""Run the current outer42 fusion best on Potch clean epoch features.

This is a compatibility test script for Colab. It maps the Potch epoch feature
CSV into the feature schema saved in the DreamT training NPZ files, imputes
features that Potch does not yet provide with the DreamT train mean, forwards
the outer42 checkpoint set, and writes per-epoch probabilities.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from sse_sleep.evaluate_direct4_hybrid_deep_fusion import current_probs_to_four, hybrid_fusion
from sse_sleep.evaluate_direct4_source_blend_hybrid import blend_direct4_sources
from sse_sleep.evaluate_four_model_fusion import build_grouped_class_weights, four_model_classwise_fusion
from sse_sleep.evaluate_light_deep_specialist_fusion import fuse_light_deep_conditional
from sse_sleep.potch_raw import QualityFilter, build_potch_epoch_features
from sse_sleep.train_lstm import RecurrentSleepClassifier
from sse_sleep.export_lstm_predictions import normalize_state_dict_keys


STAGE4_NAMES = ("Wake", "Light", "Deep", "REM")
INIT_SEEDS = (1001, 2002, 3003, 4004, 5005)

CURRENT_SOURCE_BETAS = np.asarray((0.0, 0.0, 0.25, 0.50), dtype=np.float32)
CURRENT_HYBRID_ALPHAS = np.asarray((0.1875, 0.54375, 0.81875, 0.0), dtype=np.float32)
CURRENT_DEEP_GAIN = 1.15
SPECIALIST_BLEND = np.asarray((0.40, 0.60), dtype=np.float32)
SPECIALIST_BETA = 0.975
SPECIALIST_SCALE = 0.75
SPECIALIST_BIAS = 0.25

BASE_FEATURE_MAP = {
    "bvp_mean": "ppg_mean",
    "bvp_std": "ppg_std",
    "bvp_median": "ppg_median",
    "bvp_iqr": "ppg_iqr",
    "bvp_min": "ppg_min",
    "bvp_max": "ppg_max",
    "bvp_slope": "ppg_slope",
}

for axis in ("x", "y", "z", "vm"):
    for suffix in ("mean", "std", "median", "iqr", "min", "max", "slope"):
        BASE_FEATURE_MAP[f"acc_{axis}_{suffix}"] = f"acc_{axis}_{suffix}"
BASE_FEATURE_MAP["acc_vm_activity"] = "acc_vm_activity"
for signal in ("hr", "ibi"):
    for suffix in ("mean", "std", "median", "iqr", "min", "max", "slope"):
        BASE_FEATURE_MAP[f"{signal}_{suffix}"] = f"{signal}_{suffix}"


TEMPORAL_DELTA_RE = re.compile(r"^(?P<base>.+)_delta_(?P<lag>[0-9]+)$")
TEMPORAL_ROLL_RE = re.compile(r"^(?P<base>.+)_roll_(?P<stat>mean|std)_(?P<window>[0-9]+)$")
FEATURE_PROFILES = (
    "current",
    "stable-vitals-v5",
    "v6-a-bvp-mean-kept",
    "v6-b-hribi-rollmean-kept",
    "v6-c-hribi-no-slope",
    "v6-d-bvp-mean-imputed",
)


def find_existing(paths: Sequence[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing {label}. Tried: {[str(path) for path in paths]}")


def load_train_schema(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {
            "feature_names": data["feature_names"].astype(str),
            "mean": data["train_feature_mean"].astype(np.float32),
            "std": data["train_feature_std"].astype(np.float32),
            "context_epochs": np.asarray(data["context_epochs"]).astype(np.int64),
        }


def split_temporal_feature(feature: str) -> tuple[str, str | None, str | None]:
    delta_match = TEMPORAL_DELTA_RE.match(feature)
    roll_match = TEMPORAL_ROLL_RE.match(feature)
    if delta_match is not None:
        return delta_match.group("base"), "delta", None
    if roll_match is not None:
        return roll_match.group("base"), "roll", roll_match.group("stat")
    return feature, None, None


def stable_vitals_v5_disabled(feature: str) -> bool:
    base, temporal_kind, _ = split_temporal_feature(feature)
    if base == "bvp_mean":
        return True
    if base.startswith("hr_") or base.startswith("ibi_"):
        if temporal_kind is not None:
            return True
        suffix = base.split("_", 1)[1]
        return suffix not in {"mean", "median", "min", "max"}
    return False


def v6_a_bvp_mean_kept_disabled(feature: str) -> bool:
    base, temporal_kind, _ = split_temporal_feature(feature)
    if base.startswith("hr_") or base.startswith("ibi_"):
        if temporal_kind is not None:
            return True
        suffix = base.split("_", 1)[1]
        return suffix not in {"mean", "median", "min", "max"}
    return False


def v6_b_hribi_rollmean_kept_disabled(feature: str) -> bool:
    base, temporal_kind, roll_stat = split_temporal_feature(feature)
    if base == "bvp_mean":
        return True
    if base.startswith("hr_") or base.startswith("ibi_"):
        suffix = base.split("_", 1)[1]
        if temporal_kind == "roll":
            return not (suffix == "mean" and roll_stat == "mean")
        if temporal_kind == "delta":
            return True
        return suffix not in {"mean", "median", "min", "max"}
    return False


def v6_c_hribi_no_slope_disabled(feature: str) -> bool:
    base, _, _ = split_temporal_feature(feature)
    if base == "bvp_mean":
        return True
    if base in {"hr_slope", "ibi_slope"}:
        return True
    return False


def v6_d_bvp_mean_imputed_disabled(feature: str) -> bool:
    base, _, _ = split_temporal_feature(feature)
    return base == "bvp_mean"


def feature_disabled(feature: str, feature_profile: str) -> bool:
    if feature_profile == "current":
        return False
    if feature_profile == "stable-vitals-v5":
        return stable_vitals_v5_disabled(feature)
    if feature_profile == "v6-a-bvp-mean-kept":
        return v6_a_bvp_mean_kept_disabled(feature)
    if feature_profile == "v6-b-hribi-rollmean-kept":
        return v6_b_hribi_rollmean_kept_disabled(feature)
    if feature_profile == "v6-c-hribi-no-slope":
        return v6_c_hribi_no_slope_disabled(feature)
    if feature_profile == "v6-d-bvp-mean-imputed":
        return v6_d_bvp_mean_imputed_disabled(feature)
    raise ValueError(f"Unknown feature profile: {feature_profile}")


def base_value(row: pd.Series, feature: str, feature_map: dict[str, str]) -> float:
    source = feature_map.get(feature)
    if source is None or source not in row:
        return float("nan")
    value = row[source]
    if pd.isna(value):
        return float("nan")
    return float(value)


def values_for_training_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    feature_map: dict[str, str],
    feature_profile: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    values: list[list[float]] = []
    missing_base_features: set[str] = set()
    missing_full_features: set[str] = set()
    profile_imputed_features: set[str] = set()

    for _, group in df.groupby("session_id", sort=True):
        history: list[dict[str, float]] = []
        for _, row in group.sort_values("epoch_index").iterrows():
            current_base = {name: base_value(row, name, feature_map) for name in feature_map}
            out_row: list[float] = []
            for feature in feature_names:
                if feature_disabled(feature, feature_profile):
                    missing_full_features.add(feature)
                    profile_imputed_features.add(feature)
                    out_row.append(float("nan"))
                    continue

                direct = base_value(row, feature, feature_map)
                if direct == direct:
                    out_row.append(direct)
                    continue

                delta_match = TEMPORAL_DELTA_RE.match(feature)
                roll_match = TEMPORAL_ROLL_RE.match(feature)
                value = float("nan")
                if delta_match is not None:
                    base = delta_match.group("base")
                    lag = int(delta_match.group("lag"))
                    current = base_value(row, base, feature_map)
                    if len(history) >= lag:
                        previous = history[-lag].get(base, float("nan"))
                        if current == current and previous == previous:
                            value = current - previous
                    if base not in feature_map:
                        missing_base_features.add(base)
                elif roll_match is not None:
                    base = roll_match.group("base")
                    stat = roll_match.group("stat")
                    window = int(roll_match.group("window"))
                    if len(history) >= window:
                        window_values = np.asarray(
                            [history[-idx].get(base, float("nan")) for idx in range(1, window + 1)],
                            dtype=np.float32,
                        )
                        window_values = window_values[np.isfinite(window_values)]
                        if window_values.size:
                            value = float(window_values.mean() if stat == "mean" else window_values.std())
                    if base not in feature_map:
                        missing_base_features.add(base)
                else:
                    missing_full_features.add(feature)
                out_row.append(value)

            history.append(current_base)
            values.append(out_row)

    report = {
        "missing_base_feature_count": len(missing_base_features),
        "missing_base_features": sorted(missing_base_features),
        "missing_full_feature_count": len(missing_full_features),
        "missing_full_features": sorted(missing_full_features),
        "profile_imputed_feature_count": len(profile_imputed_features),
        "profile_imputed_features": sorted(profile_imputed_features),
    }
    return np.asarray(values, dtype=np.float32), report


def sorted_clean_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"session_id", "epoch_index", "start_app_ts", "end_app_ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Clean CSV is missing required columns: {sorted(missing)}")
    return df.sort_values(["session_id", "epoch_index"]).reset_index(drop=True)


def build_context_windows(
    clean_csv: Path,
    train_npz: Path,
    feature_profile: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    df = sorted_clean_df(clean_csv)
    schema = load_train_schema(train_npz)
    feature_names = schema["feature_names"]
    context_epochs = int(schema["context_epochs"])
    if context_epochs <= 0:
        raise ValueError(f"Invalid context_epochs in {train_npz}: {context_epochs}")

    raw_values, feature_report = values_for_training_features(
        df,
        feature_names,
        BASE_FEATURE_MAP,
        feature_profile,
    )
    mean = schema["mean"].reshape(1, -1)
    std = schema["std"].reshape(1, -1)
    filled = np.where(np.isfinite(raw_values), raw_values, mean)
    normalized = ((filled - mean) / std).astype(np.float32)

    windows: list[np.ndarray] = []
    meta_rows: list[pd.Series] = []
    for _, group in df.groupby("session_id", sort=True):
        positions = group.index.to_numpy()
        epoch_indices = group["epoch_index"].to_numpy()
        for local_end in range(context_epochs - 1, len(positions)):
            span = epoch_indices[local_end - context_epochs + 1 : local_end + 1]
            if not np.array_equal(span, np.arange(span[0], span[0] + context_epochs)):
                continue
            selected_positions = positions[local_end - context_epochs + 1 : local_end + 1]
            windows.append(normalized[selected_positions])
            meta_rows.append(df.loc[positions[local_end]])

    if not windows:
        raise ValueError(
            f"No contiguous context{context_epochs} windows could be built from {clean_csv}"
        )
    meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    report = {
        "train_npz": str(train_npz),
        "context_epochs": context_epochs,
        "training_feature_count": int(len(feature_names)),
        "input_epoch_count": int(len(df)),
        "window_count": int(len(windows)),
        "feature_profile": feature_profile,
        **feature_report,
    }
    return np.stack(windows).astype(np.float32), meta, report


def checkpoint_value(checkpoint: dict[str, Any], key: str, default: Any) -> Any:
    return checkpoint[key] if key in checkpoint else default


def load_model(checkpoint_path: Path, device: torch.device, default_num_classes: int) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RecurrentSleepClassifier(
        input_size=int(checkpoint_value(checkpoint, "input_size", 0)),
        hidden_size=int(checkpoint_value(checkpoint, "hidden_size", 64)),
        num_layers=int(checkpoint_value(checkpoint, "num_layers", 1)),
        num_classes=int(checkpoint_value(checkpoint, "num_classes", default_num_classes)),
        dropout=float(checkpoint_value(checkpoint, "dropout", 0.4)),
        model_type=str(checkpoint_value(checkpoint, "model_type", "lstm")),
        aux_head=str(checkpoint_value(checkpoint, "aux_head", "none")),
    ).to(device)
    model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"]))
    model.eval()
    return model


def predict_checkpoint(
    checkpoint_path: Path,
    x: np.ndarray,
    device: torch.device,
    default_num_classes: int,
    batch_size: int,
) -> np.ndarray:
    model = load_model(checkpoint_path, device, default_num_classes)
    batches: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
        with torch.no_grad():
            logits = model(batch)["stage_logits"]
            probs = torch.softmax(logits, dim=1)
        batches.append(probs.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def average_checkpoints(
    paths: Sequence[Path],
    x: np.ndarray,
    device: torch.device,
    default_num_classes: int,
    batch_size: int,
) -> np.ndarray:
    probs = [
        predict_checkpoint(path, x, device, default_num_classes, batch_size)
        for path in paths
    ]
    return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float32)


def role_paths(output_root: Path, outer_seed: int, role: str) -> list[Path]:
    if outer_seed != 42:
        raise ValueError("This script currently supports outer_seed=42 only")
    if role == "original":
        base = output_root / "lstm_temporal_context20_h64_inverse" / "lstm_best.pt"
    elif role == "full_w20":
        base = output_root / "lstm_temporal_w20_context20_h64_inverse" / "lstm_best.pt"
    elif role == "capacity_h128":
        base = output_root / "lstm_temporal_w20_context20_inverse_capacity_h128" / "lstm_best.pt"
    elif role == "h128_ls003":
        base = output_root / "lstm_temporal_w20_context20_inverse_h128_ls003" / "lstm_best.pt"
    else:
        raise ValueError(f"Unknown role: {role}")
    replica_role = "original" if role == "original" else role
    replicas = [
        output_root / f"same_split_init_ensemble_{replica_role}_outer{outer_seed}_init{seed}" / "lstm_best.pt"
        for seed in INIT_SEEDS
    ]
    return [base, *replicas]


def direct4_paths(output_root: Path, outer_seed: int) -> tuple[Path, list[Path]]:
    single = output_root / f"direct4_original_outer{outer_seed}" / "lstm4_best.pt"
    ensemble_members = [
        single,
        *[
            output_root / f"same_split_init_ensemble_direct4_original_outer{outer_seed}_init{seed}" / "lstm4_best.pt"
            for seed in INIT_SEEDS
        ],
    ]
    return single, ensemble_members


def specialist_paths(output_root: Path, outer_seed: int) -> tuple[Path, Path]:
    h128 = output_root / f"light_deep_specialist_original_h128_ce_outer{outer_seed}" / "light_deep_best.pt"
    h256 = output_root / f"light_deep_specialist_light_h256_inverse_lstm2_outer{outer_seed}" / "light_deep_best.pt"
    return h128, h256


def check_paths(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(missing))


def summarize_predictions(prediction: np.ndarray) -> dict[str, Any]:
    counts = np.bincount(prediction, minlength=len(STAGE4_NAMES))
    return {
        STAGE4_NAMES[index]: int(counts[index])
        for index in range(len(STAGE4_NAMES))
    }


def run_count(prediction: np.ndarray) -> int:
    if prediction.size == 0:
        return 0
    return int(1 + np.sum(prediction[1:] != prediction[:-1]))


def probability_summary(probabilities: np.ndarray) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1).astype(np.int64)
    sorted_probs = np.sort(probabilities, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    return {
        "class_counts": summarize_predictions(prediction),
        "probability_mean": {
            name: float(probabilities[:, index].mean())
            for index, name in enumerate(STAGE4_NAMES)
        },
        "margin": {
            "mean": float(margins.mean()),
            "median": float(np.median(margins)),
            "low_lt_0_03": int(np.sum(margins < 0.03)),
            "low_lt_0_05": int(np.sum(margins < 0.05)),
            "low_lt_0_10": int(np.sum(margins < 0.10)),
        },
        "run_count": run_count(prediction),
    }


def write_component_diagnostics(
    out_dir: Path,
    meta: pd.DataFrame,
    variants: dict[str, np.ndarray],
) -> tuple[Path, Path, Path, dict[str, Any]]:
    diagnostic = meta[["session_id", "epoch_index", "start_app_ts", "end_app_ts"]].copy()
    summary: dict[str, Any] = {}
    npz_arrays: dict[str, np.ndarray] = {
        "stage_names": np.asarray(STAGE4_NAMES),
        "session_id": diagnostic["session_id"].to_numpy(dtype=np.int64),
        "epoch_index": diagnostic["epoch_index"].to_numpy(dtype=np.int64),
        "start_app_ts": diagnostic["start_app_ts"].to_numpy(dtype=np.int64),
        "end_app_ts": diagnostic["end_app_ts"].to_numpy(dtype=np.int64),
    }

    for variant_name, probabilities in variants.items():
        prediction = probabilities.argmax(axis=1).astype(np.int64)
        for index, stage_name in enumerate(STAGE4_NAMES):
            diagnostic[f"{variant_name}_p_{stage_name.lower()}"] = probabilities[:, index]
        diagnostic[f"{variant_name}_pred_id"] = prediction
        diagnostic[f"{variant_name}_pred_stage"] = [STAGE4_NAMES[index] for index in prediction]
        summary[variant_name] = probability_summary(probabilities)
        npz_arrays[f"{variant_name}_probs"] = probabilities.astype(np.float32)
        npz_arrays[f"{variant_name}_pred"] = prediction

    diagnostic_csv = out_dir / "potch_model_component_predictions.csv"
    diagnostic_npz = out_dir / "potch_model_component_predictions.npz"
    diagnostic_json = out_dir / "potch_model_component_summary.json"
    diagnostic.to_csv(diagnostic_csv, index=False)
    np.savez_compressed(diagnostic_npz, **npz_arrays)
    diagnostic_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return diagnostic_csv, diagnostic_npz, diagnostic_json, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Potch clean epochs through current outer42 best.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--clean-csv", type=Path)
    input_group.add_argument("--raw-zip", type=Path)
    input_group.add_argument("--raw-bin", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/content/drive/MyDrive/SSE_outputs"))
    parser.add_argument("--out-dir", type=Path, default=Path("/content/drive/MyDrive/SSE_outputs/potch_current_best_outer42"))
    parser.add_argument("--original-npz", type=Path, help="DreamT original temporal context20 NPZ schema.")
    parser.add_argument("--w20-npz", type=Path, help="DreamT full-w20 temporal context20 NPZ schema.")
    parser.add_argument("--outer-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--feature-profile",
        choices=FEATURE_PROFILES,
        default="current",
        help=(
            "current keeps the v4 serving feature map. stable-vitals-v5 imputes "
            "BVP mean-family features and unstable HR/IBI variability/temporal features. "
            "v6-* profiles isolate BVP mean-family and HR/IBI variability effects."
        ),
    )
    parser.add_argument("--session-gap-ms", type=int, default=30_000)
    parser.add_argument("--packet-count-min", type=int, default=216)
    parser.add_argument("--duration-ms-min", type=int, default=25_000)
    parser.add_argument("--seq-gap-count-required", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    outer_seed = args.outer_seed
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean_csv = args.clean_csv
    raw_path = args.raw_zip or args.raw_bin
    raw_report: dict[str, Any] | None = None
    if raw_path is not None:
        clean_csv = args.out_dir / "potch_clean_epoch_features.csv"
        raw_report = build_potch_epoch_features(
            raw_path=raw_path,
            out_csv=args.out_dir / "potch_epoch_features.csv",
            clean_csv=clean_csv,
            summary_json=args.out_dir / "potch_epoch_feature_summary.json",
            clean_npz=args.out_dir / "potch_clean_epoch_features.npz",
            session_gap_ms=args.session_gap_ms,
            quality=QualityFilter(
                packet_count_min=args.packet_count_min,
                duration_ms_min=args.duration_ms_min,
                seq_gap_count_required=args.seq_gap_count_required,
            ),
        )
    if clean_csv is None:
        raise ValueError("Expected --clean-csv, --raw-zip, or --raw-bin")
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
    check_paths([original_npz, w20_npz])

    x_original, meta_original, original_report = build_context_windows(
        clean_csv,
        original_npz,
        args.feature_profile,
    )
    x_w20, meta_w20, w20_report = build_context_windows(
        clean_csv,
        w20_npz,
        args.feature_profile,
    )
    if not meta_original[["session_id", "epoch_index"]].equals(meta_w20[["session_id", "epoch_index"]]):
        raise ValueError("Original and w20 context windows are not aligned")

    original_paths = role_paths(output_root, outer_seed, "original")
    full_paths = role_paths(output_root, outer_seed, "full_w20")
    capacity_paths = role_paths(output_root, outer_seed, "capacity_h128")
    ls003_paths = role_paths(output_root, outer_seed, "h128_ls003")
    direct4_single, direct4_members = direct4_paths(output_root, outer_seed)
    specialist_h128, specialist_h256 = specialist_paths(output_root, outer_seed)
    check_paths(
        [
            *original_paths,
            *full_paths,
            *capacity_paths,
            *ls003_paths,
            *direct4_members,
            specialist_h128,
            specialist_h256,
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_probs = average_checkpoints(original_paths, x_original, device, 5, args.batch_size)
    full_probs = average_checkpoints(full_paths, x_w20, device, 5, args.batch_size)
    capacity_probs = average_checkpoints(capacity_paths, x_w20, device, 5, args.batch_size)
    ls003_probs = average_checkpoints(ls003_paths, x_w20, device, 5, args.batch_size)

    primary, secondary, tertiary = build_grouped_class_weights(
        wake_primary=0.72,
        wake_secondary=0.06,
        wake_tertiary=0.00,
        light_deep_primary=0.80,
        light_deep_secondary=0.02,
        light_deep_tertiary=0.15,
        deep_primary=0.82,
        deep_secondary=0.00,
        deep_tertiary=0.18,
        rem_primary=0.00,
        rem_secondary=0.42,
        rem_tertiary=0.13,
    )
    current5 = four_model_classwise_fusion(
        original_probs,
        full_probs,
        capacity_probs,
        ls003_probs,
        primary,
        secondary,
        tertiary,
    )
    current4 = current_probs_to_four(current5)

    direct4_single_probs = predict_checkpoint(direct4_single, x_original, device, 4, args.batch_size)
    direct4_ensemble_probs = average_checkpoints(direct4_members, x_original, device, 4, args.batch_size)
    direct4_source = blend_direct4_sources(
        direct4_single_probs,
        direct4_ensemble_probs,
        CURRENT_SOURCE_BETAS,
    )
    static_best = hybrid_fusion(
        current4,
        direct4_source,
        CURRENT_HYBRID_ALPHAS,
        CURRENT_DEEP_GAIN,
    )

    specialist_h128_probs = predict_checkpoint(specialist_h128, x_original, device, 2, args.batch_size)
    specialist_h256_probs = predict_checkpoint(specialist_h256, x_original, device, 2, args.batch_size)
    specialist_blend = (
        SPECIALIST_BLEND[0] * specialist_h128_probs
        + SPECIALIST_BLEND[1] * specialist_h256_probs
    ).astype(np.float32)
    final_probs = fuse_light_deep_conditional(
        static_best,
        specialist_blend,
        SPECIALIST_BETA,
        SPECIALIST_SCALE,
        SPECIALIST_BIAS,
    ).astype(np.float32)
    final_pred = final_probs.argmax(axis=1).astype(np.int64)
    component_variants = {
        "original_ensemble": current_probs_to_four(original_probs),
        "full_w20_ensemble": current_probs_to_four(full_probs),
        "capacity_h128_ensemble": current_probs_to_four(capacity_probs),
        "h128_ls003_ensemble": current_probs_to_four(ls003_probs),
        "four_model_current": current4,
        "direct4_single": direct4_single_probs,
        "direct4_ensemble": direct4_ensemble_probs,
        "direct4_source_blend": direct4_source,
        "static_hybrid_no_specialist": static_best,
        "final_with_specialist": final_probs,
    }
    (
        component_csv,
        component_npz,
        component_json,
        component_summary,
    ) = write_component_diagnostics(args.out_dir, meta_original, component_variants)

    out = meta_original.copy()
    for index, name in enumerate(STAGE4_NAMES):
        out[f"p_{name.lower()}"] = final_probs[:, index]
    out["pred_id"] = final_pred
    out["pred_stage"] = [STAGE4_NAMES[index] for index in final_pred]

    prediction_csv = args.out_dir / "potch_current_best_outer42_predictions.csv"
    prediction_npz = args.out_dir / "potch_current_best_outer42_predictions.npz"
    report_json = args.out_dir / "potch_current_best_outer42_report.json"
    out.to_csv(prediction_csv, index=False)
    np.savez_compressed(
        prediction_npz,
        probs=final_probs,
        pred=final_pred,
        stage_names=np.asarray(STAGE4_NAMES),
        session_id=out["session_id"].to_numpy(dtype=np.int64),
        epoch_index=out["epoch_index"].to_numpy(dtype=np.int64),
        start_app_ts=out["start_app_ts"].to_numpy(dtype=np.int64),
        end_app_ts=out["end_app_ts"].to_numpy(dtype=np.int64),
    )
    report = {
        "clean_csv": str(clean_csv),
        "raw_input": str(raw_path) if raw_path is not None else None,
        "output_root": str(output_root),
        "outer_seed": outer_seed,
        "feature_profile": args.feature_profile,
        "device": str(device),
        "prediction_csv": str(prediction_csv),
        "prediction_npz": str(prediction_npz),
        "component_prediction_csv": str(component_csv),
        "component_prediction_npz": str(component_npz),
        "component_summary_json": str(component_json),
        "prediction_count": int(final_probs.shape[0]),
        "class_counts": summarize_predictions(final_pred),
        "probability_mean": {
            name: float(final_probs[:, index].mean())
            for index, name in enumerate(STAGE4_NAMES)
        },
        "original_feature_report": original_report,
        "w20_feature_report": w20_report,
        "raw_feature_report": raw_report,
        "component_summary": component_summary,
        "imputation_note": (
            "Features absent from Potch clean CSV were imputed with the DreamT "
            "training mean, which becomes 0 after normalization."
        ),
        "fusion": {
            "outer_seed": outer_seed,
            "checkpoint_count": {
                "original": len(original_paths),
                "full_w20": len(full_paths),
                "capacity_h128": len(capacity_paths),
                "h128_ls003": len(ls003_paths),
                "direct4": len(direct4_members),
                "specialists": 2,
            },
            "source_betas": CURRENT_SOURCE_BETAS.tolist(),
            "hybrid_alphas": CURRENT_HYBRID_ALPHAS.tolist(),
            "deep_gain": CURRENT_DEEP_GAIN,
            "specialist_blend": SPECIALIST_BLEND.tolist(),
            "specialist_beta": SPECIALIST_BETA,
            "specialist_scale": SPECIALIST_SCALE,
            "specialist_bias": SPECIALIST_BIAS,
        },
    }
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
