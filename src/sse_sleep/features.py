"""Small dependency-free feature extractors for 30-second wearable epochs."""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


Number = int | float


def _clean(values: Iterable[Number | None]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if value is None:
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            cleaned.append(numeric)
    return cleaned


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def basic_stats(values: Iterable[Number | None], prefix: str) -> dict[str, float | None]:
    """Return robust summary statistics for one epoch signal."""
    cleaned = _clean(values)
    if not cleaned:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
            f"{prefix}_median": None,
            f"{prefix}_iqr": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_slope": None,
        }
    ordered = sorted(cleaned)
    q25 = _quantile(ordered, 0.25)
    q75 = _quantile(ordered, 0.75)
    slope = (cleaned[-1] - cleaned[0]) / max(len(cleaned) - 1, 1)
    return {
        f"{prefix}_mean": statistics.fmean(cleaned),
        f"{prefix}_std": statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0,
        f"{prefix}_median": statistics.median(cleaned),
        f"{prefix}_iqr": None if q25 is None or q75 is None else q75 - q25,
        f"{prefix}_min": ordered[0],
        f"{prefix}_max": ordered[-1],
        f"{prefix}_slope": slope,
    }


def quality_features(values: Sequence[Number | None], prefix: str) -> dict[str, float | None]:
    """Return simple missing/flatline/clipping proxies for one epoch signal."""
    total = len(values)
    cleaned = _clean(values)
    if total == 0:
        return {
            f"{prefix}_missing_ratio": None,
            f"{prefix}_flatline_ratio": None,
            f"{prefix}_edge_ratio": None,
        }
    missing_ratio = 1.0 - (len(cleaned) / total)
    if len(cleaned) < 2:
        flatline_ratio = None
    else:
        flat_count = sum(1 for before, after in zip(cleaned, cleaned[1:], strict=False) if before == after)
        flatline_ratio = flat_count / (len(cleaned) - 1)
    if cleaned:
        low = min(cleaned)
        high = max(cleaned)
        edge_count = sum(1 for value in cleaned if value == low or value == high)
        edge_ratio = edge_count / len(cleaned)
    else:
        edge_ratio = None
    return {
        f"{prefix}_missing_ratio": missing_ratio,
        f"{prefix}_flatline_ratio": flatline_ratio,
        f"{prefix}_edge_ratio": edge_ratio,
    }


def ppg_epoch_features(green_ppg: Sequence[Number | None]) -> dict[str, float | None]:
    """Extract app-computable GREEN PPG features for one 30-second epoch."""
    features = {}
    features.update(basic_stats(green_ppg, "green_ppg"))
    features.update(quality_features(green_ppg, "green_ppg"))
    return features


def acc_epoch_features(
    acc_x: Sequence[Number | None],
    acc_y: Sequence[Number | None],
    acc_z: Sequence[Number | None],
) -> dict[str, float | None]:
    """Extract app-computable accelerometer features for one 30-second epoch."""
    features = {}
    features.update(basic_stats(acc_x, "acc_x"))
    features.update(basic_stats(acc_y, "acc_y"))
    features.update(basic_stats(acc_z, "acc_z"))
    magnitudes: list[float] = []
    for x_value, y_value, z_value in zip(acc_x, acc_y, acc_z, strict=False):
        cleaned = _clean((x_value, y_value, z_value))
        if len(cleaned) == 3:
            magnitudes.append(math.sqrt(sum(value * value for value in cleaned)))
    features.update(basic_stats(magnitudes, "acc_vm"))
    if len(magnitudes) > 1:
        movement = sum(abs(after - before) for before, after in zip(magnitudes, magnitudes[1:], strict=False))
        features["acc_vm_activity"] = movement / (len(magnitudes) - 1)
    else:
        features["acc_vm_activity"] = None
    return features


def temp_epoch_features(temp: Sequence[Number | None], session_baseline: float | None = None) -> dict[str, float | None]:
    """Extract app-computable temperature features for one 30-second epoch."""
    features = basic_stats(temp, "temp")
    if session_baseline is not None and features["temp_mean"] is not None:
        features["temp_baseline_delta"] = features["temp_mean"] - session_baseline
    else:
        features["temp_baseline_delta"] = None
    return features
