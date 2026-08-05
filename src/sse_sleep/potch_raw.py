"""Parse Potch raw binary packets into clean epoch feature CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import struct
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .features import basic_stats, quality_features, temp_epoch_features


RECORD_BYTES = 150
MAGIC = 0x5AA5
EPOCH_MS = 30_000
ACC_RAW_SCALE = 256.0
PPG_AC_SCALE = 10.0
IBI_MS_PER_SECOND = 1000.0
DREAMT_SLOPE_SAMPLE_RATE_HZ = 100.0
IBI_OUTLIER_REL_TOLERANCE = 0.20
IBI_OUTLIER_MAD_MULTIPLIER = 4.0
BEAT_MEDIAN_SMOOTH_WINDOW = 9
NTC_R0_OHMS = 100_000.0
NTC_BETA_K = 4390.0
NTC_T0_K = 298.15
NTC_FIXED_RESISTOR_OHMS = 100_000.0
NTC_ADC_MAX = 4095.0
PPG_TRANSFORMS = (
    "epoch-median",
    "rolling-mean-8s",
    "rolling-mean-15s",
    "rolling-mean-30s",
    "highpass-lowpass-15s",
    "highpass-lowpass-30s",
)

IMU_AXES = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
FEATURE_COLUMNS = (
    "battery_mean",
    "battery_min",
    "battery_max",
    "ntc_raw_mean",
    "ntc_raw_min",
    "ntc_raw_max",
    "temp_mean",
    "temp_std",
    "temp_median",
    "temp_iqr",
    "temp_min",
    "temp_max",
    "temp_slope",
    "temp_slope_v2",
    "temp_baseline_delta",
    "ppg_mean",
    "ppg_std",
    "ppg_median",
    "ppg_iqr",
    "ppg_min",
    "ppg_max",
    "ppg_slope",
    "ppg_missing_ratio",
    "ppg_flatline_ratio",
    "ppg_edge_ratio",
    "acc_x_mean",
    "acc_x_std",
    "acc_x_median",
    "acc_x_iqr",
    "acc_x_min",
    "acc_x_max",
    "acc_x_slope",
    "acc_y_mean",
    "acc_y_std",
    "acc_y_median",
    "acc_y_iqr",
    "acc_y_min",
    "acc_y_max",
    "acc_y_slope",
    "acc_z_mean",
    "acc_z_std",
    "acc_z_median",
    "acc_z_iqr",
    "acc_z_min",
    "acc_z_max",
    "acc_z_slope",
    "acc_vm_mean",
    "acc_vm_std",
    "acc_vm_median",
    "acc_vm_iqr",
    "acc_vm_min",
    "acc_vm_max",
    "acc_vm_slope",
    "acc_vm_activity",
    "hr_mean",
    "hr_mean_v2",
    "hr_std",
    "hr_std_v2",
    "hr_median",
    "hr_median_v2",
    "hr_iqr",
    "hr_iqr_v2",
    "hr_min",
    "hr_min_v2",
    "hr_max",
    "hr_max_v2",
    "hr_slope",
    "hr_slope_v2",
    "hr_missing_ratio",
    "hr_flatline_ratio",
    "hr_edge_ratio",
    "ibi_mean",
    "ibi_mean_v2",
    "ibi_std",
    "ibi_std_v2",
    "ibi_median",
    "ibi_median_v2",
    "ibi_iqr",
    "ibi_iqr_v2",
    "ibi_min",
    "ibi_min_v2",
    "ibi_max",
    "ibi_max_v2",
    "ibi_slope",
    "ibi_slope_v2",
    "ibi_missing_ratio",
    "ibi_flatline_ratio",
    "ibi_edge_ratio",
    "gyro_x_mean",
    "gyro_x_std",
    "gyro_x_median",
    "gyro_x_iqr",
    "gyro_x_min",
    "gyro_x_max",
    "gyro_x_slope",
    "gyro_y_mean",
    "gyro_y_std",
    "gyro_y_median",
    "gyro_y_iqr",
    "gyro_y_min",
    "gyro_y_max",
    "gyro_y_slope",
    "gyro_z_mean",
    "gyro_z_std",
    "gyro_z_median",
    "gyro_z_iqr",
    "gyro_z_min",
    "gyro_z_max",
    "gyro_z_slope",
)
CSV_COLUMNS = (
    "session_id",
    "epoch_index",
    "start_app_ts",
    "end_app_ts",
    "duration_ms",
    "packet_count",
    "imu_sample_count",
    "ppg_sample_count",
    "seq_first",
    "seq_last",
    "seq_gap_count",
    "app_delta_median_ms",
    "app_delta_max_ms",
    "mcu_delta_median_ms",
    "mcu_delta_max_ms",
    *FEATURE_COLUMNS,
    "quality_pass",
)


@dataclass(frozen=True)
class PotchPacket:
    app_ts_ms: int
    seq: int
    mcu_ts_ms: int
    battery_raw: int
    ntc_raw: int
    imu: tuple[tuple[int, int, int, int, int, int], ...]
    ppg: tuple[int, ...]


@dataclass(frozen=True)
class QualityFilter:
    packet_count_min: int = 216
    duration_ms_min: int = 25_000
    seq_gap_count_required: int = 0


def read_raw_payload(path: Path) -> tuple[bytes, str]:
    """Read a .bin file or a zip that contains exactly one .bin payload."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            bin_names = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".bin")
                and "__MACOSX/" not in name
                and not Path(name).name.startswith("._")
            ]
            if len(bin_names) != 1:
                raise ValueError(f"Expected exactly one .bin in {path}, found {bin_names}")
            return archive.read(bin_names[0]), bin_names[0]
    return path.read_bytes(), path.name


def parse_packet(payload: bytes, offset: int) -> PotchPacket:
    app_ts_ms = struct.unpack_from("<Q", payload, offset)[0]
    magic, seq, mcu_ts_ms, battery_raw, ntc_raw = struct.unpack_from("<HHIHH", payload, offset + 8)
    if magic != MAGIC:
        raise ValueError(f"Bad Potch packet magic at offset {offset}: 0x{magic:04x}")
    imu_values = struct.unpack_from("<" + "h" * 48, payload, offset + 20)
    imu = tuple(
        tuple(int(value) for value in imu_values[index : index + 6])
        for index in range(0, len(imu_values), 6)
    )
    ppg = tuple(int(value) for value in struct.unpack_from("<" + "H" * 16, payload, offset + 116))
    return PotchPacket(
        app_ts_ms=int(app_ts_ms),
        seq=int(seq),
        mcu_ts_ms=int(mcu_ts_ms),
        battery_raw=int(battery_raw),
        ntc_raw=int(ntc_raw),
        imu=imu,
        ppg=ppg,
    )


def parse_packets(path: Path) -> tuple[list[PotchPacket], dict[str, Any]]:
    payload, payload_name = read_raw_payload(path)
    if len(payload) % RECORD_BYTES != 0:
        raise ValueError(
            f"Raw payload length {len(payload)} is not divisible by {RECORD_BYTES}"
        )
    packets: list[PotchPacket] = []
    bad_records: list[dict[str, Any]] = []
    for record_index, offset in enumerate(range(0, len(payload), RECORD_BYTES)):
        try:
            packets.append(parse_packet(payload, offset))
        except ValueError as error:
            bad_records.append(
                {
                    "record_index": record_index,
                    "offset": offset,
                    "error": str(error),
                }
            )
    if not packets:
        raise ValueError(f"No valid Potch packets found in {path}")
    packets.sort(key=lambda packet: (packet.app_ts_ms, packet.seq))
    report = {
        "source_path": str(path),
        "payload_name": payload_name,
        "payload_bytes": len(payload),
        "record_bytes": RECORD_BYTES,
        "record_count": len(payload) // RECORD_BYTES,
        "packet_count": len(packets),
        "bad_record_count": len(bad_records),
        "bad_record_examples": bad_records[:20],
    }
    return packets, report


def split_sessions(packets: Sequence[PotchPacket], session_gap_ms: int) -> list[list[PotchPacket]]:
    if not packets:
        return []
    sessions: list[list[PotchPacket]] = [[packets[0]]]
    for packet in packets[1:]:
        previous = sessions[-1][-1]
        if packet.app_ts_ms - previous.app_ts_ms > session_gap_ms:
            sessions.append([packet])
        else:
            sessions[-1].append(packet)
    return sessions


def group_epochs(session: Sequence[PotchPacket]) -> dict[int, list[PotchPacket]]:
    if not session:
        return {}
    session_start = session[0].app_ts_ms
    epochs: dict[int, list[PotchPacket]] = {}
    for packet in session:
        epoch_index = int((packet.app_ts_ms - session_start) // EPOCH_MS)
        epochs.setdefault(epoch_index, []).append(packet)
    return epochs


def median_or_none(values: Sequence[int | float]) -> float | None:
    return float(statistics.median(values)) if values else None


def max_or_none(values: Sequence[int | float]) -> float | None:
    return float(max(values)) if values else None


def modulo_delta(after: int, before: int, modulus: int) -> int:
    return int((after - before) % modulus)


def flattened_imu_axis(packets: Sequence[PotchPacket], axis_index: int) -> list[int]:
    return [sample[axis_index] for packet in packets for sample in packet.imu]


def flattened_acc_axis(packets: Sequence[PotchPacket], axis_index: int) -> list[float]:
    return [sample[axis_index] / ACC_RAW_SCALE for packet in packets for sample in packet.imu]


def ntc_raw_to_celsius(raw: int | float) -> float | None:
    adc = float(raw)
    if not math.isfinite(adc) or adc <= 0 or adc >= NTC_ADC_MAX:
        return None
    resistance = NTC_FIXED_RESISTOR_OHMS * adc / (NTC_ADC_MAX - adc)
    if resistance <= 0:
        return None
    inv_temp_k = (1.0 / NTC_T0_K) + (1.0 / NTC_BETA_K) * math.log(resistance / NTC_R0_OHMS)
    if inv_temp_k <= 0:
        return None
    return (1.0 / inv_temp_k) - 273.15


def flatten_ppg(packets: Sequence[PotchPacket]) -> list[int]:
    return [value for packet in packets for value in packet.ppg]


def packet_key(packet: PotchPacket) -> tuple[int, int]:
    return packet.app_ts_ms, packet.seq


def centered_ppg_proxy(ppg_values: Sequence[int | float], scale: float = PPG_AC_SCALE) -> list[float]:
    if not ppg_values:
        return []
    baseline = statistics.median(ppg_values)
    return [(float(value) - baseline) / scale for value in ppg_values]


def session_sample_rate_hz(packets: Sequence[PotchPacket], sample_count: int) -> float:
    if len(packets) < 2:
        return 0.0
    duration_seconds = (packets[-1].app_ts_ms - packets[0].app_ts_ms) / 1000.0
    if duration_seconds <= 0:
        return 0.0
    return sample_count / duration_seconds


def seconds_from_transform(transform: str, default: float) -> float:
    if transform.endswith("8s"):
        return 8.0
    if transform.endswith("15s"):
        return 15.0
    if transform.endswith("30s"):
        return 30.0
    return default


def session_ppg_proxy(
    packets: Sequence[PotchPacket],
    transform: str,
    scale: float = PPG_AC_SCALE,
) -> dict[tuple[int, int], list[float]] | None:
    if transform == "epoch-median":
        return None
    if transform not in PPG_TRANSFORMS:
        raise ValueError(f"Unknown PPG transform: {transform}")

    raw_values = [float(value) for packet in packets for value in packet.ppg]
    if not raw_values:
        return {packet_key(packet): [] for packet in packets}
    sample_rate_hz = session_sample_rate_hz(packets, len(raw_values))
    if sample_rate_hz <= 0:
        transformed = centered_ppg_proxy(raw_values, scale)
    else:
        baseline_seconds = seconds_from_transform(transform, 15.0)
        baseline_window = max(3, int(round(sample_rate_hz * baseline_seconds)))
        baseline = moving_average(raw_values, baseline_window)
        transformed = [
            (value - base) / scale
            for value, base in zip(raw_values, baseline, strict=True)
        ]
        if transform.startswith("highpass-lowpass"):
            lowpass_window = max(3, int(round(sample_rate_hz * 0.15)))
            transformed = moving_average(transformed, lowpass_window)

    by_packet: dict[tuple[int, int], list[float]] = {}
    offset = 0
    for packet in packets:
        count = len(packet.ppg)
        by_packet[packet_key(packet)] = transformed[offset : offset + count]
        offset += count
    return by_packet


def flatten_ppg_proxy(
    packets: Sequence[PotchPacket],
    ppg_model_by_packet: dict[tuple[int, int], list[float]] | None,
    ppg_scale: float,
) -> list[float]:
    if ppg_model_by_packet is None:
        return centered_ppg_proxy(flatten_ppg(packets), ppg_scale)
    return [
        value
        for packet in packets
        for value in ppg_model_by_packet.get(packet_key(packet), [])
    ]


def flattened_acc_vm(packets: Sequence[PotchPacket]) -> list[float]:
    values: list[float] = []
    for packet in packets:
        for sample in packet.imu:
            acc_x, acc_y, acc_z = (value / ACC_RAW_SCALE for value in sample[:3])
            values.append(math.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z))
    return values


def activity(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    movement = sum(abs(after - before) for before, after in zip(values, values[1:], strict=False))
    return movement / (len(values) - 1)


def dreamt_grid_slope_v2(
    values: Sequence[float],
    duration_ms: int,
    sample_rate_hz: float = DREAMT_SLOPE_SAMPLE_RATE_HZ,
) -> float | None:
    if len(values) < 2 or duration_ms <= 0:
        return None
    edge_count = max(1, min(3, len(values) // 5))
    start_value = statistics.median(values[:edge_count])
    end_value = statistics.median(values[-edge_count:])
    dreamt_samples = max(2, int(round((duration_ms / 1000.0) * sample_rate_hz)))
    return (float(end_value) - float(start_value)) / (dreamt_samples - 1)


def moving_average(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or not values:
        return [float(value) for value in values]
    half = window // 2
    smoothed: list[float] = []
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + float(value))
    for index in range(len(values)):
        start = max(0, index - half)
        end = min(len(values), index + half + 1)
        smoothed.append((prefix[end] - prefix[start]) / (end - start))
    return smoothed


def quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def detect_ppg_peak_indices(ppg_values: Sequence[int | float], sample_rate_hz: float) -> list[int]:
    if len(ppg_values) < 3 or sample_rate_hz <= 0:
        return []
    smooth_window = max(3, int(round(sample_rate_hz * 0.06)))
    baseline_window = max(smooth_window + 2, int(round(sample_rate_hz * 0.75)))
    smoothed = moving_average([float(value) for value in ppg_values], smooth_window)
    baseline = moving_average(smoothed, baseline_window)
    detrended = [value - base for value, base in zip(smoothed, baseline, strict=True)]
    ordered = sorted(detrended)
    median = statistics.median(ordered)
    q75 = quantile(ordered, 0.75) or median
    q25 = quantile(ordered, 0.25) or median
    robust_spread = max(q75 - q25, statistics.pstdev(detrended), 1.0)
    threshold = median + 0.35 * robust_spread
    min_distance = max(1, int(round(sample_rate_hz * 0.30)))

    candidates: list[int] = []
    for index in range(1, len(detrended) - 1):
        if (
            detrended[index] > threshold
            and detrended[index] >= detrended[index - 1]
            and detrended[index] > detrended[index + 1]
        ):
            candidates.append(index)

    selected: list[int] = []
    for index in candidates:
        if not selected or index - selected[-1] >= min_distance:
            selected.append(index)
            continue
        if detrended[index] > detrended[selected[-1]]:
            selected[-1] = index
    return selected


def median_filter(values: Sequence[float], window: int) -> list[float]:
    if window <= 1 or len(values) < 3:
        return [float(value) for value in values]
    radius = window // 2
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(float(statistics.median(values[start:end])))
    return smoothed


def reject_ibi_outliers(ibi_values: Sequence[float]) -> list[float]:
    physiological = [float(value) for value in ibi_values if 0.3 <= float(value) <= 2.0]
    if len(physiological) < 5:
        return physiological

    median = statistics.median(physiological)
    abs_deviations = [abs(value - median) for value in physiological]
    mad = statistics.median(abs_deviations)
    robust_sigma = 1.4826 * mad
    filtered: list[float] = []
    for index, value in enumerate(physiological):
        start = max(0, index - 2)
        end = min(len(physiological), index + 3)
        local_median = statistics.median(physiological[start:end])
        rel_ok = local_median <= 0 or abs(value - local_median) / local_median <= IBI_OUTLIER_REL_TOLERANCE
        mad_ok = robust_sigma <= 0 or abs(value - median) <= IBI_OUTLIER_MAD_MULTIPLIER * robust_sigma
        if rel_ok and mad_ok:
            filtered.append(value)
    return filtered


def smoothed_hr_ibi_v2(ibi_values: Sequence[float]) -> tuple[list[float], list[float]]:
    filtered_ibi = reject_ibi_outliers(ibi_values)
    smoothed_ibi = median_filter(filtered_ibi, BEAT_MEDIAN_SMOOTH_WINDOW)
    smoothed_hr = [60.0 / ibi_seconds for ibi_seconds in smoothed_ibi if ibi_seconds > 0]
    return smoothed_hr, smoothed_ibi


def ppg_derived_hr_ibi(
    ppg_values: Sequence[int | float],
    duration_ms: int,
) -> tuple[list[float], list[float], list[float], list[float], dict[str, float | int]]:
    if duration_ms <= 0 or len(ppg_values) < 3:
        return [], [], [], [], {
            "peak_count": 0,
            "valid_ibi_count": 0,
            "v2_valid_ibi_count": 0,
            "sample_rate_hz": 0.0,
        }
    sample_rate_hz = len(ppg_values) / (duration_ms / 1000.0)
    peaks = detect_ppg_peak_indices(ppg_values, sample_rate_hz)
    ibi_values: list[float] = []
    for before, after in zip(peaks, peaks[1:], strict=False):
        ibi_ms = (after - before) / sample_rate_hz * 1000.0
        if 300.0 <= ibi_ms <= 2000.0:
            ibi_values.append(ibi_ms / IBI_MS_PER_SECOND)
    hr_values = [60.0 / ibi_seconds for ibi_seconds in ibi_values if ibi_seconds > 0]
    hr_values_v2, ibi_values_v2 = smoothed_hr_ibi_v2(ibi_values)
    report = {
        "peak_count": len(peaks),
        "valid_ibi_count": len(ibi_values),
        "v2_valid_ibi_count": len(ibi_values_v2),
        "sample_rate_hz": sample_rate_hz,
    }
    return hr_values, ibi_values, hr_values_v2, ibi_values_v2, report


def stats_for_raw(values: Iterable[int | float], prefix: str) -> dict[str, float | None]:
    stats = basic_stats(values, prefix)
    return {key.removeprefix(f"{prefix}_"): value for key, value in stats.items()}


def quality_for_raw(values: Sequence[int | float], prefix: str) -> dict[str, float | None]:
    features = quality_features(values, prefix)
    return {key.removeprefix(f"{prefix}_"): value for key, value in features.items()}


def epoch_row(
    session_id: int,
    epoch_index: int,
    packets: Sequence[PotchPacket],
    quality: QualityFilter,
    ppg_model_by_packet: dict[tuple[int, int], list[float]] | None = None,
    ppg_scale: float = PPG_AC_SCALE,
    temp_session_baseline: float | None = None,
) -> dict[str, Any]:
    packets = sorted(packets, key=lambda packet: (packet.app_ts_ms, packet.seq))
    app_times = [packet.app_ts_ms for packet in packets]
    mcu_times = [packet.mcu_ts_ms for packet in packets]
    seqs = [packet.seq for packet in packets]
    app_deltas = [after - before for before, after in zip(app_times, app_times[1:], strict=False)]
    mcu_deltas = [modulo_delta(after, before, 2**32) for before, after in zip(mcu_times, mcu_times[1:], strict=False)]
    seq_gap_count = sum(
        1 for before, after in zip(seqs, seqs[1:], strict=False)
        if modulo_delta(after, before, 2**16) != 1
    )
    duration_ms = app_times[-1] - app_times[0] if app_times else 0
    row: dict[str, Any] = {
        "session_id": session_id,
        "epoch_index": epoch_index,
        "start_app_ts": app_times[0],
        "end_app_ts": app_times[-1],
        "duration_ms": duration_ms,
        "packet_count": len(packets),
        "imu_sample_count": len(packets) * 8,
        "ppg_sample_count": len(packets) * 16,
        "seq_first": seqs[0],
        "seq_last": seqs[-1],
        "seq_gap_count": seq_gap_count,
        "app_delta_median_ms": median_or_none(app_deltas),
        "app_delta_max_ms": max_or_none(app_deltas),
        "mcu_delta_median_ms": median_or_none(mcu_deltas),
        "mcu_delta_max_ms": max_or_none(mcu_deltas),
    }
    for name in ("battery", "ntc_raw"):
        values = [packet.battery_raw if name == "battery" else packet.ntc_raw for packet in packets]
        stats = stats_for_raw(values, name)
        row[f"{name}_mean"] = stats["mean"]
        row[f"{name}_min"] = stats["min"]
        row[f"{name}_max"] = stats["max"]
    temp_values = [
        converted
        for packet in packets
        if (converted := ntc_raw_to_celsius(packet.ntc_raw)) is not None
    ]
    row.update(temp_epoch_features(temp_values, temp_session_baseline))
    row["temp_slope_v2"] = dreamt_grid_slope_v2(temp_values, duration_ms)
    ppg_raw_values = flatten_ppg(packets)
    ppg_model_values = flatten_ppg_proxy(packets, ppg_model_by_packet, ppg_scale)
    for suffix, value in stats_for_raw(ppg_model_values, "ppg").items():
        row[f"ppg_{suffix}"] = value
    for suffix, value in quality_for_raw(ppg_raw_values, "ppg").items():
        row[f"ppg_{suffix}"] = value
    for axis_index, axis_name in enumerate(IMU_AXES):
        axis_values = (
            flattened_acc_axis(packets, axis_index)
            if axis_name.startswith("acc_")
            else flattened_imu_axis(packets, axis_index)
        )
        for suffix, value in stats_for_raw(axis_values, axis_name).items():
            row[f"{axis_name}_{suffix}"] = value
    acc_vm_values = flattened_acc_vm(packets)
    for suffix, value in stats_for_raw(acc_vm_values, "acc_vm").items():
        row[f"acc_vm_{suffix}"] = value
    row["acc_vm_activity"] = activity(acc_vm_values)
    ppg_peak_values: Sequence[int | float] = ppg_raw_values if ppg_model_by_packet is None else ppg_model_values
    hr_values, ibi_values, hr_values_v2, ibi_values_v2, ppg_derived_report = ppg_derived_hr_ibi(
        ppg_peak_values,
        duration_ms,
    )
    for prefix, values, values_v2 in (
        ("hr", hr_values, hr_values_v2),
        ("ibi", ibi_values, ibi_values_v2),
    ):
        for suffix, value in stats_for_raw(values, prefix).items():
            row[f"{prefix}_{suffix}"] = value
        for suffix, value in stats_for_raw(values_v2, prefix).items():
            if suffix != "slope":
                row[f"{prefix}_{suffix}_v2"] = value
        row[f"{prefix}_slope_v2"] = dreamt_grid_slope_v2(values_v2, duration_ms)
        for suffix, value in quality_for_raw(values, prefix).items():
            row[f"{prefix}_{suffix}"] = value
    row["ppg_peak_count"] = ppg_derived_report["peak_count"]
    row["ppg_valid_ibi_count"] = ppg_derived_report["valid_ibi_count"]
    row["ppg_sample_rate_hz"] = ppg_derived_report["sample_rate_hz"]
    quality_pass = (
        len(packets) >= quality.packet_count_min
        and duration_ms >= quality.duration_ms_min
        and seq_gap_count == quality.seq_gap_count_required
    )
    row["quality_pass"] = quality_pass
    return row


def build_epoch_rows(
    packets: Sequence[PotchPacket],
    session_gap_ms: int,
    quality: QualityFilter,
    ppg_transform: str = "epoch-median",
    ppg_scale: float = PPG_AC_SCALE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session_id, session in enumerate(split_sessions(packets, session_gap_ms)):
        ppg_model_by_packet = session_ppg_proxy(session, ppg_transform, ppg_scale)
        session_temp_values = [
            converted
            for packet in session
            if (converted := ntc_raw_to_celsius(packet.ntc_raw)) is not None
        ]
        temp_session_baseline = (
            float(statistics.median(session_temp_values))
            if session_temp_values
            else None
        )
        for epoch_index, epoch_packets in sorted(group_epochs(session).items()):
            rows.append(
                epoch_row(
                    session_id,
                    epoch_index,
                    epoch_packets,
                    quality,
                    ppg_model_by_packet=ppg_model_by_packet,
                    ppg_scale=ppg_scale,
                    temp_session_baseline=temp_session_baseline,
                )
            )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in CSV_COLUMNS})


def finite_summary(values: Sequence[int | float]) -> dict[str, float | int | None]:
    cleaned = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    if not cleaned:
        return {"min": None, "median": None, "max": None}
    return {
        "min": int(min(cleaned)) if min(cleaned).is_integer() else min(cleaned),
        "median": float(statistics.median(cleaned)),
        "max": int(max(cleaned)) if max(cleaned).is_integer() else max(cleaned),
    }


def rejection_reasons(row: dict[str, Any], quality: QualityFilter) -> list[str]:
    reasons: list[str] = []
    if int(row["packet_count"]) < quality.packet_count_min:
        reasons.append("packet_count_below_min")
    if int(row["duration_ms"]) < quality.duration_ms_min:
        reasons.append("duration_below_min")
    if int(row["seq_gap_count"]) != quality.seq_gap_count_required:
        reasons.append("seq_gap_nonzero")
    return reasons


def maybe_write_npz(path: Path, rows: Sequence[dict[str, Any]]) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_columns = (
        "session_id",
        "epoch_index",
        "start_app_ts",
        "end_app_ts",
        "duration_ms",
        "packet_count",
        "seq_gap_count",
        "app_delta_max_ms",
    )
    arrays = {
        "feature_names": np.asarray(FEATURE_COLUMNS),
        "features": np.asarray(
            [[row[column] for column in FEATURE_COLUMNS] for row in rows],
            dtype=np.float32,
        ),
    }
    for column in metadata_columns:
        arrays[column] = np.asarray([row[column] for row in rows])
    np.savez_compressed(path, **arrays)
    return True


def build_potch_epoch_features(
    raw_path: Path,
    out_csv: Path,
    clean_csv: Path,
    summary_json: Path,
    clean_npz: Path | None = None,
    session_gap_ms: int = 30_000,
    quality: QualityFilter = QualityFilter(),
    ppg_transform: str = "epoch-median",
    ppg_scale: float = PPG_AC_SCALE,
) -> dict[str, Any]:
    packets, parse_report = parse_packets(raw_path)
    rows = build_epoch_rows(
        packets,
        session_gap_ms=session_gap_ms,
        quality=quality,
        ppg_transform=ppg_transform,
        ppg_scale=ppg_scale,
    )
    clean_rows = [row for row in rows if row["quality_pass"]]
    write_csv(out_csv, rows)
    write_csv(clean_csv, clean_rows)
    clean_npz_written = False
    if clean_npz is not None:
        clean_npz_written = maybe_write_npz(clean_npz, clean_rows)

    reason_counts: dict[str, int] = {}
    rejected_epochs: list[dict[str, Any]] = []
    for row in rows:
        reasons = rejection_reasons(row, quality)
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            rejected_epochs.append(
                {
                    "session_id": int(row["session_id"]),
                    "epoch_index": int(row["epoch_index"]),
                    "duration_ms": int(row["duration_ms"]),
                    "packet_count": int(row["packet_count"]),
                    "seq_gap_count": int(row["seq_gap_count"]),
                    "app_delta_max_ms": row["app_delta_max_ms"],
                    "seq_first": int(row["seq_first"]),
                    "seq_last": int(row["seq_last"]),
                    "reasons": reasons,
                }
            )

    summary = {
        **parse_report,
        "out_csv": str(out_csv),
        "clean_csv": str(clean_csv),
        "clean_npz": str(clean_npz) if clean_npz is not None and clean_npz_written else None,
        "clean_npz_written": clean_npz_written,
        "session_gap_ms": session_gap_ms,
        "ppg_transform": ppg_transform,
        "ppg_scale": ppg_scale,
        "quality_filter": asdict(quality),
        "input_epoch_count": len(rows),
        "clean_epoch_count": len(clean_rows),
        "rejected_epoch_count": len(rows) - len(clean_rows),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_columns": list(FEATURE_COLUMNS),
        "metadata_columns_in_npz": [
            "session_id",
            "epoch_index",
            "start_app_ts",
            "end_app_ts",
            "duration_ms",
            "packet_count",
            "seq_gap_count",
            "app_delta_max_ms",
        ],
        "clean_packet_count": finite_summary([row["packet_count"] for row in clean_rows]),
        "clean_duration_ms": finite_summary([row["duration_ms"] for row in clean_rows]),
        "rejection_reasons": reason_counts,
        "rejected_epochs": rejected_epochs[:100],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Potch epoch features from a raw .bin or .zip file.")
    parser.add_argument("--raw", type=Path, required=True, help="Potch raw .bin or zip containing one .bin file.")
    parser.add_argument("--out-csv", type=Path, required=True, help="All epoch features before quality filtering.")
    parser.add_argument("--clean-csv", type=Path, required=True, help="Epoch features after quality filtering.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--clean-npz", type=Path)
    parser.add_argument("--session-gap-ms", type=int, default=30_000)
    parser.add_argument("--ppg-transform", choices=PPG_TRANSFORMS, default="epoch-median")
    parser.add_argument("--ppg-scale", type=float, default=PPG_AC_SCALE)
    parser.add_argument("--packet-count-min", type=int, default=216)
    parser.add_argument("--duration-ms-min", type=int, default=25_000)
    parser.add_argument("--seq-gap-count-required", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality = QualityFilter(
        packet_count_min=args.packet_count_min,
        duration_ms_min=args.duration_ms_min,
        seq_gap_count_required=args.seq_gap_count_required,
    )
    summary = build_potch_epoch_features(
        raw_path=args.raw,
        out_csv=args.out_csv,
        clean_csv=args.clean_csv,
        summary_json=args.summary_json,
        clean_npz=args.clean_npz,
        session_gap_ms=args.session_gap_ms,
        quality=quality,
        ppg_transform=args.ppg_transform,
        ppg_scale=args.ppg_scale,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
