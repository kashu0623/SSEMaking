"""Signal and feature provenance definitions for the wearable sleep pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


FeatureProvenance = Literal["app_raw", "app_derived", "dreamt_only"]


@dataclass(frozen=True)
class ChannelSpec:
    canonical_name: str
    aliases: tuple[str, ...]
    app_raw_available: bool
    notes: str


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    required_channels: tuple[str, ...]
    provenance: FeatureProvenance
    notes: str


CHANNEL_SPECS = (
    ChannelSpec("IR_PPG", ("ir ppg", "ppg ir", "ir", "pleth", "ppg"), True, "App raw input if optical wavelength matches."),
    ChannelSpec("RED_PPG", ("red ppg", "ppg red", "red"), True, "App raw input if red channel is available."),
    ChannelSpec("BVP", ("bvp", "blood volume pulse"), False, "DreamT wearable proxy derived from optical PPG; app can derive a comparable signal from IR/RED PPG."),
    ChannelSpec("ACC_X", ("acc x", "acc_x", "accelerometer x", "x"), True, "App raw accelerometer x-axis."),
    ChannelSpec("ACC_Y", ("acc y", "acc_y", "accelerometer y", "y"), True, "App raw accelerometer y-axis."),
    ChannelSpec("ACC_Z", ("acc z", "acc_z", "accelerometer z", "z"), True, "App raw accelerometer z-axis."),
    ChannelSpec("TEMP", ("temp", "temperature", "skin temp", "skin temperature"), True, "App raw skin/device temperature."),
    ChannelSpec("HR", ("hr", "heart rate"), False, "DreamT derived channel; app can derive from IR/RED PPG when quality is sufficient."),
    ChannelSpec("IBI", ("ibi", "inter beat interval", "interbeat interval"), False, "DreamT derived channel; app can derive from pulse peaks when quality is sufficient."),
    ChannelSpec("EDA", ("eda", "electrodermal activity"), False, "Not part of the target app raw input."),
    ChannelSpec("ECG", ("ecg", "ekg"), False, "DreamT/PSG-only unless future wearable includes ECG."),
    ChannelSpec("EEG", ("eeg", "c3", "c4", "o1", "o2", "f3", "f4"), False, "DreamT/PSG-only; exclude from app-serving model."),
    ChannelSpec("EOG", ("eog", "loc", "roc"), False, "DreamT/PSG-only; exclude from app-serving model."),
    ChannelSpec("EMG", ("emg", "chin"), False, "DreamT/PSG-only; exclude from app-serving model."),
    ChannelSpec("SPO2", ("spo2", "sao2", "oxygen"), False, "Use only if derivable and calibrated from app RED/IR PPG."),
)


FEATURE_SPECS = (
    FeatureSpec("ppg_amplitude_stats", ("IR_PPG",), "app_raw", "Mean/std/median/IQR/min/max from PPG epoch."),
    FeatureSpec("bvp_proxy", ("IR_PPG",), "app_derived", "Filtered PPG waveform proxy for blood-volume pulse."),
    FeatureSpec("ppg_signal_quality", ("IR_PPG",), "app_raw", "Missing, clipping, flatline, and noise proxy ratios."),
    FeatureSpec("heart_rate_stats", ("IR_PPG",), "app_derived", "Peak-derived HR summary when PPG quality is sufficient."),
    FeatureSpec("ibi_time_domain", ("IR_PPG",), "app_derived", "IBI mean/std, SDNN, RMSSD, pNN50 from valid PPG peaks."),
    FeatureSpec("hrv_frequency_domain", ("IR_PPG",), "app_derived", "LF/HF only over a sufficiently long multi-epoch window."),
    FeatureSpec("dual_ppg_ratio", ("IR_PPG", "RED_PPG"), "app_derived", "IR/RED correlation and normalized AC/DC ratio."),
    FeatureSpec("spo2_proxy", ("IR_PPG", "RED_PPG"), "app_derived", "Requires device calibration before use as a core model feature."),
    FeatureSpec("acc_motion_stats", ("ACC_X", "ACC_Y", "ACC_Z"), "app_raw", "Axis and vector-magnitude movement summaries."),
    FeatureSpec("acc_posture_proxy", ("ACC_X", "ACC_Y", "ACC_Z"), "app_derived", "Gravity component and orientation proxy."),
    FeatureSpec("temp_trend", ("TEMP",), "app_raw", "Temperature mean/std/slope and baseline deviation."),
    FeatureSpec("eeg_bandpower", ("EEG",), "dreamt_only", "Useful for PSG upper-bound only; not app-computable."),
    FeatureSpec("eog_rem_features", ("EOG",), "dreamt_only", "Useful for PSG upper-bound only; not app-computable."),
    FeatureSpec("emg_tone_features", ("EMG",), "dreamt_only", "Useful for PSG upper-bound only; not app-computable."),
)


def channel_alias_index() -> dict[str, str]:
    """Return a normalized alias -> canonical channel lookup."""
    index: dict[str, str] = {}
    for spec in CHANNEL_SPECS:
        index[spec.canonical_name.lower()] = spec.canonical_name
        for alias in spec.aliases:
            index[alias.lower()] = spec.canonical_name
    return index


def feature_catalog_dicts() -> list[dict[str, object]]:
    """Return feature catalog as JSON-serializable dictionaries."""
    return [asdict(spec) for spec in FEATURE_SPECS]
