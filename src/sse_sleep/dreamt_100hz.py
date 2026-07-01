"""Constants for the confirmed DreamT data_100Hz CSV layout."""

from __future__ import annotations

import re
from pathlib import Path


SAMPLING_RATE_HZ = 100
EPOCH_SECONDS = 30
ROWS_PER_EPOCH = SAMPLING_RATE_HZ * EPOCH_SECONDS

FILENAME_RE = re.compile(r"^(S[0-9]+)_PSG_df_updated\.csv$")

TIMESTAMP_COLUMN = "TIMESTAMP"
LABEL_COLUMN = "Sleep_Stage"

APP_RAW_COLUMNS = ("ACC_X", "ACC_Y", "ACC_Z", "TEMP")
APP_PROXY_DERIVED_COLUMNS = ("BVP", "HR", "IBI")
OPTIONAL_APP_DERIVED_COLUMNS = ("SAO2",)

DREAMT_ONLY_COLUMNS = (
    "C4-M1",
    "F4-M1",
    "O2-M1",
    "Fp1-O2",
    "T3 - CZ",
    "CZ - T4",
    "CHIN",
    "E1",
    "E2",
    "ECG",
    "LAT",
    "RAT",
    "SNORE",
    "PTAF",
    "FLOW",
    "THORAX",
    "ABDOMEN",
    "EDA",
)

EVENT_COLUMNS = ("Obstructive_Apnea", "Central_Apnea", "Hypopnea", "Multiple_Events")

DEFAULT_MODEL_INPUT_COLUMNS = APP_PROXY_DERIVED_COLUMNS + APP_RAW_COLUMNS


def subject_id_from_path(path: str | Path) -> str:
    """Extract subject id such as S002 from a DreamT 100Hz CSV path."""
    name = Path(path).name
    match = FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected DreamT 100Hz filename: {name}")
    return match.group(1)


def rows_to_epoch_index(row_index_zero_based: int) -> int:
    """Map a 100Hz sample row index to a 30-second epoch index."""
    if row_index_zero_based < 0:
        raise ValueError("row_index_zero_based must be non-negative")
    return row_index_zero_based // ROWS_PER_EPOCH

