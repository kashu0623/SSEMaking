"""Canonical sleep-stage label mapping for 5-class training and 4-class evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


STAGE5_NAMES = ("Wake", "N1", "N2", "N3", "REM")
STAGE4_NAMES = ("Wake", "Light", "Deep", "REM")

STAGE5_TO_ID = {name: idx for idx, name in enumerate(STAGE5_NAMES)}
STAGE4_TO_ID = {name: idx for idx, name in enumerate(STAGE4_NAMES)}
IGNORE_LABELS = {"", "?", "unknown", "artifact", "movement", "mt", "p"}

_ALIASES = {
    "Wake": {"w", "wake", "awake", "stage w", "0"},
    "N1": {"n1", "s1", "stage 1", "stage n1", "1"},
    "N2": {"n2", "s2", "stage 2", "stage n2", "2"},
    "N3": {"n3", "n4", "s3", "s4", "stage 3", "stage 4", "stage n3", "deep", "3", "4"},
    "REM": {"r", "rem", "stage r", "stage rem", "5"},
}

_STAGE5_TO_STAGE4 = {
    STAGE5_TO_ID["Wake"]: STAGE4_TO_ID["Wake"],
    STAGE5_TO_ID["N1"]: STAGE4_TO_ID["Light"],
    STAGE5_TO_ID["N2"]: STAGE4_TO_ID["Light"],
    STAGE5_TO_ID["N3"]: STAGE4_TO_ID["Deep"],
    STAGE5_TO_ID["REM"]: STAGE4_TO_ID["REM"],
}


@dataclass(frozen=True)
class LabelMappingResult:
    raw_label: object
    canonical: str | None
    class_id_5: int | None
    ignored: bool


def normalize_label(raw_label: object) -> str:
    """Normalize a raw label value for alias matching."""
    text = str(raw_label).strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    return " ".join(text.split())


def canonical_stage(raw_label: object) -> str | None:
    """Map a raw DreamT/PSG label to Wake/N1/N2/N3/REM, or None if ignored."""
    normalized = normalize_label(raw_label)
    if normalized in IGNORE_LABELS:
        return None
    for stage, aliases in _ALIASES.items():
        if normalized in aliases:
            return stage
    return None


def map_label_5(raw_label: object) -> LabelMappingResult:
    """Return canonical 5-class mapping metadata for one raw label."""
    stage = canonical_stage(raw_label)
    if stage is None:
        return LabelMappingResult(raw_label=raw_label, canonical=None, class_id_5=None, ignored=True)
    return LabelMappingResult(
        raw_label=raw_label,
        canonical=stage,
        class_id_5=STAGE5_TO_ID[stage],
        ignored=False,
    )


def merge_5_to_4(class_id_5: int) -> int:
    """Merge 5-class id into 4-class id: Wake, Light, Deep, REM."""
    return _STAGE5_TO_STAGE4[class_id_5]


def merge_many_5_to_4(labels_5: Iterable[int]) -> list[int]:
    """Merge an iterable of 5-class ids into 4-class ids."""
    return [merge_5_to_4(label) for label in labels_5]
