"""Utilities for building fixed 30-second sleep-stage epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class EpochWindow:
    index: int
    start_seconds: float
    end_seconds: float


def make_epoch_windows(
    start_seconds: float,
    end_seconds: float,
    epoch_seconds: float = 30.0,
) -> list[EpochWindow]:
    """Create half-open epoch windows [start, end)."""
    if epoch_seconds <= 0:
        raise ValueError("epoch_seconds must be positive")
    windows: list[EpochWindow] = []
    current = start_seconds
    index = 0
    while current + epoch_seconds <= end_seconds:
        windows.append(EpochWindow(index=index, start_seconds=current, end_seconds=current + epoch_seconds))
        current += epoch_seconds
        index += 1
    return windows


def slice_by_time(
    times_seconds: Sequence[float],
    values: Sequence[T],
    start_seconds: float,
    end_seconds: float,
) -> list[T]:
    """Return values with timestamps inside [start_seconds, end_seconds)."""
    if len(times_seconds) != len(values):
        raise ValueError("times_seconds and values must have the same length")
    return [
        value
        for time_seconds, value in zip(times_seconds, values, strict=True)
        if start_seconds <= time_seconds < end_seconds
    ]

