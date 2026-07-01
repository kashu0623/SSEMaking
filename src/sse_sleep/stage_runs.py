"""Convert repeated row-level Sleep_Stage values into 30-second epoch labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .dreamt_100hz import ROWS_PER_EPOCH
from .labels import map_label_5


@dataclass(frozen=True)
class StageRun:
    raw_label: str
    start_row: int
    end_row: int

    @property
    def length_rows(self) -> int:
        return self.end_row - self.start_row


@dataclass(frozen=True)
class LabeledEpoch:
    raw_label: str
    canonical_label: str
    class_id_5: int
    start_row: int
    end_row: int


def runs_from_values(values: Iterable[str]) -> list[StageRun]:
    """Build run-length segments from a sequence of raw stage values."""
    runs: list[StageRun] = []
    current_label: str | None = None
    current_start = 0

    for row_index, value in enumerate(values):
        if current_label is None:
            current_label = value
            current_start = row_index
            continue
        if value != current_label:
            runs.append(StageRun(raw_label=current_label, start_row=current_start, end_row=row_index))
            current_label = value
            current_start = row_index

    if current_label is not None:
        runs.append(StageRun(raw_label=current_label, start_row=current_start, end_row=row_index + 1))
    return runs


def runs_to_labeled_epochs(
    runs: Iterable[StageRun],
    rows_per_epoch: int = ROWS_PER_EPOCH,
    drop_partial_runs: bool = True,
) -> list[LabeledEpoch]:
    """Convert stage runs to full 30-second labeled epochs.

    Ignored labels such as P are skipped. A run longer than one epoch is split
    into repeated full epochs with the same label.
    """
    if rows_per_epoch <= 0:
        raise ValueError("rows_per_epoch must be positive")

    epochs: list[LabeledEpoch] = []
    for run in runs:
        mapped = map_label_5(run.raw_label)
        if mapped.ignored or mapped.canonical is None or mapped.class_id_5 is None:
            continue
        full_epoch_count = run.length_rows // rows_per_epoch
        remainder = run.length_rows % rows_per_epoch
        if remainder and not drop_partial_runs:
            full_epoch_count += 1
        for epoch_offset in range(full_epoch_count):
            start_row = run.start_row + epoch_offset * rows_per_epoch
            end_row = min(start_row + rows_per_epoch, run.end_row)
            if drop_partial_runs and end_row - start_row != rows_per_epoch:
                continue
            epochs.append(
                LabeledEpoch(
                    raw_label=run.raw_label,
                    canonical_label=mapped.canonical,
                    class_id_5=mapped.class_id_5,
                    start_row=start_row,
                    end_row=end_row,
                )
            )
    return epochs

