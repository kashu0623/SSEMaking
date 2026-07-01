"""Smart alarm decision policy over 5-class sleep-stage probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .labels import STAGE5_TO_ID


class AlarmDecision(str, Enum):
    TRIGGER = "trigger"
    WAIT = "wait"
    AVOID_DEEP = "avoid_deep"


@dataclass(frozen=True)
class AlarmPolicyConfig:
    smoothing_epochs: int = 3
    wake_threshold: float = 0.65
    deep_avoid_threshold: float = 0.45
    light_threshold: float = 0.45
    n2_crossing_weight: float = 0.05
    min_n1_gradient: float = 0.0
    max_n2_gradient: float = 0.0


@dataclass(frozen=True)
class AlarmPolicyResult:
    decision: AlarmDecision
    reason: str
    smoothed_probabilities: tuple[float, float, float, float, float]
    n1_gradient: float
    n2_gradient: float


def _mean_probabilities(probability_history: Sequence[Sequence[float]], window: int) -> tuple[float, float, float, float, float]:
    selected = probability_history[-window:]
    if not selected:
        raise ValueError("probability_history must not be empty")
    for probabilities in selected:
        if len(probabilities) != 5:
            raise ValueError("Each probability vector must have 5 values: W, N1, N2, N3, REM")
    return tuple(sum(epoch[class_idx] for epoch in selected) / len(selected) for class_idx in range(5))  # type: ignore[return-value]


def _gradient(probability_history: Sequence[Sequence[float]], class_id: int, window: int) -> float:
    selected = probability_history[-window:]
    if len(selected) < 2:
        return 0.0
    return selected[-1][class_id] - selected[0][class_id]


def decide_alarm(
    probability_history: Sequence[Sequence[float]],
    config: AlarmPolicyConfig | None = None,
) -> AlarmPolicyResult:
    """Decide whether a smart alarm should trigger from recent 5-class probabilities.

    Probability order must be Wake, N1, N2, N3, REM.
    """
    cfg = config or AlarmPolicyConfig()
    if cfg.smoothing_epochs <= 0:
        raise ValueError("smoothing_epochs must be positive")

    smoothed = _mean_probabilities(probability_history, cfg.smoothing_epochs)
    wake = smoothed[STAGE5_TO_ID["Wake"]]
    n1 = smoothed[STAGE5_TO_ID["N1"]]
    n2 = smoothed[STAGE5_TO_ID["N2"]]
    n3 = smoothed[STAGE5_TO_ID["N3"]]
    light = n1 + n2
    n1_gradient = _gradient(probability_history, STAGE5_TO_ID["N1"], cfg.smoothing_epochs)
    n2_gradient = _gradient(probability_history, STAGE5_TO_ID["N2"], cfg.smoothing_epochs)

    if wake >= cfg.wake_threshold:
        return AlarmPolicyResult(
            decision=AlarmDecision.TRIGGER,
            reason="wake_probability_threshold",
            smoothed_probabilities=smoothed,
            n1_gradient=n1_gradient,
            n2_gradient=n2_gradient,
        )

    if n3 >= cfg.deep_avoid_threshold:
        return AlarmPolicyResult(
            decision=AlarmDecision.AVOID_DEEP,
            reason="deep_sleep_avoidance",
            smoothed_probabilities=smoothed,
            n1_gradient=n1_gradient,
            n2_gradient=n2_gradient,
        )

    n1_crossed_n2 = n1 > n2 + cfg.n2_crossing_weight
    light_sleep = light >= cfg.light_threshold
    getting_lighter = n1_gradient >= cfg.min_n1_gradient and n2_gradient <= cfg.max_n2_gradient

    if light_sleep and n1_crossed_n2 and getting_lighter:
        return AlarmPolicyResult(
            decision=AlarmDecision.TRIGGER,
            reason="n1_n2_light_sleep_crossing",
            smoothed_probabilities=smoothed,
            n1_gradient=n1_gradient,
            n2_gradient=n2_gradient,
        )

    return AlarmPolicyResult(
        decision=AlarmDecision.WAIT,
        reason="no_trigger_condition",
        smoothed_probabilities=smoothed,
        n1_gradient=n1_gradient,
        n2_gradient=n2_gradient,
    )

