"""Dependency-free evaluation helpers for 5-class and merged 4-class staging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .labels import STAGE4_NAMES, STAGE5_NAMES, merge_many_5_to_4


@dataclass(frozen=True)
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationResult:
    class_names: tuple[str, ...]
    accuracy: float
    macro_f1: float
    cohen_kappa: float
    confusion_matrix: list[list[int]]
    class_wise: dict[str, ClassMetrics]


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> list[list[int]]:
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[true_label][pred_label] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(y_true: Sequence[int], y_pred: Sequence[int], class_names: tuple[str, ...]) -> EvaluationResult:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot evaluate empty predictions")

    num_classes = len(class_names)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    total = len(y_true)
    correct = sum(matrix[idx][idx] for idx in range(num_classes))
    accuracy = correct / total

    class_wise: dict[str, ClassMetrics] = {}
    f1_values: list[float] = []
    for idx, name in enumerate(class_names):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(num_classes) if row != idx)
        fn = sum(matrix[idx][col] for col in range(num_classes) if col != idx)
        support = sum(matrix[idx])
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)
        f1_values.append(f1)
        class_wise[name] = ClassMetrics(precision=precision, recall=recall, f1=f1, support=support)

    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[row][col] for row in range(num_classes)) for col in range(num_classes)]
    expected_accuracy = sum(row_totals[idx] * col_totals[idx] for idx in range(num_classes)) / (total * total)
    cohen_kappa = _safe_divide(accuracy - expected_accuracy, 1.0 - expected_accuracy)

    return EvaluationResult(
        class_names=class_names,
        accuracy=accuracy,
        macro_f1=sum(f1_values) / num_classes,
        cohen_kappa=cohen_kappa,
        confusion_matrix=matrix,
        class_wise=class_wise,
    )


def evaluate_5_and_4(y_true_5: Sequence[int], y_pred_5: Sequence[int]) -> dict[str, EvaluationResult]:
    """Evaluate original 5-class labels and merged 4-class labels."""
    return {
        "5_class": evaluate(y_true_5, y_pred_5, STAGE5_NAMES),
        "4_class": evaluate(merge_many_5_to_4(y_true_5), merge_many_5_to_4(y_pred_5), STAGE4_NAMES),
    }

