"""Stack aligned Light-vs-Deep specialists with OOF logistic regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_prediction_fusion import load_split


def validate(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    label: str,
) -> None:
    if not np.array_equal(reference["y_true"], candidate["y_true"]):
        raise ValueError(f"Unaligned labels for {label}")
    if reference["probs"].shape != candidate["probs"].shape:
        raise ValueError(f"Unaligned probabilities for {label}")
    for key in ("subject_ids", "epoch_indices"):
        if (
            key in reference
            and key in candidate
            and not np.array_equal(reference[key], candidate[key])
        ):
            raise ValueError(f"Unaligned {key} for {label}")


def load_members(
    paths: Sequence[Path],
    split: str,
) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]]]:
    if not paths:
        raise ValueError("At least one specialist prediction is required")
    members = [load_split(path, split) for path in paths]
    reference = members[0]
    for index, member in enumerate(members, start=1):
        if member["probs"].ndim != 2 or member["probs"].shape[1] != 2:
            raise ValueError(
                f"Expected two specialist probabilities for member {index}, "
                f"got {member['probs'].shape}"
            )
        if index > 1:
            validate(reference, member, f"{split} member {index}")
    return reference, members


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def build_features(members: Sequence[dict[str, np.ndarray]]) -> np.ndarray:
    deep_probabilities = np.column_stack(
        [member["probs"][:, 1] for member in members]
    ).astype(np.float32)
    logits = safe_logit(deep_probabilities).astype(np.float32)
    summaries = np.column_stack(
        (
            deep_probabilities.mean(axis=1),
            deep_probabilities.std(axis=1),
            deep_probabilities.min(axis=1),
            deep_probabilities.max(axis=1),
        )
    ).astype(np.float32)
    return np.concatenate((logits, deep_probabilities, summaries), axis=1)


class LogisticStacker:
    def __init__(
        self,
        c_value: float,
        class_weight: str | None,
        max_iter: int = 100,
        tolerance: float = 1e-7,
    ) -> None:
        self.c_value = c_value
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tolerance = tolerance

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _objective(
        design: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        coefficients: np.ndarray,
        regularization: float,
    ) -> float:
        logits = design @ coefficients
        probabilities = LogisticStacker._sigmoid(logits)
        epsilon = 1e-12
        loss = -np.sum(
            sample_weights
            * (
                labels * np.log(np.clip(probabilities, epsilon, 1.0))
                + (1.0 - labels)
                * np.log(np.clip(1.0 - probabilities, epsilon, 1.0))
            )
        )
        return float(
            loss
            + 0.5 * regularization * np.dot(coefficients[1:], coefficients[1:])
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticStacker":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
            raise ValueError("Invalid feature or label shape")
        if np.unique(y).size != 2:
            raise ValueError("Logistic stacker requires both classes")

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-8] = 1.0
        standardized = (x - self.mean_) / self.scale_
        design = np.column_stack((np.ones(x.shape[0]), standardized))

        sample_weights = np.ones(y.shape[0], dtype=np.float64)
        if self.class_weight == "balanced":
            counts = np.bincount(y.astype(np.int64), minlength=2)
            sample_weights = np.where(
                y == 1.0,
                y.size / (2.0 * max(int(counts[1]), 1)),
                y.size / (2.0 * max(int(counts[0]), 1)),
            )
        elif self.class_weight is not None:
            raise ValueError(f"Unknown class weight: {self.class_weight}")

        regularization = 1.0 / self.c_value
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        weighted_positive_rate = np.average(y, weights=sample_weights)
        weighted_positive_rate = float(
            np.clip(weighted_positive_rate, 1e-6, 1.0 - 1e-6)
        )
        coefficients[0] = np.log(
            weighted_positive_rate / (1.0 - weighted_positive_rate)
        )
        penalty_diagonal = np.ones(design.shape[1], dtype=np.float64)
        penalty_diagonal[0] = 0.0

        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            probabilities = self._sigmoid(design @ coefficients)
            gradient = design.T @ (sample_weights * (probabilities - y))
            gradient += regularization * penalty_diagonal * coefficients
            curvature = sample_weights * probabilities * (1.0 - probabilities)
            hessian = design.T @ (design * curvature[:, None])
            hessian += np.diag(regularization * penalty_diagonal + 1e-8)
            step = np.linalg.solve(hessian, gradient)
            if np.max(np.abs(step)) < self.tolerance:
                break

            current_objective = self._objective(
                design,
                y,
                sample_weights,
                coefficients,
                regularization,
            )
            step_scale = 1.0
            while step_scale >= 1e-6:
                candidate = coefficients - step_scale * step
                if self._objective(
                    design,
                    y,
                    sample_weights,
                    candidate,
                    regularization,
                ) <= current_objective:
                    coefficients = candidate
                    break
                step_scale *= 0.5
            else:
                break

        self.intercept_ = np.asarray((coefficients[0],), dtype=np.float64)
        self.coef_ = coefficients[1:].reshape(1, -1)
        self.iterations_ = iterations
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        standardized = (x - self.mean_) / self.scale_
        deep = self._sigmoid(self.intercept_[0] + standardized @ self.coef_[0])
        return np.column_stack((1.0 - deep, deep))


def make_model(c_value: float, class_weight: str | None) -> LogisticStacker:
    return LogisticStacker(c_value, class_weight)


def group_splits(
    groups: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    split_count = min(folds, unique_groups.size)
    if split_count < 2:
        raise ValueError("Need at least two validation subjects for OOF stacking")
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique_groups.size)
    order = order[np.argsort(-counts[order], kind="stable")]
    fold_groups: list[list[int]] = [[] for _ in range(split_count)]
    fold_sizes = np.zeros(split_count, dtype=np.int64)
    for group_index in order:
        fold_index = int(np.argmin(fold_sizes))
        fold_groups[fold_index].append(int(group_index))
        fold_sizes[fold_index] += counts[group_index]
    result = []
    for held_group_indices in fold_groups:
        holdout = np.isin(inverse, held_group_indices)
        result.append((np.flatnonzero(~holdout), np.flatnonzero(holdout)))
    return result


def stratified_splits(
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    class_counts = np.bincount(labels, minlength=2)
    split_count = min(folds, int(class_counts.min()))
    if split_count < 2:
        raise ValueError("Need at least two examples per class for OOF stacking")
    rng = np.random.default_rng(seed)
    holdouts: list[list[np.ndarray]] = [[] for _ in range(split_count)]
    for class_id in (0, 1):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        for fold_index, part in enumerate(np.array_split(indices, split_count)):
            holdouts[fold_index].append(part)
    all_indices = np.arange(labels.size)
    result = []
    for parts in holdouts:
        holdout = np.sort(np.concatenate(parts))
        result.append((np.setdiff1d(all_indices, holdout), holdout))
    return result


def binary_labels(labels_4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (labels_4 == 1) | (labels_4 == 2)
    return mask, (labels_4[mask] == 2).astype(np.int64)


def oof_val_probabilities(
    features: np.ndarray,
    labels_4: np.ndarray,
    subject_ids: np.ndarray | None,
    c_value: float,
    class_weight: str | None,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    light_deep_mask, y_binary = binary_labels(labels_4)
    light_deep_indices = np.flatnonzero(light_deep_mask)
    if np.unique(y_binary).size != 2:
        raise ValueError("Validation Light/Deep subset must contain both classes")

    probabilities = np.full(labels_4.shape[0], np.nan, dtype=np.float32)
    fold_reports: list[dict[str, Any]] = []
    if subject_ids is not None:
        groups = np.asarray(subject_ids)[light_deep_mask]
        split_iterator = group_splits(groups, folds, seed)
        split_count = len(split_iterator)
        for fold_index, (train_local, holdout_local) in enumerate(
            split_iterator,
            start=1,
        ):
            train_indices = light_deep_indices[train_local]
            holdout_groups = np.unique(groups[holdout_local])
            holdout_indices = np.flatnonzero(
                np.isin(np.asarray(subject_ids), holdout_groups)
            )
            model = make_model(c_value, class_weight)
            model.fit(features[train_indices], (labels_4[train_indices] == 2))
            probabilities[holdout_indices] = model.predict_proba(
                features[holdout_indices]
            )[:, 1]
            fold_reports.append(
                {
                    "fold": fold_index,
                    "train_light_deep_rows": int(train_indices.size),
                    "holdout_rows": int(holdout_indices.size),
                    "holdout_subjects": int(holdout_groups.size),
                }
            )
    else:
        split_iterator = stratified_splits(y_binary, folds, seed)
        split_count = len(split_iterator)
        for fold_index, (train_local, holdout_local) in enumerate(
            split_iterator,
            start=1,
        ):
            train_indices = light_deep_indices[train_local]
            holdout_indices = light_deep_indices[holdout_local]
            model = make_model(c_value, class_weight)
            model.fit(features[train_indices], (labels_4[train_indices] == 2))
            probabilities[holdout_indices] = model.predict_proba(
                features[holdout_indices]
            )[:, 1]
            fold_reports.append(
                {
                    "fold": fold_index,
                    "train_light_deep_rows": int(train_indices.size),
                    "holdout_rows": int(holdout_indices.size),
                }
            )

    full_model = make_model(c_value, class_weight)
    full_model.fit(features[light_deep_mask], y_binary)
    missing = np.isnan(probabilities)
    if missing.any():
        probabilities[missing] = full_model.predict_proba(features[missing])[:, 1]
    return probabilities, {
        "splitter": "group" if subject_ids is not None else "stratified",
        "fold_count": split_count,
        "folds": fold_reports,
    }


def stack_predictions(
    paths: Sequence[Path],
    out_path: Path,
    c_value: float,
    class_weight: str | None,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    val_reference, val_members = load_members(paths, "val")
    test_reference, test_members = load_members(paths, "test")
    val_features = build_features(val_members)
    test_features = build_features(test_members)
    val_mask, val_binary = binary_labels(val_reference["y_true"])

    val_deep, oof_report = oof_val_probabilities(
        val_features,
        val_reference["y_true"],
        val_reference.get("subject_ids"),
        c_value,
        class_weight,
        folds,
        seed,
    )
    final_model = make_model(c_value, class_weight)
    final_model.fit(val_features[val_mask], val_binary)
    test_deep = final_model.predict_proba(test_features)[:, 1].astype(np.float32)

    arrays: dict[str, np.ndarray] = {
        "specialist_names": np.asarray(("Light", "Deep")),
    }
    for split, reference, deep in (
        ("val", val_reference, val_deep),
        ("test", test_reference, test_deep),
    ):
        probabilities = np.column_stack((1.0 - deep, deep)).astype(np.float32)
        arrays[f"{split}_y_true"] = reference["y_true"]
        arrays[f"{split}_probs"] = probabilities
        arrays[f"{split}_y_pred"] = probabilities.argmax(axis=1).astype(np.int64)
        for key in ("subject_ids", "epoch_indices"):
            if key in reference:
                arrays[f"{split}_{key}"] = reference[key]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return {
        "members": [str(path) for path in paths],
        "member_count": len(paths),
        "out_path": str(out_path),
        "c": c_value,
        "class_weight": class_weight,
        "feature_count": int(val_features.shape[1]),
        "val_rows": int(val_features.shape[0]),
        "test_rows": int(test_features.shape[0]),
        "oof": oof_report,
        "final_intercept": final_model.intercept_.tolist(),
        "final_coefficients": final_model.coef_.tolist(),
        "final_iterations": final_model.iterations_,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OOF-stack aligned Light-vs-Deep specialist predictions."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument(
        "--class-weight",
        choices=("none", "balanced"),
        default="none",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.c <= 0.0:
        raise ValueError("C must be positive")
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    class_weight = None if args.class_weight == "none" else args.class_weight
    report = stack_predictions(
        args.predictions,
        args.out,
        args.c,
        class_weight,
        args.folds,
        args.seed,
    )
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "members"},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
