"""Train validation-only causal gates over the current four-model prediction pool."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate_four_model_fusion import build_grouped_class_weights, four_model_classwise_fusion
from .evaluate_prediction_fusion import json_ready, load_split, validate_alignment
from .labels import STAGE4_NAMES, STAGE5_NAMES
from .metrics import evaluate_5_and_4


MODEL_ROLES = ("original_temporal", "full_w20", "capacity_h128", "h128_ls003")
GATE_VARIANTS = ("static", "causal")


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    metrics = json_ready(evaluate_5_and_4(y_true.tolist(), y_pred.tolist()))
    class_wise = metrics["4_class"]["class_wise"]
    return {
        "metrics": metrics,
        "summary": {
            "4_macro_f1": float(metrics["4_class"]["macro_f1"]),
            "4_kappa": float(metrics["4_class"]["cohen_kappa"]),
            "4_macro_f1_plus_4_kappa": float(
                metrics["4_class"]["macro_f1"] + metrics["4_class"]["cohen_kappa"]
            ),
            "wake_f1": float(class_wise["Wake"]["f1"]),
            "light_f1": float(class_wise["Light"]["f1"]),
            "deep_f1": float(class_wise["Deep"]["f1"]),
            "rem_f1": float(class_wise["REM"]["f1"]),
        },
    }


def score(summary: dict[str, float]) -> float:
    return float(summary["4_macro_f1_plus_4_kappa"])


def entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1, keepdims=True)


def margin(probabilities: np.ndarray) -> np.ndarray:
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    return (top_two[:, 1] - top_two[:, 0]).reshape(-1, 1)


def contiguous_groups(subject_ids: np.ndarray, epoch_indices: np.ndarray) -> list[np.ndarray]:
    groups: list[np.ndarray] = []
    for subject_id in sorted(set(subject_ids.tolist())):
        subject_rows = np.flatnonzero(subject_ids == subject_id)
        ordered = subject_rows[np.argsort(epoch_indices[subject_rows], kind="stable")]
        start = 0
        for position in range(1, ordered.shape[0] + 1):
            is_end = position == ordered.shape[0]
            has_gap = not is_end and epoch_indices[ordered[position]] != epoch_indices[ordered[position - 1]] + 1
            if is_end or has_gap:
                groups.append(ordered[start:position])
                start = position
    return groups


def causal_history_features(
    fused_probs: np.ndarray,
    subject_ids: np.ndarray,
    epoch_indices: np.ndarray,
    windows: Sequence[int],
) -> np.ndarray:
    class_count = fused_probs.shape[1]
    history = np.zeros((fused_probs.shape[0], len(windows) * class_count + class_count + 1), dtype=np.float32)
    for group in contiguous_groups(subject_ids, epoch_indices):
        for position, row_index in enumerate(group):
            offset = 0
            for window in windows:
                previous = group[max(0, position - window) : position]
                if previous.size:
                    history[row_index, offset : offset + class_count] = fused_probs[previous].mean(axis=0)
                offset += class_count
            if position:
                previous_stage = int(fused_probs[group[position - 1]].argmax())
                history[row_index, offset + previous_stage] = 1.0
                run_length = 1
                scan = position - 2
                while scan >= 0 and int(fused_probs[group[scan]].argmax()) == previous_stage:
                    run_length += 1
                    scan -= 1
                history[row_index, -1] = min(run_length, 20) / 20.0
    return history


def build_features(
    model_probs: dict[str, np.ndarray],
    fused_probs: np.ndarray,
    subject_ids: np.ndarray,
    epoch_indices: np.ndarray,
    variant: str,
    history_windows: Sequence[int],
) -> np.ndarray:
    blocks = [model_probs[role] for role in MODEL_ROLES]
    blocks.append(fused_probs)
    blocks.extend(entropy(model_probs[role]) for role in MODEL_ROLES)
    blocks.extend(margin(model_probs[role]) for role in MODEL_ROLES)
    vote_counts = np.zeros_like(fused_probs)
    for role in MODEL_ROLES:
        vote_counts[np.arange(fused_probs.shape[0]), model_probs[role].argmax(axis=1)] += 1.0
    blocks.append(vote_counts / len(MODEL_ROLES))
    static_features = np.concatenate(blocks, axis=1).astype(np.float32)
    if variant == "static":
        return static_features
    if variant == "causal":
        return np.concatenate(
            [
                static_features,
                causal_history_features(fused_probs, subject_ids, epoch_indices, history_windows),
            ],
            axis=1,
        ).astype(np.float32)
    raise ValueError(f"Unknown gate variant: {variant}")


def inner_subject_masks(subject_ids: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    subjects = sorted(set(subject_ids.tolist()))
    if len(subjects) < 2:
        raise ValueError("Causal gate needs at least two validation subjects for internal selection")
    rng = np.random.default_rng(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    holdout_count = min(len(subjects) - 1, max(1, round(len(subjects) * fraction)))
    holdout_subjects = set(shuffled[:holdout_count])
    holdout_mask = np.asarray([subject in holdout_subjects for subject in subject_ids], dtype=bool)
    return ~holdout_mask, holdout_mask


def fit_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c_value: float,
    class_weight: str | None,
) -> Any:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=0,
        ),
    ).fit(x_train, y_train)


def select_gate_config(
    x_val: np.ndarray,
    y_val: np.ndarray,
    subject_ids: np.ndarray,
    inner_fraction: float,
    seed: int,
    c_values: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    train_mask, holdout_mask = inner_subject_masks(subject_ids, fraction=inner_fraction, seed=seed)
    candidates: list[dict[str, Any]] = []
    for class_weight in (None, "balanced"):
        for c_value in c_values:
            model = fit_gate(x_val[train_mask], y_val[train_mask], c_value=c_value, class_weight=class_weight)
            holdout_prediction = model.predict(x_val[holdout_mask]).astype(np.int64)
            holdout = metric_summary(y_val[holdout_mask], holdout_prediction)
            candidates.append(
                {
                    "c": c_value,
                    "class_weight": class_weight,
                    "inner_holdout_score": score(holdout["summary"]),
                    "inner_holdout_summary": holdout["summary"],
                }
            )
    best = max(candidates, key=lambda item: item["inner_holdout_score"])
    return best, {
        "subject_count": len(set(subject_ids.tolist())),
        "train_samples": int(train_mask.sum()),
        "holdout_samples": int(holdout_mask.sum()),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def load_aligned_splits(prediction_paths: dict[str, Path], split: str) -> dict[str, dict[str, np.ndarray]]:
    loaded = {role: load_split(path, split) for role, path in prediction_paths.items()}
    base = loaded[MODEL_ROLES[0]]
    for role in MODEL_ROLES[1:]:
        validate_alignment(base, loaded[role], split)
    missing = [key for key in ("subject_ids", "epoch_indices") if key not in base]
    if missing:
        raise ValueError(f"{split} predictions need metadata for causal gate: {', '.join(missing)}")
    return loaded


def fusion_probs(
    loaded: dict[str, dict[str, np.ndarray]],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> np.ndarray:
    return four_model_classwise_fusion(
        loaded["original_temporal"]["probs"],
        loaded["full_w20"]["probs"],
        loaded["capacity_h128"]["probs"],
        loaded["h128_ls003"]["probs"],
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )


def evaluate_seed(
    seed_label: str,
    prediction_paths: dict[str, Path],
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
    inner_fraction: float,
    history_windows: Sequence[int],
    c_values: Sequence[float],
) -> dict[str, Any]:
    val = load_aligned_splits(prediction_paths, "val")
    test = load_aligned_splits(prediction_paths, "test")
    val_fused = fusion_probs(val, primary_alphas, secondary_alphas, tertiary_alphas)
    test_fused = fusion_probs(test, primary_alphas, secondary_alphas, tertiary_alphas)
    y_val = val[MODEL_ROLES[0]]["y_true"]
    y_test = test[MODEL_ROLES[0]]["y_true"]
    val_base_prediction = val_fused.argmax(axis=1).astype(np.int64)
    test_base_prediction = test_fused.argmax(axis=1).astype(np.int64)
    variants: dict[str, Any] = {
        "current_best_fusion": {
            "val": metric_summary(y_val, val_base_prediction),
            "test": metric_summary(y_test, test_base_prediction),
        }
    }

    for variant_index, variant in enumerate(GATE_VARIANTS):
        x_val = build_features(
            {role: val[role]["probs"] for role in MODEL_ROLES},
            val_fused,
            val[MODEL_ROLES[0]]["subject_ids"],
            val[MODEL_ROLES[0]]["epoch_indices"],
            variant,
            history_windows,
        )
        x_test = build_features(
            {role: test[role]["probs"] for role in MODEL_ROLES},
            test_fused,
            test[MODEL_ROLES[0]]["subject_ids"],
            test[MODEL_ROLES[0]]["epoch_indices"],
            variant,
            history_windows,
        )
        best_config, selection = select_gate_config(
            x_val,
            y_val,
            val[MODEL_ROLES[0]]["subject_ids"],
            inner_fraction=inner_fraction,
            seed=sum(ord(character) for character in seed_label) + variant_index,
            c_values=c_values,
        )
        gate = fit_gate(
            x_val,
            y_val,
            c_value=float(best_config["c"]),
            class_weight=best_config["class_weight"],
        )
        variants[f"gate_{variant}"] = {
            "selected_config": best_config,
            "selection": selection,
            "val": metric_summary(y_val, gate.predict(x_val).astype(np.int64)),
            "test": metric_summary(y_test, gate.predict(x_test).astype(np.int64)),
            "feature_count": int(x_val.shape[1]),
        }
    return {"seed": seed_label, "variants": variants}


def mean_std(values: Sequence[float]) -> dict[str, float]:
    return {"mean": mean(float(value) for value in values), "std": pstdev(float(value) for value in values)}


def summarize(seed_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    variant_names = seed_reports[0]["variants"].keys()
    fields = ("4_macro_f1", "4_kappa", "4_macro_f1_plus_4_kappa", "wake_f1", "light_f1", "deep_f1", "rem_f1")
    for variant in variant_names:
        result[variant] = {
            split: {
                field: mean_std([report["variants"][variant][split]["summary"][field] for report in seed_reports])
                for field in fields
            }
            for split in ("val", "test")
        }
    return result


def print_summary(report: dict[str, Any]) -> None:
    print("| candidate | test 4M | test 4K | test 4M+4K | Wake | Light | Deep | REM |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, candidate in report["summary"].items():
        test = candidate["test"]
        print(
            f"| {name} | {test['4_macro_f1']['mean']:.4f} | {test['4_kappa']['mean']:.4f} | "
            f"{test['4_macro_f1_plus_4_kappa']['mean']:.4f} | {test['wake_f1']['mean']:.4f} | "
            f"{test['light_f1']['mean']:.4f} | {test['deep_f1']['mean']:.4f} | {test['rem_f1']['mean']:.4f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation-trained causal gates for four-model fusion.")
    parser.add_argument("--base-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--primary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--secondary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--tertiary-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--inner-val-subject-fraction", type=float, default=0.25)
    parser.add_argument("--history-windows", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--c-values", type=float, nargs="+", default=(0.02, 0.1, 0.5, 1.0))
    parser.add_argument("--wake-primary", type=float, default=0.72)
    parser.add_argument("--wake-secondary", type=float, default=0.06)
    parser.add_argument("--wake-tertiary", type=float, default=0.00)
    parser.add_argument("--light-primary", type=float, default=0.80)
    parser.add_argument("--light-secondary", type=float, default=0.02)
    parser.add_argument("--light-tertiary", type=float, default=0.15)
    parser.add_argument("--deep-primary", type=float, default=0.82)
    parser.add_argument("--deep-secondary", type=float, default=0.00)
    parser.add_argument("--deep-tertiary", type=float, default=0.18)
    parser.add_argument("--rem-primary", type=float, default=0.00)
    parser.add_argument("--rem-secondary", type=float, default=0.42)
    parser.add_argument("--rem-tertiary", type=float, default=0.13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path_lists = (args.base_predictions, args.primary_predictions, args.secondary_predictions, args.tertiary_predictions)
    count = len(args.base_predictions)
    if any(len(paths) != count for paths in path_lists):
        raise ValueError("All prediction path lists must have the same length")
    labels = args.seed_labels or [str(index + 1) for index in range(count)]
    if len(labels) != count:
        raise ValueError("--seed-labels must match the number of prediction sets")
    if not 0.0 < args.inner_val_subject_fraction < 0.5:
        raise ValueError("--inner-val-subject-fraction must be between 0 and 0.5")

    primary_alphas, secondary_alphas, tertiary_alphas = build_grouped_class_weights(
        args.wake_primary, args.wake_secondary, args.wake_tertiary,
        args.light_primary, args.light_secondary, args.light_tertiary,
        args.deep_primary, args.deep_secondary, args.deep_tertiary,
        args.rem_primary, args.rem_secondary, args.rem_tertiary,
    )
    prediction_sets = [
        {
            "original_temporal": args.base_predictions[index],
            "full_w20": args.primary_predictions[index],
            "capacity_h128": args.secondary_predictions[index],
            "h128_ls003": args.tertiary_predictions[index],
        }
        for index in range(count)
    ]
    seed_reports = [
        evaluate_seed(
            seed_label,
            paths,
            primary_alphas,
            secondary_alphas,
            tertiary_alphas,
            args.inner_val_subject_fraction,
            args.history_windows,
            args.c_values,
        )
        for seed_label, paths in zip(labels, prediction_sets, strict=True)
    ]
    report = {
        "experiment": "four_model_validation_trained_causal_gate",
        "gate_variants": list(GATE_VARIANTS),
        "model_roles": list(MODEL_ROLES),
        "history_windows": list(args.history_windows),
        "inner_val_subject_fraction": args.inner_val_subject_fraction,
        "c_values": list(args.c_values),
        "stage5_names": list(STAGE5_NAMES),
        "fusion_weights": {
            "primary_full_w20": primary_alphas.tolist(),
            "secondary_capacity_h128": secondary_alphas.tolist(),
            "tertiary_h128_ls003": tertiary_alphas.tolist(),
        },
        "seed_reports": seed_reports,
        "summary": summarize(seed_reports),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print_summary(report)


if __name__ == "__main__":
    main()
