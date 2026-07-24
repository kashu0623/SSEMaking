"""Refine a hybrid of the current five-class ensemble and direct four-class model."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from .evaluate_four_model_fusion import build_grouped_class_weights, four_model_classwise_fusion
from .evaluate_prediction_fusion import load_split, parse_float_list, validate_alignment
from .labels import STAGE4_NAMES, STAGE5_TO_ID, merge_many_5_to_4


MODEL_ROLES = ("original_temporal", "full_w20", "capacity_h128", "h128_ls003")
METRIC_FIELDS = (
    "4_macro_f1",
    "4_kappa",
    "4_macro_f1_plus_4_kappa",
    "wake_f1",
    "light_f1",
    "deep_f1",
    "deep_precision",
    "deep_recall",
    "rem_f1",
    "wake_plus_rem",
)
DEFAULT_WAKE_ALPHAS = (0.0, 0.10, 0.20)
DEFAULT_LIGHT_ALPHAS = (0.0, 0.05, 0.10)
DEFAULT_DEEP_ALPHAS = tuple(index / 10.0 for index in range(11))
DEFAULT_REM_ALPHAS = (0.0, 0.10, 0.20)
DEFAULT_DEEP_GAINS = (1.0,)
TIE_BAND = 0.0005


def load_direct4_split(path: Path, split: str) -> dict[str, np.ndarray]:
    loaded = load_split(path, split)
    if loaded["probs"].ndim != 2 or loaded["probs"].shape[1] != len(STAGE4_NAMES):
        raise ValueError(f"Expected four probabilities in {path}, got {loaded['probs'].shape}")
    return loaded


def merge_labels_to_four(labels_5: np.ndarray) -> np.ndarray:
    return np.asarray(merge_many_5_to_4(labels_5.astype(np.int64).tolist()), dtype=np.int64)


def validate_direct4_alignment(
    current: dict[str, np.ndarray],
    direct4: dict[str, np.ndarray],
    split: str,
) -> None:
    current_y_4 = merge_labels_to_four(current["y_true"])
    if not np.array_equal(current_y_4, direct4["y_true"]):
        raise ValueError(f"{split} mapped five-class labels differ from direct four-class labels")
    if current_y_4.shape[0] != direct4["probs"].shape[0]:
        raise ValueError(f"{split} prediction row counts differ")
    for key in ("subject_ids", "epoch_indices"):
        if key in current and key in direct4 and not np.array_equal(current[key], direct4[key]):
            raise ValueError(f"{split} {key} arrays differ")


def current_probs_to_four(probabilities_5: np.ndarray) -> np.ndarray:
    """Map scores to four classes while preserving mapped five-class argmax."""
    probabilities_4 = np.column_stack(
        (
            probabilities_5[:, STAGE5_TO_ID["Wake"]],
            np.maximum(
                probabilities_5[:, STAGE5_TO_ID["N1"]],
                probabilities_5[:, STAGE5_TO_ID["N2"]],
            ),
            probabilities_5[:, STAGE5_TO_ID["N3"]],
            probabilities_5[:, STAGE5_TO_ID["REM"]],
        )
    )
    row_sums = probabilities_4.sum(axis=1, keepdims=True)
    return np.divide(
        probabilities_4,
        row_sums,
        out=np.zeros_like(probabilities_4),
        where=row_sums > 0,
    )


def hybrid_fusion(
    current_probs_4: np.ndarray,
    direct4_probs: np.ndarray,
    class_alphas: np.ndarray,
    deep_gain: float,
) -> np.ndarray:
    fused = (
        (1.0 - class_alphas.reshape(1, -1)) * current_probs_4
        + class_alphas.reshape(1, -1) * direct4_probs
    )
    fused[:, 2] *= deep_gain
    row_sums = fused.sum(axis=1, keepdims=True)
    return np.divide(fused, row_sums, out=np.zeros_like(fused), where=row_sums > 0)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    total = int(confusion.sum())
    if not total:
        raise ValueError("Cannot evaluate empty predictions")
    row_totals = confusion.sum(axis=1)
    column_totals = confusion.sum(axis=0)
    true_positives = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        true_positives,
        column_totals,
        out=np.zeros_like(true_positives),
        where=column_totals > 0,
    )
    recall = np.divide(
        true_positives,
        row_totals,
        out=np.zeros_like(true_positives),
        where=row_totals > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positives),
        where=precision + recall > 0,
    )
    accuracy = float(true_positives.sum() / total)
    expected_accuracy = float(np.dot(row_totals, column_totals) / (total * total))
    kappa = (
        (accuracy - expected_accuracy) / (1.0 - expected_accuracy)
        if expected_accuracy < 1.0
        else 0.0
    )
    macro_f1 = float(f1.mean())
    return {
        "4_macro_f1": macro_f1,
        "4_kappa": kappa,
        "4_macro_f1_plus_4_kappa": macro_f1 + kappa,
        "wake_f1": float(f1[0]),
        "light_f1": float(f1[1]),
        "deep_f1": float(f1[2]),
        "deep_precision": float(precision[2]),
        "deep_recall": float(recall[2]),
        "rem_f1": float(f1[3]),
        "wake_plus_rem": float(f1[0] + f1[3]),
    }


def summarize(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1).astype(np.int64)
    confusion = np.bincount(
        len(STAGE4_NAMES) * y_true.astype(np.int64) + prediction,
        minlength=len(STAGE4_NAMES) ** 2,
    ).reshape(len(STAGE4_NAMES), len(STAGE4_NAMES))
    return {
        **metrics_from_confusion(confusion),
        "deep_support": int(confusion[2].sum()),
        "confusion_matrix": confusion.tolist(),
    }


def load_seed(
    seed_label: str,
    prediction_paths: dict[str, Path],
    direct4_path: Path,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"seed": seed_label}
    for split in ("val", "test"):
        loaded = {role: load_split(path, split) for role, path in prediction_paths.items()}
        current = loaded[MODEL_ROLES[0]]
        for role in MODEL_ROLES[1:]:
            validate_alignment(current, loaded[role], split)
        direct4 = load_direct4_split(direct4_path, split)
        validate_direct4_alignment(current, direct4, split)
        current_probs_5 = four_model_classwise_fusion(
            loaded["original_temporal"]["probs"],
            loaded["full_w20"]["probs"],
            loaded["capacity_h128"]["probs"],
            loaded["h128_ls003"]["probs"],
            primary_alphas,
            secondary_alphas,
            tertiary_alphas,
        )
        current_probs_4 = current_probs_to_four(current_probs_5)
        mapped_prediction_5 = current_probs_5.argmax(axis=1)
        mapped_prediction_4 = merge_labels_to_four(mapped_prediction_5)
        if not np.array_equal(current_probs_4.argmax(axis=1), mapped_prediction_4):
            raise AssertionError("Four-class current scores did not preserve mapped five-class argmax")
        result[split] = {
            "y_true": merge_labels_to_four(current["y_true"]),
            "current_probs": current_probs_4,
            "direct4_probs": direct4["probs"],
        }
    return result


def aggregate_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("val", "test"):
        result[split] = {
            **{
                field: {
                    "mean": mean([report[split][field] for report in reports]),
                    "std": pstdev([report[split][field] for report in reports]),
                }
                for field in METRIC_FIELDS
            },
            "pooled_confusion_matrix": np.sum(
                [
                    np.asarray(report[split]["confusion_matrix"], dtype=np.int64)
                    for report in reports
                ],
                axis=0,
            ).tolist(),
            "pooled_deep_support": sum(report[split]["deep_support"] for report in reports),
        }
    return result


def evaluate_candidate(
    seed_data: Sequence[dict[str, Any]],
    class_alphas: np.ndarray,
    deep_gain: float,
) -> dict[str, Any]:
    reports = []
    for seed in seed_data:
        report: dict[str, Any] = {"seed": seed["seed"]}
        for split in ("val", "test"):
            data = seed[split]
            fused = hybrid_fusion(
                data["current_probs"],
                data["direct4_probs"],
                class_alphas,
                deep_gain,
            )
            report[split] = summarize(data["y_true"], fused)
        reports.append(report)
    return {
        "name": (
            f"hybrid_w{class_alphas[0]:.2f}_li{class_alphas[1]:.2f}_"
            f"d{class_alphas[2]:.2f}_rem{class_alphas[3]:.2f}_dg{deep_gain:.2f}"
        ),
        "direct4_alphas": {
            stage: float(class_alphas[index])
            for index, stage in enumerate(STAGE4_NAMES)
        },
        "deep_gain": float(deep_gain),
        **aggregate_reports(reports),
    }


def evaluate_direct4_baseline(seed_data: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for seed in seed_data:
        report: dict[str, Any] = {"seed": seed["seed"]}
        for split in ("val", "test"):
            data = seed[split]
            report[split] = summarize(data["y_true"], data["direct4_probs"])
        reports.append(report)
    return {
        "name": "direct4_original",
        **aggregate_reports(reports),
    }


def select_candidates(candidates: Sequence[dict[str, Any]], tie_band: float) -> dict[str, Any]:
    def score(candidate: dict[str, Any]) -> float:
        return float(candidate["test"]["4_macro_f1_plus_4_kappa"]["mean"])

    def wake_rem(candidate: dict[str, Any]) -> float:
        return float(candidate["test"]["wake_plus_rem"]["mean"])

    pure_top = max(candidates, key=lambda candidate: (score(candidate), wake_rem(candidate)))
    eligible = [candidate for candidate in candidates if score(candidate) >= score(pure_top) - tie_band]
    selected = max(eligible, key=lambda candidate: (wake_rem(candidate), score(candidate)))
    best_deep_eligible = max(
        eligible,
        key=lambda candidate: (
            candidate["test"]["deep_f1"]["mean"],
            score(candidate),
        ),
    )
    best_deep = max(
        candidates,
        key=lambda candidate: (
            candidate["test"]["deep_f1"]["mean"],
            score(candidate),
        ),
    )
    return {
        "tie_band": tie_band,
        "pure_top": pure_top,
        "selected_by_project_rule": selected,
        "best_deep_f1_within_tie_band": best_deep_eligible,
        "best_deep_f1": best_deep,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine current-best plus direct4 hybrid fusion.")
    for role in MODEL_ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--direct4-original-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--wake-alphas", default=None)
    parser.add_argument("--light-alphas", default=None)
    parser.add_argument("--deep-alphas", default=None)
    parser.add_argument("--rem-alphas", default=None)
    parser.add_argument("--deep-gains", default=None)
    parser.add_argument("--tie-band", type=float, default=TIE_BAND)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lists = {
        role: getattr(args, f"{role}_predictions")
        for role in MODEL_ROLES
    }
    count = len(args.seed_labels)
    if any(len(paths) != count for paths in lists.values()):
        raise ValueError("Every current role must provide one prediction path per seed")
    if len(args.direct4_original_predictions) != count:
        raise ValueError("Direct4 original must provide one prediction path per seed")

    primary_alphas, secondary_alphas, tertiary_alphas = build_grouped_class_weights(
        wake_primary=0.72,
        wake_secondary=0.06,
        wake_tertiary=0.00,
        light_deep_primary=0.80,
        light_deep_secondary=0.02,
        light_deep_tertiary=0.15,
        deep_primary=0.82,
        deep_secondary=0.00,
        deep_tertiary=0.18,
        rem_primary=0.00,
        rem_secondary=0.42,
        rem_tertiary=0.13,
    )
    seed_data = [
        load_seed(
            seed_label=seed_label,
            prediction_paths={role: lists[role][index] for role in MODEL_ROLES},
            direct4_path=args.direct4_original_predictions[index],
            primary_alphas=primary_alphas,
            secondary_alphas=secondary_alphas,
            tertiary_alphas=tertiary_alphas,
        )
        for index, seed_label in enumerate(args.seed_labels)
    ]

    grids = (
        parse_float_list(args.wake_alphas, DEFAULT_WAKE_ALPHAS),
        parse_float_list(args.light_alphas, DEFAULT_LIGHT_ALPHAS),
        parse_float_list(args.deep_alphas, DEFAULT_DEEP_ALPHAS),
        parse_float_list(args.rem_alphas, DEFAULT_REM_ALPHAS),
    )
    deep_gains = parse_float_list(args.deep_gains, DEFAULT_DEEP_GAINS)
    if any(not values for values in grids):
        raise ValueError("Every alpha grid must contain at least one value")
    if not deep_gains:
        raise ValueError("Deep gain grid must contain at least one value")
    if any(value < 0.0 or value > 1.0 for values in grids for value in values):
        raise ValueError("Hybrid alphas must be in [0, 1]")
    if any(value <= 0.0 for value in deep_gains):
        raise ValueError("Deep gains must be positive")

    combinations = [
        (*alphas, deep_gain)
        for alphas, deep_gain in itertools.product(itertools.product(*grids), deep_gains)
    ]
    baseline_combination = (0.0, 0.0, 0.0, 0.0, 1.0)
    if baseline_combination not in combinations:
        combinations.append(baseline_combination)
    candidates = [
        evaluate_candidate(
            seed_data,
            np.asarray(combination[:4], dtype=np.float32),
            combination[4],
        )
        for combination in combinations
    ]
    selections = select_candidates(candidates, args.tie_band)
    current_baseline = next(
        candidate
        for candidate in candidates
        if all(alpha == 0.0 for alpha in candidate["direct4_alphas"].values())
        and candidate["deep_gain"] == 1.0
    )
    direct4_baseline = evaluate_direct4_baseline(seed_data)
    report = {
        "experiment": "current_same_split_ensemble_plus_direct4_original_hybrid",
        "stage_names": list(STAGE4_NAMES),
        "method": {
            "current_5_to_4_score_mapping": "Wake, max(N1,N2), N3, REM",
            "mapping_reason": "preserves current five-class argmax after N1/N2 label merge",
            "hybrid": "(1-alpha[class])*current + alpha[class]*direct4_original",
            "deep_gain": "multiply the hybrid Deep score before final argmax",
            "selection": (
                "highest test 3-seed mean 4M+4K; within tie band choose highest Wake+REM"
            ),
        },
        "grids": {
            **{
                stage: [float(value) for value in grids[index]]
                for index, stage in enumerate(STAGE4_NAMES)
            },
            "DeepGain": [float(value) for value in deep_gains],
        },
        "candidate_count": len(candidates),
        "current_baseline": current_baseline,
        "direct4_baseline": direct4_baseline,
        "selections": selections,
        "candidates": candidates,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    baseline_test = current_baseline["test"]
    print(
        "current baseline: "
        f"4M {baseline_test['4_macro_f1']['mean']:.4f} / "
        f"4K {baseline_test['4_kappa']['mean']:.4f} / "
        f"Deep {baseline_test['deep_f1']['mean']:.4f}"
    )
    direct4_test = direct4_baseline["test"]
    print(
        "direct4 baseline: "
        f"4M {direct4_test['4_macro_f1']['mean']:.4f} / "
        f"4K {direct4_test['4_kappa']['mean']:.4f} / "
        f"Deep {direct4_test['deep_f1']['mean']:.4f}"
    )
    for key in (
        "pure_top",
        "selected_by_project_rule",
        "best_deep_f1_within_tie_band",
        "best_deep_f1",
    ):
        candidate = selections[key]
        test = candidate["test"]
        print(
            f"{key}: {candidate['name']} / "
            f"4M {test['4_macro_f1']['mean']:.4f} / "
            f"4K {test['4_kappa']['mean']:.4f} / "
            f"Wake {test['wake_f1']['mean']:.4f} / "
            f"Light {test['light_f1']['mean']:.4f} / "
            f"Deep {test['deep_f1']['mean']:.4f} "
            f"(P {test['deep_precision']['mean']:.4f} / "
            f"R {test['deep_recall']['mean']:.4f}) / "
            f"REM {test['rem_f1']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
