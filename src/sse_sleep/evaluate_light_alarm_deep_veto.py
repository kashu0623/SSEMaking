"""Fuse Light-vs-rest proposals with the current staging model's Deep veto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_direct4_hybrid_deep_fusion import MODEL_ROLES
from .evaluate_four_model_fusion import build_grouped_class_weights
from .evaluate_light_alarm import (
    DEFAULT_DEEP_LIMITS,
    aggregate_split,
    current_baseline,
    load_prediction,
    parse_float_list,
    select_records,
    validate_alignment,
)
from .evaluate_light_deep_specialist_fusion import (
    current_best_seed_data,
    fuse_light_deep_conditional,
    load_specialist_split,
    validate_specialist_alignment,
)


DEFAULT_ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_GAMMAS = (0.0, 0.50, 1.0, 2.0, 4.0)
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.arange(0.10, 0.901, 0.05))


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def veto_score(
    alarm_light: np.ndarray,
    current_light: np.ndarray,
    current_deep_conditional: np.ndarray,
    alarm_alpha: float,
    veto_gamma: float,
) -> np.ndarray:
    proposal = sigmoid(
        (1.0 - alarm_alpha) * logit(current_light)
        + alarm_alpha * logit(alarm_light)
    )
    veto = np.power(
        np.clip(1.0 - current_deep_conditional, 1e-6, 1.0),
        veto_gamma,
    )
    return np.clip(proposal * veto, 0.0, 1.0)


def current_seed_data(
    seed_label: str,
    prediction_paths: dict[str, Path],
    direct4_single_path: Path,
    direct4_ensemble_path: Path,
    specialist_path: Path,
    beta: float,
    scale: float,
    bias: float,
) -> dict[str, Any]:
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
    base = current_best_seed_data(
        seed_label,
        prediction_paths,
        direct4_single_path,
        direct4_ensemble_path,
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    result: dict[str, Any] = {"seed": seed_label}
    for split in ("val", "test"):
        specialist = load_specialist_split(specialist_path, split)
        validate_specialist_alignment(base[split], specialist, specialist_path, split)
        probabilities = fuse_light_deep_conditional(
            base[split]["current_best_probs"],
            specialist["probs"],
            beta,
            scale,
            bias,
        )
        light_deep_mass = np.maximum(
            probabilities[:, 1] + probabilities[:, 2],
            1e-12,
        )
        result[split] = {
            "y_true4": base[split]["y_true"],
            "current_light": probabilities[:, 1],
            "current_deep_conditional": probabilities[:, 2] / light_deep_mass,
        }
    return result


def candidate_record(
    label: str,
    alarm_seeds: Sequence[dict[str, dict[str, np.ndarray]]],
    current_seeds: Sequence[dict[str, Any]],
    alarm_alpha: float,
    veto_gamma: float,
    threshold: float,
) -> dict[str, Any]:
    transformed: dict[str, list[dict[str, np.ndarray]]] = {
        "val": [],
        "test": [],
    }
    for alarm_seed, current_seed in zip(alarm_seeds, current_seeds, strict=True):
        for split in ("val", "test"):
            transformed[split].append(
                {
                    "y_true4": current_seed[split]["y_true4"],
                    "light_probs": veto_score(
                        alarm_seed[split]["light_probs"],
                        current_seed[split]["current_light"],
                        current_seed[split]["current_deep_conditional"],
                        alarm_alpha,
                        veto_gamma,
                    ),
                }
            )
    return {
        "name": (
            f"{label}__alarm{alarm_alpha:.2f}_veto{veto_gamma:.2f}"
            f"_threshold{threshold:.2f}"
        ),
        "config": label,
        "alarm_alpha": float(alarm_alpha),
        "veto_gamma": float(veto_gamma),
        "threshold": float(threshold),
        "val": aggregate_split(transformed["val"], threshold),
        "test": aggregate_split(transformed["test"], threshold),
    }


def constrained_selections(
    candidates: Sequence[dict[str, Any]],
    baseline_deep_leak: float,
) -> dict[str, Any]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["val"]["deep_to_light_rate"]["mean"] <= baseline_deep_leak
    ]
    if not eligible:
        return {
            "validation_deep_leak_limit": baseline_deep_leak,
            "best_light_objective": None,
            "best_light_recall": None,
        }
    return {
        "validation_deep_leak_limit": baseline_deep_leak,
        "best_light_objective": max(
            eligible,
            key=lambda item: (
                item["val"]["light_objective"]["mean"],
                -item["val"]["deep_to_light_rate"]["mean"],
                item["val"]["light_precision"]["mean"],
            ),
        ),
        "best_light_recall": max(
            eligible,
            key=lambda item: (
                item["val"]["light_recall"]["mean"],
                item["val"]["light_precision"]["mean"],
                item["val"]["binary_kappa"]["mean"],
            ),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Light alarm proposals with a current-stage Deep veto."
    )
    for role in MODEL_ROLES:
        parser.add_argument(
            f"--{role.replace('_', '-')}-predictions",
            type=Path,
            nargs="+",
            required=True,
        )
    parser.add_argument(
        "--direct4-single-predictions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--direct4-ensemble-predictions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--current-specialist-predictions",
        type=Path,
        nargs="+",
        required=True,
    )
    parser.add_argument("--current-beta", type=float, default=0.975)
    parser.add_argument("--current-scale", type=float, default=0.75)
    parser.add_argument("--current-bias", type=float, default=0.25)
    parser.add_argument("--config-labels", nargs="+", required=True)
    parser.add_argument("--alarm-prediction-paths", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--alarm-alphas", default=None)
    parser.add_argument("--veto-gammas", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--deep-leak-limits", default=None)
    parser.add_argument("--current-summary-json", type=Path, required=True)
    parser.add_argument("--archive-top", type=int, default=50)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_count = len(args.seed_labels)
    role_lists = {
        role: getattr(args, f"{role}_predictions")
        for role in MODEL_ROLES
    }
    if any(len(paths) != seed_count for paths in role_lists.values()):
        raise ValueError("Every current role must provide one path per seed")
    if len(args.direct4_single_predictions) != seed_count:
        raise ValueError("Direct4 single must provide one path per seed")
    if len(args.direct4_ensemble_predictions) != seed_count:
        raise ValueError("Direct4 ensemble must provide one path per seed")
    if len(args.current_specialist_predictions) != seed_count:
        raise ValueError("Current specialist must provide one path per seed")
    expected_alarm_paths = len(args.config_labels) * seed_count
    if len(args.alarm_prediction_paths) != expected_alarm_paths:
        raise ValueError(
            f"Expected {expected_alarm_paths} alarm paths, "
            f"got {len(args.alarm_prediction_paths)}"
        )

    current_seeds = [
        current_seed_data(
            seed_label,
            {role: role_lists[role][index] for role in MODEL_ROLES},
            args.direct4_single_predictions[index],
            args.direct4_ensemble_predictions[index],
            args.current_specialist_predictions[index],
            args.current_beta,
            args.current_scale,
            args.current_bias,
        )
        for index, seed_label in enumerate(args.seed_labels)
    ]
    alarms: dict[str, list[dict[str, dict[str, np.ndarray]]]] = {}
    reference_by_seed: list[dict[str, dict[str, np.ndarray]] | None] = [
        None for _ in range(seed_count)
    ]
    for config_index, label in enumerate(args.config_labels):
        seeds = []
        for seed_index in range(seed_count):
            path = args.alarm_prediction_paths[
                config_index * seed_count + seed_index
            ]
            loaded = load_prediction(path)
            reference = reference_by_seed[seed_index]
            if reference is None:
                reference_by_seed[seed_index] = loaded
            else:
                validate_alignment(reference, loaded, path)
            for split in ("val", "test"):
                if not np.array_equal(
                    loaded[split]["y_true4"],
                    current_seeds[seed_index][split]["y_true4"],
                ):
                    raise ValueError(f"{path} {split} does not align with current fusion")
            seeds.append(loaded)
        alarms[label] = seeds

    alarm_alphas = parse_float_list(args.alarm_alphas, DEFAULT_ALPHAS)
    veto_gammas = parse_float_list(args.veto_gammas, DEFAULT_GAMMAS)
    thresholds = parse_float_list(args.thresholds, DEFAULT_THRESHOLDS)
    deep_limits = parse_float_list(args.deep_leak_limits, DEFAULT_DEEP_LIMITS)
    if any(not 0.0 <= value <= 1.0 for value in alarm_alphas):
        raise ValueError("Alarm alphas must be in [0, 1]")
    if any(value < 0.0 for value in veto_gammas):
        raise ValueError("Veto gammas must be non-negative")
    if any(not 0.0 < value < 1.0 for value in thresholds):
        raise ValueError("Thresholds must be in (0, 1)")

    candidates = [
        candidate_record(
            label,
            alarm_seeds,
            current_seeds,
            alarm_alpha,
            veto_gamma,
            threshold,
        )
        for label, alarm_seeds in alarms.items()
        for alarm_alpha in alarm_alphas
        for veto_gamma in veto_gammas
        for threshold in thresholds
    ]
    baseline = current_baseline(args.current_summary_json)
    global_selections = select_records(candidates, deep_limits)
    baseline_constrained = constrained_selections(
        candidates,
        float(baseline["val"]["deep_to_light_rate"]),
    )
    source_selections = {
        label: {
            "unconstrained": select_records(
                [candidate for candidate in candidates if candidate["config"] == label],
                deep_limits,
            ),
            "baseline_constrained": constrained_selections(
                [candidate for candidate in candidates if candidate["config"] == label],
                float(baseline["val"]["deep_to_light_rate"]),
            ),
        }
        for label in args.config_labels
    }

    archived_candidates = [
        *sorted(
            candidates,
            key=lambda item: item["val"]["light_objective"]["mean"],
            reverse=True,
        )[: args.archive_top],
        *sorted(
            candidates,
            key=lambda item: item["val"]["light_recall"]["mean"],
            reverse=True,
        )[: args.archive_top],
        *sorted(
            candidates,
            key=lambda item: item["val"]["deep_to_light_rate"]["mean"],
        )[: args.archive_top],
    ]
    selection_groups = [
        global_selections,
        baseline_constrained,
        *[
            group
            for source in source_selections.values()
            for group in source.values()
        ],
    ]
    for selection in selection_groups:
        for key, value in selection.items():
            if isinstance(value, dict) and "name" in value:
                archived_candidates.append(value)
            if key == "safe_profiles":
                archived_candidates.extend(
                    item for item in value.values() if item is not None
                )
    archived = {item["name"]: item for item in archived_candidates}
    report = {
        "experiment": "light_alarm_deep_veto_fusion",
        "method": {
            "proposal": "logit blend of current P(Light) and direct alarm P(Light)",
            "veto": "proposal * (1 - current P(Deep|Light,Deep)) ** gamma",
            "selection": "validation 3-seed mean only; test is reporting only",
        },
        "config_labels": args.config_labels,
        "seed_labels": args.seed_labels,
        "grids": {
            "alarm_alpha": [float(value) for value in alarm_alphas],
            "veto_gamma": [float(value) for value in veto_gammas],
            "threshold": [float(value) for value in thresholds],
        },
        "candidate_count": len(candidates),
        "current_best_argmax_baseline": baseline,
        "global_selections": global_selections,
        "baseline_deep_leak_constrained": baseline_constrained,
        "source_selections": source_selections,
        "archived_candidates": list(archived.values()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selected = baseline_constrained["best_light_objective"]
    if selected is None:
        print(f"candidates {len(candidates)} / no baseline-constrained candidate")
    else:
        print(
            f"candidates {len(candidates)} / constrained {selected['name']} / "
            f"test Light F1 {selected['test']['light_f1']['mean']:.6f} / "
            f"test binary Kappa {selected['test']['binary_kappa']['mean']:.6f} / "
            f"test Deep->Light {selected['test']['deep_to_light_rate']['mean']:.6f}"
        )


if __name__ == "__main__":
    main()
