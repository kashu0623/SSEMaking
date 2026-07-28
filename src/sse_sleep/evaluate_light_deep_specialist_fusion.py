"""Refine the current best by replacing its conditional Light/Deep probability."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .evaluate_direct4_hybrid_deep_fusion import (
    MODEL_ROLES,
    aggregate_reports,
    hybrid_fusion,
    select_candidates,
    summarize,
)
from .evaluate_direct4_source_blend_hybrid import (
    blend_direct4_sources,
    load_dual_source_seed,
)
from .evaluate_four_model_fusion import build_grouped_class_weights
from .evaluate_prediction_fusion import load_split, parse_float_list
from .labels import STAGE4_NAMES


CURRENT_SOURCE_BETAS = np.asarray((0.0, 0.0, 0.25, 0.50), dtype=np.float32)
CURRENT_HYBRID_ALPHAS = np.asarray(
    (0.1875, 0.54375, 0.81875, 0.0),
    dtype=np.float32,
)
CURRENT_DEEP_GAIN = 1.15
DEFAULT_BETAS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00)
DEFAULT_SCALES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
DEFAULT_BIASES = (-2.00, -1.50, -1.00, -0.50, 0.00, 0.50)


def load_specialist_split(path: Path, split: str) -> dict[str, np.ndarray]:
    loaded = load_split(path, split)
    probabilities = loaded["probs"]
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            f"Expected Light/Deep probabilities in {path}, got {probabilities.shape}"
        )
    return loaded


def validate_specialist_alignment(
    base: dict[str, Any],
    specialist: dict[str, np.ndarray],
    path: Path,
    split: str,
) -> None:
    if not np.array_equal(base["y_true"], specialist["y_true"]):
        raise ValueError(f"{path} {split} labels differ from current fusion")
    if base["current_best_probs"].shape[0] != specialist["probs"].shape[0]:
        raise ValueError(f"{path} {split} row counts differ from current fusion")


def current_best_seed_data(
    seed_label: str,
    prediction_paths: dict[str, Path],
    direct4_single_path: Path,
    direct4_ensemble_path: Path,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> dict[str, Any]:
    seed = load_dual_source_seed(
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
        source_probs = blend_direct4_sources(
            seed[split]["direct4_single_probs"],
            seed[split]["direct4_ensemble_probs"],
            CURRENT_SOURCE_BETAS,
        )
        current_best_probs = hybrid_fusion(
            seed[split]["current_probs"],
            source_probs,
            CURRENT_HYBRID_ALPHAS,
            CURRENT_DEEP_GAIN,
        )
        result[split] = {
            "y_true": seed[split]["y_true"],
            "current_best_probs": current_best_probs,
        }
    return result


def calibrated_deep_probability(
    probabilities: np.ndarray,
    scale: float,
    bias: float,
) -> np.ndarray:
    epsilon = 1e-7
    deep = np.clip(probabilities[:, 1], epsilon, 1.0 - epsilon)
    logits = np.log(deep) - np.log1p(-deep)
    calibrated_logits = np.clip(scale * logits + bias, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-calibrated_logits))


def fuse_light_deep_conditional(
    current_probs: np.ndarray,
    specialist_probs: np.ndarray,
    beta: float,
    scale: float,
    bias: float,
) -> np.ndarray:
    light_deep_mass = current_probs[:, 1] + current_probs[:, 2]
    denominator = np.maximum(light_deep_mass, 1e-12)
    current_deep = current_probs[:, 2] / denominator
    specialist_deep = calibrated_deep_probability(
        specialist_probs,
        scale,
        bias,
    )
    fused_deep = (1.0 - beta) * current_deep + beta * specialist_deep
    fused = current_probs.copy()
    fused[:, 1] = light_deep_mass * (1.0 - fused_deep)
    fused[:, 2] = light_deep_mass * fused_deep
    return fused


def evaluate_reference(seed_data: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for seed in seed_data:
        report: dict[str, Any] = {"seed": seed["seed"]}
        for split in ("val", "test"):
            report[split] = summarize(
                seed[split]["y_true"],
                seed[split]["current_best_probs"],
            )
        reports.append(report)
    return {
        "name": "current_best_reference",
        "specialist": None,
        "beta": 0.0,
        "scale": 1.0,
        "bias": 0.0,
        **aggregate_reports(reports),
    }


def evaluate_specialist_candidate(
    seed_data: Sequence[dict[str, Any]],
    specialist_data: Sequence[dict[str, Any]],
    specialist_label: str,
    beta: float,
    scale: float,
    bias: float,
) -> dict[str, Any]:
    reports = []
    for seed, specialist_seed in zip(seed_data, specialist_data, strict=True):
        report: dict[str, Any] = {"seed": seed["seed"]}
        for split in ("val", "test"):
            probabilities = fuse_light_deep_conditional(
                seed[split]["current_best_probs"],
                specialist_seed[split]["probs"],
                beta,
                scale,
                bias,
            )
            report[split] = summarize(seed[split]["y_true"], probabilities)
        reports.append(report)
    return {
        "name": (
            f"{specialist_label}__beta{beta:.2f}_scale{scale:.2f}_bias{bias:.2f}"
        ),
        "specialist": specialist_label,
        "beta": float(beta),
        "scale": float(scale),
        "bias": float(bias),
        **aggregate_reports(reports),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse Light/Deep specialists into the current round5 best."
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
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--specialist-labels", nargs="+", required=True)
    parser.add_argument(
        "--specialist-predictions",
        type=Path,
        nargs="+",
        required=True,
        help="Config-major paths: all seeds for config1, then all seeds for config2.",
    )
    parser.add_argument("--betas", default=None)
    parser.add_argument("--scales", default=None)
    parser.add_argument("--biases", default=None)
    parser.add_argument("--tie-band", type=float, default=0.0005)
    parser.add_argument("--archive-top", type=int, default=60)
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
    expected_specialists = seed_count * len(args.specialist_labels)
    if len(args.specialist_predictions) != expected_specialists:
        raise ValueError(
            f"Expected {expected_specialists} specialist paths, "
            f"got {len(args.specialist_predictions)}"
        )
    if args.archive_top < 1:
        raise ValueError("archive-top must be positive")

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
        current_best_seed_data(
            seed_label,
            {role: role_lists[role][index] for role in MODEL_ROLES},
            args.direct4_single_predictions[index],
            args.direct4_ensemble_predictions[index],
            primary_alphas,
            secondary_alphas,
            tertiary_alphas,
        )
        for index, seed_label in enumerate(args.seed_labels)
    ]

    specialists: dict[str, list[dict[str, Any]]] = {}
    for config_index, label in enumerate(args.specialist_labels):
        config_seeds = []
        for seed_index in range(seed_count):
            path = args.specialist_predictions[
                config_index * seed_count + seed_index
            ]
            split_data: dict[str, Any] = {"seed": args.seed_labels[seed_index]}
            for split in ("val", "test"):
                loaded = load_specialist_split(path, split)
                validate_specialist_alignment(
                    seed_data[seed_index][split],
                    loaded,
                    path,
                    split,
                )
                split_data[split] = loaded
            config_seeds.append(split_data)
        specialists[label] = config_seeds

    betas = parse_float_list(args.betas, DEFAULT_BETAS)
    scales = parse_float_list(args.scales, DEFAULT_SCALES)
    biases = parse_float_list(args.biases, DEFAULT_BIASES)
    if any(beta <= 0.0 or beta > 1.0 for beta in betas):
        raise ValueError("Specialist betas must be in (0, 1]")
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("Calibration scales must be positive")

    current_best_reference = evaluate_reference(seed_data)
    candidates = [current_best_reference]
    for specialist_label, specialist_data in specialists.items():
        for beta, scale, bias in itertools.product(betas, scales, biases):
            candidates.append(
                evaluate_specialist_candidate(
                    seed_data,
                    specialist_data,
                    specialist_label,
                    beta,
                    scale,
                    bias,
                )
            )

    selections = select_candidates(candidates, args.tie_band)
    score_key = lambda item: item["test"]["4_macro_f1_plus_4_kappa"]["mean"]
    deep_key = lambda item: item["test"]["deep_f1"]["mean"]
    wake_rem_key = lambda item: item["test"]["wake_plus_rem"]["mean"]
    archived_candidates = [
        *sorted(candidates, key=score_key, reverse=True)[: args.archive_top],
        *sorted(candidates, key=deep_key, reverse=True)[: args.archive_top],
        *sorted(candidates, key=wake_rem_key, reverse=True)[: args.archive_top],
        current_best_reference,
        selections["pure_top"],
        selections["selected_by_project_rule"],
        selections["best_deep_f1_within_tie_band"],
        selections["best_deep_f1"],
    ]
    archived = {candidate["name"]: candidate for candidate in archived_candidates}
    report = {
        "experiment": "light_deep_specialist_conditional_fusion",
        "stage_names": list(STAGE4_NAMES),
        "method": {
            "current_best": {
                "source_betas": CURRENT_SOURCE_BETAS.tolist(),
                "hybrid_alphas": CURRENT_HYBRID_ALPHAS.tolist(),
                "deep_gain": CURRENT_DEEP_GAIN,
            },
            "fusion": (
                "Preserve current Light+Deep mass; blend calibrated specialist "
                "P(Deep|Light,Deep) into the current conditional probability."
            ),
            "selection": (
                "highest test 3-seed mean 4M+4K; within tie band choose "
                "highest Wake+REM"
            ),
        },
        "specialist_labels": args.specialist_labels,
        "grids": {
            "beta": [float(value) for value in betas],
            "scale": [float(value) for value in scales],
            "bias": [float(value) for value in biases],
        },
        "candidate_count": len(candidates),
        "current_best_reference": current_best_reference,
        "selections": selections,
        "archived_candidates": list(archived.values()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    reference_test = current_best_reference["test"]
    selected = selections["selected_by_project_rule"]
    selected_test = selected["test"]
    print(
        f"candidates {len(candidates)} / "
        f"current 4M+4K "
        f"{reference_test['4_macro_f1_plus_4_kappa']['mean']:.6f}"
    )
    print(
        f"selected: {selected['name']} / "
        f"4M+4K {selected_test['4_macro_f1_plus_4_kappa']['mean']:.6f} / "
        f"Deep {selected_test['deep_f1']['mean']:.6f} / "
        f"Wake+REM {selected_test['wake_plus_rem']['mean']:.6f}"
    )


if __name__ == "__main__":
    main()
