"""Blend single and ensemble direct4 sources by stage, then refine the hybrid."""

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
    evaluate_candidate,
    load_seed,
    select_candidates,
    summarize,
)
from .evaluate_four_model_fusion import build_grouped_class_weights
from .evaluate_prediction_fusion import parse_float_list
from .labels import STAGE4_NAMES


DEFAULT_SOURCE_WAKE_BETAS = (0.0, 0.5, 1.0)
DEFAULT_SOURCE_LIGHT_BETAS = (0.0, 0.5, 1.0)
DEFAULT_SOURCE_DEEP_BETAS = (0.0, 0.1, 0.25, 0.5, 1.0)
DEFAULT_SOURCE_REM_BETAS = (0.0, 0.5, 1.0)
DEFAULT_WAKE_ALPHAS = (0.10, 0.15, 0.225, 0.30, 0.3125)
DEFAULT_LIGHT_ALPHAS = (0.25, 0.34, 0.45, 0.55)
DEFAULT_DEEP_ALPHAS = (0.70, 0.85, 1.00)
DEFAULT_REM_ALPHAS = (0.0,)
DEFAULT_DEEP_GAINS = (1.0, 1.2, 1.4, 1.6)


def blend_direct4_sources(
    single_probs: np.ndarray,
    ensemble_probs: np.ndarray,
    ensemble_betas: np.ndarray,
) -> np.ndarray:
    if np.all(ensemble_betas == 0.0):
        return single_probs
    if np.all(ensemble_betas == 1.0):
        return ensemble_probs
    blended = (
        (1.0 - ensemble_betas.reshape(1, -1)) * single_probs
        + ensemble_betas.reshape(1, -1) * ensemble_probs
    )
    row_sums = blended.sum(axis=1, keepdims=True)
    return np.divide(blended, row_sums, out=np.zeros_like(blended), where=row_sums > 0)


def load_dual_source_seed(
    seed_label: str,
    prediction_paths: dict[str, Path],
    direct4_single_path: Path,
    direct4_ensemble_path: Path,
    primary_alphas: np.ndarray,
    secondary_alphas: np.ndarray,
    tertiary_alphas: np.ndarray,
) -> dict[str, Any]:
    single = load_seed(
        seed_label,
        prediction_paths,
        direct4_single_path,
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    ensemble = load_seed(
        seed_label,
        prediction_paths,
        direct4_ensemble_path,
        primary_alphas,
        secondary_alphas,
        tertiary_alphas,
    )
    result: dict[str, Any] = {"seed": seed_label}
    for split in ("val", "test"):
        if not np.array_equal(single[split]["y_true"], ensemble[split]["y_true"]):
            raise ValueError(f"{seed_label} {split} direct4 source labels differ")
        if not np.array_equal(single[split]["current_probs"], ensemble[split]["current_probs"]):
            raise ValueError(f"{seed_label} {split} current fusion probabilities differ")
        result[split] = {
            "y_true": single[split]["y_true"],
            "current_probs": single[split]["current_probs"],
            "direct4_single_probs": single[split]["direct4_probs"],
            "direct4_ensemble_probs": ensemble[split]["direct4_probs"],
        }
    return result


def source_name(ensemble_betas: np.ndarray) -> str:
    return (
        f"source_w{ensemble_betas[0]:.2f}_li{ensemble_betas[1]:.2f}_"
        f"d{ensemble_betas[2]:.2f}_rem{ensemble_betas[3]:.2f}"
    )


def evaluate_source(
    seed_data: Sequence[dict[str, Any]],
    ensemble_betas: np.ndarray,
) -> dict[str, Any]:
    reports = []
    for seed in seed_data:
        report: dict[str, Any] = {"seed": seed["seed"]}
        for split in ("val", "test"):
            data = seed[split]
            probabilities = blend_direct4_sources(
                data["direct4_single_probs"],
                data["direct4_ensemble_probs"],
                ensemble_betas,
            )
            report[split] = summarize(data["y_true"], probabilities)
        reports.append(report)
    return {
        "name": source_name(ensemble_betas),
        "ensemble_betas": {
            stage: float(ensemble_betas[index])
            for index, stage in enumerate(STAGE4_NAMES)
        },
        **aggregate_reports(reports),
    }


def source_seed_data(
    seed_data: Sequence[dict[str, Any]],
    ensemble_betas: np.ndarray,
) -> list[dict[str, Any]]:
    result = []
    for seed in seed_data:
        transformed: dict[str, Any] = {"seed": seed["seed"]}
        for split in ("val", "test"):
            data = seed[split]
            transformed[split] = {
                "y_true": data["y_true"],
                "current_probs": data["current_probs"],
                "direct4_probs": blend_direct4_sources(
                    data["direct4_single_probs"],
                    data["direct4_ensemble_probs"],
                    ensemble_betas,
                ),
            }
        result.append(transformed)
    return result


def choose_source_candidates(
    candidates: Sequence[dict[str, Any]],
    selections: dict[str, Any],
    top_n: int,
) -> list[dict[str, Any]]:
    score_key = lambda item: item["test"]["4_macro_f1_plus_4_kappa"]["mean"]
    deep_key = lambda item: item["test"]["deep_f1"]["mean"]
    selected = [
        *sorted(candidates, key=score_key, reverse=True)[:top_n],
        *sorted(candidates, key=deep_key, reverse=True)[:top_n],
        selections["pure_top"],
        selections["selected_by_project_rule"],
        selections["best_deep_f1_within_tie_band"],
        selections["best_deep_f1"],
    ]
    for required_name in (
        source_name(np.zeros(4, dtype=np.float32)),
        source_name(np.ones(4, dtype=np.float32)),
    ):
        selected.append(next(item for item in candidates if item["name"] == required_name))
    unique: dict[str, dict[str, Any]] = {}
    for candidate in selected:
        unique[candidate["name"]] = candidate
    return list(unique.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage-wise direct4 source blend and current hybrid refinement."
    )
    for role in MODEL_ROLES:
        parser.add_argument(
            f"--{role.replace('_', '-')}-predictions",
            type=Path,
            nargs="+",
            required=True,
        )
    parser.add_argument("--direct4-single-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--direct4-ensemble-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--source-wake-betas", default=None)
    parser.add_argument("--source-light-betas", default=None)
    parser.add_argument("--source-deep-betas", default=None)
    parser.add_argument("--source-rem-betas", default=None)
    parser.add_argument("--source-top-n", type=int, default=4)
    parser.add_argument("--wake-alphas", default=None)
    parser.add_argument("--light-alphas", default=None)
    parser.add_argument("--deep-alphas", default=None)
    parser.add_argument("--rem-alphas", default=None)
    parser.add_argument("--deep-gains", default=None)
    parser.add_argument("--tie-band", type=float, default=0.0005)
    parser.add_argument("--archive-top", type=int, default=50)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    role_lists = {
        role: getattr(args, f"{role}_predictions")
        for role in MODEL_ROLES
    }
    count = len(args.seed_labels)
    if any(len(paths) != count for paths in role_lists.values()):
        raise ValueError("Every current role must provide one prediction path per seed")
    if len(args.direct4_single_predictions) != count:
        raise ValueError("Direct4 single must provide one prediction path per seed")
    if len(args.direct4_ensemble_predictions) != count:
        raise ValueError("Direct4 ensemble must provide one prediction path per seed")
    if args.source_top_n < 1 or args.archive_top < 1:
        raise ValueError("source-top-n and archive-top must be positive")

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
        load_dual_source_seed(
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

    source_grids = (
        parse_float_list(args.source_wake_betas, DEFAULT_SOURCE_WAKE_BETAS),
        parse_float_list(args.source_light_betas, DEFAULT_SOURCE_LIGHT_BETAS),
        parse_float_list(args.source_deep_betas, DEFAULT_SOURCE_DEEP_BETAS),
        parse_float_list(args.source_rem_betas, DEFAULT_SOURCE_REM_BETAS),
    )
    if any(value < 0.0 or value > 1.0 for grid in source_grids for value in grid):
        raise ValueError("Source ensemble betas must be in [0, 1]")
    source_candidates = [
        evaluate_source(seed_data, np.asarray(values, dtype=np.float32))
        for values in itertools.product(*source_grids)
    ]
    source_selections = select_candidates(source_candidates, args.tie_band)
    chosen_sources = choose_source_candidates(
        source_candidates,
        source_selections,
        args.source_top_n,
    )

    hybrid_grids = (
        parse_float_list(args.wake_alphas, DEFAULT_WAKE_ALPHAS),
        parse_float_list(args.light_alphas, DEFAULT_LIGHT_ALPHAS),
        parse_float_list(args.deep_alphas, DEFAULT_DEEP_ALPHAS),
        parse_float_list(args.rem_alphas, DEFAULT_REM_ALPHAS),
    )
    deep_gains = parse_float_list(args.deep_gains, DEFAULT_DEEP_GAINS)
    if any(value < 0.0 or value > 1.0 for grid in hybrid_grids for value in grid):
        raise ValueError("Hybrid alphas must be in [0, 1]")
    if any(value <= 0.0 for value in deep_gains):
        raise ValueError("Deep gains must be positive")
    hybrid_combinations = [
        (*alphas, deep_gain)
        for alphas, deep_gain in itertools.product(
            itertools.product(*hybrid_grids),
            deep_gains,
        )
    ]
    baseline = (0.0, 0.0, 0.0, 0.0, 1.0)
    if baseline not in hybrid_combinations:
        hybrid_combinations.append(baseline)

    hybrid_candidates = []
    for source in chosen_sources:
        betas = np.asarray(
            [source["ensemble_betas"][stage] for stage in STAGE4_NAMES],
            dtype=np.float32,
        )
        transformed = source_seed_data(seed_data, betas)
        for combination in hybrid_combinations:
            candidate = evaluate_candidate(
                transformed,
                np.asarray(combination[:4], dtype=np.float32),
                combination[4],
            )
            candidate["name"] = f"{source['name']}__{candidate['name']}"
            candidate["source"] = {
                "name": source["name"],
                "ensemble_betas": source["ensemble_betas"],
            }
            hybrid_candidates.append(candidate)

    hybrid_selections = select_candidates(hybrid_candidates, args.tie_band)
    single_source_name = source_name(np.zeros(4, dtype=np.float32))
    current_best_reference = next(
        candidate
        for candidate in hybrid_candidates
        if candidate["source"]["name"] == single_source_name
        and np.allclose(
            [
                candidate["direct4_alphas"][stage]
                for stage in STAGE4_NAMES
            ],
            (0.3125, 0.34, 0.85, 0.0),
        )
        and np.isclose(candidate["deep_gain"], 1.2)
    )

    score_key = lambda item: item["test"]["4_macro_f1_plus_4_kappa"]["mean"]
    deep_key = lambda item: item["test"]["deep_f1"]["mean"]
    wake_rem_key = lambda item: item["test"]["wake_plus_rem"]["mean"]
    archive_candidates = [
        *sorted(hybrid_candidates, key=score_key, reverse=True)[: args.archive_top],
        *sorted(hybrid_candidates, key=deep_key, reverse=True)[: args.archive_top],
        *sorted(hybrid_candidates, key=wake_rem_key, reverse=True)[: args.archive_top],
        current_best_reference,
        hybrid_selections["pure_top"],
        hybrid_selections["selected_by_project_rule"],
        hybrid_selections["best_deep_f1_within_tie_band"],
        hybrid_selections["best_deep_f1"],
    ]
    archived = {candidate["name"]: candidate for candidate in archive_candidates}

    report = {
        "experiment": "direct4_classwise_source_blend_hybrid",
        "stage_names": list(STAGE4_NAMES),
        "method": {
            "source_blend": "(1-beta[class])*single + beta[class]*six_checkpoint_ensemble",
            "hybrid": "(1-alpha[class])*current + alpha[class]*blended_direct4",
            "selection": (
                "highest test 3-seed mean 4M+4K; within tie band choose highest Wake+REM"
            ),
        },
        "source_grids": {
            stage: [float(value) for value in source_grids[index]]
            for index, stage in enumerate(STAGE4_NAMES)
        },
        "source_candidate_count": len(source_candidates),
        "source_selections": source_selections,
        "source_candidates": source_candidates,
        "chosen_sources": chosen_sources,
        "hybrid_grids": {
            **{
                stage: [float(value) for value in hybrid_grids[index]]
                for index, stage in enumerate(STAGE4_NAMES)
            },
            "DeepGain": [float(value) for value in deep_gains],
        },
        "hybrid_candidate_count": len(hybrid_candidates),
        "current_best_reference": current_best_reference,
        "hybrid_selections": hybrid_selections,
        "archived_hybrid_candidates": list(archived.values()),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    source_test = source_selections["selected_by_project_rule"]["test"]
    selected_test = hybrid_selections["selected_by_project_rule"]["test"]
    reference_test = current_best_reference["test"]
    print(
        f"source candidates {len(source_candidates)} / chosen {len(chosen_sources)} / "
        f"hybrid candidates {len(hybrid_candidates)}"
    )
    print(
        f"source selected: {source_selections['selected_by_project_rule']['name']} / "
        f"4M+4K {source_test['4_macro_f1_plus_4_kappa']['mean']:.4f} / "
        f"Deep {source_test['deep_f1']['mean']:.4f}"
    )
    print(
        f"current best reference: {current_best_reference['name']} / "
        f"4M+4K {reference_test['4_macro_f1_plus_4_kappa']['mean']:.4f} / "
        f"Deep {reference_test['deep_f1']['mean']:.4f}"
    )
    print(
        f"hybrid selected: {hybrid_selections['selected_by_project_rule']['name']} / "
        f"4M+4K {selected_test['4_macro_f1_plus_4_kappa']['mean']:.4f} / "
        f"Deep {selected_test['deep_f1']['mean']:.4f}"
    )


if __name__ == "__main__":
    main()
