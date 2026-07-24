"""Evaluate direct four-class models and mapped current-best fusion weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import numpy as np

from .evaluate_four_model_fusion import four_model_classwise_fusion
from .labels import STAGE4_NAMES
from .metrics import evaluate


ROLES = ("original_4class", "full_w20_4class", "capacity_h128_4class", "h128_ls003_4class")
FUSION_NAME = "direct4_current_mapped_weight_fusion"


def load_split(path: Path, split: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        result = {
            "y_true": data[f"{split}_y_true"].astype(np.int64),
            "probs": data[f"{split}_probs"].astype(np.float32),
        }
        for suffix in ("subject_ids", "epoch_indices"):
            key = f"{split}_{suffix}"
            if key in data.files:
                result[suffix] = data[key]
    if result["probs"].shape[1] != len(STAGE4_NAMES):
        raise ValueError(f"Expected four probabilities in {path}, got {result['probs'].shape}")
    return result


def validate(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], role: str) -> None:
    if not np.array_equal(reference["y_true"], candidate["y_true"]):
        raise ValueError(f"Unaligned labels for {role}")
    if reference["probs"].shape != candidate["probs"].shape:
        raise ValueError(f"Unaligned probabilities for {role}")
    for key in ("subject_ids", "epoch_indices"):
        if key in reference and key in candidate and not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"Unaligned {key} for {role}")


def summarize(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1).astype(np.int64)
    metrics = asdict(evaluate(y_true.tolist(), prediction.tolist(), STAGE4_NAMES))
    return {
        "4_macro_f1": float(metrics["macro_f1"]),
        "4_kappa": float(metrics["cohen_kappa"]),
        "4_macro_f1_plus_4_kappa": float(metrics["macro_f1"] + metrics["cohen_kappa"]),
        "wake_f1": float(metrics["class_wise"]["Wake"]["f1"]),
        "light_f1": float(metrics["class_wise"]["Light"]["f1"]),
        "deep_f1": float(metrics["class_wise"]["Deep"]["f1"]),
        "deep_precision": float(metrics["class_wise"]["Deep"]["precision"]),
        "deep_recall": float(metrics["class_wise"]["Deep"]["recall"]),
        "deep_support": int(metrics["class_wise"]["Deep"]["support"]),
        "rem_f1": float(metrics["class_wise"]["REM"]["f1"]),
        "confusion_matrix": metrics["confusion_matrix"],
    }


def evaluate_seed(paths: dict[str, Path], seed: str) -> dict[str, Any]:
    splits = {split: {role: load_split(path, split) for role, path in paths.items()} for split in ("val", "test")}
    primary = np.asarray((0.72, 0.80, 0.82, 0.00), dtype=np.float32)
    secondary = np.asarray((0.06, 0.02, 0.00, 0.42), dtype=np.float32)
    tertiary = np.asarray((0.00, 0.15, 0.18, 0.13), dtype=np.float32)
    candidates: dict[str, Any] = {}
    for split in ("val", "test"):
        reference = splits[split][ROLES[0]]
        for role in ROLES[1:]:
            validate(reference, splits[split][role], role)
        for role in ROLES:
            candidates.setdefault(role, {})[split] = summarize(reference["y_true"], splits[split][role]["probs"])
        fused = four_model_classwise_fusion(
            splits[split][ROLES[0]]["probs"],
            splits[split][ROLES[1]]["probs"],
            splits[split][ROLES[2]]["probs"],
            splits[split][ROLES[3]]["probs"],
            primary,
            secondary,
            tertiary,
        )
        candidates.setdefault(FUSION_NAME, {})[split] = summarize(reference["y_true"], fused)
    return {"seed": seed, "candidates": candidates}


def mean_std(values: Sequence[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": pstdev(values)}


def aggregate(seed_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "4_macro_f1",
        "4_kappa",
        "4_macro_f1_plus_4_kappa",
        "wake_f1",
        "light_f1",
        "deep_f1",
        "deep_precision",
        "deep_recall",
        "rem_f1",
    )
    names = seed_reports[0]["candidates"].keys()
    return {
        name: {
            split: {
                **{
                    field: mean_std(
                        [report["candidates"][name][split][field] for report in seed_reports]
                    )
                    for field in fields
                },
                "pooled_confusion_matrix": np.sum(
                    [
                        np.asarray(
                            report["candidates"][name][split]["confusion_matrix"],
                            dtype=np.int64,
                        )
                        for report in seed_reports
                    ],
                    axis=0,
                ).tolist(),
                "pooled_deep_support": sum(
                    int(report["candidates"][name][split]["deep_support"])
                    for report in seed_reports
                ),
            }
            for split in ("val", "test")
        }
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate direct four-class four-model fusion.")
    for role in ROLES:
        parser.add_argument(f"--{role.replace('_', '-')}-predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--seed-labels", nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    lists = {role: getattr(args, f"{role}_predictions") for role in ROLES}
    count = len(args.seed_labels)
    if any(len(paths) != count for paths in lists.values()):
        raise ValueError("Every role must provide one prediction path per seed")
    seed_reports = [
        evaluate_seed({role: lists[role][index] for role in ROLES}, seed)
        for index, seed in enumerate(args.seed_labels)
    ]
    report = {
        "experiment": "direct_four_class_four_model_fusion",
        "stage_names": list(STAGE4_NAMES),
        "seed_reports": seed_reports,
        "summary": aggregate(seed_reports),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, candidate in report["summary"].items():
        test = candidate["test"]
        print(
            f"{name}: 4M {test['4_macro_f1']['mean']:.4f} / 4K {test['4_kappa']['mean']:.4f} / "
            f"Wake {test['wake_f1']['mean']:.4f} / Light {test['light_f1']['mean']:.4f} / "
            f"Deep {test['deep_f1']['mean']:.4f} "
            f"(P {test['deep_precision']['mean']:.4f} / R {test['deep_recall']['mean']:.4f}) / "
            f"REM {test['rem_f1']['mean']:.4f}"
        )


if __name__ == "__main__":
    main()
