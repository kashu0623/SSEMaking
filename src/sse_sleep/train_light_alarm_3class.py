"""Train an app-oriented Other/Light/Deep recurrent classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .labels import STAGE4_NAMES, STAGE4_TO_ID
from .train_light_alarm import binary_metrics, map_labels_4, select_threshold
from .train_lstm import (
    RecurrentSleepClassifier,
    json_ready,
    load_npz,
    make_criterion,
    set_seed,
)


ALARM3_NAMES = ("Other", "Light", "Deep")
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.arange(0.10, 0.901, 0.025))


def map_alarm3(labels_4: np.ndarray) -> np.ndarray:
    labels = np.zeros(labels_4.shape, dtype=np.int64)
    labels[labels_4 == STAGE4_TO_ID["Light"]] = 1
    labels[labels_4 == STAGE4_TO_ID["Deep"]] = 2
    return labels


def map_alarm3_tensor(labels_4: torch.Tensor) -> torch.Tensor:
    labels = torch.zeros_like(labels_4)
    labels[labels_4 == STAGE4_TO_ID["Light"]] = 1
    labels[labels_4 == STAGE4_TO_ID["Deep"]] = 2
    return labels


def alarm3_class_weights(
    labels_3: np.ndarray,
    mode: str,
    deep_multiplier: float,
) -> torch.Tensor | None:
    if deep_multiplier <= 0:
        raise ValueError("Deep class multiplier must be positive")
    if mode == "none":
        if deep_multiplier != 1.0:
            weights = np.ones(len(ALARM3_NAMES), dtype=np.float32)
        else:
            return None
    else:
        counts = np.bincount(labels_3, minlength=len(ALARM3_NAMES)).astype(np.float32)
        weights = counts.sum() / np.maximum(counts, 1.0)
        if mode == "sqrt":
            weights = np.sqrt(weights)
        elif mode != "inverse":
            raise ValueError(f"Unknown class weight mode: {mode}")
    weights[2] *= deep_multiplier
    return torch.as_tensor(weights / weights.mean(), dtype=torch.float32)


def make_loader(
    features: np.ndarray,
    labels_4: np.ndarray,
    batch_size: int,
    sampler_mode: str,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels_4).long(),
    )
    if sampler_mode == "none":
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
    if sampler_mode != "weighted":
        raise ValueError(f"Unknown sampler mode: {sampler_mode}")
    labels_3 = map_alarm3(labels_4)
    counts = np.bincount(labels_3, minlength=len(ALARM3_NAMES)).astype(np.float32)
    sample_weights = 1.0 / np.maximum(counts[labels_3], 1.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(labels_4.shape[0]),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for features, labels_4 in loader:
        features = features.to(device)
        targets = map_alarm3_tensor(labels_4).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)["stage_logits"]
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        count = int(labels_4.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
    return total_loss / max(total_count, 1)


def infer(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    labels_4_batches: list[np.ndarray] = []
    logit_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for features, labels_4 in loader:
        features = features.to(device)
        targets = map_alarm3_tensor(labels_4).to(device)
        with torch.no_grad():
            logits = model(features)["stage_logits"]
            loss = criterion(logits, targets)
            probabilities = torch.softmax(logits, dim=1)
        count = int(labels_4.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        labels_4_batches.append(labels_4.numpy().astype(np.int64))
        logit_batches.append(logits.detach().cpu().numpy().astype(np.float32))
        probability_batches.append(
            probabilities.detach().cpu().numpy().astype(np.float32)
        )
    alarm3_probabilities = np.concatenate(probability_batches)
    return {
        "loss": total_loss / max(total_count, 1),
        "y_true4": np.concatenate(labels_4_batches),
        "alarm3_logits": np.concatenate(logit_batches),
        "alarm3_probs": alarm3_probabilities,
        "light_probs": alarm3_probabilities[:, 1],
    }


def train_light_alarm_3class(
    npz_path: Path,
    out_dir: Path,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    model_type: str,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    class_weight_mode: str,
    deep_class_multiplier: float,
    train_sampler: str,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    set_seed(seed)
    arrays = load_npz(npz_path)
    features = {
        split: arrays[f"X_{split}"].astype(np.float32)
        for split in ("train", "val", "test")
    }
    labels_4 = {
        split: map_labels_4(arrays[f"y_{split}"])
        for split in ("train", "val", "test")
    }
    labels_3_train = map_alarm3(labels_4["train"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentSleepClassifier(
        input_size=features["train"].shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(ALARM3_NAMES),
        dropout=dropout,
        model_type=model_type,
        aux_head="none",
    ).to(device)
    weights = alarm3_class_weights(
        labels_3_train,
        class_weight_mode,
        deep_class_multiplier,
    )
    criterion = make_criterion(
        loss_type=loss_type,
        weights_for_loss=None if weights is None else weights.to(device),
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    loaders = {
        split: make_loader(
            features[split],
            labels_4[split],
            batch_size,
            train_sampler if split == "train" else "none",
            seed,
            shuffle=split == "train" and train_sampler == "none",
        )
        for split in ("train", "val", "test")
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "light_alarm_3class_best.pt"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = -1
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = run_training_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
        )
        val_result = infer(model, loaders["val"], criterion, device)
        threshold_result = select_threshold(
            val_result["y_true4"],
            val_result["light_probs"],
            thresholds,
        )
        selection_key = (
            float(threshold_result["light_objective"]),
            -float(threshold_result["deep_to_light_rate"]),
            float(threshold_result["light_precision"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(val_result["loss"]),
                "val_selection": threshold_result,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"light_objective={threshold_result['light_objective']:.4f} "
            f"threshold={threshold_result['threshold']:.3f} "
            f"deep_to_light={threshold_result['deep_to_light_rate']:.4f}",
            flush=True,
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": features["train"].shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": len(ALARM3_NAMES),
                    "dropout": dropout,
                    "model_type": model_type,
                    "stage_names": ALARM3_NAMES,
                    "label_mode": "other_light_deep",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final = {
        split: infer(model, loaders[split], criterion, device)
        for split in ("val", "test")
    }
    selected_threshold = select_threshold(
        final["val"]["y_true4"],
        final["val"]["light_probs"],
        thresholds,
    )["threshold"]
    final_metrics = {
        split: binary_metrics(
            final[split]["y_true4"],
            final[split]["light_probs"],
            selected_threshold,
        )
        for split in ("val", "test")
    }
    report = {
        "experiment": "light_alarm_3class",
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "device": str(device),
        "alarm3_names": list(ALARM3_NAMES),
        "stage4_names": list(STAGE4_NAMES),
        "hyperparameters": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "model_type": model_type,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "class_weight_mode": class_weight_mode,
            "deep_class_multiplier": deep_class_multiplier,
            "train_sampler": train_sampler,
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
        },
        "train_stage4_counts": np.bincount(
            labels_4["train"],
            minlength=len(STAGE4_NAMES),
        ).tolist(),
        "train_alarm3_counts": np.bincount(
            labels_3_train,
            minlength=len(ALARM3_NAMES),
        ).tolist(),
        "alarm3_class_weights": None if weights is None else weights.tolist(),
        "best_epoch": best_epoch,
        "selected_validation_threshold": float(selected_threshold),
        "history": history,
        "final": {
            split: {
                "loss": float(final[split]["loss"]),
                "metrics": final_metrics[split],
            }
            for split in ("val", "test")
        },
    }
    (out_dir / "light_alarm_3class_metrics.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prediction_arrays: dict[str, np.ndarray] = {
        "alarm3_names": np.asarray(ALARM3_NAMES),
        "stage4_names": np.asarray(STAGE4_NAMES),
    }
    for split in ("val", "test"):
        prediction_arrays[f"{split}_y_true4"] = final[split]["y_true4"]
        prediction_arrays[f"{split}_light_probs"] = final[split]["light_probs"]
        prediction_arrays[f"{split}_alarm3_logits"] = final[split]["alarm3_logits"]
        prediction_arrays[f"{split}_alarm3_probs"] = final[split]["alarm3_probs"]
        for suffix in ("subject_ids", "epoch_indices"):
            source_key = f"{split}_{suffix}"
            if source_key in arrays:
                prediction_arrays[source_key] = arrays[source_key]
    np.savez_compressed(
        out_dir / "light_alarm_3class_predictions.npz",
        **prediction_arrays,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an Other(Wake+REM)/Light/Deep alarm classifier."
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--model-type", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight-mode",
        choices=("inverse", "sqrt", "none"),
        default="inverse",
    )
    parser.add_argument("--deep-class-multiplier", type=float, default=1.0)
    parser.add_argument("--train-sampler", choices=("none", "weighted"), default="none")
    parser.add_argument(
        "--loss-type",
        choices=("cross_entropy", "focal"),
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()
    report = train_light_alarm_3class(
        args.npz,
        args.out_dir,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.model_type,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        args.seed,
        args.class_weight_mode,
        args.deep_class_multiplier,
        args.train_sampler,
        args.loss_type,
        args.focal_gamma,
        args.label_smoothing,
    )
    print(
        json.dumps(
            {
                "best_epoch": report["best_epoch"],
                "final_test": report["final"]["test"]["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
