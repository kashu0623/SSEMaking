"""Train a binary specialist that separates Light sleep from Deep sleep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .labels import merge_many_5_to_4
from .train_lstm import (
    RecurrentSleepClassifier,
    json_ready,
    load_npz,
    make_criterion,
    make_loader,
    make_train_loader,
    run_epoch,
    set_seed,
)


BINARY_NAMES = ("Light", "Deep")


def map_labels(labels_5: np.ndarray) -> np.ndarray:
    return np.asarray(
        merge_many_5_to_4(labels_5.astype(np.int64).tolist()),
        dtype=np.int64,
    )


def light_deep_subset(
    features: np.ndarray,
    labels_4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = (labels_4 == 1) | (labels_4 == 2)
    return features[mask], (labels_4[mask] == 2).astype(np.int64)


def binary_class_weights(y_train: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "inverse":
        raise ValueError(f"Unknown class weight mode: {mode}")
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    total = int(confusion.sum())
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
    accuracy = float(true_positives.sum() / max(total, 1))
    expected = (
        float(np.dot(row_totals, column_totals) / (total * total))
        if total
        else 0.0
    )
    kappa = (
        (accuracy - expected) / (1.0 - expected)
        if expected < 1.0
        else 0.0
    )
    macro_f1 = float(f1.mean())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "macro_f1_plus_kappa": macro_f1 + kappa,
        "deep_precision": float(precision[1]),
        "deep_recall": float(recall[1]),
        "deep_f1": float(f1[1]),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate_binary_loader(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            logits = model(x_batch)["stage_logits"]
            loss = criterion(logits, y_batch)
        total_loss += float(loss.detach().cpu()) * y_batch.shape[0]
        total_count += y_batch.shape[0]
        true_batches.append(y_batch.detach().cpu().numpy().astype(np.int64))
        pred_batches.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(pred_batches)
    confusion = np.bincount(
        2 * y_true + y_pred,
        minlength=4,
    ).reshape(2, 2)
    return {
        "loss": total_loss / max(total_count, 1),
        "metrics": metrics_from_confusion(confusion),
    }


def infer_full(
    model: nn.Module,
    features: np.ndarray,
    labels_4: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = make_loader(features, labels_4, batch_size, shuffle=False)
    model.eval()
    logit_batches: list[np.ndarray] = []
    for x_batch, _ in loader:
        with torch.no_grad():
            logits = model(x_batch.to(device))["stage_logits"]
        logit_batches.append(logits.detach().cpu().numpy().astype(np.float32))
    all_logits = np.concatenate(logit_batches)
    shifted = all_logits - all_logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    return {
        "y_true": labels_4,
        "y_pred": probabilities.argmax(axis=1).astype(np.int64),
        "logits": all_logits,
        "probabilities": probabilities.astype(np.float32),
    }


def train_light_deep_specialist(
    npz_path: Path,
    out_dir: Path,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    class_weight_mode: str,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
) -> dict[str, Any]:
    set_seed(seed)
    arrays = load_npz(npz_path)
    features = {
        split: arrays[f"X_{split}"].astype(np.float32)
        for split in ("train", "val", "test")
    }
    labels_4 = {
        split: map_labels(arrays[f"y_{split}"])
        for split in ("train", "val", "test")
    }
    subsets = {
        split: light_deep_subset(features[split], labels_4[split])
        for split in ("train", "val", "test")
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentSleepClassifier(
        input_size=features["train"].shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        model_type="lstm",
        aux_head="none",
    ).to(device)
    weights = binary_class_weights(subsets["train"][1], class_weight_mode)
    criterion = make_criterion(
        loss_type=loss_type,
        weights_for_loss=None if weights is None else weights.to(device),
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    train_loader = make_train_loader(
        *subsets["train"],
        batch_size,
        sampler_mode="none",
    )
    val_loader = make_loader(*subsets["val"], batch_size, shuffle=False)
    test_loader = make_loader(*subsets["test"], batch_size, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "light_deep_best.pt"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            stage_criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_result = evaluate_binary_loader(model, val_loader, criterion, device)
        selection_score = float(val_result["metrics"]["macro_f1_plus_kappa"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_result["loss"]),
                "val_loss": float(val_result["loss"]),
                "val_metrics": val_result["metrics"],
                "selection_score": selection_score,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"ld_macro_f1_plus_kappa={selection_score:.4f}",
            flush=True,
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": features["train"].shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": 2,
                    "dropout": dropout,
                    "model_type": "lstm",
                    "stage_names": BINARY_NAMES,
                    "label_mode": "light_vs_deep",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_final = evaluate_binary_loader(model, val_loader, criterion, device)
    test_final = evaluate_binary_loader(model, test_loader, criterion, device)
    full_predictions = {
        split: infer_full(
            model,
            features[split],
            labels_4[split],
            batch_size,
            device,
        )
        for split in ("val", "test")
    }

    report = {
        "experiment": "light_deep_binary_specialist",
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "device": str(device),
        "stage_names": list(BINARY_NAMES),
        "hyperparameters": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "class_weight_mode": class_weight_mode,
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
        },
        "class_counts": np.bincount(subsets["train"][1], minlength=2).tolist(),
        "class_weights": None if weights is None else weights.tolist(),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "history": history,
        "final_val": val_final,
        "final_test": test_final,
    }
    (out_dir / "light_deep_metrics.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    prediction_arrays: dict[str, np.ndarray] = {
        "specialist_names": np.asarray(BINARY_NAMES),
    }
    for split, result in full_predictions.items():
        prediction_arrays[f"{split}_y_true"] = result["y_true"]
        prediction_arrays[f"{split}_y_pred"] = result["y_pred"]
        prediction_arrays[f"{split}_logits"] = result["logits"]
        prediction_arrays[f"{split}_probs"] = result["probabilities"]
        for suffix in ("subject_ids", "epoch_indices"):
            source_key = f"{split}_{suffix}"
            if source_key in arrays:
                prediction_arrays[source_key] = arrays[source_key]
    np.savez_compressed(
        out_dir / "light_deep_predictions.npz",
        **prediction_arrays,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a binary recurrent Light-vs-Deep specialist."
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-weight-mode",
        choices=("inverse", "sqrt", "none"),
        default="inverse",
    )
    parser.add_argument(
        "--loss-type",
        choices=("cross_entropy", "focal"),
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()
    report = train_light_deep_specialist(
        args.npz,
        args.out_dir,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        args.seed,
        args.class_weight_mode,
        args.loss_type,
        args.focal_gamma,
        args.label_smoothing,
    )
    print(
        json.dumps(
            {
                "best_epoch": report["best_epoch"],
                "final_test": report["final_test"]["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
