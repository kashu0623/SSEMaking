"""Train a recurrent classifier directly on Wake/Light/Deep/REM labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch import nn

from .labels import STAGE4_NAMES, merge_many_5_to_4
from .metrics import evaluate
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


def map_labels(labels_5: np.ndarray) -> np.ndarray:
    return np.asarray(merge_many_5_to_4(labels_5.astype(np.int64).tolist()), dtype=np.int64)


def class_weights_4(y_train: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=len(STAGE4_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "inverse":
        raise ValueError(f"Unknown class weight mode: {mode}")
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


def evaluate_loader_4(
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
    logit_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            logits = model(x_batch)["stage_logits"]
            loss = criterion(logits, y_batch)
            probabilities = torch.softmax(logits, dim=1)
        total_loss += float(loss.detach().cpu()) * y_batch.shape[0]
        total_count += y_batch.shape[0]
        true_batches.append(y_batch.detach().cpu().numpy().astype(np.int64))
        pred_batches.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
        logit_batches.append(logits.detach().cpu().numpy().astype(np.float32))
        prob_batches.append(probabilities.detach().cpu().numpy().astype(np.float32))
    y_true = np.concatenate(true_batches)
    y_pred = np.concatenate(pred_batches)
    metrics = json_ready(evaluate(y_true.tolist(), y_pred.tolist(), STAGE4_NAMES))
    return {
        "loss": total_loss / max(total_count, 1),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": np.concatenate(logit_batches),
        "probabilities": np.concatenate(prob_batches),
        "metrics": metrics,
    }


def metric_summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "4_macro_f1": float(metrics["macro_f1"]),
        "4_kappa": float(metrics["cohen_kappa"]),
        "4_macro_f1_plus_4_kappa": float(metrics["macro_f1"] + metrics["cohen_kappa"]),
        "wake_f1": float(metrics["class_wise"]["Wake"]["f1"]),
        "light_f1": float(metrics["class_wise"]["Light"]["f1"]),
        "deep_f1": float(metrics["class_wise"]["Deep"]["f1"]),
        "rem_f1": float(metrics["class_wise"]["REM"]["f1"]),
    }


def train_lstm_4class(
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
    label_smoothing: float,
) -> dict[str, Any]:
    set_seed(seed)
    arrays = load_npz(npz_path)
    x_train = arrays["X_train"].astype(np.float32)
    x_val = arrays["X_val"].astype(np.float32)
    x_test = arrays["X_test"].astype(np.float32)
    y_train = map_labels(arrays["y_train"])
    y_val = map_labels(arrays["y_val"])
    y_test = map_labels(arrays["y_test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentSleepClassifier(
        input_size=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=len(STAGE4_NAMES),
        dropout=dropout,
        model_type="lstm",
        aux_head="none",
    ).to(device)
    weights = class_weights_4(y_train, class_weight_mode)
    criterion = make_criterion(
        loss_type="cross_entropy",
        weights_for_loss=None if weights is None else weights.to(device),
        focal_gamma=0.0,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_train_loader(x_train, y_train, batch_size, sampler_mode="none")
    val_loader = make_loader(x_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size, shuffle=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "lstm4_best.pt"
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
        val_result = evaluate_loader_4(model, val_loader, criterion, device)
        val_summary = metric_summary(val_result["metrics"])
        selection_score = val_summary["4_macro_f1_plus_4_kappa"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_result["loss"]),
                "val_loss": float(val_result["loss"]),
                "val_summary": val_summary,
                "selection_score": selection_score,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} val_loss={val_result['loss']:.4f} "
            f"4_macro_f1_plus_4_kappa={selection_score:.4f}",
            flush=True,
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": x_train.shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": len(STAGE4_NAMES),
                    "dropout": dropout,
                    "model_type": "lstm",
                    "stage_names": STAGE4_NAMES,
                    "label_mode": "4_class",
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_final = evaluate_loader_4(model, val_loader, criterion, device)
    test_final = evaluate_loader_4(model, test_loader, criterion, device)
    report = {
        "experiment": "direct_4_class_lstm",
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "device": str(device),
        "stage_names": list(STAGE4_NAMES),
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
            "label_smoothing": label_smoothing,
        },
        "class_counts": np.bincount(y_train, minlength=len(STAGE4_NAMES)).tolist(),
        "class_weights": None if weights is None else weights.tolist(),
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "history": history,
        "final_val": {"loss": val_final["loss"], "metrics": val_final["metrics"], "summary": metric_summary(val_final["metrics"])},
        "final_test": {"loss": test_final["loss"], "metrics": test_final["metrics"], "summary": metric_summary(test_final["metrics"])},
    }
    (out_dir / "lstm4_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    prediction_arrays: dict[str, np.ndarray] = {"stage4_names": np.asarray(STAGE4_NAMES)}
    for split, result in (("val", val_final), ("test", test_final)):
        prediction_arrays[f"{split}_y_true"] = result["y_true"]
        prediction_arrays[f"{split}_y_pred"] = result["y_pred"]
        prediction_arrays[f"{split}_logits"] = result["logits"]
        prediction_arrays[f"{split}_probs"] = result["probabilities"]
        for suffix in ("subject_ids", "epoch_indices"):
            source_key = f"{split}_{suffix}"
            if source_key in arrays:
                prediction_arrays[source_key] = arrays[source_key]
    np.savez_compressed(out_dir / "lstm4_predictions.npz", **prediction_arrays)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a direct Wake/Light/Deep/REM LSTM.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight-mode", choices=("inverse", "sqrt", "none"), default="inverse")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    args = parser.parse_args()
    report = train_lstm_4class(
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
        args.label_smoothing,
    )
    print(json.dumps({"best_epoch": report["best_epoch"], "final_test": report["final_test"]["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
