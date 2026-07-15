"""Train one-vs-rest recurrent specialists for individual sleep stages."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .labels import STAGE5_NAMES, STAGE5_TO_ID
from .metrics import evaluate
from .train_lstm import RecurrentSleepClassifier, load_npz


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def binary_targets(y_5: np.ndarray, target_stage: str) -> np.ndarray:
    return (y_5.astype(np.int64) == STAGE5_TO_ID[target_stage]).astype(np.int64)


def binary_class_names(target_stage: str) -> tuple[str, str]:
    return (f"not_{target_stage}", target_stage)


def make_loader(
    x: np.ndarray,
    y_binary: np.ndarray,
    batch_size: int,
    shuffle: bool,
    sampler_mode: str = "none",
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y_binary).long())
    if not shuffle:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    if sampler_mode == "none":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    if sampler_mode != "weighted":
        raise ValueError(f"Unknown train_sampler: {sampler_mode}")
    counts = np.bincount(y_binary.astype(np.int64), minlength=2).astype(np.float32)
    sample_weights = 1.0 / np.maximum(counts[y_binary.astype(np.int64)], 1.0)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(y_binary.shape[0]),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def binary_class_weights(y_binary: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y_binary.astype(np.int64), minlength=2).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "inverse":
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def validation_score(metrics: dict[str, Any], loss: float, selection_metric: str) -> float:
    if selection_metric == "positive_f1":
        return float(metrics["class_wise"][metrics["class_names"][1]]["f1"])
    if selection_metric == "macro_f1":
        return float(metrics["macro_f1"])
    if selection_metric == "kappa":
        return float(metrics["cohen_kappa"])
    if selection_metric == "macro_f1_plus_kappa":
        return float(metrics["macro_f1"]) + float(metrics["cohen_kappa"])
    if selection_metric == "negative_loss":
        return -float(loss)
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x_batch)["stage_logits"]
            loss = criterion(logits, y_batch)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = y_batch.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())

    return {
        "loss": total_loss / max(total_count, 1),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    target_stage: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            logits = model(x_batch)["stage_logits"]
            loss = criterion(logits, y_batch)
            probabilities = torch.softmax(logits, dim=1)
        batch_size = y_batch.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        logits_batches.append(logits.detach().cpu().numpy().astype(np.float32))
        prob_batches.append(probabilities.detach().cpu().numpy().astype(np.float32))

    logits_array = (
        np.concatenate(logits_batches, axis=0)
        if logits_batches
        else np.empty((0, 2), dtype=np.float32)
    )
    prob_array = (
        np.concatenate(prob_batches, axis=0)
        if prob_batches
        else np.empty((0, 2), dtype=np.float32)
    )
    return {
        "loss": total_loss / max(total_count, 1),
        "metrics": json_ready(evaluate(y_true, y_pred, binary_class_names(target_stage))),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits_array,
        "probabilities": prob_array,
    }


def train_binary_specialist(
    npz_path: Path,
    out_dir: Path,
    target_stage: str,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    seed: int,
    model_type: str,
    class_weight_mode: str,
    train_sampler: str,
    selection_metric: str,
) -> dict[str, Any]:
    if target_stage not in STAGE5_TO_ID:
        raise ValueError(f"Unknown target_stage: {target_stage}")
    set_seed(seed)
    arrays = load_npz(npz_path)
    x_train = arrays["X_train"].astype(np.float32)
    y_train_5 = arrays["y_train"].astype(np.int64)
    x_val = arrays["X_val"].astype(np.float32)
    y_val_5 = arrays["y_val"].astype(np.int64)
    x_test = arrays["X_test"].astype(np.float32)
    y_test_5 = arrays["y_test"].astype(np.int64)
    y_train = binary_targets(y_train_5, target_stage)
    y_val = binary_targets(y_val_5, target_stage)
    y_test = binary_targets(y_test_5, target_stage)
    feature_names = arrays["feature_names"].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentSleepClassifier(
        input_size=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout=dropout,
        model_type=model_type,
        aux_head="none",
    ).to(device)
    weights = binary_class_weights(y_train, mode=class_weight_mode)
    weights_for_loss = weights.to(device) if weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights_for_loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(x_train, y_train, batch_size, shuffle=True, sampler_mode=train_sampler)
    train_eval_loader = make_loader(x_train, y_train, batch_size, shuffle=False)
    val_loader = make_loader(x_val, y_val, batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "specialist_best.pt"
    history: list[dict[str, Any]] = []
    best_selection_score = -float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        train_metrics = json_ready(evaluate(train_result["y_true"], train_result["y_pred"], binary_class_names(target_stage)))
        val_result = evaluate_model(model, val_loader, criterion, device, target_stage)
        score = validation_score(val_result["metrics"], val_result["loss"], selection_metric)
        record = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_metrics": train_metrics,
            "val_loss": val_result["loss"],
            "val_metrics": val_result["metrics"],
            "selection_metric": selection_metric,
            "selection_score": score,
        }
        history.append(record)
        print(
            f"target={target_stage} epoch={epoch} train_loss={train_result['loss']:.4f} "
            f"val_loss={val_result['loss']:.4f} {selection_metric}={score:.4f}",
            flush=True,
        )
        if score > best_selection_score:
            best_selection_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": x_train.shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "num_classes": 2,
                    "dropout": dropout,
                    "model_type": model_type,
                    "target_stage": target_stage,
                    "class_weight_mode": class_weight_mode,
                    "train_sampler": train_sampler,
                    "selection_metric": selection_metric,
                    "feature_names": feature_names,
                    "stage5_names": STAGE5_NAMES,
                    "binary_class_names": binary_class_names(target_stage),
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early_stopping target={target_stage} epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    train_final = evaluate_model(model, train_eval_loader, criterion, device, target_stage)
    val_final = evaluate_model(model, val_loader, criterion, device, target_stage)
    test_final = evaluate_model(model, test_loader, criterion, device, target_stage)

    report = {
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "best_model_path": str(best_model_path),
        "device": str(device),
        "seed": seed,
        "target_stage": target_stage,
        "target_stage_id": STAGE5_TO_ID[target_stage],
        "binary_class_names": list(binary_class_names(target_stage)),
        "hyperparameters": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "model_type": model_type,
            "class_weight_mode": class_weight_mode,
            "train_sampler": train_sampler,
            "selection_metric": selection_metric,
        },
        "array_shapes": {
            "X_train": list(x_train.shape),
            "X_val": list(x_val.shape),
            "X_test": list(x_test.shape),
        },
        "feature_names": feature_names,
        "stage5_names": list(STAGE5_NAMES),
        "class_weights": None if weights is None else weights.detach().cpu().numpy().tolist(),
        "best_epoch": best_epoch,
        "best_selection_metric": selection_metric,
        "best_selection_score": best_selection_score,
        "history": history,
        "final_train": {
            "loss": train_final["loss"],
            "metrics": train_final["metrics"],
        },
        "final_val": {
            "loss": val_final["loss"],
            "metrics": val_final["metrics"],
        },
        "final_test": {
            "loss": test_final["loss"],
            "metrics": test_final["metrics"],
        },
    }
    (out_dir / "specialist_metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    prediction_arrays: dict[str, np.ndarray] = {
        "target_stage": np.asarray(target_stage),
        "target_stage_id": np.asarray(STAGE5_TO_ID[target_stage], dtype=np.int64),
        "stage5_names": np.asarray(STAGE5_NAMES),
        "binary_class_names": np.asarray(binary_class_names(target_stage)),
        "train_y_true": y_train_5.astype(np.int64),
        "train_y_binary": np.asarray(train_final["y_true"], dtype=np.int64),
        "train_y_pred_binary": np.asarray(train_final["y_pred"], dtype=np.int64),
        "train_logits": train_final["logits"],
        "train_probs": train_final["probabilities"][:, 1].astype(np.float32),
        "val_y_true": y_val_5.astype(np.int64),
        "val_y_binary": np.asarray(val_final["y_true"], dtype=np.int64),
        "val_y_pred_binary": np.asarray(val_final["y_pred"], dtype=np.int64),
        "val_logits": val_final["logits"],
        "val_probs": val_final["probabilities"][:, 1].astype(np.float32),
        "test_y_true": y_test_5.astype(np.int64),
        "test_y_binary": np.asarray(test_final["y_true"], dtype=np.int64),
        "test_y_pred_binary": np.asarray(test_final["y_pred"], dtype=np.int64),
        "test_logits": test_final["logits"],
        "test_probs": test_final["probabilities"][:, 1].astype(np.float32),
    }
    for split_name in ("train", "val", "test"):
        for suffix in ("subject_ids", "epoch_indices"):
            key = f"{split_name}_{suffix}"
            if key in arrays:
                prediction_arrays[key] = arrays[key]
    np.savez_compressed(out_dir / "specialist_predictions.npz", **prediction_arrays)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a one-vs-rest recurrent sleep-stage specialist.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-stage", choices=STAGE5_NAMES, required=True)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--class-weight-mode", choices=("inverse", "sqrt", "none"), default="inverse")
    parser.add_argument("--train-sampler", choices=("none", "weighted"), default="none")
    parser.add_argument(
        "--selection-metric",
        choices=("positive_f1", "macro_f1", "kappa", "macro_f1_plus_kappa", "negative_loss"),
        default="positive_f1",
    )
    args = parser.parse_args()

    train_binary_specialist(
        npz_path=args.npz,
        out_dir=args.out_dir,
        target_stage=args.target_stage,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        model_type=args.model_type,
        class_weight_mode=args.class_weight_mode,
        train_sampler=args.train_sampler,
        selection_metric=args.selection_metric,
    )


if __name__ == "__main__":
    main()
