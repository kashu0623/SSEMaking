"""Train and evaluate a recurrent sleep-stage classifier from a DreamT NPZ dataset."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .labels import STAGE5_NAMES
from .metrics import evaluate_5_and_4


class RecurrentSleepClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 5,
        dropout: float = 0.0,
        model_type: str = "lstm",
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        if model_type == "lstm":
            recurrent_cls = nn.LSTM
        elif model_type == "gru":
            recurrent_cls = nn.GRU
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.recurrent(x)
        latest = output[:, -1, :]
        return self.head(latest)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_train_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    sampler_mode: str,
) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    if sampler_mode == "none":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    if sampler_mode != "weighted":
        raise ValueError(f"Unknown train_sampler: {sampler_mode}")

    counts = np.bincount(y.astype(np.int64), minlength=len(STAGE5_NAMES)).astype(np.float32)
    sample_weights = 1.0 / np.maximum(counts[y.astype(np.int64)], 1.0)
    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(y.shape[0]),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def class_weights(
    y_train: np.ndarray,
    num_classes: int,
    mode: str,
    n3_weight_multiplier: float = 1.0,
) -> torch.Tensor | None:
    """Return class weights for cross entropy, or None for unweighted loss."""
    if n3_weight_multiplier <= 0:
        raise ValueError(f"n3_weight_multiplier must be positive: {n3_weight_multiplier}")
    if mode == "none":
        return None
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "inverse":
        raise ValueError(f"Unknown class_weight_mode: {mode}")
    weights = weights / weights.mean()
    if n3_weight_multiplier != 1.0:
        n3_index = STAGE5_NAMES.index("N3")
        weights[n3_index] *= n3_weight_multiplier
        weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"focal gamma must be non-negative: {gamma}")
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        target_log_probability = F.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1)
        target_probability = target_log_probability.exp()
        focal_factor = (1.0 - target_probability).pow(self.gamma)
        return (focal_factor * ce_loss).mean()


def make_criterion(
    loss_type: str,
    weights_for_loss: torch.Tensor | None,
    focal_gamma: float,
    label_smoothing: float,
) -> nn.Module:
    if not 0 <= label_smoothing < 1:
        raise ValueError(f"label_smoothing must be in [0, 1): {label_smoothing}")
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weights_for_loss, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLoss(weight=weights_for_loss, gamma=focal_gamma, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss_type: {loss_type}")


def validation_score(metrics: dict[str, Any], selection_metric: str) -> float:
    metric_map = {
        "5_macro_f1": ("5_class", "macro_f1"),
        "4_macro_f1": ("4_class", "macro_f1"),
        "5_kappa": ("5_class", "cohen_kappa"),
        "4_kappa": ("4_class", "cohen_kappa"),
    }
    if selection_metric in metric_map:
        group, metric_name = metric_map[selection_metric]
        return float(metrics[group][metric_name])
    if selection_metric == "5_macro_f1_plus_4_kappa":
        return float(metrics["5_class"]["macro_f1"]) + float(metrics["4_class"]["cohen_kappa"])
    if selection_metric == "4_macro_f1_plus_4_kappa":
        return float(metrics["4_class"]["macro_f1"]) + float(metrics["4_class"]["cohen_kappa"])
    raise ValueError(f"Unknown selection_metric: {selection_metric}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, list[int], list[int]]:
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
            logits = model(x_batch)
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

    return total_loss / max(total_count, 1), y_true, y_pred


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


def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, Any]:
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
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            probabilities = torch.softmax(logits, dim=1)

        batch_size = y_batch.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        logits_batches.append(logits.detach().cpu().numpy().astype(np.float32))
        prob_batches.append(probabilities.detach().cpu().numpy().astype(np.float32))

    loss = total_loss / max(total_count, 1)
    metrics = evaluate_5_and_4(y_true, y_pred)
    logits_array = (
        np.concatenate(logits_batches, axis=0)
        if logits_batches
        else np.empty((0, len(STAGE5_NAMES)), dtype=np.float32)
    )
    prob_array = (
        np.concatenate(prob_batches, axis=0)
        if prob_batches
        else np.empty((0, len(STAGE5_NAMES)), dtype=np.float32)
    )
    return {
        "loss": loss,
        "metrics": json_ready(metrics),
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits_array,
        "probabilities": prob_array,
    }


def train_lstm(
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
    n3_weight_multiplier: float,
    model_type: str,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    train_sampler: str,
    selection_metric: str,
) -> dict[str, Any]:
    set_seed(seed)
    arrays = load_npz(npz_path)
    x_train = arrays["X_train"].astype(np.float32)
    y_train = arrays["y_train"].astype(np.int64)
    x_val = arrays["X_val"].astype(np.float32)
    y_val = arrays["y_val"].astype(np.int64)
    x_test = arrays["X_test"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.int64)
    feature_names = arrays["feature_names"].astype(str).tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentSleepClassifier(
        input_size=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type,
    ).to(device)

    weights = class_weights(
        y_train,
        len(STAGE5_NAMES),
        mode=class_weight_mode,
        n3_weight_multiplier=n3_weight_multiplier,
    )
    weights_for_loss = weights.to(device) if weights is not None else None
    criterion = make_criterion(
        loss_type=loss_type,
        weights_for_loss=weights_for_loss,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_train_loader(x_train, y_train, batch_size=batch_size, sampler_mode=train_sampler)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "lstm_best.pt"
    history: list[dict[str, Any]] = []
    best_selection_score = -float("inf")
    best_val_5_macro_f1 = -1.0
    best_val_4_macro_f1 = -1.0
    best_val_5_kappa = -1.0
    best_val_4_kappa = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        train_metrics = evaluate_5_and_4(train_true, train_pred)
        val_result = evaluate_loader(model=model, loader=val_loader, criterion=criterion, device=device)
        selection_score = validation_score(val_result["metrics"], selection_metric)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": json_ready(train_metrics),
            "val_loss": val_result["loss"],
            "val_metrics": val_result["metrics"],
            "selection_metric": selection_metric,
            "selection_score": selection_score,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_result['loss']:.4f} "
            f"{selection_metric}={selection_score:.4f}",
            flush=True,
        )

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_val_5_macro_f1 = float(val_result["metrics"]["5_class"]["macro_f1"])
            best_val_4_macro_f1 = float(val_result["metrics"]["4_class"]["macro_f1"])
            best_val_5_kappa = float(val_result["metrics"]["5_class"]["cohen_kappa"])
            best_val_4_kappa = float(val_result["metrics"]["4_class"]["cohen_kappa"])
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": x_train.shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "model_type": model_type,
                    "class_weight_mode": class_weight_mode,
                    "n3_weight_multiplier": n3_weight_multiplier,
                    "loss_type": loss_type,
                    "focal_gamma": focal_gamma,
                    "label_smoothing": label_smoothing,
                    "train_sampler": train_sampler,
                    "selection_metric": selection_metric,
                    "feature_names": feature_names,
                    "stage5_names": STAGE5_NAMES,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"early_stopping epoch={epoch}", flush=True)
                break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_final = evaluate_loader(model=model, loader=val_loader, criterion=criterion, device=device)
    test_final = evaluate_loader(model=model, loader=test_loader, criterion=criterion, device=device)

    report = {
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "best_model_path": str(best_model_path),
        "device": str(device),
        "seed": seed,
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
            "n3_weight_multiplier": n3_weight_multiplier,
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
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
        "class_weight_mode": class_weight_mode,
        "n3_weight_multiplier": n3_weight_multiplier,
        "class_weights": None if weights is None else weights.detach().cpu().numpy().tolist(),
        "best_epoch": best_epoch,
        "best_selection_metric": selection_metric,
        "best_selection_score": best_selection_score,
        "best_val_macro_f1": best_val_5_macro_f1,
        "best_val_5_macro_f1": best_val_5_macro_f1,
        "best_val_4_macro_f1": best_val_4_macro_f1,
        "best_val_5_kappa": best_val_5_kappa,
        "best_val_4_kappa": best_val_4_kappa,
        "history": history,
        "final_val": {
            "loss": val_final["loss"],
            "metrics": val_final["metrics"],
        },
        "final_test": {
            "loss": test_final["loss"],
            "metrics": test_final["metrics"],
        },
    }

    metrics_path = out_dir / "lstm_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    prediction_arrays = {
        "val_y_true": np.asarray(val_final["y_true"], dtype=np.int64),
        "val_y_pred": np.asarray(val_final["y_pred"], dtype=np.int64),
        "val_logits": val_final["logits"],
        "val_probs": val_final["probabilities"],
        "test_y_true": np.asarray(test_final["y_true"], dtype=np.int64),
        "test_y_pred": np.asarray(test_final["y_pred"], dtype=np.int64),
        "test_logits": test_final["logits"],
        "test_probs": test_final["probabilities"],
        "stage5_names": np.asarray(STAGE5_NAMES),
    }
    for split_name in ("val", "test"):
        subject_key = f"{split_name}_subject_ids"
        epoch_key = f"{split_name}_epoch_indices"
        if subject_key in arrays:
            prediction_arrays[subject_key] = arrays[subject_key]
        if epoch_key in arrays:
            prediction_arrays[epoch_key] = arrays[epoch_key]

    np.savez_compressed(out_dir / "lstm_predictions.npz", **prediction_arrays)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTM sleep-stage classifier.")
    parser.add_argument("--npz", type=Path, required=True, help="NPZ dataset from build_npz_dataset.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for model and metrics.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-type",
        choices=("lstm", "gru"),
        default="lstm",
        help="Recurrent cell type. Defaults to lstm to preserve previous experiments.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=("inverse", "sqrt", "none"),
        default="inverse",
        help="Class weighting for CrossEntropyLoss. inverse matches the original behavior.",
    )
    parser.add_argument(
        "--n3-weight-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplier for the N3 class weight after the selected class weighting mode.",
    )
    parser.add_argument(
        "--loss-type",
        choices=("cross_entropy", "focal"),
        default="cross_entropy",
        help="Training loss. cross_entropy preserves previous experiments.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma. Used only with --loss-type focal.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing passed to cross entropy or focal loss.",
    )
    parser.add_argument(
        "--train-sampler",
        choices=("none", "weighted"),
        default="none",
        help="Training sampler. weighted oversamples minority classes with replacement.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "5_macro_f1",
            "4_macro_f1",
            "5_kappa",
            "4_kappa",
            "5_macro_f1_plus_4_kappa",
            "4_macro_f1_plus_4_kappa",
        ),
        default="5_macro_f1",
        help="Validation metric used for best checkpoint selection and early stopping.",
    )
    args = parser.parse_args()

    report = train_lstm(
        npz_path=args.npz,
        out_dir=args.out_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        seed=args.seed,
        class_weight_mode=args.class_weight_mode,
        n3_weight_multiplier=args.n3_weight_multiplier,
        model_type=args.model_type,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        train_sampler=args.train_sampler,
        selection_metric=args.selection_metric,
    )
    print(json.dumps({"best_epoch": report["best_epoch"], "final_test": report["final_test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
