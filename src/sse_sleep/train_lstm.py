"""Train and evaluate an LSTM sleep-stage classifier from a DreamT NPZ dataset."""

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
from torch.utils.data import DataLoader, TensorDataset

from .labels import STAGE5_NAMES
from .metrics import evaluate_5_and_4


class LSTMSleepClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
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
        output, _ = self.lstm(x)
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


def class_weights(y_train: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


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
    loss, y_true, y_pred = run_epoch(model=model, loader=loader, criterion=criterion, device=device)
    metrics = evaluate_5_and_4(y_true, y_pred)
    return {
        "loss": loss,
        "metrics": json_ready(metrics),
        "y_true": y_true,
        "y_pred": y_pred,
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
    model = LSTMSleepClassifier(
        input_size=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    weights = class_weights(y_train, len(STAGE5_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = out_dir / "lstm_best.pt"
    history: list[dict[str, Any]] = []
    best_val_macro_f1 = -1.0
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
        val_macro_f1 = val_result["metrics"]["5_class"]["macro_f1"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_metrics": json_ready(train_metrics),
            "val_loss": val_result["loss"],
            "val_metrics": val_result["metrics"],
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={val_result['loss']:.4f} val_macro_f1={val_macro_f1:.4f}",
            flush=True,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": x_train.shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
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
        },
        "array_shapes": {
            "X_train": list(x_train.shape),
            "X_val": list(x_val.shape),
            "X_test": list(x_test.shape),
        },
        "feature_names": feature_names,
        "stage5_names": list(STAGE5_NAMES),
        "class_weights": weights.detach().cpu().numpy().tolist(),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
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
    np.savez_compressed(
        out_dir / "lstm_predictions.npz",
        val_y_true=np.asarray(val_final["y_true"], dtype=np.int64),
        val_y_pred=np.asarray(val_final["y_pred"], dtype=np.int64),
        test_y_true=np.asarray(test_final["y_true"], dtype=np.int64),
        test_y_pred=np.asarray(test_final["y_pred"], dtype=np.int64),
    )
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
    )
    print(json.dumps({"best_epoch": report["best_epoch"], "final_test": report["final_test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

