"""Export logits/probabilities from a saved recurrent sleep-stage checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .labels import STAGE5_NAMES
from .train_lstm import RecurrentSleepClassifier, load_npz


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def predict_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        with torch.no_grad():
            outputs = model(x_batch)
            logits = outputs["stage_logits"]
            probabilities = torch.softmax(logits, dim=1)
        y_true_batches.append(y_batch.detach().cpu().numpy().astype(np.int64))
        y_pred_batches.append(logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64))
        logits_batches.append(logits.detach().cpu().numpy().astype(np.float32))
        prob_batches.append(probabilities.detach().cpu().numpy().astype(np.float32))

    return {
        "y_true": np.concatenate(y_true_batches, axis=0),
        "y_pred": np.concatenate(y_pred_batches, axis=0),
        "logits": np.concatenate(logits_batches, axis=0),
        "probs": np.concatenate(prob_batches, axis=0),
    }


def checkpoint_value(checkpoint: dict[str, Any], key: str, default: Any) -> Any:
    return checkpoint[key] if key in checkpoint else default


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Support checkpoints saved before the recurrent module was renamed."""
    if any(key.startswith("recurrent.") for key in state_dict):
        return state_dict
    return {
        key.replace("lstm.", "recurrent.", 1) if key.startswith("lstm.") else key: value
        for key, value in state_dict.items()
    }


def export_lstm_predictions(
    npz_path: Path,
    checkpoint_path: Path,
    out_npz: Path,
    batch_size: int,
) -> dict[str, Any]:
    arrays = load_npz(npz_path)
    x_train = arrays["X_train"].astype(np.float32)
    y_train = arrays["y_train"].astype(np.int64)
    x_val = arrays["X_val"].astype(np.float32)
    y_val = arrays["y_val"].astype(np.int64)
    x_test = arrays["X_test"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RecurrentSleepClassifier(
        input_size=int(checkpoint_value(checkpoint, "input_size", x_val.shape[-1])),
        hidden_size=int(checkpoint_value(checkpoint, "hidden_size", 64)),
        num_layers=int(checkpoint_value(checkpoint, "num_layers", 1)),
        dropout=float(checkpoint_value(checkpoint, "dropout", 0.4)),
        model_type=str(checkpoint_value(checkpoint, "model_type", "lstm")),
        aux_head=str(checkpoint_value(checkpoint, "aux_head", "none")),
    ).to(device)
    model.load_state_dict(normalize_state_dict_keys(checkpoint["model_state_dict"]))

    train = predict_split(model, make_loader(x_train, y_train, batch_size=batch_size), device=device)
    val = predict_split(model, make_loader(x_val, y_val, batch_size=batch_size), device=device)
    test = predict_split(model, make_loader(x_test, y_test, batch_size=batch_size), device=device)

    prediction_arrays: dict[str, np.ndarray] = {
        "train_y_true": train["y_true"],
        "train_y_pred": train["y_pred"],
        "train_logits": train["logits"],
        "train_probs": train["probs"],
        "val_y_true": val["y_true"],
        "val_y_pred": val["y_pred"],
        "val_logits": val["logits"],
        "val_probs": val["probs"],
        "test_y_true": test["y_true"],
        "test_y_pred": test["y_pred"],
        "test_logits": test["logits"],
        "test_probs": test["probs"],
        "stage5_names": np.asarray(STAGE5_NAMES),
    }
    for split_name in ("train", "val", "test"):
        for suffix in ("subject_ids", "epoch_indices"):
            key = f"{split_name}_{suffix}"
            if key in arrays:
                prediction_arrays[key] = arrays[key]

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **prediction_arrays)
    return {
        "npz_path": str(npz_path),
        "checkpoint_path": str(checkpoint_path),
        "out_npz": str(out_npz),
        "device": str(device),
        "batch_size": batch_size,
        "train_samples": int(train["y_true"].shape[0]),
        "val_samples": int(val["y_true"].shape[0]),
        "test_samples": int(test["y_true"].shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LSTM/GRU prediction probabilities from a saved checkpoint.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    summary = export_lstm_predictions(
        npz_path=args.npz,
        checkpoint_path=args.checkpoint,
        out_npz=args.out,
        batch_size=args.batch_size,
    )
    print(summary)


if __name__ == "__main__":
    main()
