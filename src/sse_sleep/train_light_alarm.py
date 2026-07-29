"""Train an app-oriented Light-vs-rest alarm classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .labels import STAGE4_NAMES, STAGE4_TO_ID, merge_many_5_to_4
from .metrics import evaluate
from .train_lstm import json_ready, load_npz, set_seed


BINARY_NAMES = ("Other", "Light")
DEFAULT_THRESHOLDS = tuple(float(value) for value in np.arange(0.10, 0.901, 0.025))


class LightAlarmClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        model_type: str,
        use_stage4_aux: bool,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        if model_type == "lstm":
            recurrent_cls = nn.LSTM
        elif model_type == "gru":
            recurrent_cls = nn.GRU
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        self.recurrent = recurrent_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.light_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.stage4_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, len(STAGE4_NAMES)),
            )
            if use_stage4_aux
            else None
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        output, _ = self.recurrent(features)
        latest = output[:, -1, :]
        result = {"light_logits": self.light_head(latest).squeeze(1)}
        if self.stage4_head is not None:
            result["stage4_logits"] = self.stage4_head(latest)
        return result


def map_labels_4(labels_5: np.ndarray) -> np.ndarray:
    return np.asarray(
        merge_many_5_to_4(labels_5.astype(np.int64).tolist()),
        dtype=np.int64,
    )


def light_labels(labels_4: np.ndarray) -> np.ndarray:
    return (labels_4 == STAGE4_TO_ID["Light"]).astype(np.int64)


def class_weight_vector(labels: np.ndarray, count: int, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(count, dtype=np.float32)
    counts = np.bincount(labels.astype(np.int64), minlength=count).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    if mode == "sqrt":
        weights = np.sqrt(weights)
    elif mode != "inverse":
        raise ValueError(f"Unknown class weight mode: {mode}")
    return (weights / weights.mean()).astype(np.float32)


def primary_weight_lookup(
    labels_4: np.ndarray,
    binary_weight_mode: str,
    wake_negative_multiplier: float,
    deep_negative_multiplier: float,
    rem_negative_multiplier: float,
) -> tuple[np.ndarray, float]:
    multipliers = np.asarray(
        (
            wake_negative_multiplier,
            1.0,
            deep_negative_multiplier,
            rem_negative_multiplier,
        ),
        dtype=np.float32,
    )
    if np.any(multipliers <= 0):
        raise ValueError("All stage multipliers must be positive")
    binary_weights = class_weight_vector(light_labels(labels_4), 2, binary_weight_mode)
    per_stage = np.asarray(
        (
            binary_weights[0] * multipliers[0],
            binary_weights[1],
            binary_weights[0] * multipliers[2],
            binary_weights[0] * multipliers[3],
        ),
        dtype=np.float32,
    )
    normalizer = float(np.mean(per_stage[labels_4]))
    return per_stage / max(normalizer, 1e-12), normalizer


def make_alarm_loader(
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
    if sampler_mode != "stage_balanced":
        raise ValueError(f"Unknown sampler mode: {sampler_mode}")
    counts = np.bincount(labels_4, minlength=len(STAGE4_NAMES)).astype(np.float32)
    weights = 1.0 / np.maximum(counts[labels_4], 1.0)
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(labels_4.shape[0]),
        replacement=True,
        generator=generator,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)


def binary_metrics(
    labels_4: np.ndarray,
    light_probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    true_binary = light_labels(labels_4)
    predicted_binary = (light_probabilities >= threshold).astype(np.int64)
    result = evaluate(
        true_binary.tolist(),
        predicted_binary.tolist(),
        BINARY_NAMES,
    )
    light = result.class_wise["Light"]
    other = result.class_wise["Other"]

    def positive_rate(stage_name: str) -> float:
        mask = labels_4 == STAGE4_TO_ID[stage_name]
        return float(predicted_binary[mask].mean()) if np.any(mask) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(result.accuracy),
        "balanced_accuracy": float((light.recall + other.recall) / 2.0),
        "macro_f1": float(result.macro_f1),
        "binary_kappa": float(result.cohen_kappa),
        "light_objective": float(light.f1 + result.cohen_kappa),
        "light_precision": float(light.precision),
        "light_recall": float(light.recall),
        "light_f1": float(light.f1),
        "other_recall": float(other.recall),
        "wake_to_light_rate": positive_rate("Wake"),
        "deep_to_light_rate": positive_rate("Deep"),
        "rem_to_light_rate": positive_rate("REM"),
        "confusion_matrix": result.confusion_matrix,
    }


def select_threshold(
    labels_4: np.ndarray,
    light_probabilities: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    candidates = [
        binary_metrics(labels_4, light_probabilities, threshold)
        for threshold in thresholds
    ]
    return max(
        candidates,
        key=lambda item: (
            item["light_objective"],
            -item["deep_to_light_rate"],
            item["light_precision"],
            -abs(item["threshold"] - 0.5),
        ),
    )


def primary_loss(
    light_logits: torch.Tensor,
    labels_4: torch.Tensor,
    stage_weights: torch.Tensor,
) -> torch.Tensor:
    targets = (labels_4 == STAGE4_TO_ID["Light"]).float()
    losses = F.binary_cross_entropy_with_logits(
        light_logits,
        targets,
        reduction="none",
    )
    return (losses * stage_weights[labels_4]).mean()


def run_training_epoch(
    model: LightAlarmClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    stage_weights: torch.Tensor,
    aux_criterion: nn.Module | None,
    aux_weight: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_primary = 0.0
    total_aux = 0.0
    total_count = 0
    for feature_batch, label_batch in loader:
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(feature_batch)
        primary = primary_loss(outputs["light_logits"], label_batch, stage_weights)
        auxiliary = (
            aux_criterion(outputs["stage4_logits"], label_batch)
            if aux_criterion is not None
            else None
        )
        loss = primary if auxiliary is None else primary + aux_weight * auxiliary
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        count = int(label_batch.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_primary += float(primary.detach().cpu()) * count
        if auxiliary is not None:
            total_aux += float(auxiliary.detach().cpu()) * count
        total_count += count
    return {
        "loss": total_loss / max(total_count, 1),
        "primary_loss": total_primary / max(total_count, 1),
        "aux_loss": (
            total_aux / max(total_count, 1)
            if aux_criterion is not None
            else 0.0
        ),
    }


def infer(
    model: LightAlarmClassifier,
    loader: DataLoader,
    stage_weights: torch.Tensor,
    aux_criterion: nn.Module | None,
    aux_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    labels: list[np.ndarray] = []
    light_probabilities: list[np.ndarray] = []
    stage4_probabilities: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    for feature_batch, label_batch in loader:
        feature_batch = feature_batch.to(device)
        label_batch = label_batch.to(device)
        with torch.no_grad():
            outputs = model(feature_batch)
            primary = primary_loss(outputs["light_logits"], label_batch, stage_weights)
            auxiliary = (
                aux_criterion(outputs["stage4_logits"], label_batch)
                if aux_criterion is not None
                else None
            )
            loss = primary if auxiliary is None else primary + aux_weight * auxiliary
        count = int(label_batch.shape[0])
        total_loss += float(loss.detach().cpu()) * count
        total_count += count
        labels.append(label_batch.detach().cpu().numpy().astype(np.int64))
        light_probabilities.append(
            torch.sigmoid(outputs["light_logits"]).detach().cpu().numpy().astype(np.float32)
        )
        if "stage4_logits" in outputs:
            stage4_probabilities.append(
                torch.softmax(outputs["stage4_logits"], dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return {
        "loss": total_loss / max(total_count, 1),
        "y_true4": np.concatenate(labels),
        "light_probs": np.concatenate(light_probabilities),
        "stage4_probs": (
            np.concatenate(stage4_probabilities)
            if stage4_probabilities
            else None
        ),
    }


def train_light_alarm(
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
    binary_weight_mode: str,
    wake_negative_multiplier: float,
    deep_negative_multiplier: float,
    rem_negative_multiplier: float,
    train_sampler: str,
    stage4_aux_weight: float,
    stage4_aux_class_weight_mode: str,
) -> dict[str, Any]:
    if stage4_aux_weight < 0:
        raise ValueError("stage4_aux_weight must be non-negative")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightAlarmClassifier(
        input_size=features["train"].shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type,
        use_stage4_aux=stage4_aux_weight > 0,
    ).to(device)
    primary_weights_np, weight_normalizer = primary_weight_lookup(
        labels_4["train"],
        binary_weight_mode,
        wake_negative_multiplier,
        deep_negative_multiplier,
        rem_negative_multiplier,
    )
    primary_weights = torch.as_tensor(
        primary_weights_np,
        dtype=torch.float32,
        device=device,
    )
    aux_weights_np = class_weight_vector(
        labels_4["train"],
        len(STAGE4_NAMES),
        stage4_aux_class_weight_mode,
    )
    aux_criterion = (
        nn.CrossEntropyLoss(
            weight=torch.as_tensor(aux_weights_np, dtype=torch.float32, device=device)
        )
        if stage4_aux_weight > 0
        else None
    )
    loaders = {
        split: make_alarm_loader(
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
    checkpoint_path = out_dir / "light_alarm_best.pt"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        train_result = run_training_epoch(
            model,
            loaders["train"],
            optimizer,
            primary_weights,
            aux_criterion,
            stage4_aux_weight,
            device,
        )
        val_result = infer(
            model,
            loaders["val"],
            primary_weights,
            aux_criterion,
            stage4_aux_weight,
            device,
        )
        threshold_result = select_threshold(
            val_result["y_true4"],
            val_result["light_probs"],
        )
        selection_key = (
            float(threshold_result["light_objective"]),
            -float(threshold_result["deep_to_light_rate"]),
            float(threshold_result["light_precision"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_result,
                "val_loss": float(val_result["loss"]),
                "val_selection": threshold_result,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_result['loss']:.4f} "
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
                    "dropout": dropout,
                    "model_type": model_type,
                    "use_stage4_aux": stage4_aux_weight > 0,
                    "label_mode": "light_vs_rest",
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
        split: infer(
            model,
            loaders[split],
            primary_weights,
            aux_criterion,
            stage4_aux_weight,
            device,
        )
        for split in ("val", "test")
    }
    selected_threshold = select_threshold(
        final["val"]["y_true4"],
        final["val"]["light_probs"],
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
        "experiment": "light_alarm_objective",
        "npz_path": str(npz_path),
        "out_dir": str(out_dir),
        "seed": seed,
        "device": str(device),
        "binary_names": list(BINARY_NAMES),
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
            "binary_weight_mode": binary_weight_mode,
            "wake_negative_multiplier": wake_negative_multiplier,
            "deep_negative_multiplier": deep_negative_multiplier,
            "rem_negative_multiplier": rem_negative_multiplier,
            "train_sampler": train_sampler,
            "stage4_aux_weight": stage4_aux_weight,
            "stage4_aux_class_weight_mode": stage4_aux_class_weight_mode,
        },
        "train_stage4_counts": np.bincount(
            labels_4["train"],
            minlength=len(STAGE4_NAMES),
        ).tolist(),
        "train_binary_counts": np.bincount(
            light_labels(labels_4["train"]),
            minlength=2,
        ).tolist(),
        "primary_stage_weights": primary_weights_np.tolist(),
        "primary_weight_normalizer": weight_normalizer,
        "stage4_aux_class_weights": (
            aux_weights_np.tolist() if stage4_aux_weight > 0 else None
        ),
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
    (out_dir / "light_alarm_metrics.json").write_text(
        json.dumps(json_ready(report), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    prediction_arrays: dict[str, np.ndarray] = {
        "binary_names": np.asarray(BINARY_NAMES),
        "stage4_names": np.asarray(STAGE4_NAMES),
    }
    for split in ("val", "test"):
        prediction_arrays[f"{split}_y_true4"] = final[split]["y_true4"]
        prediction_arrays[f"{split}_light_probs"] = final[split]["light_probs"]
        if final[split]["stage4_probs"] is not None:
            prediction_arrays[f"{split}_stage4_probs"] = final[split]["stage4_probs"]
        for suffix in ("subject_ids", "epoch_indices"):
            source_key = f"{split}_{suffix}"
            if source_key in arrays:
                prediction_arrays[source_key] = arrays[source_key]
    np.savez_compressed(
        out_dir / "light_alarm_predictions.npz",
        **prediction_arrays,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Light-vs-rest alarm classifier with optional 4-class auxiliary loss."
    )
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--model-type", choices=("lstm", "gru"), default="lstm")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--binary-weight-mode",
        choices=("inverse", "sqrt", "none"),
        default="inverse",
    )
    parser.add_argument("--wake-negative-multiplier", type=float, default=1.0)
    parser.add_argument("--deep-negative-multiplier", type=float, default=1.0)
    parser.add_argument("--rem-negative-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--train-sampler",
        choices=("none", "stage_balanced"),
        default="none",
    )
    parser.add_argument("--stage4-aux-weight", type=float, default=0.0)
    parser.add_argument(
        "--stage4-aux-class-weight-mode",
        choices=("inverse", "sqrt", "none"),
        default="inverse",
    )
    args = parser.parse_args()
    report = train_light_alarm(
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
        args.binary_weight_mode,
        args.wake_negative_multiplier,
        args.deep_negative_multiplier,
        args.rem_negative_multiplier,
        args.train_sampler,
        args.stage4_aux_weight,
        args.stage4_aux_class_weight_mode,
    )
    print(
        json.dumps(
            {
                "best_epoch": report["best_epoch"],
                "selected_validation_threshold": report["selected_validation_threshold"],
                "final_test": report["final"]["test"]["metrics"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
