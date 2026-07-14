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
from .metrics import evaluate, evaluate_5_and_4


DEEP_BINARY_NAMES = ("not_N3", "N3")
REM_BINARY_NAMES = ("not_REM", "REM")
AUX_HEAD_TARGETS = {
    "none": (),
    "deep": ("deep",),
    "rem": ("rem",),
    "deep_rem": ("deep", "rem"),
}
AUX_TARGETS_FOR_OUTPUT = ("deep", "rem")
AUX_TARGET_STAGE = {
    "deep": "N3",
    "rem": "REM",
}
AUX_TARGET_NAMES = {
    "deep": DEEP_BINARY_NAMES,
    "rem": REM_BINARY_NAMES,
}


class RecurrentSleepClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int = 5,
        dropout: float = 0.0,
        model_type: str = "lstm",
        aux_head: str = "none",
    ) -> None:
        super().__init__()
        if aux_head not in AUX_HEAD_TARGETS:
            raise ValueError(f"Unknown aux_head: {aux_head}")
        self.aux_head = aux_head
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
        self.aux_heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, 1),
                )
                for target in AUX_HEAD_TARGETS[aux_head]
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, _ = self.recurrent(x)
        latest = output[:, -1, :]
        outputs = {"stage_logits": self.head(latest)}
        for target, head in self.aux_heads.items():
            outputs[f"{target}_logits"] = head(latest).squeeze(1)
        return outputs


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
    teacher_probs: np.ndarray | None = None,
) -> DataLoader:
    tensors: list[torch.Tensor] = [torch.from_numpy(x).float(), torch.from_numpy(y).long()]
    if teacher_probs is not None:
        tensors.append(torch.from_numpy(teacher_probs).float())
    dataset = TensorDataset(*tensors)
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
    rem_weight_multiplier: float = 1.0,
) -> torch.Tensor | None:
    """Return class weights for cross entropy, or None for unweighted loss."""
    if n3_weight_multiplier <= 0:
        raise ValueError(f"n3_weight_multiplier must be positive: {n3_weight_multiplier}")
    if rem_weight_multiplier <= 0:
        raise ValueError(f"rem_weight_multiplier must be positive: {rem_weight_multiplier}")
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
    if rem_weight_multiplier != 1.0:
        rem_index = STAGE5_NAMES.index("REM")
        weights[rem_index] *= rem_weight_multiplier
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


def make_binary_aux_criterion(
    y_train: np.ndarray,
    device: torch.device,
    pos_weight_mode: str,
    target_stage: str,
) -> nn.Module:
    if pos_weight_mode == "none":
        return nn.BCEWithLogitsLoss()
    if pos_weight_mode != "balanced":
        raise ValueError(f"Unknown auxiliary pos_weight_mode: {pos_weight_mode}")

    target_index = STAGE5_NAMES.index(target_stage)
    positives = float(np.sum(y_train == target_index))
    negatives = float(y_train.shape[0] - positives)
    if positives <= 0:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def make_aux_criteria(
    y_train: np.ndarray,
    device: torch.device,
    aux_head: str,
    aux_weight: float,
    pos_weight_mode: str,
) -> dict[str, nn.Module]:
    if aux_weight <= 0:
        return {}
    return {
        target: make_binary_aux_criterion(
            y_train,
            device=device,
            pos_weight_mode=pos_weight_mode,
            target_stage=AUX_TARGET_STAGE[target],
        )
        for target in AUX_HEAD_TARGETS[aux_head]
    }


def binary_labels(y_true_5: list[int], target_stage: str) -> list[int]:
    target_index = STAGE5_NAMES.index(target_stage)
    return [1 if label == target_index else 0 for label in y_true_5]


def aux_binary_labels(y_true_5: list[int], target: str) -> list[int]:
    return binary_labels(y_true_5, AUX_TARGET_STAGE[target])


def evaluate_deep_binary(y_true_5: list[int], y_pred_binary: list[int]) -> dict[str, Any]:
    return evaluate_aux_binary(y_true_5, y_pred_binary, "deep")


def evaluate_aux_binary(y_true_5: list[int], y_pred_binary: list[int], target: str) -> dict[str, Any]:
    return json_ready(evaluate(aux_binary_labels(y_true_5, target), y_pred_binary, AUX_TARGET_NAMES[target]))


def deep_predictions_from_stage(y_pred_5: list[int]) -> list[int]:
    return aux_predictions_from_stage(y_pred_5, "deep")


def aux_predictions_from_stage(y_pred_5: list[int], target: str) -> list[int]:
    return aux_binary_labels(y_pred_5, target)


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
    stage_criterion: nn.Module,
    device: torch.device,
    aux_criteria: dict[str, nn.Module] | None = None,
    aux_weight: float = 0.0,
    optimizer: torch.optim.Optimizer | None = None,
    feature_dropout_indices: torch.Tensor | None = None,
    feature_dropout_prob: float = 0.0,
    distill_weight: float = 0.0,
    teacher_hard_weight: float = 0.0,
    teacher_hard_mode: str = "all",
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_stage_loss = 0.0
    total_aux_loss = 0.0
    total_distill_loss = 0.0
    total_teacher_hard_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    aux_y_pred: dict[str, list[int]] = {target: [] for target in (aux_criteria or {})}

    for batch in loader:
        x_batch = batch[0]
        y_batch = batch[1]
        teacher_batch = batch[2] if len(batch) > 2 else None
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        if teacher_batch is not None:
            teacher_batch = teacher_batch.to(device)
        if (
            training
            and feature_dropout_indices is not None
            and feature_dropout_indices.numel() > 0
            and feature_dropout_prob > 0.0
        ):
            keep_mask = (torch.rand((x_batch.shape[0], 1, 1), device=device) >= feature_dropout_prob).float()
            x_batch = x_batch.clone()
            x_batch[:, :, feature_dropout_indices] *= keep_mask
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(x_batch)
            logits = outputs["stage_logits"]
            stage_loss = stage_criterion(logits, y_batch)
            loss = stage_loss
            aux_loss = None
            distill_loss = None
            teacher_hard_loss = None
            if teacher_batch is not None and distill_weight > 0:
                distill_loss = F.kl_div(
                    F.log_softmax(logits, dim=1),
                    teacher_batch,
                    reduction="batchmean",
                )
                loss = loss + distill_weight * distill_loss
            if teacher_batch is not None and teacher_hard_weight > 0:
                teacher_labels = teacher_batch.argmax(dim=1)
                teacher_hard_losses = F.cross_entropy(logits, teacher_labels, reduction="none")
                if teacher_hard_mode == "rem_only":
                    rem_mask = teacher_labels == STAGE5_NAMES.index("REM")
                    if rem_mask.any():
                        teacher_hard_loss = teacher_hard_losses[rem_mask].mean()
                    else:
                        teacher_hard_loss = logits.sum() * 0.0
                elif teacher_hard_mode == "all":
                    teacher_hard_loss = teacher_hard_losses.mean()
                else:
                    raise ValueError(f"Unknown teacher_hard_mode: {teacher_hard_mode}")
                loss = loss + teacher_hard_weight * teacher_hard_loss
            if aux_criteria and aux_weight > 0:
                aux_losses = []
                for target, criterion in aux_criteria.items():
                    binary_target = (y_batch == STAGE5_NAMES.index(AUX_TARGET_STAGE[target])).float()
                    aux_losses.append(criterion(outputs[f"{target}_logits"], binary_target))
                if aux_losses:
                    aux_loss = torch.stack(aux_losses).sum()
                    loss = loss + aux_weight * aux_loss
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        batch_size = y_batch.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_stage_loss += float(stage_loss.detach().cpu()) * batch_size
        if aux_loss is not None:
            total_aux_loss += float(aux_loss.detach().cpu()) * batch_size
        if distill_loss is not None:
            total_distill_loss += float(distill_loss.detach().cpu()) * batch_size
        if teacher_hard_loss is not None:
            total_teacher_hard_loss += float(teacher_hard_loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        for target in aux_y_pred:
            aux_y_pred[target].extend(
                (torch.sigmoid(outputs[f"{target}_logits"]) >= 0.5).detach().cpu().numpy().astype(int).tolist()
            )

    result: dict[str, Any] = {
        "loss": total_loss / max(total_count, 1),
        "stage_loss": total_stage_loss / max(total_count, 1),
        "aux_loss": total_aux_loss / max(total_count, 1) if aux_criteria and aux_weight > 0 else None,
        "distill_loss": total_distill_loss / max(total_count, 1) if distill_weight > 0 else None,
        "teacher_hard_loss": total_teacher_hard_loss / max(total_count, 1) if teacher_hard_weight > 0 else None,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    for target, predictions in aux_y_pred.items():
        if predictions:
            result[f"{target}_y_pred"] = predictions
    return result


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


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    stage_criterion: nn.Module,
    device: torch.device,
    aux_criteria: dict[str, nn.Module] | None = None,
    aux_weight: float = 0.0,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_stage_loss = 0.0
    total_aux_loss = 0.0
    total_count = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    aux_y_pred: dict[str, list[int]] = {target: [] for target in (aux_criteria or {})}
    logits_batches: list[np.ndarray] = []
    prob_batches: list[np.ndarray] = []
    aux_logit_batches: dict[str, list[np.ndarray]] = {target: [] for target in (aux_criteria or {})}
    aux_prob_batches: dict[str, list[np.ndarray]] = {target: [] for target in (aux_criteria or {})}

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            outputs = model(x_batch)
            logits = outputs["stage_logits"]
            stage_loss = stage_criterion(logits, y_batch)
            loss = stage_loss
            aux_loss = None
            if aux_criteria and aux_weight > 0:
                aux_losses = []
                for target, criterion in aux_criteria.items():
                    binary_target = (y_batch == STAGE5_NAMES.index(AUX_TARGET_STAGE[target])).float()
                    aux_losses.append(criterion(outputs[f"{target}_logits"], binary_target))
                if aux_losses:
                    aux_loss = torch.stack(aux_losses).sum()
                    loss = loss + aux_weight * aux_loss
            probabilities = torch.softmax(logits, dim=1)

        batch_size = y_batch.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_stage_loss += float(stage_loss.detach().cpu()) * batch_size
        if aux_loss is not None:
            total_aux_loss += float(aux_loss.detach().cpu()) * batch_size
        total_count += batch_size
        y_true.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
        logits_batches.append(logits.detach().cpu().numpy().astype(np.float32))
        prob_batches.append(probabilities.detach().cpu().numpy().astype(np.float32))
        for target in aux_y_pred:
            aux_logits = outputs[f"{target}_logits"]
            aux_probabilities = torch.sigmoid(aux_logits)
            aux_y_pred[target].extend((aux_probabilities >= 0.5).detach().cpu().numpy().astype(int).tolist())
            aux_logit_batches[target].append(aux_logits.detach().cpu().numpy().astype(np.float32))
            aux_prob_batches[target].append(aux_probabilities.detach().cpu().numpy().astype(np.float32))

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
    result = {
        "loss": loss,
        "stage_loss": total_stage_loss / max(total_count, 1),
        "aux_loss": total_aux_loss / max(total_count, 1) if aux_criteria and aux_weight > 0 else None,
        "metrics": json_ready(metrics),
        "deep_binary_from_stage_metrics": evaluate_deep_binary(y_true, deep_predictions_from_stage(y_pred)),
        "rem_binary_from_stage_metrics": evaluate_aux_binary(y_true, aux_predictions_from_stage(y_pred, "rem"), "rem"),
        "deep_binary_aux_metrics": evaluate_deep_binary(y_true, aux_y_pred["deep"]) if aux_y_pred.get("deep") else None,
        "rem_binary_aux_metrics": evaluate_aux_binary(y_true, aux_y_pred["rem"], "rem") if aux_y_pred.get("rem") else None,
        "y_true": y_true,
        "y_pred": y_pred,
        "logits": logits_array,
        "probabilities": prob_array,
    }
    for target, predictions in aux_y_pred.items():
        result[f"{target}_y_pred"] = predictions
        result[f"{target}_logits"] = np.concatenate(aux_logit_batches[target], axis=0) if aux_logit_batches[target] else None
        result[f"{target}_probabilities"] = (
            np.concatenate(aux_prob_batches[target], axis=0) if aux_prob_batches[target] else None
        )
    for target in AUX_TARGETS_FOR_OUTPUT:
        result.setdefault(f"{target}_y_pred", [])
        result.setdefault(f"{target}_logits", None)
        result.setdefault(f"{target}_probabilities", None)
    return result


def load_teacher_train_probs(path: Path, arrays: dict[str, np.ndarray]) -> np.ndarray:
    teacher = load_npz(path)
    required = ["train_y_true", "train_probs"]
    missing = [key for key in required if key not in teacher]
    if missing:
        raise ValueError(f"Missing required teacher arrays in {path}: {missing}")
    teacher_y = teacher["train_y_true"].astype(np.int64)
    y_train = arrays["y_train"].astype(np.int64)
    if teacher_y.shape != y_train.shape or not np.array_equal(teacher_y, y_train):
        raise ValueError("Teacher train_y_true does not align with dataset y_train")
    for key in ("train_subject_ids", "train_epoch_indices"):
        if key in teacher and key in arrays and not np.array_equal(teacher[key], arrays[key]):
            raise ValueError(f"Teacher {key} does not align with dataset {key}")
    teacher_probs = teacher["train_probs"].astype(np.float32)
    if teacher_probs.shape != (y_train.shape[0], len(STAGE5_NAMES)):
        raise ValueError(f"Unexpected teacher train_probs shape: {teacher_probs.shape}")
    row_sums = teacher_probs.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Teacher train_probs contains rows with non-positive sums")
    return (teacher_probs / row_sums).astype(np.float32)


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
    rem_weight_multiplier: float,
    model_type: str,
    loss_type: str,
    focal_gamma: float,
    label_smoothing: float,
    train_sampler: str,
    selection_metric: str,
    aux_head: str,
    aux_weight: float,
    aux_deep_pos_weight_mode: str,
    feature_dropout_pattern: str,
    feature_dropout_prob: float,
    teacher_probs_npz: Path | None,
    distill_weight: float,
    teacher_hard_weight: float,
    teacher_hard_mode: str,
) -> dict[str, Any]:
    if aux_weight < 0:
        raise ValueError(f"aux_weight must be non-negative: {aux_weight}")
    if aux_weight > 0 and aux_head == "none":
        raise ValueError("--aux-head must not be none when aux_weight is positive")
    if not 0.0 <= feature_dropout_prob < 1.0:
        raise ValueError(f"feature_dropout_prob must be in [0, 1): {feature_dropout_prob}")
    if distill_weight < 0:
        raise ValueError(f"distill_weight must be non-negative: {distill_weight}")
    if teacher_hard_weight < 0:
        raise ValueError(f"teacher_hard_weight must be non-negative: {teacher_hard_weight}")
    if teacher_hard_mode not in {"all", "rem_only"}:
        raise ValueError(f"Unknown teacher_hard_mode: {teacher_hard_mode}")
    if teacher_probs_npz is None and (distill_weight > 0 or teacher_hard_weight > 0):
        raise ValueError("--teacher-probs-npz is required when teacher loss weight is positive")
    set_seed(seed)
    arrays = load_npz(npz_path)
    x_train = arrays["X_train"].astype(np.float32)
    y_train = arrays["y_train"].astype(np.int64)
    x_val = arrays["X_val"].astype(np.float32)
    y_val = arrays["y_val"].astype(np.int64)
    x_test = arrays["X_test"].astype(np.float32)
    y_test = arrays["y_test"].astype(np.int64)
    feature_names = arrays["feature_names"].astype(str).tolist()
    teacher_train_probs = (
        load_teacher_train_probs(teacher_probs_npz, arrays)
        if teacher_probs_npz is not None and (distill_weight > 0 or teacher_hard_weight > 0)
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecurrentSleepClassifier(
        input_size=x_train.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        model_type=model_type,
        aux_head=aux_head,
    ).to(device)

    weights = class_weights(
        y_train,
        len(STAGE5_NAMES),
        mode=class_weight_mode,
        n3_weight_multiplier=n3_weight_multiplier,
        rem_weight_multiplier=rem_weight_multiplier,
    )
    weights_for_loss = weights.to(device) if weights is not None else None
    stage_criterion = make_criterion(
        loss_type=loss_type,
        weights_for_loss=weights_for_loss,
        focal_gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )
    aux_criteria = make_aux_criteria(
        y_train,
        device=device,
        aux_head=aux_head,
        aux_weight=aux_weight,
        pos_weight_mode=aux_deep_pos_weight_mode,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = make_train_loader(
        x_train,
        y_train,
        batch_size=batch_size,
        sampler_mode=train_sampler,
        teacher_probs=teacher_train_probs,
    )
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)
    feature_dropout_indices_np = np.asarray(
        [idx for idx, name in enumerate(feature_names) if feature_dropout_pattern and feature_dropout_pattern in name],
        dtype=np.int64,
    )
    feature_dropout_indices = torch.as_tensor(feature_dropout_indices_np, dtype=torch.long, device=device)

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
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            stage_criterion=stage_criterion,
            device=device,
            aux_criteria=aux_criteria,
            aux_weight=aux_weight,
            optimizer=optimizer,
            feature_dropout_indices=feature_dropout_indices,
            feature_dropout_prob=feature_dropout_prob,
            distill_weight=distill_weight,
            teacher_hard_weight=teacher_hard_weight,
            teacher_hard_mode=teacher_hard_mode,
        )
        train_loss = train_result["loss"]
        train_true = train_result["y_true"]
        train_pred = train_result["y_pred"]
        train_metrics = evaluate_5_and_4(train_true, train_pred)
        val_result = evaluate_loader(
            model=model,
            loader=val_loader,
            stage_criterion=stage_criterion,
            device=device,
            aux_criteria=aux_criteria,
            aux_weight=aux_weight,
        )
        selection_score = validation_score(val_result["metrics"], selection_metric)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_stage_loss": train_result["stage_loss"],
            "train_aux_loss": train_result["aux_loss"],
            "train_distill_loss": train_result["distill_loss"],
            "train_teacher_hard_loss": train_result["teacher_hard_loss"],
            "train_metrics": json_ready(train_metrics),
            "val_loss": val_result["loss"],
            "val_stage_loss": val_result["stage_loss"],
            "val_aux_loss": val_result["aux_loss"],
            "val_metrics": val_result["metrics"],
            "val_deep_binary_from_stage_metrics": val_result["deep_binary_from_stage_metrics"],
            "val_rem_binary_from_stage_metrics": val_result["rem_binary_from_stage_metrics"],
            "val_deep_binary_aux_metrics": val_result["deep_binary_aux_metrics"],
            "val_rem_binary_aux_metrics": val_result["rem_binary_aux_metrics"],
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
                    "rem_weight_multiplier": rem_weight_multiplier,
                    "loss_type": loss_type,
                    "focal_gamma": focal_gamma,
                    "label_smoothing": label_smoothing,
                    "train_sampler": train_sampler,
                    "selection_metric": selection_metric,
                    "aux_head": aux_head,
                    "aux_weight": aux_weight,
                    "aux_deep_pos_weight_mode": aux_deep_pos_weight_mode,
                    "feature_dropout_pattern": feature_dropout_pattern,
                    "feature_dropout_prob": feature_dropout_prob,
                    "feature_dropout_count": int(feature_dropout_indices_np.shape[0]),
                    "teacher_probs_npz": None if teacher_probs_npz is None else str(teacher_probs_npz),
                    "distill_weight": distill_weight,
                    "teacher_hard_weight": teacher_hard_weight,
                    "teacher_hard_mode": teacher_hard_mode,
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
    val_final = evaluate_loader(
        model=model,
        loader=val_loader,
        stage_criterion=stage_criterion,
        device=device,
        aux_criteria=aux_criteria,
        aux_weight=aux_weight,
    )
    test_final = evaluate_loader(
        model=model,
        loader=test_loader,
        stage_criterion=stage_criterion,
        device=device,
        aux_criteria=aux_criteria,
        aux_weight=aux_weight,
    )

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
            "rem_weight_multiplier": rem_weight_multiplier,
            "loss_type": loss_type,
            "focal_gamma": focal_gamma,
            "label_smoothing": label_smoothing,
            "train_sampler": train_sampler,
            "selection_metric": selection_metric,
            "aux_head": aux_head,
            "aux_weight": aux_weight,
            "aux_deep_pos_weight_mode": aux_deep_pos_weight_mode,
            "feature_dropout_pattern": feature_dropout_pattern,
            "feature_dropout_prob": feature_dropout_prob,
            "feature_dropout_count": int(feature_dropout_indices_np.shape[0]),
            "teacher_probs_npz": None if teacher_probs_npz is None else str(teacher_probs_npz),
            "distill_weight": distill_weight,
            "teacher_hard_weight": teacher_hard_weight,
            "teacher_hard_mode": teacher_hard_mode,
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
        "rem_weight_multiplier": rem_weight_multiplier,
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
            "stage_loss": val_final["stage_loss"],
            "aux_loss": val_final["aux_loss"],
            "metrics": val_final["metrics"],
            "deep_binary_from_stage_metrics": val_final["deep_binary_from_stage_metrics"],
            "rem_binary_from_stage_metrics": val_final["rem_binary_from_stage_metrics"],
            "deep_binary_aux_metrics": val_final["deep_binary_aux_metrics"],
            "rem_binary_aux_metrics": val_final["rem_binary_aux_metrics"],
        },
        "final_test": {
            "loss": test_final["loss"],
            "stage_loss": test_final["stage_loss"],
            "aux_loss": test_final["aux_loss"],
            "metrics": test_final["metrics"],
            "deep_binary_from_stage_metrics": test_final["deep_binary_from_stage_metrics"],
            "rem_binary_from_stage_metrics": test_final["rem_binary_from_stage_metrics"],
            "deep_binary_aux_metrics": test_final["deep_binary_aux_metrics"],
            "rem_binary_aux_metrics": test_final["rem_binary_aux_metrics"],
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
    for target in AUX_TARGETS_FOR_OUTPUT:
        if val_final[f"{target}_logits"] is not None:
            prediction_arrays[f"val_{target}_logits"] = val_final[f"{target}_logits"]
            prediction_arrays[f"val_{target}_probs"] = val_final[f"{target}_probabilities"]
            prediction_arrays[f"val_{target}_y_pred"] = np.asarray(val_final[f"{target}_y_pred"], dtype=np.int64)
        if test_final[f"{target}_logits"] is not None:
            prediction_arrays[f"test_{target}_logits"] = test_final[f"{target}_logits"]
            prediction_arrays[f"test_{target}_probs"] = test_final[f"{target}_probabilities"]
            prediction_arrays[f"test_{target}_y_pred"] = np.asarray(test_final[f"{target}_y_pred"], dtype=np.int64)
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
        "--rem-weight-multiplier",
        type=float,
        default=1.0,
        help="Additional multiplier for the REM class weight after the selected class weighting mode.",
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
    parser.add_argument(
        "--aux-head",
        choices=("none", "deep", "rem", "deep_rem"),
        default="none",
        help="Optional auxiliary head. deep=N3-vs-rest, rem=REM-vs-rest, deep_rem=both heads.",
    )
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.0,
        help="Weight for the auxiliary loss. Used when --aux-head is not none.",
    )
    parser.add_argument(
        "--aux-deep-pos-weight-mode",
        choices=("balanced", "none"),
        default="balanced",
        help="Positive-class weighting for the Deep/N3 auxiliary BCE loss.",
    )
    parser.add_argument(
        "--feature-dropout-pattern",
        default="",
        help="Substring used to select input features for train-time group dropout. Empty disables selection.",
    )
    parser.add_argument(
        "--feature-dropout-prob",
        type=float,
        default=0.0,
        help="Probability of zeroing selected features for each training sample. Evaluation is unchanged.",
    )
    parser.add_argument(
        "--teacher-probs-npz",
        type=Path,
        default=None,
        help="Optional NPZ containing train_probs soft targets for distillation.",
    )
    parser.add_argument(
        "--distill-weight",
        type=float,
        default=0.0,
        help="Weight for KL(teacher_probs || student_probs) distillation loss on the train split.",
    )
    parser.add_argument(
        "--teacher-hard-weight",
        type=float,
        default=0.0,
        help="Weight for hard CE loss against argmax teacher labels on the train split.",
    )
    parser.add_argument(
        "--teacher-hard-mode",
        choices=("all", "rem_only"),
        default="all",
        help="Apply teacher hard CE to all samples or only samples whose teacher label is REM.",
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
        rem_weight_multiplier=args.rem_weight_multiplier,
        model_type=args.model_type,
        loss_type=args.loss_type,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        train_sampler=args.train_sampler,
        selection_metric=args.selection_metric,
        aux_head=args.aux_head,
        aux_weight=args.aux_weight,
        aux_deep_pos_weight_mode=args.aux_deep_pos_weight_mode,
        feature_dropout_pattern=args.feature_dropout_pattern,
        feature_dropout_prob=args.feature_dropout_prob,
        teacher_probs_npz=args.teacher_probs_npz,
        distill_weight=args.distill_weight,
        teacher_hard_weight=args.teacher_hard_weight,
        teacher_hard_mode=args.teacher_hard_mode,
    )
    print(json.dumps({"best_epoch": report["best_epoch"], "final_test": report["final_test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
