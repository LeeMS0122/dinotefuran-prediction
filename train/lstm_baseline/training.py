from __future__ import annotations

import copy
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch import nn

from .data import StaticPreprocessor, load_aligned_target, make_loader
from .model import HybridSequenceLSTM


TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def select_fbeta_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true, probability, sample_weight=sample_weight
    )
    if len(thresholds) == 0:
        return 0.5, 0.0
    precision = precision[:-1]
    recall = recall[:-1]
    beta_sq = beta ** 2
    denominator = beta_sq * precision + recall
    fbeta = np.divide(
        (1 + beta_sq) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.nanargmax(fbeta))
    return float(thresholds[best]), float(fbeta[best])


def top_k_capture(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
    percentages: list[int],
) -> list[dict[str, float]]:
    order = np.argsort(-probability, kind="stable")
    positive_total = float(y_true.sum())
    weighted_positive_total = float((y_true * sample_weight).sum())
    rows = []
    for percentage in percentages:
        count = max(1, int(math.ceil(len(y_true) * percentage / 100)))
        selected = order[:count]
        captured = float(y_true[selected].sum())
        weighted_captured = float((y_true[selected] * sample_weight[selected]).sum())
        rows.append({
            "top_k_pct": float(percentage),
            "selected_rows": int(count),
            "captured_positive": int(captured),
            "capture_rate_pct": captured / positive_total * 100 if positive_total else np.nan,
            "weighted_capture_rate_pct": (
                weighted_captured / weighted_positive_total * 100
                if weighted_positive_total else np.nan
            ),
        })
    return rows


def classification_metrics(
    y_true: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
    beta: float,
) -> dict[str, float]:
    predicted = probability >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = float(sample_weight[predicted & positive].sum())
    fp = float(sample_weight[predicted & negative].sum())
    fn = float(sample_weight[~predicted & positive].sum())
    tn = float(sample_weight[~predicted & negative].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    beta_sq = beta ** 2
    fbeta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if beta_sq * precision + recall else 0.0
    )
    return {
        "n_rows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate_pct": float(y_true.mean() * 100),
        "weighted_pr_auc": float(
            average_precision_score(y_true, probability, sample_weight=sample_weight)
        ),
        "weighted_roc_auc": float(
            roc_auc_score(y_true, probability, sample_weight=sample_weight)
        ) if len(np.unique(y_true)) == 2 else np.nan,
        "threshold": float(threshold),
        "weighted_accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1": float(f1),
        "weighted_f2": float(fbeta),
        "weighted_tp": tp,
        "weighted_fp": fp,
        "weighted_fn": fn,
        "weighted_tn": tn,
        "predicted_positive_rows": int(predicted.sum()),
    }


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    outputs: dict[str, list[np.ndarray]] = {
        "array_row": [], "target": [], "sample_weight": [], "probability": []
    }
    for batch in loader:
        logits = model(
            batch["sequence"].to(device),
            batch["categorical"].to(device),
            batch["numeric"].to(device),
        )
        outputs["array_row"].append(batch["array_row"].numpy())
        outputs["target"].append(batch["target"].numpy())
        outputs["sample_weight"].append(batch["sample_weight"].numpy())
        outputs["probability"].append(torch.sigmoid(logits).cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in outputs.items()}


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float,
) -> float:
    model.train()
    weighted_loss_sum = 0.0
    weight_sum = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["sequence"].to(device),
            batch["categorical"].to(device),
            batch["numeric"].to(device),
        )
        target = batch["target"].to(device)
        weights = batch["sample_weight"].to(device)
        loss_values = loss_function(logits, target)
        loss = (loss_values * weights).sum() / weights.sum().clamp_min(1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        weighted_loss_sum += float((loss_values.detach() * weights).sum().cpu())
        weight_sum += float(weights.sum().cpu())
    return weighted_loss_sum / weight_sum if weight_sum else np.nan


def stratified_cap_rows(
    index: pd.DataFrame,
    split_name: str,
    cap: int | None,
    seed: int,
) -> np.ndarray:
    rows = index.index[index["split"].eq(split_name)].to_numpy(dtype=np.int64)
    if cap is None or len(rows) <= cap:
        return rows
    rng = np.random.default_rng(seed)
    positive = rows[index.loc[rows, "target"].to_numpy() == 1]
    negative = rows[index.loc[rows, "target"].to_numpy() == 0]
    positive_take = min(len(positive), max(1, cap // 4))
    negative_take = min(len(negative), cap - positive_take)
    if positive_take + negative_take < cap:
        positive_take = min(len(positive), cap - negative_take)
    selected = np.concatenate([
        rng.choice(positive, positive_take, replace=False),
        rng.choice(negative, negative_take, replace=False),
    ])
    rng.shuffle(selected)
    return selected.astype(np.int64)


def run_target(
    target_name: str,
    sequence_dir: Path,
    feature_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    device: torch.device,
    max_epochs_override: int | None = None,
    quick_cap: int | None = None,
    evaluate_test: bool = True,
) -> dict[str, Any]:
    seed = int(config["seed"])
    set_seed(seed)
    arrays, index, features = load_aligned_target(sequence_dir, feature_dir, target_name)
    split_cfg = config["splits"]
    train_rows = stratified_cap_rows(index, split_cfg["train"], quick_cap, seed)
    validation_rows = stratified_cap_rows(index, split_cfg["validation"], quick_cap, seed + 1)
    test_rows = (
        stratified_cap_rows(index, split_cfg["test"], quick_cap, seed + 2)
        if evaluate_test else np.asarray([], dtype=np.int64)
    )

    categorical_columns = list(config["static_features"]["categorical"])
    numeric_columns = list(config["static_features"]["numeric"])
    static = features[categorical_columns + numeric_columns].copy()
    static["key_valid"] = arrays["key_valid"].astype(np.float32)
    static["date_valid"] = arrays["date_valid"].astype(np.float32)
    numeric_columns += list(config["static_features"]["masks"])
    preprocessor = StaticPreprocessor.fit(
        static.loc[train_rows], categorical_columns, numeric_columns
    )
    categorical, numeric = preprocessor.transform(static)

    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    loaders = {
        "train": make_loader(
            arrays, categorical, numeric, train_rows, batch_size, True,
            int(training_cfg["num_workers"]), seed,
        ),
        "validation": make_loader(
            arrays, categorical, numeric, validation_rows, batch_size, False,
            int(training_cfg["num_workers"]), seed,
        ),
    }
    if evaluate_test:
        loaders["test"] = make_loader(
            arrays, categorical, numeric, test_rows, batch_size, False,
            int(training_cfg["num_workers"]), seed,
        )

    architecture = config["architecture"]
    model = HybridSequenceLSTM(
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        numeric_size=len(numeric_columns),
        sequence_channels=int(config["sequence"]["input_channels"]),
        hidden_size=int(architecture["hidden_size"]),
        num_layers=int(architecture["num_layers"]),
        bidirectional=bool(architecture["bidirectional"]),
        embedding_max_dim=int(architecture["embedding_max_dim"]),
        mlp_hidden_size=int(architecture["mlp_hidden_size"]),
        dropout=float(architecture["dropout"]),
    ).to(device)

    train_y = arrays["target"][train_rows].astype(np.float32)
    train_weight = arrays["group_sample_weight"][train_rows].astype(np.float32)
    weighted_positive = float((train_y * train_weight).sum())
    weighted_negative = float(((1 - train_y) * train_weight).sum())
    if weighted_positive <= 0:
        raise ValueError(f"{target_name} training split has no positive rows")
    pos_weight = weighted_negative / weighted_positive
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    max_epochs = int(max_epochs_override or training_cfg["max_epochs"])
    patience = int(training_cfg["early_stopping_patience"])
    min_delta = float(training_cfg["early_stopping_min_delta"])
    beta = float(config["selection"]["beta"])
    history_rows = []
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, loaders["train"], optimizer, loss_function, device,
            float(training_cfg["gradient_clip_norm"]),
        )
        validation_prediction = predict_loader(model, loaders["validation"], device)
        validation_ap = average_precision_score(
            validation_prediction["target"],
            validation_prediction["probability"],
            sample_weight=validation_prediction["sample_weight"],
        )
        history_rows.append({
            "target": target_name,
            "epoch": epoch,
            "train_weighted_loss": train_loss,
            "validation_weighted_pr_auc": float(validation_ap),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - started,
        })
        print(
            f"[{target_name}] epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"validation_pr_auc={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_score + min_delta:
            best_score = float(validation_ap)
            best_epoch = epoch
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("No LSTM checkpoint was selected")
    model.load_state_dict(best_state)
    model.to(device)
    validation_prediction = predict_loader(model, loaders["validation"], device)
    threshold, validation_best_f2 = select_fbeta_threshold(
        validation_prediction["target"],
        validation_prediction["probability"],
        validation_prediction["sample_weight"],
        beta=beta,
    )
    validation_metrics = classification_metrics(
        validation_prediction["target"], validation_prediction["probability"],
        validation_prediction["sample_weight"], threshold, beta,
    )
    validation_metrics["selection_best_f2"] = validation_best_f2

    test_prediction = None
    test_metrics = None
    if evaluate_test:
        # Test is touched only after epoch and threshold selection are complete.
        test_prediction = predict_loader(model, loaders["test"], device)
        test_metrics = classification_metrics(
            test_prediction["target"], test_prediction["probability"],
            test_prediction["sample_weight"], threshold, beta,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "target": target_name,
            "best_epoch": best_epoch,
            "threshold": threshold,
            "categorical_cardinalities": preprocessor.categorical_cardinalities,
            "numeric_size": len(numeric_columns),
            "architecture": architecture,
        },
        output_dir / f"{target_name}_lstm_baseline_v1.pt",
    )
    pd.DataFrame(history_rows).to_csv(
        output_dir / f"{target_name}_training_history_v1.csv", index=False, encoding="utf-8-sig"
    )
    prediction_items = [("validation", validation_prediction)]
    if evaluate_test and test_prediction is not None:
        prediction_items.append(("test", test_prediction))
    prediction_tables = []
    for split_name, prediction in prediction_items:
        rows = prediction["array_row"].astype(np.int64)
        prediction_tables.append(pd.DataFrame({
            "target_name": target_name,
            "split": split_name,
            "array_row": rows,
            "record_id": index.loc[rows, "record_id"].astype(str).to_numpy(),
            "target": prediction["target"].astype(np.int8),
            "sample_weight": prediction["sample_weight"].astype(np.float32),
            "probability": prediction["probability"].astype(np.float32),
            "predicted_at_validation_threshold": (
                prediction["probability"] >= threshold
            ).astype(np.int8),
        }))
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.to_parquet(output_dir / f"{target_name}_predictions_v1.parquet", index=False)

    window_days = int(arrays["inspection_count"].shape[1])
    candidate_name = (
        "lstm_hybrid_baseline"
        if evaluate_test else f"lstm_hybrid_window_{window_days}d"
    )
    metric_items = [("validation", validation_metrics)]
    if evaluate_test and test_metrics is not None:
        metric_items.append(("test", test_metrics))
    metrics = []
    for split_name, current in metric_items:
        row = {
            "target": target_name,
            "target_label": TARGET_LABELS[target_name],
            "model_type": "LSTM",
            "candidate": candidate_name,
            "feature_set": f"{window_days}d_sequence+static_core",
            "split": split_name,
            "best_epoch": best_epoch,
            "fit_seconds": time.perf_counter() - started,
            "pos_weight": pos_weight,
            **current,
        }
        metrics.append(row)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(
        output_dir / f"{target_name}_metrics_v1.csv", index=False, encoding="utf-8-sig"
    )
    top_k_rows = []
    for split_name, prediction in prediction_items:
        for row in top_k_capture(
            prediction["target"], prediction["probability"], prediction["sample_weight"],
            [int(value) for value in config["evaluation"]["top_k_percent"]],
        ):
            top_k_rows.append({
                "target": target_name,
                "target_label": TARGET_LABELS[target_name],
                "model_type": "LSTM",
                "split": split_name,
                **row,
            })
    top_k = pd.DataFrame(top_k_rows)
    top_k.to_csv(output_dir / f"{target_name}_top_k_v1.csv", index=False, encoding="utf-8-sig")

    return {
        "target": target_name,
        "device": str(device),
        "best_epoch": best_epoch,
        "epochs_ran": len(history_rows),
        "threshold": threshold,
        "fit_seconds": time.perf_counter() - started,
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "test_rows": int(len(test_rows)),
        "test_evaluated": bool(evaluate_test),
        "window_days": window_days,
        "preprocessor": preprocessor.to_dict(),
        "metrics": metrics,
        "top_k": top_k_rows,
    }
