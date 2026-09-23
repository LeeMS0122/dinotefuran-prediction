from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import nn

from baseline_modeling.pipeline import (
    choose_fbeta_threshold,
    evaluate_scores,
    feature_sets,
    fit_model,
    top_k_table,
)
from lstm_baseline.data import StaticPreprocessor, load_aligned_target, make_loader
from lstm_baseline.model import HybridSequenceLSTM
from lstm_baseline.training import predict_loader, set_seed, train_one_epoch


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("occurrence", "screening", "noncompliance")
TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}
TABULAR_MODELS = ("logistic", "catboost", "lightgbm", "xgboost")
MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
COHORTS = ("all", "high")
COHORT_LABELS = {
    "all": "전체 연결국가",
    "high": "고신뢰 국가",
}
VARIANTS = ("internal_only", "internal_plus_climate")
VARIANT_LABELS = {
    "internal_only": "내부변수-only",
    "internal_plus_climate": "내부+기후평년",
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(model: Any, model_name: str, path_stem: Path) -> Path:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        path = path_stem.with_suffix(".cbm")
        model.save_model(path)
        return path
    path = path_stem.with_suffix(".joblib")
    joblib.dump(model, path)
    return path


def load_comparison_frame(
    target: str,
    cohort: str,
    climate_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_path = ROOT / "output" / "features_v1" / f"{target}_features_v1.parquet"
    climate_path = (
        ROOT
        / "output"
        / "external_variables_v1"
        / "country_month_climate_pilot_v1"
        / f"{target}_country_climate_features_v1.parquet"
    )
    features = pd.read_parquet(feature_path)
    climate = pd.read_parquet(climate_path)
    if not features["record_id"].is_unique:
        raise ValueError(f"{target}: 내부변수 record_id 중복")
    if not climate["record_id"].is_unique:
        raise ValueError(f"{target}: 기후변수 record_id 중복")
    payload_columns = [
        "record_id",
        "country_mapping_confidence",
        "country_climate_matched",
        *climate_columns,
    ]
    merged = features.merge(
        climate[payload_columns],
        on="record_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(merged) != len(features):
        raise ValueError(f"{target}: 기후 연결 후 행 수 변경")
    if not merged["record_id"].astype(str).equals(features["record_id"].astype(str)):
        raise ValueError(f"{target}: 기후 연결 후 record_id 순서 변경")

    matched = merged["country_climate_matched"].fillna(False).astype(bool)
    if cohort == "high":
        matched &= merged["country_mapping_confidence"].eq("high")
    elif cohort != "all":
        raise ValueError(f"지원하지 않는 cohort: {cohort}")
    model_split = merged["split"].isin(["train", "validation"])
    comparison = merged.loc[matched & model_split].copy()
    if comparison[climate_columns].isna().any().any():
        missing = comparison[climate_columns].isna().sum()
        raise ValueError(f"{target}/{cohort}: 연결 모집단 기후 결측 {missing[missing.gt(0)].to_dict()}")
    split_per_group = comparison.groupby("duplicate_group_id", observed=True)["split"].nunique()
    cross_split_groups = int(split_per_group.gt(1).sum())
    if cross_split_groups:
        raise ValueError(f"{target}/{cohort}: duplicate_group_id 분할 교차 {cross_split_groups}")

    split_rows: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        part = comparison.loc[comparison["split"].eq(split_name)]
        if part.empty or part["target"].nunique() < 2:
            raise ValueError(f"{target}/{cohort}/{split_name}: 두 라벨이 모두 필요")
        split_rows[split_name] = {
            "rows": int(len(part)),
            "positive": int(part["target"].sum()),
            "positive_rate": float(part["target"].mean()),
            "weighted_positive_rate": float(
                np.average(part["target"], weights=part["group_sample_weight"])
            ),
            "source_counts": {
                str(key): int(value)
                for key, value in part["source_system"].value_counts().items()
            },
        }
    quality = {
        "target": target,
        "cohort": cohort,
        "feature_rows": int(len(features)),
        "climate_rows": int(len(climate)),
        "joined_rows": int(len(merged)),
        "comparison_rows": int(len(comparison)),
        "row_count_preserved": bool(len(merged) == len(features)),
        "feature_record_id_duplicates": int(features["record_id"].duplicated().sum()),
        "climate_record_id_duplicates": int(climate["record_id"].duplicated().sum()),
        "comparison_record_id_duplicates": int(comparison["record_id"].duplicated().sum()),
        "cross_split_duplicate_groups": cross_split_groups,
        "climate_missing_cells": int(comparison[climate_columns].isna().sum().sum()),
        "split_rows": split_rows,
        "test_rows_scored": 0,
        "test_data_used_for_selection": False,
    }
    return comparison, quality


def metric_row(
    target: str,
    cohort: str,
    model_name: str,
    variant: str,
    scores: np.ndarray,
    validation: pd.DataFrame,
    threshold: float,
    beta: float,
    fit_seconds: float,
    best_epoch: int | None = None,
    window_days: int | None = None,
) -> dict[str, Any]:
    y_true = validation["target"].astype(int).to_numpy()
    weights = validation["group_sample_weight"].astype(float).to_numpy()
    metrics = evaluate_scores(y_true, scores, weights, threshold, beta)
    top10 = top_k_table(y_true, scores, weights, [0.10]).iloc[0]
    return {
        "target": target,
        "target_label": TARGET_LABELS[target],
        "cohort": cohort,
        "cohort_label": COHORT_LABELS[cohort],
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "split": "validation",
        "fit_seconds": float(fit_seconds),
        "best_epoch": best_epoch,
        "window_days": window_days,
        **metrics,
        "top10_capture_rate": float(top10["capture_rate"]),
        "top10_precision": float(top10["precision"]),
        "top10_lift": float(top10["lift"]),
    }


def run_tabular_pairs(
    target: str,
    cohort: str,
    comparison: pd.DataFrame,
    manifest: dict[str, Any],
    baseline_config: dict[str, Any],
    climate_columns: list[str],
    beta: float,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    categorical, internal_numeric = feature_sets(manifest["targets"][target])["extended"]
    train = comparison.loc[comparison["split"].eq("train")].copy()
    validation = comparison.loc[comparison["split"].eq("validation")].copy()
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    top_k_rows: list[pd.DataFrame] = []
    for model_name in TABULAR_MODELS:
        for variant in VARIANTS:
            numeric = list(internal_numeric)
            if variant == "internal_plus_climate":
                numeric += climate_columns
            started = time.perf_counter()
            print(f"[{target}][{cohort}][{model_name}][{variant}] 시작", flush=True)
            model, scores = fit_model(
                model_name,
                train,
                validation,
                list(categorical),
                numeric,
                baseline_config,
            )
            fit_seconds = time.perf_counter() - started
            weights = validation["group_sample_weight"].astype(float).to_numpy()
            threshold = choose_fbeta_threshold(
                validation["target"].astype(int).to_numpy(), scores, weights, beta
            )
            row = metric_row(
                target,
                cohort,
                model_name,
                variant,
                scores,
                validation,
                threshold,
                beta,
                fit_seconds,
            )
            rows.append(row)
            prediction = validation[
                ["record_id", "duplicate_group_id", "target", "group_sample_weight"]
            ].copy()
            prediction["target_name"] = target
            prediction["cohort"] = cohort
            prediction["model"] = model_name
            prediction["variant"] = variant
            prediction["score"] = scores
            prediction["threshold"] = threshold
            predictions.append(prediction)
            top_k = top_k_table(
                validation["target"].astype(int).to_numpy(),
                scores,
                weights,
                [0.05, 0.10, 0.20],
            )
            top_k.insert(0, "variant", variant)
            top_k.insert(0, "model", model_name)
            top_k.insert(0, "cohort", cohort)
            top_k.insert(0, "target", target)
            top_k_rows.append(top_k)
            model_path = save_model(
                model,
                model_name,
                output_dir / "models" / f"{target}_{cohort}_{model_name}_{variant}",
            )
            print(
                f"[{target}][{cohort}][{model_name}][{variant}] 완료 · "
                f"Recall={row['recall']:.6f} · PR-AUC={row['average_precision']:.6f} · "
                f"model={model_path.name}",
                flush=True,
            )
    return (
        pd.DataFrame(rows),
        pd.concat(predictions, ignore_index=True),
        pd.concat(top_k_rows, ignore_index=True),
    )


def run_lstm_variant(
    target: str,
    cohort: str,
    arrays: dict[str, np.ndarray],
    index: pd.DataFrame,
    features: pd.DataFrame,
    comparison_ids: set[str],
    variant: str,
    config: dict[str, Any],
    manifest: dict[str, Any],
    climate_columns: list[str],
    output_dir: Path,
    device: torch.device,
    window_days: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    lstm_config = config["lstm"]
    seed = int(config["seed"])
    set_seed(seed)
    allowed = features["record_id"].astype(str).isin(comparison_ids).to_numpy()
    train_rows = np.flatnonzero(index["split"].eq("train").to_numpy() & allowed).astype(np.int64)
    validation_rows = np.flatnonzero(
        index["split"].eq("validation").to_numpy() & allowed
    ).astype(np.int64)
    if len(train_rows) == 0 or len(validation_rows) == 0:
        raise ValueError(f"{target}/{cohort}: LSTM 학습 또는 검증 행 없음")

    categorical_columns, internal_numeric = feature_sets(manifest["targets"][target])["extended"]
    numeric_columns = list(internal_numeric)
    if variant == "internal_plus_climate":
        numeric_columns += climate_columns
    static = features[categorical_columns + numeric_columns].copy()
    for mask in lstm_config["masks"]:
        static[mask] = arrays[mask].astype(np.float32)
    numeric_columns += list(lstm_config["masks"])
    preprocessor = StaticPreprocessor.fit(
        static.loc[train_rows], list(categorical_columns), numeric_columns
    )
    categorical, numeric = preprocessor.transform(static)
    training = lstm_config["training"]
    loaders = {
        "train": make_loader(
            arrays,
            categorical,
            numeric,
            train_rows,
            int(training["batch_size"]),
            True,
            int(training["num_workers"]),
            seed,
        ),
        "validation": make_loader(
            arrays,
            categorical,
            numeric,
            validation_rows,
            int(training["batch_size"]),
            False,
            int(training["num_workers"]),
            seed,
        ),
    }
    architecture = lstm_config["architecture"]
    model = HybridSequenceLSTM(
        categorical_cardinalities=preprocessor.categorical_cardinalities,
        numeric_size=len(numeric_columns),
        sequence_channels=2,
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
        raise ValueError(f"{target}/{cohort}: LSTM 학습 양성 가중치가 0")
    pos_weight = weighted_negative / weighted_positive
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device),
        reduction="none",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            loss_function,
            device,
            float(training["gradient_clip_norm"]),
        )
        validation_prediction = predict_loader(model, loaders["validation"], device)
        validation_ap = average_precision_score(
            validation_prediction["target"],
            validation_prediction["probability"],
            sample_weight=validation_prediction["sample_weight"],
        )
        history.append(
            {
                "target": target,
                "cohort": cohort,
                "variant": variant,
                "window_days": window_days,
                "epoch": epoch,
                "train_weighted_loss": train_loss,
                "validation_weighted_pr_auc": float(validation_ap),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        print(
            f"[{target}][{cohort}][lstm][{variant}] epoch={epoch:02d} "
            f"loss={train_loss:.6f} PR-AUC={validation_ap:.6f}",
            flush=True,
        )
        if validation_ap > best_score + float(training["early_stopping_min_delta"]):
            best_score = float(validation_ap)
            best_epoch = epoch
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            without_improvement = 0
        else:
            without_improvement += 1
            if without_improvement >= int(training["early_stopping_patience"]):
                break
    if best_state is None:
        raise RuntimeError(f"{target}/{cohort}/{variant}: LSTM 최적 epoch 선택 실패")
    model.load_state_dict(best_state)
    model.to(device)
    prediction = predict_loader(model, loaders["validation"], device)
    beta = float(config["comparison"]["threshold_beta"])
    threshold = choose_fbeta_threshold(
        prediction["target"],
        prediction["probability"],
        prediction["sample_weight"],
        beta,
    )
    validation = features.loc[validation_rows].copy()
    result = metric_row(
        target,
        cohort,
        "lstm",
        variant,
        prediction["probability"],
        validation,
        threshold,
        beta,
        time.perf_counter() - started,
        best_epoch,
        window_days,
    )
    prediction_frame = validation[
        ["record_id", "duplicate_group_id", "target", "group_sample_weight"]
    ].copy()
    prediction_frame["target_name"] = target
    prediction_frame["cohort"] = cohort
    prediction_frame["model"] = "lstm"
    prediction_frame["variant"] = variant
    prediction_frame["score"] = prediction["probability"]
    prediction_frame["threshold"] = threshold
    top_k = top_k_table(
        prediction["target"],
        prediction["probability"],
        prediction["sample_weight"],
        [0.05, 0.10, 0.20],
    )
    top_k.insert(0, "variant", variant)
    top_k.insert(0, "model", "lstm")
    top_k.insert(0, "cohort", cohort)
    top_k.insert(0, "target", target)
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "target": target,
            "cohort": cohort,
            "variant": variant,
            "window_days": window_days,
            "best_epoch": best_epoch,
            "threshold": threshold,
            "categorical_cardinalities": preprocessor.categorical_cardinalities,
            "numeric_size": len(numeric_columns),
            "architecture": architecture,
            "test_data_used": False,
        },
        model_dir / f"{target}_{cohort}_lstm_{variant}.pt",
    )
    pd.DataFrame(history).to_csv(
        output_dir / f"{target}_{cohort}_lstm_{variant}_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (output_dir / f"{target}_{cohort}_lstm_{variant}_preprocessor.json").write_text(
        json.dumps(preprocessor.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, prediction_frame, top_k


def run_lstm_pairs(
    target: str,
    cohort: str,
    comparison: pd.DataFrame,
    config: dict[str, Any],
    manifest: dict[str, Any],
    climate_columns: list[str],
    output_dir: Path,
    device: torch.device,
    selected_windows: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_days = int(selected_windows[target]["window_days"])
    sequence_dir = ROOT / config["lstm"]["sequence_root"] / f"window_{window_days}d"
    feature_dir = ROOT / "output" / "features_v1"
    arrays, index, features = load_aligned_target(sequence_dir, feature_dir, target)
    climate_path = (
        ROOT
        / "output"
        / "external_variables_v1"
        / "country_month_climate_pilot_v1"
        / f"{target}_country_climate_features_v1.parquet"
    )
    climate = pd.read_parquet(
        climate_path,
        columns=["record_id", "country_mapping_confidence", "country_climate_matched", *climate_columns],
    )
    features = features.merge(climate, on="record_id", how="left", validate="one_to_one", sort=False)
    if not index["record_id"].astype(str).equals(features["record_id"].astype(str)):
        raise ValueError(f"{target}: LSTM 기후 연결 후 record_id 정렬 변경")
    comparison_ids = set(comparison["record_id"].astype(str))
    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    top_k: list[pd.DataFrame] = []
    for variant in VARIANTS:
        result, prediction, top = run_lstm_variant(
            target,
            cohort,
            arrays,
            index,
            features,
            comparison_ids,
            variant,
            config,
            manifest,
            climate_columns,
            output_dir,
            device,
            window_days,
        )
        metrics.append(result)
        predictions.append(prediction)
        top_k.append(top)
    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True),
        pd.concat(top_k, ignore_index=True),
    )


def comparison_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "f2",
        "average_precision",
        "roc_auc",
        "top10_capture_rate",
        "top10_precision",
        "top10_lift",
    ]
    rows: list[dict[str, Any]] = []
    for (target, cohort, model), group in metrics.groupby(
        ["target", "cohort", "model"], observed=True
    ):
        indexed = group.set_index("variant")
        row: dict[str, Any] = {
            "target": target,
            "target_label": TARGET_LABELS[target],
            "cohort": cohort,
            "cohort_label": COHORT_LABELS[cohort],
            "model": model,
            "model_label": MODEL_LABELS[model],
        }
        for column in value_columns:
            row[f"internal_{column}"] = float(indexed.loc["internal_only", column])
            row[f"climate_{column}"] = float(indexed.loc["internal_plus_climate", column])
            row[f"delta_{column}"] = row[f"climate_{column}"] - row[f"internal_{column}"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result["_target_order"] = result["target"].map(
        {value: index for index, value in enumerate(TARGETS)}
    )
    result["_cohort_order"] = result["cohort"].map(
        {value: index for index, value in enumerate(COHORTS)}
    )
    result["_model_order"] = result["model"].map(
        {value: index for index, value in enumerate(MODEL_LABELS)}
    )
    return (
        result.sort_values(["_target_order", "_cohort_order", "_model_order"])
        .drop(columns=["_target_order", "_cohort_order", "_model_order"])
        .reset_index(drop=True)
    )


def paired_bootstrap_deltas(
    predictions: pd.DataFrame,
    repeats: int,
    confidence: float,
    seed: int,
    beta: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = predictions.groupby(["target_name", "cohort", "model"], observed=True)
    for group_index, ((target, cohort, model), group) in enumerate(groups, start=1):
        pair = group.pivot(
            index=["record_id", "target", "group_sample_weight"],
            columns="variant",
            values=["score", "threshold"],
        ).reset_index()
        y = pair["target"].to_numpy(dtype=int)
        weight = pair["group_sample_weight"].to_numpy(dtype=float)
        score_internal = pair[("score", "internal_only")].to_numpy(dtype=float)
        score_climate = pair[("score", "internal_plus_climate")].to_numpy(dtype=float)
        threshold_internal = float(pair[("threshold", "internal_only")].iloc[0])
        threshold_climate = float(pair[("threshold", "internal_plus_climate")].iloc[0])
        strata = [np.flatnonzero(y == value) for value in (0, 1)]
        rng = np.random.default_rng(seed + group_index)
        samples = {
            "average_precision": [],
            "recall": [],
            "f2": [],
            "top10_capture_rate": [],
        }
        for _ in range(repeats):
            sampled = np.concatenate(
                [rng.choice(indices, len(indices), replace=True) for indices in strata]
            )
            rng.shuffle(sampled)
            yi = y[sampled]
            wi = weight[sampled]
            internal = evaluate_scores(
                yi, score_internal[sampled], wi, threshold_internal, beta
            )
            climate = evaluate_scores(
                yi, score_climate[sampled], wi, threshold_climate, beta
            )
            internal_top10 = top_k_table(
                yi, score_internal[sampled], wi, [0.10]
            ).iloc[0]["capture_rate"]
            climate_top10 = top_k_table(
                yi, score_climate[sampled], wi, [0.10]
            ).iloc[0]["capture_rate"]
            samples["average_precision"].append(
                climate["average_precision"] - internal["average_precision"]
            )
            samples["recall"].append(climate["recall"] - internal["recall"])
            samples["f2"].append(climate["f2"] - internal["f2"])
            samples["top10_capture_rate"].append(climate_top10 - internal_top10)
        alpha = (1 - confidence) / 2
        for metric, values in samples.items():
            array = np.asarray(values, dtype=float)
            rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "metric": metric,
                    "delta_mean": float(array.mean()),
                    "ci_low": float(np.quantile(array, alpha)),
                    "ci_high": float(np.quantile(array, 1 - alpha)),
                    "bootstrap_repeats": int(repeats),
                    "confidence": float(confidence),
                }
            )
        print(f"bootstrap {group_index}/{groups.ngroups}: {target}/{cohort}/{model}", flush=True)
    return pd.DataFrame(rows)


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Malgun Gothic",
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 180,
        }
    )


def render_figures(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    model_order = list(MODEL_LABELS)
    model_labels = [MODEL_LABELS[name] for name in model_order]
    colors = {"internal_only": "#64748B", "internal_plus_climate": "#0F766E"}
    paths: list[Path] = []

    fig, axes = plt.subplots(3, 2, figsize=(16, 14), sharey=True)
    for row, target in enumerate(TARGETS):
        for column, cohort in enumerate(COHORTS):
            axis = axes[row, column]
            subset = metrics.loc[
                metrics["target"].eq(target) & metrics["cohort"].eq(cohort)
            ]
            x = np.arange(len(model_order))
            width = 0.36
            for offset, variant in zip((-width / 2, width / 2), VARIANTS):
                values = [
                    float(
                        subset.loc[
                            subset["model"].eq(model) & subset["variant"].eq(variant),
                            "recall",
                        ].iloc[0]
                    )
                    for model in model_order
                ]
                axis.bar(
                    x + offset,
                    values,
                    width,
                    label=VARIANT_LABELS[variant],
                    color=colors[variant],
                )
            n_rows = int(subset["n_rows"].iloc[0])
            n_positive = int(subset["n_positive"].iloc[0])
            axis.set_title(
                f"{TARGET_LABELS[target]} · {COHORT_LABELS[cohort]}\n"
                f"검증 n={n_rows:,}, 양성={n_positive:,}"
            )
            axis.set_xticks(x, model_labels, rotation=20, ha="right")
            axis.set_ylim(0, 1.05)
            axis.set_ylabel("Recall")
            axis.grid(axis="y", alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("원산국 기후평년값 추가 전후 Recall · 동일 검증표본", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = figure_dir / "country_climate_recall_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=False)
    for row, target in enumerate(TARGETS):
        for column, cohort in enumerate(COHORTS):
            axis = axes[row, column]
            subset = bootstrap.loc[
                bootstrap["target"].eq(target)
                & bootstrap["cohort"].eq(cohort)
                & bootstrap["metric"].eq("recall")
            ].set_index("model").loc[model_order]
            y = np.arange(len(model_order))
            x = subset["delta_mean"].to_numpy()
            low = subset["ci_low"].to_numpy()
            high = subset["ci_high"].to_numpy()
            axis.errorbar(
                x,
                y,
                xerr=np.vstack([x - low, high - x]),
                fmt="o",
                color="#0F766E",
                ecolor="#5EEAD4",
                capsize=4,
            )
            axis.axvline(0, color="#334155", linestyle="--", linewidth=1)
            axis.set_yticks(y, model_labels)
            axis.set_title(f"{TARGET_LABELS[target]} · {COHORT_LABELS[cohort]}")
            axis.set_xlabel("Recall 변화(내부+기후 - 내부-only)")
            axis.grid(axis="x", alpha=0.3)
    fig.suptitle("기후평년값 추가 Recall 변화 · paired bootstrap 95% CI", fontsize=17, fontweight="bold")
    fig.tight_layout()
    path = figure_dir / "country_climate_recall_delta_ci.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    heat = deltas.copy()
    heat["row_label"] = heat.apply(
        lambda row: f"{row['target_label']} | {row['cohort_label']} | {row['model_label']}",
        axis=1,
    )
    columns = [
        "delta_recall",
        "delta_f2",
        "delta_average_precision",
        "delta_top10_capture_rate",
    ]
    values = heat[columns].to_numpy(dtype=float)
    limit = max(0.01, float(np.nanmax(np.abs(values))))
    fig, axis = plt.subplots(figsize=(10, 14))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(np.arange(len(heat)), heat["row_label"])
    axis.set_xticks(
        np.arange(len(columns)),
        ["Δ Recall", "Δ F2", "Δ PR-AUC", "Δ Top10% 포착률"],
        rotation=15,
        ha="right",
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", fontsize=8)
    axis.set_title("원산국 기후평년값 추가 성능 변화 요약")
    fig.colorbar(image, ax=axis, label="내부+기후 - 내부-only")
    fig.tight_layout()
    path = figure_dir / "country_climate_delta_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def render_report(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: pd.DataFrame,
    quality: list[dict[str, Any]],
    climate_columns: list[str],
    selected_windows: dict[str, Any],
    report_path: Path,
) -> None:
    recall_ci = bootstrap.loc[bootstrap["metric"].eq("recall"), [
        "target", "cohort", "model", "ci_low", "ci_high"
    ]]
    summary = deltas.merge(recall_ci, on=["target", "cohort", "model"], how="left")
    summary["Recall 95% CI"] = summary.apply(
        lambda row: f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]", axis=1
    )
    summary["판정"] = np.select(
        [
            summary["delta_recall"].gt(0)
            & summary["ci_low"].gt(0)
            & summary["delta_average_precision"].ge(0)
            & summary["delta_f2"].ge(-0.01),
            summary["delta_recall"].gt(0),
            summary["delta_recall"].abs().lt(1e-12)
            & summary["delta_average_precision"].gt(0)
            & summary["delta_f2"].ge(-0.01),
        ],
        ["기후 포함 우선", "튜닝에서 병행", "동률·조건부 병행"],
        default="내부-only 우선",
    )
    table = summary[[
        "target_label",
        "cohort_label",
        "model_label",
        "internal_recall",
        "climate_recall",
        "delta_recall",
        "Recall 95% CI",
        "delta_f2",
        "delta_average_precision",
        "delta_top10_capture_rate",
        "판정",
    ]].copy()
    table.columns = [
        "목표",
        "모집단",
        "모델",
        "내부 Recall",
        "기후 Recall",
        "Δ Recall",
        "Δ Recall 95% CI",
        "Δ F2",
        "Δ PR-AUC",
        "Δ Top10% 포착률",
        "판정",
    ]
    numeric = ["내부 Recall", "기후 Recall", "Δ Recall", "Δ F2", "Δ PR-AUC", "Δ Top10% 포착률"]
    table[numeric] = table[numeric].round(4)
    primary = table.loc[table["모집단"].eq("전체 연결국가")]
    sensitivity = table.loc[table["모집단"].eq("고신뢰 국가")]

    quality_rows = []
    for item in quality:
        for split_name in ("train", "validation"):
            split = item["split_rows"][split_name]
            quality_rows.append(
                {
                    "목표": TARGET_LABELS[item["target"]],
                    "모집단": COHORT_LABELS[item["cohort"]],
                    "구간": "학습" if split_name == "train" else "검증",
                    "n": split["rows"],
                    "양성": split["positive"],
                    "양성률(%)": round(100 * split["positive_rate"], 3),
                }
            )
    quality_table = pd.DataFrame(quality_rows)
    favorable = primary.loc[primary["판정"].eq("기후 포함 우선")]
    lines = [
        "# 07-10. 원산국 기후평년값 3개 목표×5개 모델 효과 비교",
        "",
        "## 한눈에 보기",
        "",
        "- 비교: 내부변수-only vs 내부변수+원산국 월별 기후평년값",
        "- 주 분석: 기후가 연결된 전체 국가코드",
        "- 민감도 분석: 생산국·원산국 직접 근거의 고신뢰 국가코드",
        f"- 주 분석에서 기후 포함 우선 기준 충족: {len(favorable)}/15개 목표·모델 조합",
        "- 모든 5개 모델은 향후 튜닝 대상에 유지하며, 기후 포함 여부만 목표·모델별 분기함.",
        "- 2026년 테스트셋: 미예측·미평가",
        "- DB·통합원장 변경: 없음",
        "",
        "## 비교 설계",
        "",
        "- 학습기간: 2015~2024년",
        "- 검증기간: 2025년",
        "- 내부 변수군: 확장형",
        "- 임계값: 각 입력군의 2025년 검증 weighted F2 최대",
        "- 1순위 지표: Recall",
        "- 보조 지표: Precision·F1·F2·PR-AUC·Top 10% 포착률",
        "- 불확실성: 동일 검증행 paired stratified bootstrap 200회",
        "- LSTM window: " + ", ".join(
            f"{TARGET_LABELS[target]} {int(selected_windows[target]['window_days'])}일"
            for target in TARGETS
        ),
        "",
        "## 사용 기후변수",
        "",
        *[f"- `{column}`" for column in climate_columns],
        "",
        "## 주 분석 결과 · 전체 연결국가",
        "",
        primary.to_markdown(index=False),
        "",
        "## 민감도 분석 · 고신뢰 국가",
        "",
        sensitivity.to_markdown(index=False),
        "",
        "## 모집단 규모",
        "",
        quality_table.to_markdown(index=False),
        "",
        "## 판정 기준",
        "",
        "- 기후 포함 우선: Recall 증가의 95% CI가 0 초과·PR-AUC 비감소·F2 감소 0.01 이내",
        "- 튜닝에서 병행: Recall은 증가했으나 불확실성 또는 다른 지표의 상충이 존재",
        "- 동률·조건부 병행: Recall 동률이며 PR-AUC 증가·F2 감소 0.01 이내",
        "- 내부-only 우선: Recall 감소 또는 동률에서 보조지표 개선 근거 부족",
        "- 위 판정은 외부변수 분기용 중간 판단이며 모델 탈락 기준이 아님.",
        "",
        "## 데이터 품질 점검",
        "",
        "- 3개 목표의 연결 전후 행 수와 record_id 순서 보존",
        "- 비교 모집단 기후 6개 변수 결측 0건",
        "- record_id 중복 0건",
        "- duplicate_group_id 학습·검증 교차 0건",
        "- 내부-only와 내부+기후는 목표·모집단별 동일 행 사용",
        "- 검사 결과·판정·MRL 등 사후정보는 입력변수에서 제외",
        "",
        "## 해석 주의사항",
        "",
        "- 수도 좌표의 월별 평년값은 실제 생산지·해당 연도 날씨가 아닌 국가 계절성 대리변수",
        "- 전체 연결국가에는 수입국 코드 대체값이 포함되어 원산국 직접값보다 신뢰도가 낮을 수 있음",
        "- 고신뢰 screening 검증은 88건·양성 16건으로 표본이 작아 신뢰구간 변동이 큼",
        "- 기준 부적합 역시 검증 양성이 적으므로 점추정만으로 기후 채택 여부를 확정하지 않음",
        "- 동일 검증셋에서 임계값까지 선택했으므로 최종 일반화 성능이 아님",
        "- 2026년 테스트셋은 외부변수와 하이퍼파라미터 확정 후 1회 평가",
        "",
        "## 산출물",
        "",
        "- 실행 코드: `country_climate_effect/`",
        "- 결과: `output/country_climate_effect_v1/`",
        "- 재현 노트북: `Country_Climate_Effect_Comparison.ipynb`",
        "- 외부변수 연결 근거: `07-9_원산국_월별_기후평년값_연결_파일럿.md`",
        "",
        "## 다음 작업",
        "",
        "- 목표·모델별 기후 포함/제외 분기를 5개 모델 전체 튜닝 실험대장에 반영",
        "- 모든 모델을 튜닝하되 각 모델의 내부-only와 기후 포함 후보를 독립적으로 기록",
        "- 최종 설정 확정 뒤 2026년 테스트셋 1회 평가",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_notebook(output_dir: Path, report_path: Path) -> Path:
    import nbformat as nbf

    metrics = pd.read_csv(output_dir / "validation_metrics.csv", encoding="utf-8-sig")
    deltas = pd.read_csv(output_dir / "climate_effect_deltas.csv", encoding="utf-8-sig")
    primary = deltas.loc[deltas["cohort"].eq("all")].copy()
    best = (
        primary.sort_values(["target", "delta_recall", "delta_average_precision"], ascending=[True, False, False])
        .groupby("target", observed=True)
        .head(1)
    )
    summary_lines = [
        f"- {TARGET_LABELS[row.target]}: Recall 변화가 가장 큰 모델은 {row.model_label} ({row.delta_recall:+.4f})"
        for row in best.itertuples(index=False)
    ]
    cells = [
        nbf.v4.new_markdown_cell(
            "# 원산국 기후평년값 모델 효과 비교\n\n"
            "## tl;dr\n\n" + "\n".join(summary_lines) +
            "\n- 결과는 2025년 검증셋 기준이며 2026년 테스트셋은 사용하지 않았다."
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "- 목표: 3개 목표×5개 모델에서 내부변수-only와 내부+기후평년을 동일 행으로 비교\n"
            "- 주 모집단: 전체 연결국가\n"
            "- 민감도 모집단: 고신뢰 국가코드\n"
            "- 기후값은 1991~2020 국가 대표지점 월별 평년값으로 실제 연도 날씨가 아니다.\n\n"
            "### Key Assumptions\n\n"
            "- 수도 좌표가 국가의 전형적인 계절 기후를 근사한다고 가정한다.\n"
            "- 수입국 코드 대체값은 별도 민감도 비교로 영향을 확인한다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n"
            "ROOT = Path.cwd()\n"
            "OUTPUT = ROOT / 'output' / 'country_climate_effect_v1'\n"
            "metrics = pd.read_csv(OUTPUT / 'validation_metrics.csv', encoding='utf-8-sig')\n"
            "deltas = pd.read_csv(OUTPUT / 'climate_effect_deltas.csv', encoding='utf-8-sig')\n"
            "quality = pd.read_csv(OUTPUT / 'population_summary.csv', encoding='utf-8-sig')"
        ),
        nbf.v4.new_markdown_cell("## Data"),
        nbf.v4.new_code_cell("quality"),
        nbf.v4.new_markdown_cell("## Results"),
        nbf.v4.new_code_cell(
            "deltas[['target_label','cohort_label','model_label','delta_recall','delta_f2','delta_average_precision','delta_top10_capture_rate']]"
        ),
        nbf.v4.new_code_cell(
            "for name in ['country_climate_recall_comparison.png','country_climate_recall_delta_ci.png','country_climate_delta_heatmap.png']:\n"
            "    display(Image(filename=str(OUTPUT / 'figures' / name)))"
        ),
        nbf.v4.new_markdown_cell(
            "## Takeaways\n\n"
            "- Recall을 1순위로 보되 PR-AUC·F2·Top 10% 포착률과 bootstrap 신뢰구간을 함께 해석한다.\n"
            "- 모든 모델은 튜닝 대상에 남기고, 기후 포함 여부만 목표·모델별로 분기한다.\n"
            "- 최종 설정 전까지 2026년 테스트셋을 사용하지 않는다."
        ),
        nbf.v4.new_markdown_cell(
            "## Reproduce\n\n"
            "아래 셀의 `RUN_TRAINING`을 `True`로 바꾸면 저장된 설정으로 전체 학습을 재실행한다."
        ),
        nbf.v4.new_code_cell(
            "RUN_TRAINING = False\n"
            "if RUN_TRAINING:\n"
            "    from country_climate_effect.experiment import run_experiment\n"
            "    run_experiment(ROOT / 'country_climate_effect' / 'config.yaml', device_name='auto')"
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    path = ROOT / "Country_Climate_Effect_Comparison.ipynb"
    nbf.write(notebook, path)
    return path


def run_experiment(
    config_path: Path,
    device_name: str = "auto",
    skip_lstm: bool = False,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    output_dir = ROOT / "output" / "country_climate_effect_v1"
    report_path = ROOT / "docs" / "외부변수_연결" / "07-10_원산국_기후평년_3개목표_5개모델_효과비교.md"
    figure_dir = output_dir / "figures"
    for path in (output_dir, output_dir / "models", figure_dir):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "output" / "features_v1" / "feature_engineering_manifest.json"
    baseline_config_path = ROOT / "baseline_modeling" / "config.yaml"
    selected_window_path = ROOT / config["lstm"]["selected_window_manifest"]
    feature_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    baseline_config = load_yaml(baseline_config_path)
    selected_windows = json.loads(selected_window_path.read_text(encoding="utf-8"))["selected_windows"]
    climate_columns = list(config["climate_numeric"])
    beta = float(config["comparison"]["threshold_beta"])
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")

    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    top_k_parts: list[pd.DataFrame] = []
    quality: list[dict[str, Any]] = []
    for target in TARGETS:
        for cohort in COHORTS:
            comparison, current_quality = load_comparison_frame(target, cohort, climate_columns)
            quality.append(current_quality)
            tab_metrics, tab_predictions, tab_top_k = run_tabular_pairs(
                target,
                cohort,
                comparison,
                feature_manifest,
                baseline_config,
                climate_columns,
                beta,
                output_dir,
            )
            metric_parts.append(tab_metrics)
            prediction_parts.append(tab_predictions)
            top_k_parts.append(tab_top_k)
            if not skip_lstm:
                lstm_metrics, lstm_predictions, lstm_top_k = run_lstm_pairs(
                    target,
                    cohort,
                    comparison,
                    config,
                    feature_manifest,
                    climate_columns,
                    output_dir,
                    device,
                    selected_windows,
                )
                metric_parts.append(lstm_metrics)
                prediction_parts.append(lstm_predictions)
                top_k_parts.append(lstm_top_k)
            pd.concat(metric_parts, ignore_index=True).to_csv(
                output_dir / "checkpoint_metrics.csv", index=False, encoding="utf-8-sig"
            )
            pd.concat(prediction_parts, ignore_index=True).to_parquet(
                output_dir / "checkpoint_predictions.parquet", index=False
            )

    metrics = pd.concat(metric_parts, ignore_index=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    top_k = pd.concat(top_k_parts, ignore_index=True)
    deltas = comparison_deltas(metrics)
    bootstrap = paired_bootstrap_deltas(
        predictions,
        int(config["comparison"]["bootstrap_repeats"]),
        float(config["comparison"]["bootstrap_confidence"]),
        int(config["seed"]),
        beta,
    )
    metrics.to_csv(output_dir / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    top_k.to_csv(output_dir / "validation_top_k.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(output_dir / "climate_effect_deltas.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "climate_effect_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    (output_dir / "data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    population_rows = []
    for item in quality:
        for split_name, split in item["split_rows"].items():
            population_rows.append(
                {
                    "target": item["target"],
                    "target_label": TARGET_LABELS[item["target"]],
                    "cohort": item["cohort"],
                    "cohort_label": COHORT_LABELS[item["cohort"]],
                    "split": split_name,
                    **split,
                }
            )
    pd.DataFrame(population_rows).to_csv(
        output_dir / "population_summary.csv", index=False, encoding="utf-8-sig"
    )
    figures = render_figures(metrics, deltas, bootstrap, figure_dir)
    render_report(
        metrics,
        deltas,
        bootstrap,
        quality,
        climate_columns,
        selected_windows,
        report_path,
    )
    notebook_path = build_notebook(output_dir, report_path)
    manifest = {
        "version": config["version"],
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "created_with": "country_climate_effect.experiment.run_experiment",
        "device": str(device),
        "skip_lstm": bool(skip_lstm),
        "test_data_used": False,
        "selection_split": "validation",
        "models": list(MODEL_LABELS if not skip_lstm else TABULAR_MODELS),
        "targets": list(TARGETS),
        "cohorts": list(COHORTS),
        "variants": list(VARIANTS),
        "climate_columns": climate_columns,
        "selected_windows": selected_windows,
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "feature_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "baseline_config": {"path": str(baseline_config_path), "sha256": sha256_file(baseline_config_path)},
            "window_manifest": {"path": str(selected_window_path), "sha256": sha256_file(selected_window_path)},
        },
        "quality": quality,
        "report": str(report_path),
        "notebook": str(notebook_path),
        "figures": [str(path) for path in figures],
    }
    manifest_output = output_dir / "manifest.json"
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "metrics": metrics,
        "deltas": deltas,
        "bootstrap": bootstrap,
        "quality": quality,
        "report_path": report_path,
        "manifest_path": manifest_output,
        "notebook_path": notebook_path,
        "device": str(device),
    }
