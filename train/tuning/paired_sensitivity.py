from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from baseline_modeling.pipeline import (
    choose_fbeta_threshold,
    evaluate_scores,
    feature_sets,
    fit_model,
    top_k_table,
)
from country_climate_effect import experiment as country_experiment
from lstm_baseline.data import load_aligned_target
from lstm_baseline.training import resolve_device

from .data import external_lstm_columns, load_tabular_bundle
from .runner import (
    _core_only_manifest,
    _load_yaml,
    _normalise_lstm_metrics,
    _save_tabular_model,
    _top10_from_table,
    _write_json,
)
from .spaces import apply_lstm_params, apply_tabular_params


METRICS = ("recall", "f2", "average_precision", "top10_capture_rate")


def _internal_feature_columns(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
) -> tuple[list[str], list[str]]:
    manifest = json.loads(
        (root / config["inputs"]["feature_manifest"]).read_text(encoding="utf-8")
    )
    if row["evidence_family"] == "route_climate_paired":
        set_name = "core"
    elif row["evidence_family"] == "domestic_weather_paired":
        set_name = str(row["feature_set"])
    else:
        set_name = "extended"
    categorical, numeric = feature_sets(manifest["targets"][str(row["target"])])[set_name]
    return list(categorical), list(numeric)


def _prediction_metrics(
    predictions: pd.DataFrame,
    beta: float,
) -> dict[str, float]:
    thresholds = predictions["threshold"].drop_duplicates()
    if len(thresholds) != 1:
        raise ValueError("validation prediction threshold가 단일 값이 아님")
    y_true = predictions["target"].astype(int).to_numpy()
    scores = predictions["score"].astype(float).to_numpy()
    weights = predictions["group_sample_weight"].astype(float).to_numpy()
    threshold = float(thresholds.iloc[0])
    result = evaluate_scores(y_true, scores, weights, threshold, beta)
    top_k = top_k_table(y_true, scores, weights, [0.10])
    result["top10_capture_rate"] = _top10_from_table(top_k)
    return {key: float(value) for key, value in result.items()}


def _normalise_predictions(
    predictions: pd.DataFrame,
    variant: str,
) -> pd.DataFrame:
    required = {
        "record_id",
        "target",
        "group_sample_weight",
        "score",
        "threshold",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"validation prediction 필수 열 누락: {missing}")
    result = predictions[
        ["record_id", "target", "group_sample_weight", "score", "threshold"]
    ].copy()
    result["record_id"] = result["record_id"].astype(str)
    if result["record_id"].duplicated().any():
        raise ValueError(f"{variant}: validation record_id 중복")
    result["variant"] = variant
    return result


def _run_tabular_internal_control(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    params: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    bundle = load_tabular_bundle(root, row, config)
    categorical, numeric = _internal_feature_columns(root, row, config)
    train = bundle.frame.loc[bundle.frame["split"].eq("train")].copy()
    validation = bundle.frame.loc[bundle.frame["split"].eq("validation")].copy()
    model_name = str(row["model"])
    baseline_config = _load_yaml(root / config["inputs"]["baseline_config"])
    model_config = apply_tabular_params(
        baseline_config,
        model_name,
        params,
        int(config["seed"]),
    )
    if model_name == "xgboost":
        model_config["xgboost"]["device"] = str(device)
    started = time.perf_counter()
    model, scores = fit_model(
        model_name,
        train,
        validation,
        categorical,
        numeric,
        model_config,
    )
    fit_seconds = time.perf_counter() - started
    y_true = validation["target"].astype(int).to_numpy()
    weights = validation["group_sample_weight"].astype(float).to_numpy()
    beta = float(config["evaluation"]["threshold_beta"])
    threshold = choose_fbeta_threshold(y_true, scores, weights, beta)
    metrics = evaluate_scores(y_true, scores, weights, threshold, beta)
    top_k = top_k_table(y_true, scores, weights, [0.10])
    metrics["top10_capture_rate"] = _top10_from_table(top_k)
    checkpoint = _save_tabular_model(model, model_name, output_dir / "model")
    predictions = validation[
        ["record_id", "target", "group_sample_weight"]
    ].copy()
    predictions["score"] = scores
    predictions["threshold"] = threshold
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    result = {
        "variant": "internal_only",
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "checkpoint_path": checkpoint,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_hash": bundle.train_hash,
        "validation_hash": bundle.validation_hash,
        "population": bundle.population,
        "categorical": categorical,
        "numeric": numeric,
        "test_data_used": False,
    }
    _write_json(output_dir / "control_result.json", result)
    return result


def _run_lstm_internal_control(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    params: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    bundle = load_tabular_bundle(root, row, config)
    target = str(row["target"])
    window = int(float(row["window_days"]))
    sequence_dir = (
        root / "output" / "lstm_window_tuning_v1" / "sequences" / f"window_{window}d"
    )
    arrays, index, features = load_aligned_target(
        sequence_dir,
        root / "output" / "features_v1",
        target,
    )
    external_columns, _, cohort = external_lstm_columns(root, row, config)
    manifest = json.loads(
        (root / config["inputs"]["feature_manifest"]).read_text(encoding="utf-8")
    )
    if str(row["evidence_family"]) == "route_climate_paired":
        manifest = _core_only_manifest(manifest)
    base_config = _load_yaml(root / config["inputs"]["lstm_config"])
    tuned = apply_lstm_params(base_config, params, int(config["seed"]))
    shared_config = {
        "seed": int(config["seed"]),
        "comparison": {"threshold_beta": float(config["evaluation"]["threshold_beta"])},
        "lstm": {
            "masks": list(tuned["static_features"]["masks"]),
            "architecture": tuned["architecture"],
            "training": tuned["training"],
        },
    }
    country_experiment.COHORT_LABELS.setdefault(cohort, cohort)
    result, predictions, top_k = country_experiment.run_lstm_variant(
        target,
        cohort,
        arrays,
        index,
        features,
        set(bundle.frame["record_id"].astype(str)),
        "internal_only",
        shared_config,
        manifest,
        external_columns,
        output_dir,
        device,
        window,
    )
    metrics = _normalise_lstm_metrics(result, top_k)
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    checkpoint = output_dir / "models" / f"{target}_{cohort}_lstm_internal_only.pt"
    payload = {
        "variant": "internal_only",
        "metrics": metrics,
        "fit_seconds": float(result["fit_seconds"]),
        "checkpoint_path": checkpoint,
        "train_rows": int(bundle.frame["split"].eq("train").sum()),
        "validation_rows": int(bundle.frame["split"].eq("validation").sum()),
        "train_hash": bundle.train_hash,
        "validation_hash": bundle.validation_hash,
        "population": bundle.population,
        "test_data_used": False,
    }
    _write_json(output_dir / "control_result.json", payload)
    return payload


def paired_bootstrap_deltas(
    internal: pd.DataFrame,
    external: pd.DataFrame,
    repeats: int,
    confidence: float,
    seed: int,
    beta: float,
) -> pd.DataFrame:
    left = _normalise_predictions(internal, "internal_only").drop(columns="variant")
    right = _normalise_predictions(external, "internal_plus_external").drop(columns="variant")
    pair = left.merge(
        right,
        on=["record_id", "target", "group_sample_weight"],
        how="inner",
        validate="one_to_one",
        suffixes=("_internal", "_external"),
    )
    if len(pair) != len(left) or len(pair) != len(right):
        raise ValueError("internal/external validation 모집단이 일치하지 않음")
    y = pair["target"].to_numpy(dtype=int)
    weight = pair["group_sample_weight"].to_numpy(dtype=float)
    score_internal = pair["score_internal"].to_numpy(dtype=float)
    score_external = pair["score_external"].to_numpy(dtype=float)
    threshold_internal = float(pair["threshold_internal"].iloc[0])
    threshold_external = float(pair["threshold_external"].iloc[0])
    strata = [np.flatnonzero(y == value) for value in (0, 1)]
    if any(len(indices) == 0 for indices in strata):
        raise ValueError("paired bootstrap에 양성과 음성 strata가 모두 필요함")
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in METRICS}
    for _ in range(repeats):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in strata]
        )
        rng.shuffle(sampled)
        yi = y[sampled]
        wi = weight[sampled]
        internal_metrics = evaluate_scores(
            yi,
            score_internal[sampled],
            wi,
            threshold_internal,
            beta,
        )
        external_metrics = evaluate_scores(
            yi,
            score_external[sampled],
            wi,
            threshold_external,
            beta,
        )
        internal_top10 = _top10_from_table(
            top_k_table(yi, score_internal[sampled], wi, [0.10])
        )
        external_top10 = _top10_from_table(
            top_k_table(yi, score_external[sampled], wi, [0.10])
        )
        samples["recall"].append(external_metrics["recall"] - internal_metrics["recall"])
        samples["f2"].append(external_metrics["f2"] - internal_metrics["f2"])
        samples["average_precision"].append(
            external_metrics["average_precision"] - internal_metrics["average_precision"]
        )
        samples["top10_capture_rate"].append(external_top10 - internal_top10)
    alpha = (1 - confidence) / 2
    rows = []
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "delta_mean": float(array.mean()),
                "ci_low": float(np.quantile(array, alpha)),
                "ci_high": float(np.quantile(array, 1 - alpha)),
                "bootstrap_repeats": int(repeats),
                "confidence": float(confidence),
            }
        )
    return pd.DataFrame(rows)


def classify_paired_result(
    deltas: dict[str, float],
    bootstrap: pd.DataFrame,
) -> str:
    intervals = bootstrap.set_index("metric")
    if (
        deltas["recall"] > 0
        and deltas["f2"] >= 0
        and deltas["average_precision"] >= 0
        and float(intervals.loc["recall", "ci_low"]) >= 0
    ):
        return "promote_external_candidate"
    if (
        deltas["f2"] < 0
        or deltas["average_precision"] < 0
        or float(intervals.loc["recall", "ci_high"]) <= 0
    ):
        return "retain_internal_candidate"
    return "sensitivity_only_inconclusive"


def _selection_context(selection_reason: str) -> dict[str, float | None]:
    recall_match = re.search(r"Recall 변화 ([+-][0-9.]+)", selection_reason)
    linkage_match = re.search(r"연결률 ([0-9.]+)%", selection_reason)
    return {
        "upstream_paired_recall_delta": (
            float(recall_match.group(1)) if recall_match else None
        ),
        "validation_linkage_rate": (
            float(linkage_match.group(1)) / 100 if linkage_match else None
        ),
    }


def evaluate_external_sensitivity(
    root: Path,
    config: dict[str, Any],
    registry: pd.DataFrame,
    config_ids: list[str] | None = None,
    device_name: str = "auto",
    bootstrap_repeats: int = 500,
    confidence: float = 0.95,
    force: bool = False,
) -> dict[str, Any]:
    output_root = root / config["output"]["root"]
    paired_root = output_root / "paired_sensitivity"
    paired_root.mkdir(parents=True, exist_ok=True)
    rows = registry.loc[registry["tuning_tier"].eq("B_external_sensitivity")].copy()
    if config_ids:
        unknown = sorted(set(config_ids).difference(rows["config_id"]))
        if unknown:
            raise ValueError(f"B 외부 민감도 config가 아님: {unknown}")
        rows = rows.loc[rows["config_id"].isin(config_ids)].copy()
    else:
        rows = rows.loc[
            rows["config_id"].map(
                lambda value: (output_root / "runs" / value / "best_trial.json").exists()
            )
        ].copy()
    device = resolve_device(device_name)
    for row_index, (_, row) in enumerate(rows.sort_values("config_id").iterrows(), start=1):
        config_id = str(row["config_id"])
        best_path = output_root / "runs" / config_id / "best_trial.json"
        if not best_path.exists():
            raise FileNotFoundError(f"{config_id}: best_trial.json 없음")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        trial_dir = output_root / "runs" / config_id / f"trial_{int(best['trial_number']):04d}"
        external_path = trial_dir / "validation_predictions.parquet"
        control_dir = paired_root / config_id / "internal_only"
        control_dir.mkdir(parents=True, exist_ok=True)
        control_result_path = control_dir / "control_result.json"
        control_prediction_path = control_dir / "validation_predictions.parquet"
        if force or not (control_result_path.exists() and control_prediction_path.exists()):
            print(f"[{config_id}] internal-only matched control training", flush=True)
            if str(row["model"]) == "lstm":
                control = _run_lstm_internal_control(
                    root,
                    row,
                    config,
                    dict(best["params"]),
                    control_dir,
                    device,
                )
            else:
                control = _run_tabular_internal_control(
                    root,
                    row,
                    config,
                    dict(best["params"]),
                    control_dir,
                    device,
                )
        else:
            control = json.loads(control_result_path.read_text(encoding="utf-8"))
        internal_predictions = pd.read_parquet(control_prediction_path)
        external_predictions = pd.read_parquet(external_path)
        beta = float(config["evaluation"]["threshold_beta"])
        internal_metrics = _prediction_metrics(internal_predictions, beta)
        external_metrics = _prediction_metrics(external_predictions, beta)
        deltas = {
            metric: external_metrics[metric] - internal_metrics[metric] for metric in METRICS
        }
        bootstrap = paired_bootstrap_deltas(
            internal_predictions,
            external_predictions,
            bootstrap_repeats,
            confidence,
            int(config["seed"]) + row_index,
            beta,
        )
        decision = classify_paired_result(deltas, bootstrap)
        record = {
            "config_id": config_id,
            "run_id": best["run_id"],
            "target": str(row["target"]),
            "model": str(row["model"]),
            "candidate": str(row["candidate"]),
            "evidence_family": str(row["evidence_family"]),
            "population": control["population"],
            "train_hash": control["train_hash"],
            "validation_hash": control["validation_hash"],
            "train_rows": int(control["train_rows"]),
            "validation_rows": int(control["validation_rows"]),
            "internal_metrics": internal_metrics,
            "external_metrics": external_metrics,
            "deltas": deltas,
            "bootstrap": bootstrap.to_dict(orient="records"),
            "decision": decision,
            "upstream_context": _selection_context(str(row["selection_reason"])),
            "comparison_mode": "fixed_external_best_params_on_matched_population",
            "test_data_used": False,
        }
        _write_json(paired_root / config_id / "paired_result.json", record)
        print(f"[{config_id}] decision={decision}", flush=True)

    all_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(paired_root.glob("CFG-*/paired_result.json"))
    ]
    metric_rows = []
    delta_rows = []
    bootstrap_rows = []
    for record in all_results:
        base = {
            key: record[key]
            for key in (
                "config_id",
                "run_id",
                "target",
                "model",
                "candidate",
                "evidence_family",
                "population",
                "train_rows",
                "validation_rows",
                "decision",
            )
        }
        for variant_key, variant in (
            ("internal_metrics", "internal_only"),
            ("external_metrics", "internal_plus_external"),
        ):
            metric_rows.append({**base, "variant": variant, **record[variant_key]})
        delta_rows.append({**base, **{f"delta_{k}": v for k, v in record["deltas"].items()}})
        for item in record["bootstrap"]:
            bootstrap_rows.append({**base, **item})
    pd.DataFrame(metric_rows).to_csv(
        paired_root / "paired_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(delta_rows).to_csv(
        paired_root / "paired_deltas.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        paired_root / "paired_bootstrap_ci.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "version": config["version"],
        "comparison_mode": "fixed_external_best_params_on_matched_population",
        "selection_bias_caveat": (
            "The hyperparameters were selected from the external-feature trials; "
            "this paired check is sensitivity evidence, not an unbiased final test."
        ),
        "bootstrap_repeats": int(bootstrap_repeats),
        "confidence": float(confidence),
        "evaluated_configs": len(all_results),
        "decision_counts": pd.Series(
            [record["decision"] for record in all_results], dtype="object"
        ).value_counts().to_dict(),
        "test_data_used": False,
        "results": all_results,
    }
    _write_json(paired_root / "summary.json", summary)
    return summary
