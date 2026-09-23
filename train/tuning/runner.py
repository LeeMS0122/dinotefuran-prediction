from __future__ import annotations

import copy
import json
import math
import shutil
import time
import traceback
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import yaml

from baseline_modeling.pipeline import (
    choose_fbeta_threshold,
    evaluate_scores,
    fit_model,
    top_k_table,
)
from country_climate_effect import experiment as country_experiment
from lstm_baseline.data import load_aligned_target
from lstm_baseline.training import resolve_device, run_target

from .data import external_lstm_columns, load_tabular_bundle
from .spaces import apply_lstm_params, apply_tabular_params, sample_params


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_tabular_model(model: Any, model_name: str, path_stem: Path) -> Path:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "catboost":
        path = path_stem.with_suffix(".cbm")
        model.save_model(path)
        return path
    path = path_stem.with_suffix(".joblib")
    joblib.dump(model, path)
    return path


def _top10_from_table(table: pd.DataFrame) -> float:
    if "top_fraction" in table:
        row = table.loc[np.isclose(table["top_fraction"], 0.10)].iloc[0]
        return float(row["capture_rate"])
    if "top_k_pct" in table:
        row = table.loc[np.isclose(table["top_k_pct"], 10)].iloc[0]
        value = row.get("weighted_capture_rate_pct", row.get("capture_rate_pct"))
        return float(value) / 100.0
    raise ValueError("Top 10% 지표 열을 찾을 수 없음")


def _normalise_lstm_metrics(
    metrics: dict[str, Any],
    top_k: pd.DataFrame | list[dict[str, Any]],
) -> dict[str, Any]:
    if "weighted_recall" in metrics:
        result = {
            "n_rows": int(metrics["n_rows"]),
            "n_positive": int(metrics["n_positive"]),
            "threshold": float(metrics["threshold"]),
            "accuracy": float(metrics["weighted_accuracy"]),
            "precision": float(metrics["weighted_precision"]),
            "recall": float(metrics["weighted_recall"]),
            "f1": float(metrics["weighted_f1"]),
            "f2": float(metrics["weighted_f2"]),
            "average_precision": float(metrics["weighted_pr_auc"]),
            "roc_auc": float(metrics["weighted_roc_auc"]),
        }
    else:
        result = {
            key: metrics[key]
            for key in [
                "n_rows",
                "n_positive",
                "threshold",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "f2",
                "average_precision",
                "roc_auc",
            ]
        }
    table = pd.DataFrame(top_k)
    result["top10_capture_rate"] = _top10_from_table(table)
    return result


def _run_tabular_trial(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    params: dict[str, Any],
    trial_dir: Path,
    device: torch.device,
    quick_cap_per_split: int | None,
) -> dict[str, Any]:
    bundle = load_tabular_bundle(
        root,
        row,
        config,
        quick_cap_per_split=quick_cap_per_split,
    )
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
        bundle.categorical,
        bundle.numeric,
        model_config,
    )
    fit_seconds = time.perf_counter() - started
    y_true = validation["target"].astype(int).to_numpy()
    weights = validation["group_sample_weight"].astype(float).to_numpy()
    beta = float(config["evaluation"]["threshold_beta"])
    threshold = choose_fbeta_threshold(y_true, scores, weights, beta)
    metrics = evaluate_scores(y_true, scores, weights, threshold, beta)
    top_k = top_k_table(
        y_true,
        scores,
        weights,
        [float(value) for value in config["evaluation"]["top_k_fractions"]],
    )
    metrics["top10_capture_rate"] = _top10_from_table(top_k)
    checkpoint = _save_tabular_model(
        model,
        model_name,
        trial_dir / "model",
    )
    predictions = validation[
        ["record_id", "duplicate_group_id", "target", "group_sample_weight"]
    ].copy()
    predictions["score"] = scores
    predictions["threshold"] = threshold
    predictions["prediction"] = (scores >= threshold).astype("int8")
    predictions.to_parquet(trial_dir / "validation_predictions.parquet", index=False)
    top_k.to_csv(trial_dir / "validation_top_k.csv", index=False, encoding="utf-8-sig")
    return {
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "checkpoint_path": checkpoint,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_hash": bundle.train_hash,
        "validation_hash": bundle.validation_hash,
        "population": bundle.population,
        "input_files": bundle.input_files,
    }


def _sequence_hash(index: pd.DataFrame, split: str) -> str:
    import hashlib

    part = (
        index.loc[index["split"].eq(split), ["record_id", "target"]]
        .assign(record_id=lambda value: value["record_id"].astype(str))
        .sort_values("record_id")
    )
    text = "\n".join(
        f"{row.record_id}|{int(row.target)}" for row in part.itertuples(index=False)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_internal_lstm_trial(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    params: dict[str, Any],
    trial_dir: Path,
    device: torch.device,
    quick_cap_per_split: int | None,
    smoke: bool,
) -> dict[str, Any]:
    base_config = _load_yaml(root / config["inputs"]["lstm_config"])
    lstm_config = apply_lstm_params(
        base_config,
        params,
        int(config["seed"]),
        max_epochs_override=2 if smoke else None,
    )
    window = int(float(row["window_days"]))
    sequence_dir = (
        root / "output" / "lstm_window_tuning_v1" / "sequences" / f"window_{window}d"
    )
    result = run_target(
        target_name=str(row["target"]),
        sequence_dir=sequence_dir,
        feature_dir=root / "output" / "features_v1",
        output_dir=trial_dir,
        config=lstm_config,
        device=device,
        quick_cap=quick_cap_per_split,
        evaluate_test=False,
    )
    metrics = _normalise_lstm_metrics(result["metrics"][0], result["top_k"])
    index = pd.read_parquet(
        sequence_dir / f"{row['target']}_sequence_index_v1.parquet",
        columns=["record_id", "target", "split"],
    )
    checkpoint = trial_dir / f"{row['target']}_lstm_baseline_v1.pt"
    return {
        "metrics": metrics,
        "fit_seconds": float(result["fit_seconds"]),
        "checkpoint_path": checkpoint,
        "train_rows": int(result["train_rows"]),
        "validation_rows": int(result["validation_rows"]),
        "train_hash": _sequence_hash(index, "train"),
        "validation_hash": _sequence_hash(index, "validation"),
        "population": "full_label_confirmed_train_validation",
        "input_files": [
            sequence_dir / f"{row['target']}_sequences_v1.npz",
            sequence_dir / f"{row['target']}_sequence_index_v1.parquet",
        ],
    }


def _core_only_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    for target in result["targets"]:
        sets = result["targets"][target]["feature_sets"]
        sets["extended_categorical"] = []
        sets["conditional_categorical"] = []
    return result


def _run_external_lstm_trial(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    params: dict[str, Any],
    trial_dir: Path,
    device: torch.device,
    quick_cap_per_split: int | None,
    smoke: bool,
) -> dict[str, Any]:
    bundle = load_tabular_bundle(
        root,
        row,
        config,
        quick_cap_per_split=quick_cap_per_split,
    )
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
    external_columns, external_path, cohort = external_lstm_columns(root, row, config)
    external = pd.read_parquet(
        external_path,
        columns=["record_id", *external_columns],
    )
    features = features.merge(
        external,
        on="record_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if not index["record_id"].astype(str).equals(features["record_id"].astype(str)):
        raise ValueError("외부 LSTM 연결 후 record_id 순서 변경")
    manifest = json.loads(
        (root / config["inputs"]["feature_manifest"]).read_text(encoding="utf-8")
    )
    if str(row["evidence_family"]) == "route_climate_paired":
        manifest = _core_only_manifest(manifest)
    base_config = _load_yaml(root / config["inputs"]["lstm_config"])
    tuned = apply_lstm_params(
        base_config,
        params,
        int(config["seed"]),
        max_epochs_override=2 if smoke else None,
    )
    shared_config = {
        "seed": int(config["seed"]),
        "comparison": {"threshold_beta": float(config["evaluation"]["threshold_beta"])},
        "lstm": {
            "masks": list(tuned["static_features"]["masks"]),
            "architecture": tuned["architecture"],
            "training": tuned["training"],
        },
    }
    country_experiment.COHORT_LABELS.setdefault(
        cohort,
        "국내기상 연결 모집단" if cohort == "domestic_weather" else cohort,
    )
    result, predictions, top_k = country_experiment.run_lstm_variant(
        target,
        cohort,
        arrays,
        index,
        features,
        set(bundle.frame["record_id"].astype(str)),
        "internal_plus_climate",
        shared_config,
        manifest,
        external_columns,
        trial_dir,
        device,
        window,
    )
    metrics = _normalise_lstm_metrics(result, top_k)
    predictions.to_parquet(trial_dir / "validation_predictions.parquet", index=False)
    checkpoint = (
        trial_dir
        / "models"
        / f"{target}_{cohort}_lstm_internal_plus_climate.pt"
    )
    return {
        "metrics": metrics,
        "fit_seconds": float(result["fit_seconds"]),
        "checkpoint_path": checkpoint,
        "train_rows": int(bundle.frame["split"].eq("train").sum()),
        "validation_rows": int(bundle.frame["split"].eq("validation").sum()),
        "train_hash": bundle.train_hash,
        "validation_hash": bundle.validation_hash,
        "population": bundle.population,
        "input_files": list(
            dict.fromkeys(
                [
                    *bundle.input_files,
                    external_path,
                    sequence_dir / f"{target}_sequences_v1.npz",
                    sequence_dir / f"{target}_sequence_index_v1.parquet",
                ]
            )
        ),
    }


def _trial_records(config_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(config_dir.glob("trial_*/trial.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "input_files" in record:
            record["input_files"] = list(dict.fromkeys(record["input_files"]))
        records.append(record)
    return records


def _ensure_prediction_alias(config_dir: Path, best: dict[str, Any]) -> None:
    trial_dir = config_dir / f"trial_{int(best['trial_number']):04d}"
    alias = trial_dir / "validation_predictions.parquet"
    if alias.exists():
        return
    native = sorted(trial_dir.glob("*_predictions_v1.parquet"))
    if len(native) == 1:
        shutil.copy2(native[0], alias)


def select_balanced_trial(
    records: list[dict[str, Any]],
    selection: dict[str, Any],
) -> dict[str, Any] | None:
    completed = [row for row in records if row["status"] == "complete"]
    if not completed:
        raise RuntimeError("완료된 trial이 없음")
    max_f2 = max(float(row["f2"]) for row in completed)
    max_ap = max(float(row["average_precision"]) for row in completed)
    eligible = [
        row
        for row in completed
        if float(row["f2"]) >= float(selection["f2_relative_floor"]) * max_f2
        and float(row["average_precision"])
        >= float(selection["average_precision_relative_floor"]) * max_ap
    ]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda row: (
            float(row["recall"]),
            float(row["f2"]),
            float(row["average_precision"]),
        ),
        reverse=True,
    )[0]


def _refresh_outputs(
    config_dir: Path,
    selection: dict[str, Any],
) -> dict[str, Any] | None:
    records = _trial_records(config_dir)
    if not records:
        return None
    frame = pd.DataFrame(records)
    frame["params"] = frame["params"].map(
        lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
    )
    frame["input_files"] = frame["input_files"].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )
    frame.to_csv(config_dir / "trials.csv", index=False, encoding="utf-8-sig")
    completed = [row for row in records if row["status"] == "complete"]
    if not completed:
        return None
    best = select_balanced_trial(records, selection)
    if best is None:
        (config_dir / "best_trial.json").unlink(missing_ok=True)
        return None
    _write_json(config_dir / "best_trial.json", best)
    return best


def run_config(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    spaces: dict[str, Any],
    output_root: Path,
    target_trials: int,
    device_name: str,
    quick_cap_per_split: int | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    config_id = str(row["config_id"])
    mode = "smoke" if smoke else "runs"
    mode_root = output_root / mode
    config_dir = mode_root / config_id
    config_dir.mkdir(parents=True, exist_ok=True)
    storage_path = mode_root / "optuna.db"
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_name = f"dinotefuran_{config_id.lower().replace('-', '_')}_{mode}"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{storage_path.resolve().as_posix()}",
        direction=str(config["optimization"]["direction"]),
        sampler=optuna.samplers.TPESampler(seed=int(config["seed"])),
        load_if_exists=True,
    )
    optuna_completed_before = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    completed_before = sum(
        row["status"] == "complete" for row in _trial_records(config_dir)
    )
    remaining = max(0, int(target_trials) - completed_before)
    device = resolve_device(device_name)

    def objective(trial: optuna.Trial) -> float:
        trial_dir = config_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        params = sample_params(trial, str(row["model"]), spaces)
        run_id = f"{config_id}-T{trial.number:04d}"
        started = time.perf_counter()
        base_record = {
            "run_id": run_id,
            "config_id": config_id,
            "trial_number": int(trial.number),
            "target": str(row["target"]),
            "model": str(row["model"]),
            "tuning_tier": str(row["tuning_tier"]),
            "candidate": str(row["candidate"]),
            "feature_set": str(row["feature_set"]),
            "window_days": (
                None if pd.isna(row["window_days"]) else int(float(row["window_days"]))
            ),
            "evidence_family": str(row["evidence_family"]),
            "params": params,
            "seed": int(config["seed"]),
            "smoke": bool(smoke),
            "test_data_used": False,
        }
        try:
            if str(row["model"]) == "lstm":
                if str(row["evidence_family"]) == "internal_full_population":
                    result = _run_internal_lstm_trial(
                        root,
                        row,
                        config,
                        params,
                        trial_dir,
                        device,
                        quick_cap_per_split,
                        smoke,
                    )
                else:
                    result = _run_external_lstm_trial(
                        root,
                        row,
                        config,
                        params,
                        trial_dir,
                        device,
                        quick_cap_per_split,
                        smoke,
                    )
            else:
                result = _run_tabular_trial(
                    root,
                    row,
                    config,
                    params,
                    trial_dir,
                    device,
                    quick_cap_per_split,
                )
            metrics = result["metrics"]
            record = {
                **base_record,
                "status": "complete",
                "error": None,
                "train_rows": result["train_rows"],
                "validation_rows": result["validation_rows"],
                "train_hash": result["train_hash"],
                "validation_hash": result["validation_hash"],
                "population": result["population"],
                "threshold": metrics["threshold"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "f2": metrics["f2"],
                "average_precision": metrics["average_precision"],
                "roc_auc": metrics["roc_auc"],
                "top10_capture_rate": metrics["top10_capture_rate"],
                "fit_seconds": result["fit_seconds"],
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_path": result["checkpoint_path"],
                "input_files": result["input_files"],
            }
            _write_json(trial_dir / "trial.json", record)
            _write_json(trial_dir / "params.json", params)
            trial.set_user_attr("threshold", float(metrics["threshold"]))
            trial.set_user_attr("f2", float(metrics["f2"]))
            trial.set_user_attr(
                "average_precision", float(metrics["average_precision"])
            )
            trial.set_user_attr("checkpoint_path", str(result["checkpoint_path"]))
            _refresh_outputs(config_dir, config["optimization"]["selection"])
            return float(metrics["recall"])
        except Exception as error:
            record = {
                **base_record,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_path": None,
                "input_files": [],
            }
            _write_json(trial_dir / "trial.json", record)
            _refresh_outputs(config_dir, config["optimization"]["selection"])
            raise

    if remaining:
        study.optimize(
            objective,
            n_trials=remaining,
            catch=(Exception,),
            show_progress_bar=False,
        )
    completed_after = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    best = _refresh_outputs(config_dir, config["optimization"]["selection"])
    if best is None:
        raise RuntimeError(
            f"{config_id}: F2 및 average precision 가드레일을 동시에 충족한 "
            "완료 trial이 없음"
        )
    _ensure_prediction_alias(config_dir, best)
    records = _trial_records(config_dir)
    valid_completed_after = sum(row["status"] == "complete" for row in records)
    invalid_trials = sum(row["status"] == "invalid" for row in records)
    failed_trials = sum(row["status"] == "failed" for row in records)
    summary = {
        "config_id": config_id,
        "study_name": study_name,
        "storage": str(storage_path),
        "target_trials": int(target_trials),
        "completed_before": int(completed_before),
        "completed_after": int(valid_completed_after),
        "optuna_completed_before": int(optuna_completed_before),
        "optuna_completed_after": int(completed_after),
        "invalid_trials": int(invalid_trials),
        "failed_trials": int(failed_trials),
        "resumed": bool(optuna_completed_before > 0),
        "smoke": bool(smoke),
        "test_data_used": False,
        "best_trial": best,
    }
    _write_json(config_dir / "run_summary.json", summary)
    return summary
