from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from baseline_modeling.pipeline import feature_sets, predict_scores
from tuning.paired_sensitivity import METRICS, _prediction_metrics, paired_bootstrap_deltas
from tuning.runner import _write_json

from .audit import load_config


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(model_name: str, checkpoint: Path) -> Any:
    if model_name == "catboost":
        model = CatBoostClassifier()
        model.load_model(checkpoint)
        return model
    return joblib.load(checkpoint)


def common_test_mask(
    frame: pd.DataFrame,
    windows: list[int],
    minimum_coverage_pct: float,
    start_date: str,
    end_date: str,
) -> pd.Series:
    dates = pd.to_datetime(frame["event_date"], errors="coerce")
    mask = frame["split"].eq("test") & dates.between(
        pd.Timestamp(start_date), pd.Timestamp(end_date), inclusive="both"
    )
    for window in windows:
        mask &= pd.to_numeric(
            frame[f"wx_coverage_pct_{window}d"], errors="coerce"
        ).ge(minimum_coverage_pct)
    return mask


def _cohort_hash(frame: pd.DataFrame) -> str:
    payload = frame[
        ["record_id", "target", "group_sample_weight"]
    ].sort_values("record_id", kind="stable")
    return hashlib.sha256(
        payload.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _load_locked_artifacts(root: Path) -> dict[str, Any]:
    from .cli import runtime_config

    weather = load_config(root)
    output_root = root / weather["output"]["root"]
    lock_path = output_root / "challenger_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("challenger_lock.json is required before shared test")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    selected = lock.get("selected")
    if lock.get("decision") != "weather_challenger_locked" or not selected:
        raise RuntimeError("a single weather challenger must be locked before shared test")
    if bool(lock.get("test_data_used")) or bool(selected.get("test_data_used")):
        raise RuntimeError("candidate lock must be development-only")

    config_id = str(selected["config_id"])
    best_path = output_root / "runs" / config_id / "best_trial.json"
    control_path = (
        output_root
        / "paired_sensitivity"
        / config_id
        / "internal_only"
        / "control_result.json"
    )
    best = json.loads(best_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if str(best["run_id"]) != str(selected["run_id"]):
        raise ValueError("locked run and best_trial run do not match")
    if str(best["model"]) != str(selected["model"]):
        raise ValueError("locked model and best_trial model do not match")
    if best["train_hash"] != control["train_hash"] or best["validation_hash"] != control["validation_hash"]:
        raise ValueError("weather/internal development populations do not match")

    weather_checkpoint = Path(best["checkpoint_path"])
    internal_checkpoint = Path(control["checkpoint_path"])
    if not weather_checkpoint.exists() or not internal_checkpoint.exists():
        raise FileNotFoundError("locked model checkpoint is missing")
    return {
        "weather_config": weather,
        "runtime_config": runtime_config(root),
        "output_root": output_root,
        "lock_path": lock_path,
        "selected": selected,
        "best_path": best_path,
        "best": best,
        "control_path": control_path,
        "control": control,
        "weather_checkpoint": weather_checkpoint,
        "internal_checkpoint": internal_checkpoint,
    }


def _load_shared_test_frame(
    root: Path, artifacts: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    weather = artifacts["weather_config"]
    runtime = artifacts["runtime_config"]
    best = artifacts["best"]
    manifest_path = root / runtime["inputs"]["feature_manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    categorical, internal_numeric = feature_sets(manifest["targets"]["screening"])[
        str(best["feature_set"])
    ]
    window = int(best["window_days"])
    weather_numeric = [
        f"wx_{field}_{window}d" for field in weather["weather"]["numeric_base_fields"]
    ]
    windows = [int(value) for value in weather["weather"]["windows"]]
    coverage_columns = [f"wx_coverage_pct_{value}d" for value in windows]
    feature_path = root / weather["inputs"]["features"]
    weather_path = root / weather["inputs"]["weather_v2"]
    meta = [
        "record_id",
        "split",
        "event_date",
        "target",
        "group_sample_weight",
    ]
    features = pd.read_parquet(
        feature_path,
        columns=list(dict.fromkeys([*meta, *categorical, *internal_numeric])),
        filters=[("split", "==", "test")],
    )
    external = pd.read_parquet(
        weather_path,
        columns=list(
            dict.fromkeys(
                ["record_id", "split", "event_date", *coverage_columns, *weather_numeric]
            )
        ),
        filters=[("split", "==", "test")],
    ).drop(columns=["split", "event_date"])
    if features["record_id"].duplicated().any() or external["record_id"].duplicated().any():
        raise ValueError("shared test input contains duplicate record_id")
    frame = features.merge(
        external, on="record_id", how="left", validate="one_to_one", sort=False
    )
    shared_test = weather["splits"]["shared_test"]
    mask = common_test_mask(
        frame,
        windows,
        float(weather["weather"]["minimum_coverage_pct"]),
        str(shared_test["start_date"]),
        str(shared_test["end_date"]),
    )
    frame = frame.loc[mask].copy()
    if frame.empty:
        raise ValueError("shared test common weather cohort is empty")
    if set(frame["target"].dropna().astype(int).unique()) != {0, 1}:
        raise ValueError("shared test must contain both target classes")
    if frame["record_id"].duplicated().any():
        raise ValueError("shared test cohort contains duplicate record_id")
    return frame, list(categorical), list(internal_numeric), weather_numeric


def evaluate_shared_test(
    root: Path = ROOT,
    bootstrap_repeats: int | None = None,
) -> dict[str, Any]:
    artifacts = _load_locked_artifacts(root)
    weather = artifacts["weather_config"]
    output_root = artifacts["output_root"]
    final_root = output_root / "shared_2026_benchmark"
    manifest_path = final_root / "result.json"
    lock_hash = _sha256(artifacts["lock_path"])
    test_period = {
        key: value.isoformat() if hasattr(value, "isoformat") else value
        for key, value in weather["splits"]["shared_test"].items()
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("candidate_lock_sha256") != lock_hash:
            raise RuntimeError("candidate lock changed after shared test evaluation")
        return {**existing, "reused_existing_result": True}

    started_path = final_root / "evaluation_started.json"
    if started_path.exists():
        raise RuntimeError(
            "shared test evaluation was already started but has no result; manual audit required"
        )
    final_root.mkdir(parents=True, exist_ok=True)
    feature_path = root / weather["inputs"]["features"]
    weather_path = root / weather["inputs"]["weather_v2"]
    started = {
        "version": weather["version"],
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_locked_before_test": True,
        "candidate_lock": str(artifacts["lock_path"]),
        "candidate_lock_sha256": lock_hash,
        "best_trial_sha256": _sha256(artifacts["best_path"]),
        "control_result_sha256": _sha256(artifacts["control_path"]),
        "weather_checkpoint_sha256": _sha256(artifacts["weather_checkpoint"]),
        "internal_checkpoint_sha256": _sha256(artifacts["internal_checkpoint"]),
        "feature_input_sha256": _sha256(feature_path),
        "weather_input_sha256": _sha256(weather_path),
        "test_period": test_period,
    }
    _write_json(started_path, started)

    frame, categorical, internal_numeric, weather_numeric = _load_shared_test_frame(
        root, artifacts
    )
    best = artifacts["best"]
    control = artifacts["control"]
    model_name = str(best["model"])
    internal_model = _load_model(model_name, artifacts["internal_checkpoint"])
    weather_model = _load_model(model_name, artifacts["weather_checkpoint"])
    internal_scores = predict_scores(
        internal_model, model_name, frame, categorical, internal_numeric
    )
    weather_scores = predict_scores(
        weather_model,
        model_name,
        frame,
        categorical,
        [*internal_numeric, *weather_numeric],
    )
    base = frame[["record_id", "target", "group_sample_weight"]].copy()
    internal_predictions = base.copy()
    internal_predictions["score"] = internal_scores
    internal_predictions["threshold"] = float(control["metrics"]["threshold"])
    weather_predictions = base.copy()
    weather_predictions["score"] = weather_scores
    weather_predictions["threshold"] = float(best["threshold"])
    internal_predictions.to_parquet(
        final_root / "internal_test_predictions.parquet", index=False
    )
    weather_predictions.to_parquet(
        final_root / "weather_test_predictions.parquet", index=False
    )

    beta = float(weather["experiment"]["threshold_beta"])
    internal_metrics = _prediction_metrics(internal_predictions, beta)
    weather_metrics = _prediction_metrics(weather_predictions, beta)
    deltas = {
        metric: weather_metrics[metric] - internal_metrics[metric]
        for metric in METRICS
    }
    repeat_count = int(
        bootstrap_repeats or weather["experiment"]["bootstrap_repeats"]
    )
    bootstrap = paired_bootstrap_deltas(
        internal_predictions,
        weather_predictions,
        repeat_count,
        float(weather["experiment"]["bootstrap_confidence"]),
        int(weather["seed"]) + 2026,
        beta,
    )
    bootstrap.to_csv(
        final_root / "paired_bootstrap_ci.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "version": weather["version"],
        "config_id": artifacts["selected"]["config_id"],
        "run_id": artifacts["selected"]["run_id"],
        "model": model_name,
        "window_days": int(best["window_days"]),
        "comparison": "locked_weather_vs_matched_internal_on_same_shared_test_cohort",
        "candidate_locked_before_test": True,
        "selection_changed_after_test": False,
        "candidate_lock_sha256": lock_hash,
        "test_period": test_period,
        "test_data_used": True,
        "test_rows": int(len(frame)),
        "test_positive": int(frame["target"].sum()),
        "test_cohort_sha256": _cohort_hash(frame),
        "coverage_windows": [int(value) for value in weather["weather"]["windows"]],
        "minimum_coverage_pct": float(weather["weather"]["minimum_coverage_pct"]),
        "internal_threshold_from_validation": float(control["metrics"]["threshold"]),
        "weather_threshold_from_validation": float(best["threshold"]),
        "internal_metrics": internal_metrics,
        "weather_metrics": weather_metrics,
        "deltas": deltas,
        "paired_bootstrap": bootstrap.to_dict(orient="records"),
        "bootstrap_repeats": repeat_count,
        "artifacts": {
            "evaluation_started": str(started_path),
            "internal_predictions": str(final_root / "internal_test_predictions.parquet"),
            "weather_predictions": str(final_root / "weather_test_predictions.parquet"),
            "paired_bootstrap_ci": str(final_root / "paired_bootstrap_ci.csv"),
        },
    }
    _write_json(manifest_path, result)
    return result
