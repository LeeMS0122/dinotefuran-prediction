from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml
from catboost import CatBoostClassifier

from baseline_modeling.pipeline import (
    evaluate_scores,
    feature_sets,
    predict_scores,
    stratified_bootstrap_intervals,
    top_k_table,
)

from .runner import _write_json


EXPECTED_TARGETS = {"occurrence", "screening", "noncompliance"}


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


def build_final_candidate_lock(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_root = root / config["output"]["root"]
    selection_path = output_root / "internal_final_candidates.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected", [])
    targets = {str(item["target"]) for item in selected}
    if len(selected) != 3 or targets != EXPECTED_TARGETS:
        raise ValueError("최종 내부 후보는 세 target을 정확히 한 번씩 포함해야 함")
    candidates = []
    for item in sorted(selected, key=lambda value: value["target"]):
        config_id = str(item["config_id"])
        best_path = output_root / "runs" / config_id / "best_trial.json"
        best = json.loads(best_path.read_text(encoding="utf-8"))
        checkpoint = Path(best["checkpoint_path"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        for field in ("target", "model", "config_id"):
            if str(best[field]) != str(item[field]):
                raise ValueError(f"{config_id}: 내부 후보와 best_trial의 {field} 불일치")
        if bool(best.get("test_data_used")):
            raise ValueError(f"{config_id}: 후보 선택에 test 데이터가 사용됨")
        candidates.append(
            {
                "target": best["target"],
                "config_id": config_id,
                "run_id": best["run_id"],
                "model": best["model"],
                "feature_set": best["feature_set"],
                "params": best["params"],
                "threshold": best["threshold"],
                "validation_recall": best["recall"],
                "validation_f2": best["f2"],
                "validation_average_precision": best["average_precision"],
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "best_trial_sha256": _sha256(best_path),
            }
        )
    payload = {
        "version": config["version"],
        "selection_source": str(selection_path),
        "selection_source_sha256": _sha256(selection_path),
        "selection_split": "validation",
        "selection_period": "2025",
        "test_period": "2026-01-01/2026-06-30",
        "external_candidates_promoted": 0,
        "test_data_used_for_selection": False,
        "candidates": candidates,
    }
    lock_path = output_root / "final_candidate_lock.json"
    if lock_path.exists():
        current = json.loads(lock_path.read_text(encoding="utf-8"))
        if current != payload:
            raise RuntimeError("기존 final candidate lock과 현재 후보가 다름")
        return current
    _write_json(lock_path, payload)
    return payload


def evaluate_final_test(
    root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_root = root / config["output"]["root"]
    lock_path = output_root / "final_candidate_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError("final_candidate_lock.json을 먼저 생성해야 함")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_hash = _sha256(lock_path)
    final_root = output_root / "final_test_evaluation"
    manifest_path = final_root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["candidate_lock_sha256"] != lock_hash:
            raise RuntimeError("기존 최종 평가와 candidate lock이 다름")
        return manifest
    final_root.mkdir(parents=True, exist_ok=True)
    started_path = final_root / "evaluation_started.json"
    if started_path.exists():
        started = json.loads(started_path.read_text(encoding="utf-8"))
        if started["candidate_lock_sha256"] != lock_hash:
            raise RuntimeError("중단된 평가와 candidate lock이 다름")
    else:
        _write_json(
            started_path,
            {
                "candidate_lock_sha256": lock_hash,
                "test_period": lock["test_period"],
                "selection_frozen": True,
            },
        )
    feature_manifest = json.loads(
        (root / config["inputs"]["feature_manifest"]).read_text(encoding="utf-8")
    )
    evaluation_config = yaml.safe_load(
        (root / config["inputs"]["baseline_config"]).read_text(encoding="utf-8-sig")
    )["evaluation"]
    metric_rows = []
    top_k_rows = []
    bootstrap_rows = []
    target_results = []
    for candidate in lock["candidates"]:
        target = str(candidate["target"])
        result_path = final_root / f"{target}_result.json"
        if result_path.exists():
            target_results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        feature_path = root / "output" / "features_v1" / f"{target}_features_v1.parquet"
        categorical, numeric = feature_sets(feature_manifest["targets"][target])[
            str(candidate["feature_set"])
        ]
        columns = list(
            dict.fromkeys(
                [
                    "record_id",
                    "duplicate_group_id",
                    "event_date",
                    "event_year",
                    "split",
                    "target",
                    "group_sample_weight",
                    *categorical,
                    *numeric,
                ]
            )
        )
        frame = pd.read_parquet(feature_path, columns=columns)
        test = frame.loc[frame["split"].eq("test")].copy()
        if test.empty or test["target"].nunique() != 2:
            raise ValueError(f"{target}: test split 또는 이진 라벨이 유효하지 않음")
        dates = pd.to_datetime(test["event_date"], errors="raise")
        if dates.min() < pd.Timestamp("2026-01-01") or dates.max() > pd.Timestamp("2026-06-30"):
            raise ValueError(f"{target}: test 기간이 잠금 범위를 벗어남")
        checkpoint = Path(candidate["checkpoint_path"])
        if _sha256(checkpoint) != candidate["checkpoint_sha256"]:
            raise RuntimeError(f"{target}: checkpoint hash 변경")
        model = _load_model(str(candidate["model"]), checkpoint)
        scores = predict_scores(model, str(candidate["model"]), test, categorical, numeric)
        y_true = test["target"].astype(int).to_numpy()
        weights = test["group_sample_weight"].astype(float).to_numpy()
        threshold = float(candidate["threshold"])
        beta = float(config["evaluation"]["threshold_beta"])
        metrics = evaluate_scores(y_true, scores, weights, threshold, beta)
        top_k = top_k_table(
            y_true,
            scores,
            weights,
            [float(value) for value in config["evaluation"]["top_k_fractions"]],
        )
        bootstrap = stratified_bootstrap_intervals(
            y_true,
            scores,
            weights,
            threshold,
            [float(value) for value in config["evaluation"]["top_k_fractions"]],
            int(evaluation_config["bootstrap_repeats"]),
            float(evaluation_config["bootstrap_confidence"]),
            int(config["seed"]),
        )
        prediction = test[
            [
                "record_id",
                "duplicate_group_id",
                "event_date",
                "event_year",
                "target",
                "group_sample_weight",
            ]
        ].copy()
        prediction["score"] = scores
        prediction["prediction"] = (scores >= threshold).astype("int8")
        prediction["threshold"] = threshold
        prediction.to_parquet(final_root / f"{target}_test_predictions.parquet", index=False)
        result = {
            "target": target,
            "config_id": candidate["config_id"],
            "run_id": candidate["run_id"],
            "model": candidate["model"],
            "feature_set": candidate["feature_set"],
            "threshold": threshold,
            "threshold_source": "locked_2025_validation_weighted_f2",
            "test_rows": int(len(test)),
            "test_start": str(dates.min().date()),
            "test_end": str(dates.max().date()),
            "feature_input_sha256": _sha256(feature_path),
            "metrics": metrics,
            "top_k": top_k.to_dict(orient="records"),
            "bootstrap": bootstrap.to_dict(orient="records"),
            "test_data_used": True,
        }
        _write_json(result_path, result)
        target_results.append(result)
    for result in target_results:
        base = {
            "target": result["target"],
            "config_id": result["config_id"],
            "run_id": result["run_id"],
            "model": result["model"],
            "feature_set": result["feature_set"],
            "threshold": result["threshold"],
            "test_rows": result["test_rows"],
        }
        metric_rows.append({**base, **result["metrics"]})
        top_k_rows.extend({**base, **item} for item in result["top_k"])
        bootstrap_rows.extend({**base, **item} for item in result["bootstrap"])
    pd.DataFrame(metric_rows).to_csv(
        final_root / "test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(top_k_rows).to_csv(
        final_root / "test_top_k.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(bootstrap_rows).to_csv(
        final_root / "test_bootstrap_ci.csv", index=False, encoding="utf-8-sig"
    )
    manifest = {
        "version": config["version"],
        "candidate_lock": str(lock_path),
        "candidate_lock_sha256": lock_hash,
        "selection_frozen_before_test": True,
        "test_period": lock["test_period"],
        "test_data_used": True,
        "targets_evaluated": sorted(result["target"] for result in target_results),
        "results": target_results,
    }
    _write_json(manifest_path, manifest)
    return manifest
