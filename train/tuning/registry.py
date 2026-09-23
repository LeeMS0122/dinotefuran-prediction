from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REQUIRED_COLUMNS = {
    "config_id",
    "target",
    "target_label",
    "model",
    "model_label",
    "tuning_tier",
    "candidate",
    "feature_set",
    "window_days",
    "selection_status",
    "selection_reason",
    "evidence_family",
}
TARGETS = {"occurrence", "screening", "noncompliance"}
MODELS = {"logistic", "catboost", "lightgbm", "xgboost", "lstm"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(root: Path, path: Path | None = None) -> dict[str, Any]:
    path = path or root / "tuning" / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def load_registry(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    spec = config["registry"]
    path = root / spec["source"]
    actual_hash = sha256_file(path)
    if actual_hash != spec["source_sha256"]:
        raise ValueError(
            f"튜닝 입력군 실험대장 해시 불일치: expected={spec['source_sha256']} actual={actual_hash}"
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"실험대장 필수 열 누락: {missing}")
    if len(frame) != int(spec["expected_rows"]):
        raise ValueError(f"실험대장 행 수 불일치: {len(frame)}")
    if frame["config_id"].duplicated().any():
        raise ValueError("config_id 중복")
    if set(frame["target"]) != TARGETS:
        raise ValueError(f"목표 집합 불일치: {sorted(frame['target'].unique())}")
    if set(frame["model"]) != MODELS:
        raise ValueError(f"모델 집합 불일치: {sorted(frame['model'].unique())}")
    counts = frame["tuning_tier"].value_counts().to_dict()
    if counts != {key: int(value) for key, value in spec["expected_tiers"].items()}:
        raise ValueError(f"튜닝 단계 수 불일치: {counts}")
    non_lstm_window = frame.loc[frame["model"].ne("lstm"), "window_days"].notna()
    if non_lstm_window.any():
        raise ValueError("LSTM 이외 모델에 window_days가 지정됨")
    if frame.loc[frame["model"].eq("lstm"), "window_days"].isna().any():
        raise ValueError("LSTM window_days 누락")
    return frame


def resolved_registry(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = frame.copy()
    defaults = config["optimization"]["default_trials"]
    result["study_name"] = result["config_id"].map(
        lambda value: f"dinotefuran_{str(value).lower().replace('-', '_')}"
    )
    result["default_trials"] = result["model"].map(defaults).astype(int)
    result["objective"] = config["optimization"]["objective"]
    result["threshold_rule"] = config["evaluation"]["threshold_metric"]
    result["selection_split"] = "validation"
    result["test_data_used"] = False
    return result


def write_validation_manifest(
    root: Path,
    config: dict[str, Any],
    registry: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolved_registry(registry, config)
    resolved_path = output_dir / "experiment_registry_resolved.csv"
    resolved.to_csv(resolved_path, index=False, encoding="utf-8-sig")
    manifest = {
        "version": config["version"],
        "registry_source": str(root / config["registry"]["source"]),
        "registry_sha256": config["registry"]["source_sha256"],
        "registry_rows": int(len(registry)),
        "unique_config_ids": int(registry["config_id"].nunique()),
        "tier_counts": {
            key: int(value)
            for key, value in registry["tuning_tier"].value_counts().sort_index().items()
        },
        "models": sorted(registry["model"].unique().tolist()),
        "targets": sorted(registry["target"].unique().tolist()),
        "test_data_used": False,
        "resolved_registry": str(resolved_path),
    }
    path = output_dir / "registry_validation.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
