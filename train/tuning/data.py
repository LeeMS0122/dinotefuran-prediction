from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from baseline_modeling.pipeline import FORBIDDEN_PREFIXES, feature_sets


META_COLUMNS = [
    "record_id",
    "duplicate_group_id",
    "split",
    "target",
    "group_sample_weight",
]


@dataclass
class DataBundle:
    frame: pd.DataFrame
    categorical: list[str]
    numeric: list[str]
    input_files: list[Path]
    population: str
    train_hash: str
    validation_hash: str


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _load_feature_manifest(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(
        (root / config["inputs"]["feature_manifest"]).read_text(encoding="utf-8")
    )


def _read_model_splits(path: Path, columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(columns))
    try:
        return pd.read_parquet(
            path,
            columns=columns,
            filters=[("split", "in", ["train", "validation"])],
        )
    except (TypeError, ValueError):
        frame = pd.read_parquet(path, columns=columns)
        return frame.loc[frame["split"].isin(["train", "validation"])].copy()


def _stable_split_hash(frame: pd.DataFrame, split: str) -> str:
    selected = (
        frame.loc[frame["split"].eq(split), ["record_id", "target"]]
        .assign(record_id=lambda value: value["record_id"].astype(str))
        .sort_values("record_id")
    )
    text = "\n".join(
        f"{row.record_id}|{int(row.target)}" for row in selected.itertuples(index=False)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _quick_cap(
    frame: pd.DataFrame,
    cap_per_split: int | None,
    seed: int,
) -> pd.DataFrame:
    if cap_per_split is None:
        return frame
    parts: list[pd.DataFrame] = []
    for offset, split in enumerate(("train", "validation")):
        part = frame.loc[frame["split"].eq(split)]
        if len(part) <= cap_per_split:
            parts.append(part)
            continue
        positive = part.loc[part["target"].eq(1)]
        negative = part.loc[part["target"].eq(0)]
        positive_take = min(len(positive), max(2, cap_per_split // 4))
        negative_take = min(len(negative), cap_per_split - positive_take)
        if positive_take + negative_take < cap_per_split:
            positive_take = min(len(positive), cap_per_split - negative_take)
        sampled = pd.concat(
            [
                positive.sample(positive_take, random_state=seed + offset),
                negative.sample(negative_take, random_state=seed + offset + 10),
            ]
        ).sort_index()
        parts.append(sampled)
    return pd.concat(parts, ignore_index=True)


def _validate_bundle(
    frame: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> None:
    missing = sorted(set(META_COLUMNS + categorical + numeric).difference(frame.columns))
    if missing:
        raise ValueError(f"튜닝 데이터 필수 열 누락: {missing}")
    if set(frame["split"].astype(str).unique()) != {"train", "validation"}:
        raise ValueError("튜닝 데이터에는 train/validation만 있어야 함")
    if frame["record_id"].duplicated().any():
        raise ValueError("튜닝 데이터 record_id 중복")
    cross_split = (
        frame.groupby("duplicate_group_id", observed=True)["split"].nunique().gt(1).sum()
    )
    if int(cross_split):
        raise ValueError(f"튜닝 데이터 duplicate_group_id 분할 교차: {int(cross_split)}")
    for split in ("train", "validation"):
        part = frame.loc[frame["split"].eq(split)]
        if part.empty or part["target"].nunique() != 2:
            raise ValueError(f"{split}에 두 라벨이 모두 필요")
    forbidden = [
        column
        for column in categorical + numeric
        if column.startswith(FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"튜닝 입력에 누수 위험 변수 포함: {forbidden}")


def _weather_columns(config: dict[str, Any], window_days: int) -> list[str]:
    fields = config["external"]["domestic_weather"]["weather_base_fields"]
    return [f"wx_{field}_{window_days}d" for field in fields]


def load_tabular_bundle(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
    quick_cap_per_split: int | None = None,
) -> DataBundle:
    target = str(row["target"])
    family = str(row["evidence_family"])
    manifest = _load_feature_manifest(root, config)
    sets = feature_sets(manifest["targets"][target])
    feature_path = root / "output" / "features_v1" / f"{target}_features_v1.parquet"
    input_files = [feature_path]

    if family == "internal_full_population":
        categorical, numeric = sets[str(row["feature_set"])]
        frame = _read_model_splits(
            feature_path,
            META_COLUMNS + list(categorical) + list(numeric),
        )
        population = "full_label_confirmed_train_validation"
    elif family == "country_climate_paired":
        categorical, internal_numeric = sets["extended"]
        climate_config_path = root / config["external"]["country_climate"]["config"]
        climate_config = _load_yaml(climate_config_path)
        external_numeric = list(climate_config["climate_numeric"])
        climate_path = (
            root
            / "output"
            / "external_variables_v1"
            / "country_month_climate_pilot_v1"
            / f"{target}_country_climate_features_v1.parquet"
        )
        features = _read_model_splits(
            feature_path,
            META_COLUMNS + list(categorical) + list(internal_numeric),
        )
        climate = pd.read_parquet(
            climate_path,
            columns=[
                "record_id",
                "country_mapping_confidence",
                "country_climate_matched",
                *external_numeric,
            ],
        )
        frame = features.merge(
            climate,
            on="record_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        matched = frame["country_climate_matched"].fillna(False).astype(bool)
        if config["external"]["country_climate"]["cohort"] == "high":
            matched &= frame["country_mapping_confidence"].eq("high")
        frame = frame.loc[matched].copy()
        numeric = list(internal_numeric) + external_numeric
        input_files.extend([climate_path, climate_config_path])
        population = "country_climate_all_linked_train_validation"
    elif family == "route_climate_paired":
        route_manifest_path = root / config["external"]["route_climate"]["dataset_manifest"]
        route_manifest = json.loads(route_manifest_path.read_text(encoding="utf-8"))
        target_meta = route_manifest["targets"][target]
        route_path = Path(target_meta["paired_high_path"])
        categorical, internal_numeric = sets["core"]
        route_numeric = list(target_meta["route_numeric"])
        frame = pd.read_parquet(
            route_path,
            columns=list(
                dict.fromkeys(META_COLUMNS + list(categorical) + list(internal_numeric) + route_numeric)
            ),
        )
        numeric = list(internal_numeric) + route_numeric
        input_files.extend([route_path, route_manifest_path])
        population = "route_climate_high_confidence_train_validation"
    elif family == "domestic_weather_paired":
        if target != "screening":
            raise ValueError("국내기상 paired 튜닝은 screening만 지원")
        categorical, internal_numeric = sets[str(row["feature_set"])]
        row_window = row.get("window_days")
        window = (
            int(float(row_window))
            if pd.notna(row_window)
            else int(config["external"]["domestic_weather"]["tabular_window_days"])
        )
        external_numeric = _weather_columns(config, window)
        common_windows = [
            int(value)
            for value in config["external"]["domestic_weather"].get(
                "common_population_windows", [window]
            )
        ]
        coverage_columns = [f"wx_coverage_pct_{value}d" for value in common_windows]
        weather_path = (
            root
            / config["external"]["domestic_weather"]["weather_root"]
            / f"{target}_weather_features_v1.parquet"
        )
        features = _read_model_splits(
            feature_path,
            META_COLUMNS + list(categorical) + list(internal_numeric),
        )
        weather = pd.read_parquet(
            weather_path,
            columns=list(dict.fromkeys(["record_id", *coverage_columns, *external_numeric])),
        )
        frame = features.merge(
            weather,
            on="record_id",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        eligible = pd.Series(True, index=frame.index)
        for coverage_column in coverage_columns:
            eligible &= pd.to_numeric(frame[coverage_column], errors="coerce").ge(
                float(config["external"]["domestic_weather"]["minimum_coverage_pct"])
            )
        frame = frame.loc[eligible].copy()
        numeric = list(internal_numeric) + external_numeric
        input_files.append(weather_path)
        population = (
            "domestic_weather_common_"
            + "_".join(f"{value}d" for value in common_windows)
            + "_linked_train_validation"
        )
    else:
        raise ValueError(f"지원하지 않는 evidence_family: {family}")

    frame = _quick_cap(
        frame,
        quick_cap_per_split,
        int(config["seed"]),
    )
    categorical = list(categorical)
    numeric = list(numeric)
    _validate_bundle(frame, categorical, numeric)
    return DataBundle(
        frame=frame,
        categorical=categorical,
        numeric=numeric,
        input_files=input_files,
        population=population,
        train_hash=_stable_split_hash(frame, "train"),
        validation_hash=_stable_split_hash(frame, "validation"),
    )


def external_lstm_columns(
    root: Path,
    row: pd.Series,
    config: dict[str, Any],
) -> tuple[list[str], Path, str]:
    family = str(row["evidence_family"])
    target = str(row["target"])
    if family == "country_climate_paired":
        current = _load_yaml(root / config["external"]["country_climate"]["config"])
        path = (
            root
            / "output"
            / "external_variables_v1"
            / "country_month_climate_pilot_v1"
            / f"{target}_country_climate_features_v1.parquet"
        )
        return list(current["climate_numeric"]), path, "all"
    if family == "route_climate_paired":
        manifest = json.loads(
            (root / config["external"]["route_climate"]["dataset_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        meta = manifest["targets"][target]
        return list(meta["route_numeric"]), Path(meta["feature_path"]), "high"
    if family == "domestic_weather_paired":
        window = int(float(row["window_days"]))
        path = (
            root
            / config["external"]["domestic_weather"]["weather_root"]
            / f"{target}_weather_features_v1.parquet"
        )
        return _weather_columns(config, window), path, "domestic_weather"
    raise ValueError(f"외부 LSTM에 지원하지 않는 evidence_family: {family}")
