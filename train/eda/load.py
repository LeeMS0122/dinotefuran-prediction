from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .common import CACHE_DIR, data_path_from_env, ensure_dirs, load_config, write_json


SOURCE_COLUMNS = [
    "record_id",
    "source_system",
    "source_dataset",
    "is_canonical",
    "model_eligible_flag",
    "hazard_label",
    "judge_result",
    "result_raw",
    "result_unit",
    "standard_raw",
    "result_value_numeric_enriched",
    "standard_numeric_enriched",
    "standard_source",
    "result_to_mrl_ratio",
    "exceedance_flag",
    "product_name_std",
    "product_group_raw",
    "event_date",
    "date_basis",
    "origin_country_name",
    "collection_province",
    "label_occurrence_v1",
    "label_screening_10pct_v1",
    "label_noncompliance_v1",
    "model_eligible_occurrence",
    "model_eligible_screening",
    "model_eligible_noncompliance",
    "partial_year_flag",
]

NUMERIC_COLUMNS = [
    "result_value_numeric_enriched",
    "standard_numeric_enriched",
    "result_to_mrl_ratio",
    "exceedance_flag",
    "label_occurrence_v1",
    "label_screening_10pct_v1",
    "label_noncompliance_v1",
    "model_eligible_occurrence",
    "model_eligible_screening",
    "model_eligible_noncompliance",
]


def _signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "source_path": str(path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "selected_columns": SOURCE_COLUMNS,
    }


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["event_date_dt"] = pd.to_datetime(df["event_date"], errors="coerce")
    df["event_year"] = df["event_date_dt"].dt.year.astype("Int64")
    df["event_month"] = df["event_date_dt"].dt.month.astype("Int64")
    df["event_ym"] = df["event_date_dt"].dt.to_period("M").astype("string")
    month_to_season = {
        12: "겨울",
        1: "겨울",
        2: "겨울",
        3: "봄",
        4: "봄",
        5: "봄",
        6: "여름",
        7: "여름",
        8: "여름",
        9: "가을",
        10: "가을",
        11: "가을",
    }
    df["season"] = pd.Categorical(
        df["event_month"].map(month_to_season),
        categories=["봄", "여름", "가을", "겨울"],
        ordered=True,
    )
    return df


def load_analysis_data(force: bool = False) -> tuple[pd.DataFrame, Path]:
    config = load_config()
    source_path = data_path_from_env(config)
    cache_path = CACHE_DIR / "analysis.parquet"
    meta_path = CACHE_DIR / "analysis.meta.json"
    expected = _signature(source_path)
    ensure_dirs()

    if not force and cache_path.exists() and meta_path.exists():
        current = json.loads(meta_path.read_text(encoding="utf-8"))
        if current == expected:
            return pd.read_parquet(cache_path), source_path

    header = pd.read_csv(source_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    missing = sorted(set(SOURCE_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")

    df = pd.read_csv(
        source_path,
        usecols=SOURCE_COLUMNS,
        encoding="utf-8-sig",
        low_memory=False,
    )
    df = _derive_columns(df)
    df.to_parquet(cache_path, index=False)
    write_json(expected, meta_path)
    return df, source_path
