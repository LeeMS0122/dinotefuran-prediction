from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    key: str
    column: str
    name: str
    purpose: str


def target_specs(config: dict) -> list[TargetSpec]:
    return [
        TargetSpec(
            key=key,
            column=value["column"],
            name=value["name"],
            purpose=value["purpose"],
        )
        for key, value in config["targets"].items()
    ]


def validate_binary(series: pd.Series, column: str) -> None:
    values = set(pd.to_numeric(series.dropna(), errors="coerce").dropna().unique())
    invalid = values - {0, 1}
    if invalid:
        raise ValueError(f"{column}에 0/1 이외 값이 있습니다: {sorted(invalid)}")


def label_summary(df: pd.DataFrame, spec: TargetSpec) -> dict:
    values = pd.to_numeric(df[spec.column], errors="coerce")
    validate_binary(values, spec.column)
    population = int(values.notna().sum())
    positive = int(values.eq(1).sum())
    negative = int(values.eq(0).sum())
    return {
        "target_key": spec.key,
        "target_name": spec.name,
        "target_column": spec.column,
        "purpose": spec.purpose,
        "total_rows": int(len(df)),
        "labeled_rows": population,
        "missing_rows": int(values.isna().sum()),
        "positive_rows": positive,
        "negative_rows": negative,
        "positive_rate_pct": positive / population * 100 if population else np.nan,
    }


def wilson_interval(positive: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    p = positive / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half = z * sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def grouped_target_summary(
    df: pd.DataFrame,
    spec: TargetSpec,
    group_column: str,
    min_n: int = 1,
) -> pd.DataFrame:
    work = df[[group_column, spec.column]].copy()
    work[group_column] = work[group_column].fillna("결측").astype(str)
    work[spec.column] = pd.to_numeric(work[spec.column], errors="coerce")
    work = work.dropna(subset=[spec.column])
    grouped = (
        work.groupby(group_column, dropna=False)[spec.column]
        .agg(labeled_rows="size", positive_rows="sum")
        .reset_index()
    )
    grouped["positive_rows"] = grouped["positive_rows"].astype(int)
    grouped["negative_rows"] = grouped["labeled_rows"] - grouped["positive_rows"]
    grouped["positive_rate_pct"] = grouped["positive_rows"] / grouped["labeled_rows"] * 100
    intervals = [
        wilson_interval(int(row.positive_rows), int(row.labeled_rows))
        for row in grouped.itertuples()
    ]
    grouped["ci95_low_pct"] = [low * 100 for low, _ in intervals]
    grouped["ci95_high_pct"] = [high * 100 for _, high in intervals]
    grouped = grouped[grouped["labeled_rows"] >= min_n]
    return grouped.sort_values(
        ["positive_rate_pct", "labeled_rows"],
        ascending=[False, False],
    ).reset_index(drop=True)


def ratio_band(series: pd.Series) -> pd.Categorical:
    values = pd.to_numeric(series, errors="coerce")
    labels = ["MRL 10% 미만", "MRL 10~100%", "MRL 초과"]
    assigned = np.select(
        [values.lt(0.1), values.ge(0.1) & values.le(1.0), values.gt(1.0)],
        labels,
        default=None,
    )
    return pd.Categorical(
        assigned,
        categories=labels,
        ordered=True,
    )
