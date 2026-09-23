from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .common import save_figure, save_table
from .targets import ratio_band


NUMERIC_FIELDS = {
    "result_value_numeric_enriched": "수치 결과값",
    "standard_numeric_enriched": "수치 기준값(MRL)",
    "result_to_mrl_ratio": "결과값/MRL",
}


def _coverage_by_source(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, group in df.groupby("source_system", dropna=False):
        for column, name in NUMERIC_FIELDS.items():
            available = int(group[column].notna().sum())
            rows.append(
                {
                    "source_system": source,
                    "field": column,
                    "field_name": name,
                    "total_rows": len(group),
                    "available_rows": available,
                    "missing_rows": len(group) - available,
                    "availability_pct": available / len(group) * 100,
                }
            )
    return pd.DataFrame(rows)


def _quantiles(df: pd.DataFrame) -> pd.DataFrame:
    quantile_points = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    rows = []
    groups = [("전체", df)]
    groups.extend((str(source), group) for source, group in df.groupby("source_system"))
    for source, group in groups:
        for column, name in NUMERIC_FIELDS.items():
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            record = {
                "source_system": source,
                "field": column,
                "field_name": name,
                "available_rows": len(values),
                "zero_rows": int(values.eq(0).sum()),
                "negative_rows": int(values.lt(0).sum()),
            }
            for point, value in values.quantile(quantile_points).items():
                record[f"q{int(point * 100):02d}"] = value
            rows.append(record)
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    coverage = _coverage_by_source(df)
    save_table(coverage, "step3_numeric_coverage.csv")

    quantiles = _quantiles(df)
    save_table(quantiles, "step3_numeric_quantiles.csv")

    work = df.loc[df["result_to_mrl_ratio"].notna(), ["source_system", "result_to_mrl_ratio"]].copy()
    work["ratio_band"] = ratio_band(work["result_to_mrl_ratio"])
    ratio_bands = (
        work.groupby(["source_system", "ratio_band"], observed=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    totals = ratio_bands.groupby("source_system")["rows"].transform("sum")
    ratio_bands["share_pct"] = ratio_bands["rows"] / totals * 100
    save_table(ratio_bands, "step3_ratio_bands_by_source.csv")

    pivot = coverage.pivot(
        index="source_system",
        columns="field_name",
        values="availability_pct",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, color=["#4472C4", "#ED7D31", "#70AD47"])
    ax.set_title("출처별 결과값·MRL·비율 가용률")
    ax.set_xlabel("출처")
    ax.set_ylabel("가용률 (%)")
    ax.set_ylim(0, 105)
    ax.legend(title="", frameon=False)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, "step3_numeric_coverage_by_source.png")

    result = pd.to_numeric(df["result_value_numeric_enriched"], errors="coerce")
    ratio = pd.to_numeric(df["result_to_mrl_ratio"], errors="coerce")
    positive_result = result[result > 0]
    positive_ratio = ratio[ratio > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if not positive_result.empty:
        axes[0].hist(
            np.log10(positive_result),
            bins=50,
            color="#4472C4",
            alpha=0.85,
        )
    axes[0].set_title("양수 수치 결과값 분포")
    axes[0].set_xlabel("log10(결과값)")
    axes[0].set_ylabel("행 수")
    axes[0].ticklabel_format(axis="y", style="plain")

    if not positive_ratio.empty:
        axes[1].hist(
            np.log10(positive_ratio),
            bins=50,
            color="#ED7D31",
            alpha=0.85,
        )
    axes[1].axvline(np.log10(0.1), color="#C00000", linestyle="--", label="MRL 10%")
    axes[1].axvline(0, color="#7030A0", linestyle="--", label="MRL 100%")
    axes[1].set_title("양수 결과값/MRL 비율 분포")
    axes[1].set_xlabel("log10(결과값/MRL)")
    axes[1].set_ylabel("행 수")
    axes[1].ticklabel_format(axis="y", style="plain")
    axes[1].legend(frameon=False)
    fig.suptitle("수치형 결과 분포 · 0 및 결측 제외", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "step3_numeric_distributions.png")

    band_pivot = ratio_bands.pivot(
        index="source_system",
        columns="ratio_band",
        values="share_pct",
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    band_pivot.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#A9D18E", "#FFD966", "#F4B183"],
    )
    ax.set_title("출처별 MRL 대비 농도 구간 구성")
    ax.set_xlabel("출처")
    ax.set_ylabel("구성비 (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="", frameon=False, loc="upper right")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_figure(fig, "step3_ratio_band_by_source.png")

    return {
        "numeric_coverage": coverage,
        "numeric_quantiles": quantiles,
        "ratio_bands": ratio_bands,
    }
