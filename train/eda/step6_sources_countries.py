from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .common import save_eda3_figure, save_eda3_table
from .step4_season import _association_test, summarize_target
from .targets import grouped_target_summary, target_specs


SOURCE_LABELS = {
    "MFDS": "식약처",
    "IMPORT_LIMS_ONLY": "수입식품 단독",
    "INTEGRATED_DIST_ONLY": "통합망 단독",
    "SAFEQ": "농관원 SafeQ",
}
SOURCE_COLORS = {
    "MFDS": "#4472C4",
    "IMPORT_LIMS_ONLY": "#A5A5A5",
    "INTEGRATED_DIST_ONLY": "#ED7D31",
    "SAFEQ": "#70AD47",
}


def source_display(value: object) -> str:
    return SOURCE_LABELS.get(str(value), str(value))


def _availability(frame: pd.DataFrame, column: str) -> float:
    return frame[column].notna().mean() * 100 if len(frame) else np.nan


def _source_year_quality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, year), part in df.groupby(
        ["source_system", "event_year"], dropna=False, observed=True
    ):
        rows.append(
            {
                "source_system": source,
                "source_name": source_display(source),
                "event_year": year,
                "total_rows": len(part),
                "product_group_availability_pct": _availability(part, "product_group_raw"),
                "origin_country_availability_pct": _availability(part, "origin_country_name"),
                "result_numeric_availability_pct": _availability(
                    part, "result_value_numeric_enriched"
                ),
                "standard_numeric_availability_pct": _availability(
                    part, "standard_numeric_enriched"
                ),
                "occurrence_label_availability_pct": _availability(
                    part, "label_occurrence_v1"
                ),
                "screening_label_availability_pct": _availability(
                    part, "label_screening_10pct_v1"
                ),
                "noncompliance_label_availability_pct": _availability(
                    part, "label_noncompliance_v1"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_year", "source_system"])


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    specs = target_specs(config)
    latest_year = int(config["eda"]["latest_year"])
    min_n = int(config["eda"]["min_group_n"])

    volume = (
        df.groupby(["event_year", "source_system"], dropna=False, observed=True)
        .size().rename("rows").reset_index()
    )
    volume["year_total_rows"] = volume.groupby("event_year")["rows"].transform("sum")
    volume["share_within_year_pct"] = volume["rows"] / volume["year_total_rows"] * 100
    volume["source_name"] = volume["source_system"].map(source_display)
    volume["period_note"] = np.select(
        [volume["event_year"].lt(2016), volume["event_year"].eq(latest_year)],
        ["초기 소량 구간", "부분연도"],
        default="주 비교 구간",
    )
    save_eda3_table(volume, "step6_source_year_volume.csv")

    source_year_frames = []
    test_rows = []
    for spec in specs:
        table = summarize_target(df, spec.column, ["source_system", "event_year"])
        table["source_name"] = table["source_system"].map(source_display)
        table.insert(0, "target_key", spec.key)
        table.insert(1, "target_name", spec.name)
        source_year_frames.append(table)
        populations = [
            ("source_system", "source_system", df),
            ("event_year", "event_year", df),
            (
                "event_year_2016_2025",
                "event_year",
                df[df["event_year"].between(2016, 2025)],
            ),
            (
                "origin_country_name",
                "origin_country_name",
                df[df["origin_country_name"].notna()],
            ),
        ]
        for test_name, dimension, population in populations:
            test_rows.append(
                {
                    "target_key": spec.key,
                    "target_name": spec.name,
                    "dimension": test_name,
                    **_association_test(population, spec.column, dimension),
                }
            )
    source_year_targets = pd.concat(source_year_frames, ignore_index=True)
    association_tests = pd.DataFrame(test_rows)
    save_eda3_table(source_year_targets, "step6_source_year_target_summary.csv")
    save_eda3_table(association_tests, "step6_dimension_association_tests.csv")

    quality = _source_year_quality(df)
    save_eda3_table(quality, "step6_source_year_data_quality.csv")

    country_rows = []
    for source, part in df.groupby("source_system", dropna=False):
        present = int(part["origin_country_name"].notna().sum())
        country_rows.append(
            {
                "source_system": source,
                "source_name": source_display(source),
                "total_rows": len(part),
                "country_present_rows": present,
                "country_missing_rows": len(part) - present,
                "country_availability_pct": present / len(part) * 100,
                "country_unique_count": int(part["origin_country_name"].nunique()),
            }
        )
    present = int(df["origin_country_name"].notna().sum())
    country_rows.insert(
        0,
        {
            "source_system": "전체",
            "source_name": "전체",
            "total_rows": len(df),
            "country_present_rows": present,
            "country_missing_rows": len(df) - present,
            "country_availability_pct": present / len(df) * 100,
            "country_unique_count": int(df["origin_country_name"].nunique()),
        },
    )
    country_coverage = pd.DataFrame(country_rows)
    save_eda3_table(country_coverage, "step6_country_coverage.csv")

    country_data = df[df["origin_country_name"].notna()].copy()
    country_frames = []
    for spec in specs:
        table = grouped_target_summary(
            country_data, spec, "origin_country_name", min_n=min_n
        )
        table.insert(0, "target_key", spec.key)
        table.insert(1, "target_name", spec.name)
        country_frames.append(table)
    country_targets = pd.concat(country_frames, ignore_index=True)
    save_eda3_table(country_targets, "step6_country_target_summary.csv")

    occurrence_spec = next(spec for spec in specs if spec.key == "occurrence")
    country_occurrence = country_targets[
        country_targets["target_key"].eq("occurrence")
    ].copy()
    country_top = country_occurrence[
        country_occurrence["labeled_rows"].ge(100)
    ].sort_values(
        ["positive_rows", "labeled_rows"], ascending=False
    ).head(20)
    save_eda3_table(country_top, "step6_country_occurrence_top.csv")

    leading_countries = (
        country_occurrence.sort_values("labeled_rows", ascending=False)
        .head(15)["origin_country_name"].tolist()
    )
    country_year = summarize_target(
        country_data[country_data["origin_country_name"].isin(leading_countries)],
        occurrence_spec.column,
        ["origin_country_name", "event_year"],
    )
    country_year["display_rate_pct"] = country_year["positive_rate_pct"].where(
        country_year["labeled_rows"].ge(30)
    )
    save_eda3_table(country_year, "step6_country_year_occurrence.csv")

    core_volume = volume[volume["event_year"].between(2016, latest_year)].copy()
    years = sorted(core_volume["event_year"].dropna().astype(int).unique())
    sources = list(SOURCE_LABELS)
    count_matrix = (
        core_volume.pivot(index="event_year", columns="source_system", values="rows")
        .reindex(index=years, columns=sources).fillna(0)
    )
    share_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0) * 100
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    bottoms = np.zeros(len(count_matrix))
    for source in sources:
        values = count_matrix[source].to_numpy()
        axes[0].bar(
            count_matrix.index.astype(int), values, bottom=bottoms,
            label=SOURCE_LABELS[source], color=SOURCE_COLORS[source],
        )
        bottoms += values
    axes[0].set_ylabel("검사 건수")
    axes[0].set_title("연도별 출처 구성과 검사 규모")
    axes[0].legend(frameon=False, ncol=4, loc="upper left")
    axes[0].grid(axis="y", alpha=0.2)
    bottoms = np.zeros(len(share_matrix))
    for source in sources:
        values = share_matrix[source].to_numpy()
        axes[1].bar(
            share_matrix.index.astype(int), values, bottom=bottoms,
            label=SOURCE_LABELS[source], color=SOURCE_COLORS[source],
        )
        bottoms += values
    axes[1].set_ylabel("연도 내 비중 (%)")
    axes[1].set_xlabel("연도")
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].axvspan(
        latest_year - 0.45, latest_year + 0.45, color="#FFF2CC", alpha=0.35
    )
    fig.tight_layout()
    save_eda3_figure(fig, "step6_source_composition_by_year.png")

    occurrence_year = source_year_targets[
        source_year_targets["target_key"].eq("occurrence")
        & source_year_targets["event_year"].between(2016, latest_year)
        & source_year_targets["labeled_rows"].ge(30)
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    for source in sources:
        part = occurrence_year[
            occurrence_year["source_system"].eq(source)
        ].sort_values("event_year")
        if not part.empty:
            ax.plot(
                part["event_year"], part["positive_rate_pct"], marker="o",
                linewidth=2, color=SOURCE_COLORS[source], label=SOURCE_LABELS[source],
            )
    ax.set_xlabel("연도")
    ax.set_ylabel("잔류 발생률 (%)")
    ax.set_title("출처별 연도 잔류 발생률 · 연도별 판정 확정 30건 이상")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.axvspan(latest_year - 0.45, latest_year + 0.45, color="#FFF2CC", alpha=0.5)
    fig.tight_layout()
    save_eda3_figure(fig, "step6_occurrence_rate_by_source_year.png")

    quality_core = quality[quality["event_year"].between(2016, latest_year)]
    label_matrix = quality_core.pivot(
        index="source_name",
        columns="event_year",
        values="occurrence_label_availability_pct",
    )
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.heatmap(
        label_matrix, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
        cbar_kws={"label": "잔류 발생 라벨 가용률 (%)"},
        linewidths=0.4, linecolor="white", ax=ax,
    )
    ax.set_title("출처·연도별 잔류 발생 라벨 가용률")
    ax.set_xlabel("연도")
    ax.set_ylabel("출처")
    fig.tight_layout()
    save_eda3_figure(fig, "step6_occurrence_label_coverage_heatmap.png")

    coverage_plot = country_coverage[
        ~country_coverage["source_system"].eq("전체")
    ].sort_values("total_rows")
    y = np.arange(len(coverage_plot))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(
        y, coverage_plot["country_present_rows"], color="#4472C4", label="원산국 있음"
    )
    ax.barh(
        y, coverage_plot["country_missing_rows"],
        left=coverage_plot["country_present_rows"],
        color="#D9E2F3", label="원산국 결측",
    )
    ax.set_yticks(y, coverage_plot["source_name"])
    ax.set_xlabel("행 수")
    ax.set_title("출처별 원산국 정보 가용성")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    for position, row in enumerate(coverage_plot.itertuples()):
        ax.text(
            row.total_rows, position, f" {row.country_availability_pct:.1f}%",
            va="center",
        )
    ax.set_xlim(0, coverage_plot["total_rows"].max() * 1.12)
    fig.tight_layout()
    save_eda3_figure(fig, "step6_country_coverage_by_source.png")

    plot_country = country_top.head(15).sort_values("positive_rows")
    y = np.arange(len(plot_country))
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    axes[0].barh(y, plot_country["positive_rows"], color="#ED7D31")
    axes[0].set_yticks(y, plot_country["origin_country_name"])
    axes[0].set_xlabel("검출 건수")
    axes[0].set_title("검출 규모")
    axes[0].grid(axis="x", alpha=0.2)
    for pos, row in enumerate(plot_country.itertuples()):
        axes[0].text(row.positive_rows, pos, f" {row.positive_rows:,}", va="center")
    axes[0].set_xlim(0, max(1, plot_country["positive_rows"].max() * 1.25))
    axes[1].barh(y, plot_country["positive_rate_pct"], color="#4472C4")
    axes[1].set_xlabel("잔류 발생률 (%)")
    axes[1].set_title("발생률과 판정 확정 건수")
    axes[1].grid(axis="x", alpha=0.2)
    for pos, row in enumerate(plot_country.itertuples()):
        axes[1].text(
            row.positive_rate_pct, pos,
            f" {row.positive_rate_pct:.2f}% (n={row.labeled_rows:,})",
            va="center", fontsize=8,
        )
    axes[1].set_xlim(
        0, max(1, plot_country["positive_rate_pct"].max() * 1.55)
    )
    fig.suptitle(
        "원산국별 디노테푸란 검출 특성 · 원산국 정보 보유 행",
        fontweight="bold",
    )
    fig.tight_layout()
    save_eda3_figure(fig, "step6_country_occurrence_top.png")

    country_matrix = country_year.pivot(
        index="origin_country_name", columns="event_year", values="display_rate_pct"
    ).reindex(leading_countries)
    country_matrix = country_matrix.reindex(columns=range(2016, latest_year + 1))
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        country_matrix, cmap="YlOrRd", vmin=0,
        cbar_kws={"label": "잔류 발생률 (%)"},
        linewidths=0.3, linecolor="white", ax=ax,
    )
    ax.set_title("주요 원산국의 연도별 잔류 발생률 · 셀별 판정 확정 30건 이상")
    ax.set_xlabel("연도")
    ax.set_ylabel("원산국")
    fig.tight_layout()
    save_eda3_figure(fig, "step6_country_year_occurrence_heatmap.png")

    return {
        "source_year_volume": volume,
        "source_year_target_summary": source_year_targets,
        "source_year_data_quality": quality,
        "dimension_association_tests": association_tests,
        "country_coverage": country_coverage,
        "country_target_summary": country_targets,
        "country_occurrence_top": country_top,
        "country_year_occurrence": country_year,
    }
