from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import chi2_contingency

from .common import save_eda2_figure, save_eda2_table
from .targets import target_specs, wilson_interval


SEASON_ORDER = ["봄", "여름", "가을", "겨울"]
TARGET_COLORS = {
    "occurrence": "#2F5597",
    "screening": "#ED7D31",
    "noncompliance": "#70AD47",
}


def season_from_month(month: int | float | None) -> str | None:
    if pd.isna(month):
        return None
    month = int(month)
    if month in (3, 4, 5):
        return "봄"
    if month in (6, 7, 8):
        return "여름"
    if month in (9, 10, 11):
        return "가을"
    if month in (12, 1, 2):
        return "겨울"
    return None


def summarize_target(
    df: pd.DataFrame,
    target_column: str,
    group_columns: list[str],
) -> pd.DataFrame:
    work = df[group_columns + [target_column]].copy()
    work[target_column] = pd.to_numeric(work[target_column], errors="coerce")
    grouped = (
        work.groupby(group_columns, dropna=False, observed=True)[target_column]
        .agg(total_rows="size", labeled_rows="count", positive_rows="sum")
        .reset_index()
    )
    grouped["positive_rows"] = grouped["positive_rows"].fillna(0).astype(int)
    grouped["labeled_rows"] = grouped["labeled_rows"].astype(int)
    grouped["total_rows"] = grouped["total_rows"].astype(int)
    grouped["missing_rows"] = grouped["total_rows"] - grouped["labeled_rows"]
    grouped["negative_rows"] = grouped["labeled_rows"] - grouped["positive_rows"]
    grouped["positive_rate_pct"] = np.where(
        grouped["labeled_rows"].gt(0),
        grouped["positive_rows"] / grouped["labeled_rows"] * 100,
        np.nan,
    )
    intervals = [
        wilson_interval(int(row.positive_rows), int(row.labeled_rows))
        for row in grouped.itertuples()
    ]
    grouped["ci95_low_pct"] = [low * 100 for low, _ in intervals]
    grouped["ci95_high_pct"] = [high * 100 for _, high in intervals]
    return grouped


def _association_test(df: pd.DataFrame, column: str, dimension: str) -> dict:
    work = df[[dimension, column]].dropna().copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=[column])
    table = pd.crosstab(work[dimension], work[column])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "chi2": np.nan,
            "degrees_of_freedom": np.nan,
            "p_value": np.nan,
            "cramers_v": np.nan,
            "labeled_rows": len(work),
        }
    chi2, p_value, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    cramers_v = sqrt((chi2 / n) / denominator) if n and denominator else np.nan
    return {
        "chi2": chi2,
        "degrees_of_freedom": dof,
        "p_value": p_value,
        "cramers_v": cramers_v,
        "labeled_rows": int(n),
    }


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    specs = target_specs(config)
    season_frames = []
    month_frames = []
    test_rows = []

    for spec in specs:
        season = summarize_target(df, spec.column, ["season"])
        season.insert(0, "target_key", spec.key)
        season.insert(1, "target_name", spec.name)
        season_frames.append(season)

        month = summarize_target(df, spec.column, ["event_month"])
        month.insert(0, "target_key", spec.key)
        month.insert(1, "target_name", spec.name)
        month_frames.append(month)

        test = _association_test(df, spec.column, "season")
        test_rows.append(
            {
                "target_key": spec.key,
                "target_name": spec.name,
                "dimension": "season",
                **test,
            }
        )

    season_summary = pd.concat(season_frames, ignore_index=True)
    month_summary = pd.concat(month_frames, ignore_index=True)
    tests = pd.DataFrame(test_rows)
    save_eda2_table(season_summary, "step4_season_target_summary.csv")
    save_eda2_table(month_summary, "step4_month_target_summary.csv")
    save_eda2_table(tests, "step4_season_association_tests.csv")

    date_basis = (
        df.groupby(["source_system", "date_basis"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    date_basis["share_within_source_pct"] = (
        date_basis["rows"]
        / date_basis.groupby("source_system")["rows"].transform("sum")
        * 100
    )
    save_eda2_table(date_basis, "step4_date_basis_by_source.csv")

    occurrence = next(spec for spec in specs if spec.key == "occurrence")
    sensitivity = summarize_target(
        df,
        occurrence.column,
        ["source_system", "date_basis", "season"],
    )
    save_eda2_table(
        sensitivity,
        "step4_occurrence_by_source_date_basis_season.csv",
    )

    main = season_summary[
        season_summary["target_key"].eq("occurrence")
    ].set_index("season").reindex(SEASON_ORDER).reset_index()
    x = np.arange(len(SEASON_ORDER))
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(
        x,
        main["labeled_rows"],
        color="#8EA9DB",
        label="판정 확정",
    )
    ax.bar(
        x,
        main["missing_rows"],
        bottom=main["labeled_rows"],
        color="#D9E1F2",
        label="미확정",
    )
    for position, total in zip(x, main["total_rows"]):
        ax.text(position, total, f"{int(total):,}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, SEASON_ORDER)
    ax.set_ylabel("검사 건수")
    ax.set_xlabel("계절")
    ax.ticklabel_format(axis="y", style="plain")
    ax.grid(axis="y", alpha=0.2)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        main["positive_rate_pct"],
        color="#C00000",
        marker="o",
        linewidth=2.2,
        label="잔류 발생률",
    )
    ax2.set_ylabel("잔류 발생률 (%)", color="#C00000")
    ax2.set_ylim(0, max(1.0, main["positive_rate_pct"].max() * 1.35))
    for position, rate in zip(x, main["positive_rate_pct"]):
        ax2.text(position, rate, f"{rate:.2f}%", color="#C00000", ha="center", va="bottom")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    ax.set_title("계절별 디노테푸란 검사 건수 및 잔류 발생률")
    fig.tight_layout()
    save_eda2_figure(fig, "step4_occurrence_season_ppt_style.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, spec in zip(axes, specs):
        plot_df = (
            season_summary[season_summary["target_key"].eq(spec.key)]
            .set_index("season")
            .reindex(SEASON_ORDER)
            .reset_index()
        )
        values = plot_df["positive_rate_pct"].to_numpy()
        low = plot_df["ci95_low_pct"].to_numpy()
        high = plot_df["ci95_high_pct"].to_numpy()
        ax.errorbar(
            x,
            values,
            yerr=np.vstack([values - low, high - values]),
            color=TARGET_COLORS[spec.key],
            marker="o",
            linewidth=2,
            capsize=4,
        )
        ax.set_xticks(x, SEASON_ORDER)
        ax.set_title(spec.name)
        ax.set_ylabel("양성률 (%)")
        ax.set_ylim(0, max(1.0, np.nanmax(high) * 1.2))
        ax.grid(axis="y", alpha=0.2)
        for position, row in enumerate(plot_df.itertuples()):
            ax.text(
                position,
                row.positive_rate_pct,
                f"{row.positive_rate_pct:.2f}%\nn={row.labeled_rows:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("계절별 3개 목표 라벨 양성률 · Wilson 95% 신뢰구간", fontweight="bold")
    fig.tight_layout()
    save_eda2_figure(fig, "step4_three_targets_by_season.png")

    sensitivity["group"] = (
        sensitivity["source_system"].astype(str)
        + " · "
        + sensitivity["date_basis"].fillna("날짜기준 미확인").astype(str)
    )
    group_sizes = sensitivity.groupby("group")["labeled_rows"].sum()
    eligible_groups = group_sizes[group_sizes >= 1000].index
    plot_sensitivity = sensitivity[sensitivity["group"].isin(eligible_groups)].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    for group, part in plot_sensitivity.groupby("group"):
        part = part.set_index("season").reindex(SEASON_ORDER)
        ax.plot(
            x,
            part["positive_rate_pct"],
            marker="o",
            linewidth=2,
            label=f"{group} (n={int(part['labeled_rows'].sum()):,})",
        )
    ax.set_xticks(x, SEASON_ORDER)
    ax.set_xlabel("계절")
    ax.set_ylabel("잔류 발생률 (%)")
    ax.set_title("출처·날짜 기준별 계절 잔류 발생률")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_eda2_figure(fig, "step4_occurrence_season_sensitivity.png")

    heat = summarize_target(df, occurrence.column, ["event_year", "event_month"])
    heat["display_rate_pct"] = heat["positive_rate_pct"].where(heat["labeled_rows"] >= 30)
    save_eda2_table(heat, "step4_occurrence_year_month.csv")
    rate_matrix = heat.pivot(
        index="event_year",
        columns="event_month",
        values="display_rate_pct",
    ).reindex(columns=range(1, 13))
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        rate_matrix,
        cmap="YlOrRd",
        vmin=0,
        cbar_kws={"label": "잔류 발생률 (%)"},
        linewidths=0.3,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("연도·월별 디노테푸란 잔류 발생률 · 월 확정 30건 이상")
    ax.set_xlabel("월")
    ax.set_ylabel("연도")
    fig.tight_layout()
    save_eda2_figure(fig, "step4_occurrence_year_month_heatmap.png")

    return {
        "season_target_summary": season_summary,
        "month_target_summary": month_summary,
        "season_association_tests": tests,
        "date_basis_by_source": date_basis,
        "occurrence_season_sensitivity": sensitivity,
        "occurrence_year_month": heat,
    }
