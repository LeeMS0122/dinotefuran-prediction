from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .common import save_eda3_figure, save_eda3_table
from .step4_season import SEASON_ORDER, summarize_target
from .targets import target_specs


def collapse_categories(
    series: pd.Series,
    leading_categories: list[str],
    other_label: str = "기타",
) -> pd.Series:
    values = series.fillna("결측").astype(str)
    return values.where(values.isin(leading_categories), other_label)


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    specs = target_specs(config)
    valid = df[df["product_group_raw"].notna() & df["season"].notna()].copy()

    target_frames = []
    for spec in specs:
        table = summarize_target(
            valid, spec.column, ["product_group_raw", "season"]
        )
        table.insert(0, "target_key", spec.key)
        table.insert(1, "target_name", spec.name)
        target_frames.append(table)
    group_season_targets = pd.concat(target_frames, ignore_index=True)
    save_eda3_table(
        group_season_targets,
        "step7_group_season_target_summary.csv",
    )

    occurrence = group_season_targets[
        group_season_targets["target_key"].eq("occurrence")
    ].copy()
    overall = summarize_target(
        valid, "label_occurrence_v1", ["product_group_raw"]
    ).rename(
        columns={
            "labeled_rows": "group_labeled_rows",
            "positive_rows": "group_positive_rows",
            "positive_rate_pct": "group_positive_rate_pct",
        }
    )
    stable = occurrence.merge(
        overall[
            [
                "product_group_raw",
                "group_labeled_rows",
                "group_positive_rows",
                "group_positive_rate_pct",
            ]
        ],
        on="product_group_raw",
        how="left",
    )
    stable = stable[
        stable["group_labeled_rows"].ge(1000)
        & stable["labeled_rows"].ge(100)
    ].copy()
    stable["rate_difference_pct_point"] = (
        stable["positive_rate_pct"] - stable["group_positive_rate_pct"]
    )
    stable["rate_ratio_vs_group"] = np.where(
        stable["group_positive_rate_pct"].gt(0),
        stable["positive_rate_pct"] / stable["group_positive_rate_pct"],
        np.nan,
    )
    stable = stable.sort_values(
        ["group_labeled_rows", "product_group_raw", "season"],
        ascending=[False, True, True],
    )
    save_eda3_table(
        stable,
        "step7_occurrence_group_season_stable.csv",
    )

    leading_groups = (
        overall.sort_values("group_labeled_rows", ascending=False)
        .head(10)["product_group_raw"].tolist()
    )
    valid["group_for_composition"] = collapse_categories(
        valid["product_group_raw"], leading_groups
    )
    composition = (
        valid.groupby(
            ["season", "group_for_composition"],
            observed=True,
        )
        .size()
        .rename("rows")
        .reset_index()
    )
    composition["season_total_rows"] = composition.groupby(
        "season", observed=True
    )["rows"].transform("sum")
    composition["share_within_season_pct"] = (
        composition["rows"] / composition["season_total_rows"] * 100
    )
    save_eda3_table(
        composition,
        "step7_season_group_composition.csv",
    )

    heat = stable[
        stable["product_group_raw"].isin(leading_groups)
    ].pivot(
        index="product_group_raw",
        columns="season",
        values="positive_rate_pct",
    ).reindex(index=leading_groups, columns=SEASON_ORDER)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        cbar_kws={"label": "잔류 발생률 (%)"},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("주요 원천 품목군×계절 잔류 발생률")
    ax.set_xlabel("계절")
    ax.set_ylabel("원천 품목군")
    fig.tight_layout()
    save_eda3_figure(fig, "step7_group_season_occurrence_heatmap.png")

    chart_groups = (
        overall.sort_values("group_positive_rows", ascending=False)
        .head(6)["product_group_raw"].tolist()
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    for ax, group_name in zip(axes.flat, chart_groups):
        part = stable[stable["product_group_raw"].eq(group_name)].copy()
        part["season"] = pd.Categorical(
            part["season"], categories=SEASON_ORDER, ordered=True
        )
        part = part.sort_values("season")
        positions = part["season"].cat.codes.to_numpy()
        if not part.empty:
            values = part["positive_rate_pct"].to_numpy()
            low = part["ci95_low_pct"].to_numpy()
            high = part["ci95_high_pct"].to_numpy()
            ax.errorbar(
                positions,
                values,
                yerr=np.vstack([values - low, high - values]),
                marker="o",
                linewidth=2,
                capsize=3,
                color="#4472C4",
            )
            for position, row in zip(positions, part.itertuples()):
                ax.text(
                    position,
                    row.positive_rate_pct,
                    f"{row.positive_rate_pct:.1f}%\nn={row.labeled_rows:,}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax.set_ylim(0, max(1, float(np.nanmax(high)) * 1.28))
        ax.set_xticks(range(4), SEASON_ORDER)
        ax.set_title(group_name)
        ax.set_ylabel("잔류 발생률 (%)")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "검출 건수 상위 원천 품목군의 계절별 잔류 발생률",
        fontweight="bold",
    )
    fig.tight_layout()
    save_eda3_figure(fig, "step7_top_groups_seasonal_profiles.png")

    composition_order = leading_groups + ["기타"]
    composition_matrix = composition.pivot(
        index="season",
        columns="group_for_composition",
        values="share_within_season_pct",
    ).reindex(
        index=SEASON_ORDER,
        columns=composition_order,
    ).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    palette = sns.color_palette("tab20", n_colors=len(composition_order))
    bottoms = np.zeros(len(composition_matrix))
    for color, group_name in zip(palette, composition_order):
        values = composition_matrix[group_name].to_numpy()
        ax.bar(
            SEASON_ORDER,
            values,
            bottom=bottoms,
            label=group_name,
            color=color,
        )
        bottoms += values
    ax.set_ylim(0, 100)
    ax.set_xlabel("계절")
    ax.set_ylabel("원천 품목군 구성비 (%)")
    ax.set_title("계절별 원천 품목군 구성")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_eda3_figure(fig, "step7_season_group_composition.png")

    return {
        "group_season_target_summary": group_season_targets,
        "occurrence_group_season_stable": stable,
        "season_group_composition": composition,
    }
