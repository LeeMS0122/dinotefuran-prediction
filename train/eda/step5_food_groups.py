from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .common import save_eda2_figure, save_eda2_table
from .step4_season import _association_test
from .targets import grouped_target_summary, target_specs


TARGET_COLORS = {
    "occurrence": "#2F5597",
    "screening": "#ED7D31",
    "noncompliance": "#70AD47",
}


def _coverage_row(frame: pd.DataFrame, source: str) -> dict:
    total = len(frame)
    group_present = int(frame["product_group_raw"].notna().sum())
    product_present = int(frame["product_name_std"].notna().sum())
    return {
        "source_system": source,
        "total_rows": total,
        "group_present_rows": group_present,
        "group_missing_rows": total - group_present,
        "group_availability_pct": group_present / total * 100 if total else np.nan,
        "group_unique_count": int(frame["product_group_raw"].nunique(dropna=True)),
        "product_present_rows": product_present,
        "product_missing_rows": total - product_present,
        "product_availability_pct": product_present / total * 100 if total else np.nan,
        "product_unique_count": int(frame["product_name_std"].nunique(dropna=True)),
    }


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    min_n = int(config["eda"]["min_group_n"])
    top_n = int(config["eda"]["top_n"])
    specs = target_specs(config)

    coverage_rows = [_coverage_row(df, "전체")]
    coverage_rows.extend(
        _coverage_row(group, str(source))
        for source, group in df.groupby("source_system", dropna=False)
    )
    coverage = pd.DataFrame(coverage_rows)
    save_eda2_table(coverage, "step5_food_group_coverage.csv")

    valid_groups = df[df["product_group_raw"].notna()].copy()
    total_by_group = (
        valid_groups.groupby("product_group_raw")
        .size()
        .rename("total_rows")
        .reset_index()
    )
    group_frames = []
    test_rows = []
    for spec in specs:
        table = grouped_target_summary(
            valid_groups,
            spec,
            "product_group_raw",
            min_n=min_n,
        )
        table = table.merge(total_by_group, on="product_group_raw", how="left")
        table["missing_rows"] = table["total_rows"] - table["labeled_rows"]
        table.insert(0, "target_key", spec.key)
        table.insert(1, "target_name", spec.name)
        group_frames.append(table)

        test = _association_test(valid_groups, spec.column, "product_group_raw")
        test_rows.append(
            {
                "target_key": spec.key,
                "target_name": spec.name,
                "dimension": "product_group_raw",
                **test,
            }
        )

    group_summary = pd.concat(group_frames, ignore_index=True)
    tests = pd.DataFrame(test_rows)
    save_eda2_table(group_summary, "step5_food_group_target_summary.csv")
    save_eda2_table(tests, "step5_food_group_association_tests.csv")

    occurrence = group_summary[group_summary["target_key"].eq("occurrence")].copy()
    top_count = occurrence.sort_values("labeled_rows", ascending=False).head(top_n)
    save_eda2_table(top_count, "step5_occurrence_top_groups_by_count.csv")

    plot_df = top_count.sort_values("labeled_rows")
    y = np.arange(len(plot_df))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(6.5, 0.42 * len(plot_df) + 2)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    axes[0].barh(y, plot_df["labeled_rows"], color="#8EA9DB")
    axes[0].set_yticks(y, plot_df["product_group_raw"])
    axes[0].set_xlabel("판정 확정 건수")
    axes[0].set_title("검사 규모")
    axes[0].grid(axis="x", alpha=0.2)
    for position, value in zip(y, plot_df["labeled_rows"]):
        axes[0].text(value, position, f" {int(value):,}", va="center", fontsize=8)

    axes[1].barh(y, plot_df["positive_rows"], color="#ED7D31")
    axes[1].set_xlabel("검출 건수")
    axes[1].set_title("검출 규모")
    axes[1].grid(axis="x", alpha=0.2)
    for position, row in enumerate(plot_df.itertuples()):
        axes[1].text(
            row.positive_rows,
            position,
            f" {row.positive_rows:,}건 ({row.positive_rate_pct:.2f}%)",
            va="center",
            fontsize=8,
        )
    fig.suptitle("원천 품목군별 검사 건수 및 검출 건수", fontweight="bold")
    fig.tight_layout()
    save_eda2_figure(fig, "step5_occurrence_group_counts.png")

    stable = occurrence[occurrence["labeled_rows"] >= 1000].copy()
    stable = stable.sort_values("positive_rate_pct", ascending=False).head(top_n)
    stable = stable.sort_values("positive_rate_pct")
    values = stable["positive_rate_pct"].to_numpy()
    low = stable["ci95_low_pct"].to_numpy()
    high = stable["ci95_high_pct"].to_numpy()
    y = np.arange(len(stable))
    fig, ax = plt.subplots(figsize=(11, max(6, 0.45 * len(stable) + 2)))
    ax.barh(
        y,
        values,
        xerr=np.vstack([values - low, high - values]),
        color="#4472C4",
        error_kw={"ecolor": "#555555", "capsize": 2},
    )
    ax.set_yticks(
        y,
        [
            f"{name} (n={count:,})"
            for name, count in zip(stable["product_group_raw"], stable["labeled_rows"])
        ],
    )
    ax.set_xlabel("잔류 발생률 (%) · Wilson 95% 신뢰구간")
    ax.set_title("원천 품목군별 잔류 발생률 · 판정 확정 1,000건 이상")
    ax.set_xlim(0, max(1.0, high.max() * 1.08))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_eda2_figure(fig, "step5_occurrence_group_rates_stable.png")

    leading_groups = (
        occurrence.sort_values("labeled_rows", ascending=False)
        .head(20)["product_group_raw"]
        .tolist()
    )
    heat = group_summary[group_summary["product_group_raw"].isin(leading_groups)]
    rate_matrix = heat.pivot(
        index="product_group_raw",
        columns="target_name",
        values="positive_rate_pct",
    ).reindex(leading_groups)
    rate_matrix = rate_matrix[
        [spec.name for spec in specs if spec.name in rate_matrix.columns]
    ]
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(
        rate_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        cbar_kws={"label": "양성률 (%)"},
        linewidths=0.4,
        linecolor="white",
        ax=ax,
    )
    ax.set_title("검사 규모 상위 원천 품목군의 목표 라벨별 양성률")
    ax.set_xlabel("목표 라벨")
    ax.set_ylabel("원천 품목군")
    fig.tight_layout()
    save_eda2_figure(fig, "step5_three_targets_group_heatmap.png")

    top_groups = (
        occurrence.sort_values(["positive_rows", "labeled_rows"], ascending=False)
        .head(4)["product_group_raw"]
        .tolist()
    )
    detail_frames = []
    for group_name in top_groups:
        subset = valid_groups[valid_groups["product_group_raw"].eq(group_name)]
        detail = grouped_target_summary(
            subset,
            next(spec for spec in specs if spec.key == "occurrence"),
            "product_name_std",
            min_n=5,
        )
        detail.insert(0, "product_group_raw", group_name)
        detail_frames.append(detail)
    detail_summary = pd.concat(detail_frames, ignore_index=True)
    save_eda2_table(
        detail_summary,
        "step5_occurrence_detail_products_in_top_groups.csv",
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, group_name in zip(axes.flat, top_groups):
        detail = detail_summary[
            detail_summary["product_group_raw"].eq(group_name)
        ].copy()
        detail = detail.sort_values(
            ["positive_rows", "labeled_rows"],
            ascending=False,
        ).head(8).sort_values("positive_rows")
        positions = np.arange(len(detail))
        ax.barh(positions, detail["positive_rows"], color="#ED7D31")
        ax.set_yticks(
            positions,
            [
                f"{name} (n={count:,})"
                for name, count in zip(detail["product_name_std"], detail["labeled_rows"])
            ],
        )
        ax.set_xlabel("검출 건수")
        ax.set_title(group_name)
        ax.grid(axis="x", alpha=0.2)
        for position, row in enumerate(detail.itertuples()):
            ax.text(
                row.positive_rows,
                position,
                f" {row.positive_rows:,}건 · {row.positive_rate_pct:.1f}%",
                va="center",
                fontsize=8,
            )
    fig.suptitle("검출 건수 상위 원천 품목군의 세부 품목", fontweight="bold")
    fig.tight_layout()
    save_eda2_figure(fig, "step5_occurrence_detail_products.png")

    return {
        "food_group_coverage": coverage,
        "food_group_target_summary": group_summary,
        "food_group_association_tests": tests,
        "occurrence_top_groups_by_count": top_count,
        "occurrence_detail_products": detail_summary,
    }
