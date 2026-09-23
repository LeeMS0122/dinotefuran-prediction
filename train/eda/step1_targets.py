from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .common import save_figure, save_table
from .targets import grouped_target_summary, label_summary, target_specs


COLORS = ["#2F5597", "#ED7D31", "#70AD47"]


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    specs = target_specs(config)
    summary = pd.DataFrame([label_summary(df, spec) for spec in specs])
    save_table(summary, "step1_target_summary.csv")

    source_frames = []
    year_frames = []
    for spec in specs:
        source = grouped_target_summary(df, spec, "source_system")
        source.insert(0, "target_key", spec.key)
        source.insert(1, "target_name", spec.name)
        source_frames.append(source)

        year = grouped_target_summary(df, spec, "event_year")
        year.insert(0, "target_key", spec.key)
        year.insert(1, "target_name", spec.name)
        year_frames.append(year)

    by_source = pd.concat(source_frames, ignore_index=True)
    by_year = pd.concat(year_frames, ignore_index=True)
    save_table(by_source, "step1_target_by_source.csv")
    save_table(by_year, "step1_target_by_year.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(summary))
    display_names = ["잔류 발생", "MRL 10%\n관심농도", "기준 부적합"]
    axes[0].bar(x, summary["labeled_rows"], color="#4472C4", label="확정")
    axes[0].bar(
        x,
        summary["missing_rows"],
        bottom=summary["labeled_rows"],
        color="#D9E2F3",
        label="미확정/결측",
    )
    axes[0].set_xticks(x, display_names)
    axes[0].set_ylabel("행 수")
    axes[0].set_title("목표 라벨별 사용 가능 모집단")
    axes[0].legend(frameon=False)
    axes[0].ticklabel_format(axis="y", style="plain")

    bars = axes[1].bar(
        x,
        summary["positive_rate_pct"],
        color=COLORS,
    )
    axes[1].set_xticks(x, display_names)
    axes[1].set_ylabel("양성률 (%)")
    axes[1].set_title("목표 라벨별 양성률")
    ymax = max(1.0, float(summary["positive_rate_pct"].max()) * 1.25)
    axes[1].set_ylim(0, ymax)
    for bar, value in zip(bars, summary["positive_rate_pct"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )
    fig.suptitle("디노테푸란 3개 목표 라벨 현황", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "step1_target_overview.png")

    fig, axes = plt.subplots(1, len(specs), figsize=(16, 4.8), sharey=False)
    for ax, spec, color in zip(axes, specs, COLORS):
        plot_df = by_source[by_source["target_key"] == spec.key].copy()
        plot_df = plot_df.sort_values("positive_rate_pct", ascending=False)
        sns.barplot(
            data=plot_df,
            x="source_system",
            y="positive_rate_pct",
            color=color,
            ax=ax,
        )
        ax.set_title(spec.name)
        ax.set_xlabel("출처")
        ax.set_ylabel("양성률 (%)")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, max(1.0, plot_df["positive_rate_pct"].max() * 1.2))
        for patch, row in zip(ax.patches, plot_df.itertuples()):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                f"n={row.labeled_rows:,}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("출처별 목표 라벨 양성률과 분석 모집단", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "step1_target_rate_by_source.png")

    fig, axes = plt.subplots(len(specs), 1, figsize=(12, 10), sharex=True)
    for ax, spec, color in zip(axes, specs, COLORS):
        plot_df = by_year[by_year["target_key"] == spec.key].copy()
        plot_df["event_year"] = pd.to_numeric(plot_df["event_year"], errors="coerce")
        plot_df = plot_df.dropna(subset=["event_year"]).sort_values("event_year")
        ax.plot(
            plot_df["event_year"],
            plot_df["positive_rate_pct"],
            marker="o",
            color=color,
            linewidth=2,
        )
        ax.set_title(spec.name)
        ax.set_ylabel("양성률 (%)")
        ax.grid(axis="y", alpha=0.25)
        if config["eda"]["latest_year_is_partial"]:
            latest = config["eda"]["latest_year"]
            if latest in set(plot_df["event_year"]):
                ax.axvspan(latest - 0.45, latest + 0.45, color="#FFF2CC", alpha=0.8)
                ax.text(latest, ax.get_ylim()[1] * 0.92, "부분연도", ha="center", fontsize=8)
    axes[-1].set_xlabel("연도")
    fig.suptitle("연도별 목표 라벨 양성률", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "step1_target_rate_by_year.png")

    return {
        "target_summary": summary,
        "target_by_source": by_source,
        "target_by_year": by_year,
    }
