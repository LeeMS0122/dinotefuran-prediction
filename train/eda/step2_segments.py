from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .common import save_figure, save_table
from .targets import grouped_target_summary, target_specs


SEGMENTS = {
    "product_name_std": "표준 품목",
    "product_group_raw": "품목군",
    "origin_country_name": "원산국",
    "collection_province": "수거 시도",
}

COLORS = {
    "occurrence": "#2F5597",
    "screening": "#ED7D31",
    "noncompliance": "#70AD47",
}


def _plot_top_groups(
    table: pd.DataFrame,
    group_column: str,
    group_name: str,
    target_key: str,
    target_name: str,
    top_n: int,
) -> None:
    plot_df = table.head(top_n).sort_values("positive_rate_pct")
    if plot_df.empty:
        return
    y = np.arange(len(plot_df))
    values = plot_df["positive_rate_pct"].to_numpy()
    low = plot_df["ci95_low_pct"].to_numpy()
    high = plot_df["ci95_high_pct"].to_numpy()
    errors = np.vstack([values - low, high - values])

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(plot_df) + 1.5)))
    ax.barh(
        y,
        values,
        xerr=errors,
        color=COLORS[target_key],
        alpha=0.9,
        error_kw={"ecolor": "#555555", "capsize": 2, "elinewidth": 1},
    )
    labels = [
        f"{name}  (n={count:,})"
        for name, count in zip(plot_df[group_column], plot_df["labeled_rows"])
    ]
    ax.set_yticks(y, labels)
    ax.set_xlabel("양성률 (%) · 오차선은 Wilson 95% 신뢰구간")
    ax.set_ylabel(group_name)
    ax.set_title(f"{target_name}: {group_name}별 상위 {len(plot_df)}개")
    ax.set_xlim(0, max(1.0, float(high.max()) * 1.08))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, f"step2_{target_key}_top_{group_column}.png")


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    min_n = int(config["eda"]["min_group_n"])
    top_n = int(config["eda"]["top_n"])
    outputs: dict[str, pd.DataFrame] = {}

    for spec in target_specs(config):
        for group_column, group_name in SEGMENTS.items():
            table = grouped_target_summary(
                df,
                spec,
                group_column,
                min_n=min_n,
            )
            key = f"{spec.key}_{group_column}"
            outputs[key] = table
            save_table(table, f"step2_{key}.csv")
            _plot_top_groups(
                table,
                group_column,
                group_name,
                spec.key,
                spec.name,
                top_n,
            )
    return outputs
