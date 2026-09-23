from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TARGET_ORDER = ["occurrence", "screening", "noncompliance"]
TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}
MODEL_ORDER = [
    "Logistic Regression",
    "CatBoost",
    "LightGBM",
    "XGBoost",
    "LSTM",
]


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4A5568",
        "axes.labelcolor": "#2D3748",
        "xtick.color": "#4A5568",
        "ytick.color": "#4A5568",
        "grid.color": "#D9E0E8",
        "font.size": 11,
    })


def load_window_metrics(output_root: Path, windows: list[int]) -> pd.DataFrame:
    frames = []
    for window in windows:
        path = output_root / "training" / f"window_{window}d" / "window_validation_metrics.csv"
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if set(frame["split"]) != {"validation"}:
            raise ValueError(f"window {window} 결과에 validation 외 split이 존재합니다.")
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    if len(metrics) != len(windows) * len(TARGET_ORDER):
        raise ValueError(f"window 결과 행 수 불일치: {len(metrics)}")
    return metrics


def select_windows(metrics: pd.DataFrame, relative_floor: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = metrics.copy()
    ranked["target_order"] = ranked["target"].map(
        {target: index for index, target in enumerate(TARGET_ORDER)}
    )
    ranked["max_weighted_f1"] = ranked.groupby("target", observed=True)[
        "weighted_f1"
    ].transform("max")
    ranked["f1_selection_floor"] = ranked["max_weighted_f1"] * relative_floor
    ranked["eligible_by_f1"] = (
        ranked["weighted_f1"] >= ranked["f1_selection_floor"]
    )
    ranked = ranked.sort_values(
        [
            "target_order",
            "eligible_by_f1",
            "weighted_recall",
            "weighted_f1",
            "weighted_pr_auc",
            "window_days",
        ],
        ascending=[True, False, False, False, False, True],
    ).reset_index(drop=True)
    ranked["selection_rank"] = (
        ranked.groupby("target", observed=True).cumcount() + 1
    )
    ranked["selected_window"] = ranked["selection_rank"].eq(1)
    selected = ranked.loc[ranked["selected_window"]].copy()
    if len(selected) != len(TARGET_ORDER):
        raise ValueError("목표별 최적 window가 정확히 1개씩 선택되지 않았습니다.")
    return ranked.drop(columns="target_order"), selected.drop(columns="target_order")


def build_interim_comparison(selected: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    source = (
        project_root
        / "docs"
        / "LSTM_베이스라인"
        / "table"
        / "all_five_model_best_baseline.csv"
    )
    other = pd.read_csv(source, encoding="utf-8-sig")
    other = other.loc[other["model_display"].ne("LSTM")].copy()
    lstm = pd.DataFrame({
        "target": selected["target"],
        "target_label": selected["target_label"],
        "candidate": selected["candidate"],
        "model_type": "LSTM",
        "model_display": "LSTM",
        "feature_set": selected["feature_set"],
        "fit_seconds": selected["fit_seconds"],
        "n_rows": selected["n_rows"],
        "n_positive": selected["n_positive"],
        "prevalence": selected["positive_rate_pct"] / 100,
        "average_precision": selected["weighted_pr_auc"],
        "roc_auc": selected["weighted_roc_auc"],
        "threshold": selected["threshold"],
        "accuracy": selected["weighted_accuracy"],
        "precision": selected["weighted_precision"],
        "recall": selected["weighted_recall"],
        "f1": selected["weighted_f1"],
        "f2": selected["weighted_f2"],
        "baseline_source": "lstm_window_tuning_v1",
        "tuning_required": True,
        "window_days": selected["window_days"],
    })
    other["window_days"] = pd.NA
    combined = pd.concat([other, lstm], ignore_index=True, sort=False)
    combined["target_order"] = combined["target"].map(
        {target: index for index, target in enumerate(TARGET_ORDER)}
    )
    combined["model_order"] = combined["model_display"].map(
        {model: index for index, model in enumerate(MODEL_ORDER)}
    )
    return (
        combined.sort_values(["target_order", "model_order"])
        .drop(columns=["target_order", "model_order"])
        .reset_index(drop=True)
    )


def render_figures(
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    figure_dir: Path,
) -> None:
    configure_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)
    windows = sorted(metrics["window_days"].unique())
    selected_map = selected.set_index("target")["window_days"].to_dict()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharey=True)
    for axis, target in zip(axes, TARGET_ORDER):
        part = metrics.loc[metrics["target"].eq(target)].sort_values("window_days")
        axis.plot(
            part["window_days"],
            part["weighted_recall"],
            marker="o",
            linewidth=2,
            color="#D88A2D",
            label="Recall",
        )
        axis.plot(
            part["window_days"],
            part["weighted_f1"],
            marker="s",
            linewidth=2,
            color="#3569A8",
            label="F1-score",
        )
        chosen = int(selected_map[target])
        chosen_row = part.loc[part["window_days"].eq(chosen)].iloc[0]
        axis.scatter(
            [chosen],
            [chosen_row["weighted_recall"]],
            s=90,
            facecolors="white",
            edgecolors="#2D3748",
            linewidths=2,
            zorder=4,
        )
        axis.annotate(
            f"선택 {chosen}일",
            (chosen, chosen_row["weighted_recall"]),
            xytext=(4, 8),
            textcoords="offset points",
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xticks(windows)
        axis.set_xlabel("Window 일수")
        axis.set_ylabel("2025년 검증 가중 성능")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.65)
        axis.legend(loc="best")
    fig.suptitle("LSTM window별 Recall·F1-score", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "lstm_window_recall_f1.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8), sharey=False)
    for axis, target in zip(axes, TARGET_ORDER):
        part = metrics.loc[metrics["target"].eq(target)].sort_values("window_days")
        bars = axis.bar(
            part["window_days"].astype(str),
            part["weighted_pr_auc"],
            color="#3569A8",
        )
        axis.set_title(TARGET_LABELS[target])
        axis.set_xlabel("Window 일수")
        axis.set_ylabel("2025년 검증 가중 PR-AUC")
        axis.set_ylim(0, max(part["weighted_pr_auc"].max() * 1.25, 0.05))
        axis.grid(axis="y", alpha=0.65)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    fig.suptitle("LSTM window별 PR-AUC", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(
        figure_dir / "lstm_window_pr_auc.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    for target in TARGET_ORDER:
        part = comparison.loc[comparison["target"].eq(target)].copy()
        part["order"] = part["model_display"].map(
            {model: index for index, model in enumerate(MODEL_ORDER)}
        )
        part = part.sort_values("order")
        fig, axis = plt.subplots(figsize=(9.8, 5.8))
        bars = axis.barh(part["model_display"], part["recall"], color="#D88A2D")
        axis.set_title(
            f"{TARGET_LABELS[target]} · 5개 모델 중간 비교 Recall"
        )
        axis.set_xlabel("2025년 검증 가중 Recall · F2 최적 임계값")
        axis.set_xlim(0, 1.05)
        axis.grid(axis="x", alpha=0.65)
        axis.bar_label(bars, fmt="%.4f", padding=3)
        fig.tight_layout()
        fig.savefig(
            figure_dir / f"{target}_five_model_interim_recall.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def write_report(
    ranked: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    docs_root: Path,
    project_root: Path,
    relative_floor: float,
) -> Path:
    judge = pd.read_csv(
        project_root / "docs" / "eda_1차" / "table" / "step0_judge_distribution.csv",
        encoding="utf-8-sig",
    ).set_index("judge_result")
    detected, nondetected, undetermined = [
        int(judge.loc[value, "rows"]) for value in ["검출", "불검출", "미확정"]
    ]
    total = detected + nondetected + undetermined
    window_columns = [
        "target_label", "window_days", "n_rows", "n_positive", "best_epoch",
        "weighted_accuracy", "weighted_precision", "weighted_recall",
        "weighted_f1", "weighted_f2", "weighted_pr_auc", "threshold",
        "eligible_by_f1", "selection_rank", "selected_window", "fit_seconds",
    ]
    selected_columns = [
        "target_label", "window_days", "weighted_accuracy", "weighted_precision",
        "weighted_recall", "weighted_f1", "weighted_f2", "weighted_pr_auc",
        "threshold", "best_epoch",
    ]
    compare_columns = [
        "target_label", "model_display", "candidate", "feature_set", "window_days",
        "accuracy", "precision", "recall", "f1", "f2",
        "average_precision", "threshold",
    ]
    report = f"""# 06-5. LSTM window 비교 및 5개 모델 중간 재비교

## 한눈에 보기

- Window 후보: 14·30·60·90일
- 비교 목표: 잔류 존재·MRL 10% 관심농도·기준 부적합
- 학습 조건: window 외 구조·정적변수·seed·학습률·early stopping 동일
- 선택 데이터: 2025년 검증셋
- 2026년 테스트셋: 예측·평가·선택에 사용하지 않음
- 1순위 지표: Recall
- 안전장치: 목표별 최고 F1-score의 {relative_floor * 100:.0f}% 이상
- 동률 기준: F1-score, PR-AUC
- 중간 비교: 선택된 LSTM window와 기존 4개 모델 베이스라인

## 중요사항 1 · 전체 검출/불검출 현황

- 전체 원장: {total:,}건
- 검출: {detected:,}건({detected / total * 100:.2f}%)
- 불검출: {nondetected:,}건({nondetected / total * 100:.2f}%)
- 미확정: {undetermined:,}건({undetermined / total * 100:.2f}%)
- 판정 확정분 검출률: {detected / (detected + nondetected) * 100:.2f}%
- 미확정은 불검출로 변환하지 않음

## Window별 검증 결과

{ranked[window_columns].to_markdown(index=False, floatfmt=".4f")}

## 목표별 선택 window

{selected[selected_columns].to_markdown(index=False, floatfmt=".4f")}

## 5개 모델 중간 재비교

{comparison[compare_columns].to_markdown(index=False, floatfmt=".4f")}

## 해석 원칙

- Recall을 1순위로 보되 F1-score 안전장치를 통과한 window만 선택함
- Accuracy는 불균형 자료에서 음성 다수의 영향을 크게 받으므로 단독 판단하지 않음
- 모든 임계값 기반 지표는 2025년 검증 F2 최적 임계값에서 계산함
- LSTM만 window가 1차 최적화된 상태이므로 이번 5개 모델 표는 최종 우열표가 아님
- 다음 단계에서 5개 모델 전체를 동일한 실험대장과 탐색 예산으로 튜닝함
"""
    report_path = docs_root / "06-5_LSTM_window_비교_5개모델_중간재비교.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def finalize_window_comparison(
    experiment: dict,
    output_root: Path,
    docs_root: Path,
    project_root: Path,
) -> dict:
    table_dir = docs_root / "table"
    figure_dir = docs_root / "figure"
    table_dir.mkdir(parents=True, exist_ok=True)
    windows = [int(value) for value in experiment["windows"]]
    metrics = load_window_metrics(output_root, windows)
    relative_floor = float(experiment["selection"]["f1_relative_floor"])
    ranked, selected = select_windows(metrics, relative_floor)
    comparison = build_interim_comparison(selected, project_root)

    ranked.to_csv(
        table_dir / "lstm_window_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        table_dir / "lstm_selected_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )
    comparison.to_csv(
        table_dir / "five_model_interim_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    render_figures(metrics, selected, comparison, figure_dir)
    report_path = write_report(
        ranked, selected, comparison, docs_root, project_root, relative_floor
    )
    selection = {
        row["target"]: {
            "target_label": row["target_label"],
            "window_days": int(row["window_days"]),
            "weighted_recall": float(row["weighted_recall"]),
            "weighted_f1": float(row["weighted_f1"]),
            "weighted_pr_auc": float(row["weighted_pr_auc"]),
        }
        for _, row in selected.iterrows()
    }
    manifest = {
        "version": experiment["version"],
        "windows": windows,
        "targets": TARGET_ORDER,
        "seed": int(experiment["seed"]),
        "test_data_used": False,
        "selection_rule": experiment["selection"],
        "selected_windows": selection,
        "interim_comparison": True,
        "report": str(report_path),
    }
    manifest_path = output_root / "lstm_window_comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "ranked": ranked,
        "selected": selected,
        "comparison": comparison,
        "report": report_path,
        "manifest": manifest_path,
    }
