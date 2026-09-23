from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import torch
import yaml

import country_climate_effect.experiment as shared
from lstm_baseline.data import load_aligned_target


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("occurrence", "screening", "noncompliance")
TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}
MODEL_ORDER = ("logistic", "catboost", "lightgbm", "xgboost", "lstm")
MODEL_LABELS = {
    "logistic": "Logistic Regression",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
}
VARIANT_LABELS = {
    "internal_only": "Paired Core",
    "internal_plus_climate": "Core+공급경로별 혼합기후",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _core_only_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(manifest)
    for target in TARGETS:
        sets = result["targets"][target]["feature_sets"]
        sets["extended_categorical"] = []
        sets["conditional_categorical"] = []
    return result


def _load_comparison(
    target: str, dataset_manifest: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    target_meta = dataset_manifest["targets"][target]
    path = Path(target_meta["paired_high_path"])
    frame = pd.read_parquet(path)
    route_numeric = list(target_meta["route_numeric"])
    if frame.empty or not frame["record_id"].is_unique:
        raise ValueError(f"{target}: paired high 행이 없거나 record_id 중복")
    if set(frame["split"].astype(str).unique()) != {"train", "validation"}:
        raise ValueError(f"{target}: 비교 모집단은 train/validation만 포함해야 함")
    cross_split = int(
        (frame.groupby("duplicate_group_id", observed=True)["split"].nunique() > 1).sum()
    )
    if cross_split:
        raise ValueError(f"{target}: duplicate_group_id 분할 교차 {cross_split}")
    split_rows: dict[str, Any] = {}
    for split_name in ("train", "validation"):
        part = frame.loc[frame["split"].eq(split_name)]
        if part["target"].nunique() != 2:
            raise ValueError(f"{target}/{split_name}: 두 라벨이 모두 필요")
        split_rows[split_name] = {
            "rows": int(len(part)),
            "positive": int(part["target"].sum()),
            "positive_rate": float(part["target"].mean()),
            "weighted_positive_rate": float(
                np.average(part["target"], weights=part["group_sample_weight"])
            ),
            "route_counts": {
                str(k): int(v) for k, v in part["route_climate_type"].value_counts().items()
            },
        }
    quality = {
        "target": target,
        "target_label": TARGET_LABELS[target],
        "rows": int(len(frame)),
        "record_id_duplicates": int(frame["record_id"].duplicated().sum()),
        "cross_split_duplicate_groups": cross_split,
        "test_rows": int(frame["split"].eq("test").sum()),
        "route_available_failures": int(
            frame["route_climate_high_confidence_flag"].ne(1).sum()
        ),
        "split_rows": split_rows,
        "paired_record_id_sha256": hashlib.sha256(
            "\n".join(frame["record_id"].astype(str)).encode("utf-8")
        ).hexdigest(),
    }
    return frame, route_numeric, quality


def _run_lstm_pairs(
    target: str,
    comparison: pd.DataFrame,
    route_numeric: list[str],
    config: dict[str, Any],
    feature_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    window_days = int(dataset_manifest["targets"][target]["window_days"])
    sequence_dir = ROOT / config["lstm"]["sequence_root"] / f"window_{window_days}d"
    arrays, index, features = load_aligned_target(
        sequence_dir, ROOT / "output" / "features_v1", target
    )
    route = pd.read_parquet(
        Path(dataset_manifest["targets"][target]["feature_path"]),
        columns=["record_id", *route_numeric],
    )
    features = features.merge(route, on="record_id", how="left", validate="one_to_one", sort=False)
    if not index["record_id"].astype(str).equals(features["record_id"].astype(str)):
        raise ValueError(f"{target}: LSTM 혼합기후 연결 후 record_id 순서 변경")
    comparison_ids = set(comparison["record_id"].astype(str))
    metric_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    top_k_rows: list[pd.DataFrame] = []
    for variant in ("internal_only", "internal_plus_climate"):
        result, prediction, top_k = shared.run_lstm_variant(
            target,
            "high",
            arrays,
            index,
            features,
            comparison_ids,
            variant,
            config,
            feature_manifest,
            route_numeric,
            output_dir,
            device,
            window_days,
        )
        metric_rows.append(result)
        predictions.append(prediction)
        top_k_rows.append(top_k)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(predictions, ignore_index=True),
        pd.concat(top_k_rows, ignore_index=True),
    )


def _confusion_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        negative = int(row["tn"] + row["fp"])
        positive = int(row["fn"] + row["tp"])
        rows.extend(
            [
                {
                    "target": row["target"],
                    "target_label": row["target_label"],
                    "model": row["model"],
                    "model_label": row["model_label"],
                    "variant": row["variant"],
                    "variant_label": VARIANT_LABELS[row["variant"]],
                    "actual_class": "음성",
                    "predicted_negative": int(row["tn"]),
                    "predicted_positive": int(row["fp"]),
                    "actual_total": negative,
                    "predicted_negative_row_pct": float(row["tn"] / negative) if negative else np.nan,
                    "predicted_positive_row_pct": float(row["fp"] / negative) if negative else np.nan,
                },
                {
                    "target": row["target"],
                    "target_label": row["target_label"],
                    "model": row["model"],
                    "model_label": row["model_label"],
                    "variant": row["variant"],
                    "variant_label": VARIANT_LABELS[row["variant"]],
                    "actual_class": "양성",
                    "predicted_negative": int(row["fn"]),
                    "predicted_positive": int(row["tp"]),
                    "actual_total": positive,
                    "predicted_negative_row_pct": float(row["fn"] / positive) if positive else np.nan,
                    "predicted_positive_row_pct": float(row["tp"] / positive) if positive else np.nan,
                },
            ]
        )
    return pd.DataFrame(rows)


def _operational_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in metrics.iterrows():
        selected = int(row["tp"] + row["fp"])
        tp = int(row["tp"])
        rows.append(
            {
                "target": row["target"],
                "target_label": row["target_label"],
                "model": row["model"],
                "model_label": row["model_label"],
                "variant": row["variant"],
                "variant_label": VARIANT_LABELS[row["variant"]],
                "validation_rows": int(row["n_rows"]),
                "validation_positive": int(row["n_positive"]),
                "threshold": float(row["threshold"]),
                "selected_rows": selected,
                "selected_rate": float(selected / row["n_rows"]),
                "true_positive_rows": tp,
                "missed_positive_rows": int(row["fn"]),
                "recall_weighted": float(row["recall"]),
                "precision_weighted": float(row["precision"]),
                "tests_per_true_positive_raw": float(selected / tp) if tp else np.nan,
                "top_10pct_capture_weighted": float(row["top10_capture_rate"]),
                "top_10pct_precision_weighted": float(row["top10_precision"]),
                "top_10pct_lift_weighted": float(row["top10_lift"]),
            }
        )
    return pd.DataFrame(rows)


def _selection_table(
    deltas: pd.DataFrame, bootstrap: pd.DataFrame, tolerance: float
) -> pd.DataFrame:
    recall_ci = bootstrap.loc[
        bootstrap["metric"].eq("recall"),
        ["target", "model", "ci_low", "ci_high", "delta_mean"],
    ].rename(
        columns={
            "ci_low": "delta_recall_ci_low",
            "ci_high": "delta_recall_ci_high",
            "delta_mean": "delta_recall_bootstrap_mean",
        }
    )
    result = deltas.merge(recall_ci, on=["target", "model"], how="left", validate="one_to_one")
    decisions: list[str] = []
    reasons: list[str] = []
    for _, row in result.iterrows():
        recall_gain = float(row["delta_recall"])
        guardrails = {
            "F2": float(row["delta_f2"]),
            "PR-AUC": float(row["delta_average_precision"]),
            "Top10": float(row["delta_top10_capture_rate"]),
        }
        guardrails_ok = all(value >= -tolerance for value in guardrails.values())
        ci_low = float(row["delta_recall_ci_low"])
        if recall_gain > 0 and guardrails_ok and ci_low > 0:
            decision = "Core+RouteClimate"
            reason = "Recall 개선, 안전지표 허용범위 유지, paired bootstrap Recall CI 하한 > 0"
        elif recall_gain > 0:
            decision = "Core와 RouteClimate 병행"
            reason = "Recall은 개선됐으나 안전지표 또는 신뢰구간이 확정적이지 않음"
        else:
            decision = "Core"
            reason = "혼합기후의 Recall 개선이 확인되지 않음"
        decisions.append(decision)
        reasons.append(reason)
    result["tuning_input_decision"] = decisions
    result["decision_reason"] = reasons
    return result


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4A5568",
            "grid.color": "#D9E0E8",
            "font.size": 10,
        }
    )


def _render_figures(
    metrics: pd.DataFrame, deltas: pd.DataFrame, selection: pd.DataFrame, figure_dir: Path
) -> list[Path]:
    _set_plot_style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = {"internal_only": "#708090", "internal_plus_climate": "#2F6B4F"}
    for target in TARGETS:
        part = metrics.loc[metrics["target"].eq(target)].copy()
        fig, axis = plt.subplots(figsize=(11.5, 5.8))
        x = np.arange(len(MODEL_ORDER))
        width = 0.36
        for offset, variant in zip((-width / 2, width / 2), colors):
            values = (
                part.loc[part["variant"].eq(variant)]
                .set_index("model")
                .reindex(MODEL_ORDER)["recall"]
            )
            bars = axis.bar(
                x + offset,
                values,
                width,
                label=VARIANT_LABELS[variant],
                color=colors[variant],
            )
            axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
        axis.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=12, ha="right")
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Recall (group_sample_weight 적용)")
        axis.set_title(f"{TARGET_LABELS[target]} · Paired Core vs 공급경로별 혼합기후")
        axis.grid(axis="y", alpha=0.65)
        axis.legend(loc="lower right")
        fig.tight_layout()
        path = figure_dir / f"{target}_paired_recall_comparison.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    matrix = (
        deltas.pivot(index="target", columns="model", values="delta_recall")
        .reindex(index=TARGETS, columns=MODEL_ORDER)
    )
    fig, axis = plt.subplots(figsize=(10.8, 4.7))
    vmax = max(0.01, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    axis.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right")
    axis.set_yticks(np.arange(len(TARGETS)), [TARGET_LABELS[t] for t in TARGETS])
    for i in range(len(TARGETS)):
        for j in range(len(MODEL_ORDER)):
            axis.text(j, i, f"{matrix.iloc[i, j]:+.3f}", ha="center", va="center", fontsize=9)
    axis.set_title("Core+공급경로별 혼합기후 Recall 변화량")
    fig.colorbar(image, ax=axis, label="후보 - Paired Core")
    fig.tight_layout()
    path = figure_dir / "route_climate_delta_recall_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    decision_codes = {"Core": 0, "Core와 RouteClimate 병행": 1, "Core+RouteClimate": 2}
    decision_matrix = (
        selection.assign(code=selection["tuning_input_decision"].map(decision_codes))
        .pivot(index="target", columns="model", values="code")
        .reindex(index=TARGETS, columns=MODEL_ORDER)
    )
    fig, axis = plt.subplots(figsize=(10.8, 4.7))
    from matplotlib.colors import ListedColormap

    axis.imshow(decision_matrix, cmap=ListedColormap(["#A9B4C2", "#E9B872", "#4C956C"]), vmin=0, vmax=2, aspect="auto")
    axis.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER], rotation=15, ha="right")
    axis.set_yticks(np.arange(len(TARGETS)), [TARGET_LABELS[t] for t in TARGETS])
    reverse = {value: key for key, value in decision_codes.items()}
    for i in range(len(TARGETS)):
        for j in range(len(MODEL_ORDER)):
            axis.text(j, i, reverse[int(decision_matrix.iloc[i, j])], ha="center", va="center", fontsize=7.5)
    axis.set_title("목표·모델별 튜닝 입력군 잠정 결정")
    fig.tight_layout()
    path = figure_dir / "tuning_input_decision_matrix.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    return paths


def _write_report(
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    selection: pd.DataFrame,
    quality: list[dict[str, Any]],
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metric_view = metrics[
        [
            "target_label",
            "model_label",
            "variant_label",
            "n_rows",
            "n_positive",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f2",
            "average_precision",
            "top10_capture_rate",
            "threshold",
        ]
    ].copy()
    selection_view = selection[
        [
            "target_label",
            "model_label",
            "delta_recall",
            "delta_f2",
            "delta_average_precision",
            "delta_top10_capture_rate",
            "delta_recall_ci_low",
            "delta_recall_ci_high",
            "tuning_input_decision",
        ]
    ].copy()
    population_rows = []
    for item in quality:
        for split_name, split in item["split_rows"].items():
            population_rows.append(
                {
                    "목표": item["target_label"],
                    "분할": split_name,
                    "전체 n": split["rows"],
                    "양성 n": split["positive"],
                    "양성률": split["positive_rate"],
                    "국내 실측": split["route_counts"].get("DOMESTIC_ACTUAL", 0),
                    "수입 기후평년": split["route_counts"].get("IMPORT_CLIMATOLOGY", 0),
                }
            )
    lines = [
        "# 07-11. Core+공급경로별 혼합기후 동일 모집단 효과 비교",
        "",
        "- 버전: `route_climate_effect_v1`",
        "- 비교 기준: 고신뢰 연결 모집단의 Paired Core",
        "- 학습: 2015~2024년",
        "- 검증: 2025년",
        "- 테스트: 사용하지 않음",
        "- 1순위 지표: Recall",
        "",
        "## 1. 공급경로별 입력 규칙",
        "",
        "- 국내: 기준일 이전 시도 실측기상 사용.",
        "- 수입: 원산국×월 기후평년 사용.",
        "- 국내 실측기상과 수입 기후평년 수치열 분리.",
        "- 국내 위치 대리변수와 수입 수도 대표점 기후평년은 proxy 플래그 보존.",
        "- 목표별 국내 기상 window: 잔류 존재 30일, MRL 10% 60일, 기준 부적합 30일.",
        "",
        "## 2. 동일 모집단",
        "",
        pd.DataFrame(population_rows).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 3. 모델 성능",
        "",
        metric_view.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 4. Paired Core 대비 변화량 및 튜닝 입력군",
        "",
        selection_view.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 5. 해석 주의사항",
        "",
        "- 공식 베이스라인은 전체 확정 모집단의 Core-only 결과이며 변경하지 않음.",
        "- 이 표의 Paired Core는 혼합기후 연결 가능 동일 행에서 다시 학습한 비교 기준임.",
        "- 국내 기상은 생산지·생산일이 완전하지 않은 경우 수거지역·수거일 이전 노출 대리변수임.",
        "- 수입 기후평년은 해당 연도의 실제 이상기상이 아닌 전형적 월별 계절성임.",
        "- 모델별 임계값은 2025년 검증셋의 가중 F2 최대점에서 각각 선택함.",
        "- 성능지표는 `group_sample_weight` 적용값이고 혼동행렬은 원자료 건수임.",
        "- 2026년 테스트는 변수군·튜닝·임계값 확정 후 1회 평가함.",
        "",
        "## 6. 다음 단계",
        "",
        "- 목표·모델별 선택 결과를 전체 튜닝 공통 실험대장에 반영.",
        "- 병행 판정은 Core와 RouteClimate를 모두 튜닝해 시간순 내부검증 안정성 비교.",
        "- 3개 목표×5개 모델 전체 튜닝 후 2026년 테스트 1회 평가.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_notebook(output_dir: Path, report_path: Path) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# Core+공급경로별 혼합기후 효과 비교\n\n"
            "## TL;DR\n"
            "- 국내 행은 기준일 이전 실측기상, 수입 행은 원산국×월 기후평년을 사용한다.\n"
            "- 동일 record_id에서 Paired Core와 비교한다.\n"
            "- 테스트셋은 사용하지 않는다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            "ROOT = Path.cwd()\nOUT = ROOT / 'output' / 'route_climate_effect_v1'"
        ),
        nbf.v4.new_markdown_cell("## 동일 모집단 품질"),
        nbf.v4.new_code_cell(
            "population = pd.read_csv(OUT/'population_summary.csv')\ndisplay(population)"
        ),
        nbf.v4.new_markdown_cell("## 3개 목표 × 5개 모델 성능"),
        nbf.v4.new_code_cell(
            "metrics = pd.read_csv(OUT/'validation_metrics.csv')\n"
            "display(metrics[['target_label','model_label','variant_label','n_rows','n_positive','accuracy','precision','recall','f1','f2','average_precision','top10_capture_rate','threshold']])"
        ),
        nbf.v4.new_markdown_cell("## Paired Core 대비 변화량과 튜닝 입력군"),
        nbf.v4.new_code_cell(
            "selection = pd.read_csv(OUT/'tuning_input_selection.csv')\n"
            "display(selection[['target_label','model_label','delta_recall','delta_f2','delta_average_precision','delta_top10_capture_rate','delta_recall_ci_low','delta_recall_ci_high','tuning_input_decision']])"
        ),
        nbf.v4.new_markdown_cell("## PPT용 그림"),
        nbf.v4.new_code_cell(
            "for name in ['occurrence_paired_recall_comparison.png','screening_paired_recall_comparison.png','noncompliance_paired_recall_comparison.png','route_climate_delta_recall_heatmap.png','tuning_input_decision_matrix.png']:\n"
            "    display(Image(filename=str(OUT/'figures'/name)))"
        ),
        nbf.v4.new_markdown_cell(
            "## 해석\n"
            "- Recall 개선을 우선 확인한다.\n"
            "- F2, PR-AUC, Top 10% 포착률과 paired bootstrap 신뢰구간을 안전지표로 사용한다.\n"
            "- 국내 위치·수입 수도 대표점은 대리변수이므로 인과효과로 해석하지 않는다."
        ),
    ]
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (toxin)",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    path = ROOT / "Route_Climate_Effect_Comparison.ipynb"
    nbf.write(notebook, path)
    return path


def run_experiment(
    config_path: Path,
    device_name: str = "auto",
    skip_lstm: bool = False,
) -> dict[str, Any]:
    config = _load_yaml(config_path)
    output_dir = ROOT / "output" / "route_climate_effect_v1"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    dataset_manifest_path = output_dir / "datasets" / "route_climate_dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    feature_manifest_path = ROOT / "output" / "features_v1" / "feature_engineering_manifest.json"
    feature_manifest = _core_only_manifest(
        json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    )
    baseline_config_path = ROOT / "baseline_modeling" / "config.yaml"
    baseline_config = _load_yaml(baseline_config_path)
    beta = float(config["comparison"]["threshold_beta"])
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA를 사용할 수 없습니다.")

    # 검증된 기존 공통 학습·LSTM 구현을 사용하되 표시와 모집단을 본 실험에 맞춘다.
    shared.COHORTS = ("high",)
    shared.COHORT_LABELS["high"] = "고신뢰 혼합기후 연결 모집단"
    shared.VARIANTS = ("internal_only", "internal_plus_climate")
    shared.VARIANT_LABELS.update(VARIANT_LABELS)

    metric_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []
    top_k_parts: list[pd.DataFrame] = []
    quality: list[dict[str, Any]] = []
    for target in TARGETS:
        comparison, route_numeric, target_quality = _load_comparison(target, dataset_manifest)
        quality.append(target_quality)
        tab_metrics, tab_predictions, tab_top_k = shared.run_tabular_pairs(
            target,
            "high",
            comparison,
            feature_manifest,
            baseline_config,
            route_numeric,
            beta,
            output_dir,
        )
        metric_parts.append(tab_metrics)
        prediction_parts.append(tab_predictions)
        top_k_parts.append(tab_top_k)
        if not skip_lstm:
            lstm_metrics, lstm_predictions, lstm_top_k = _run_lstm_pairs(
                target,
                comparison,
                route_numeric,
                config,
                feature_manifest,
                dataset_manifest,
                output_dir,
                device,
            )
            metric_parts.append(lstm_metrics)
            prediction_parts.append(lstm_predictions)
            top_k_parts.append(lstm_top_k)
        pd.concat(metric_parts, ignore_index=True).to_csv(
            output_dir / "checkpoint_metrics.csv", index=False, encoding="utf-8-sig"
        )

    metrics = pd.concat(metric_parts, ignore_index=True)
    metrics["variant_label"] = metrics["variant"].map(VARIANT_LABELS)
    metrics["pr_auc"] = metrics["average_precision"]
    predictions = pd.concat(prediction_parts, ignore_index=True)
    top_k = pd.concat(top_k_parts, ignore_index=True)
    deltas = shared.comparison_deltas(metrics)
    bootstrap = shared.paired_bootstrap_deltas(
        predictions,
        int(config["comparison"]["bootstrap_repeats"]),
        float(config["comparison"]["bootstrap_confidence"]),
        int(config["seed"]),
        beta,
    )
    selection = _selection_table(
        deltas, bootstrap, float(config["comparison"]["guardrail_absolute_tolerance"])
    )
    confusion = _confusion_table(metrics)
    operational = _operational_table(metrics)

    target_order = {target: idx for idx, target in enumerate(TARGETS)}
    model_order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    metrics = (
        metrics.assign(
            _target=metrics["target"].map(target_order),
            _model=metrics["model"].map(model_order),
            _variant=metrics["variant"].map({"internal_only": 0, "internal_plus_climate": 1}),
        )
        .sort_values(["_target", "_model", "_variant"])
        .drop(columns=["_target", "_model", "_variant"])
        .reset_index(drop=True)
    )
    metrics_path = output_dir / "validation_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    predictions.to_parquet(output_dir / "validation_predictions.parquet", index=False)
    top_k.to_csv(output_dir / "validation_top_k.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(output_dir / "route_climate_effect_deltas.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "route_climate_effect_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    selection_path = output_dir / "tuning_input_selection.csv"
    selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
    confusion.to_csv(output_dir / "confusion_matrices_raw_and_row_pct.csv", index=False, encoding="utf-8-sig")
    operational.to_csv(output_dir / "operational_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "comparison_data_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    population_rows = []
    for item in quality:
        for split_name, split in item["split_rows"].items():
            population_rows.append(
                {
                    "target": item["target"],
                    "target_label": item["target_label"],
                    "split": split_name,
                    **split,
                }
            )
    pd.DataFrame(population_rows).to_csv(
        output_dir / "population_summary.csv", index=False, encoding="utf-8-sig"
    )
    figures = _render_figures(metrics, deltas, selection, figure_dir)
    report_path = (
        ROOT
        / "docs"
        / "외부변수_연결"
        / "07-11_Core_공급경로별_혼합기후_동일모집단_효과비교.md"
    )
    _write_report(metrics, deltas, selection, quality, report_path)
    notebook_path = _build_notebook(output_dir, report_path)
    manifest = {
        "version": config["version"],
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "created_with": "route_climate_effect.experiment.run_experiment",
        "device": str(device),
        "skip_lstm": bool(skip_lstm),
        "models": list(MODEL_ORDER if not skip_lstm else MODEL_ORDER[:-1]),
        "targets": list(TARGETS),
        "cohort": "high",
        "variants": list(VARIANT_LABELS),
        "selection_split": "validation",
        "test_data_used": False,
        "threshold_rule": "2025 validation weighted F2 maximum, model/variant별",
        "selection_rule": "Recall 우선, F2/PR-AUC/Top10 절대 -0.01 이내, paired bootstrap Recall CI 확인",
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "dataset_manifest": {
                "path": str(dataset_manifest_path),
                "sha256": _sha256(dataset_manifest_path),
            },
            "feature_manifest": {
                "path": str(feature_manifest_path),
                "sha256": _sha256(feature_manifest_path),
            },
            "baseline_config": {
                "path": str(baseline_config_path),
                "sha256": _sha256(baseline_config_path),
            },
        },
        "quality": quality,
        "outputs": {
            "metrics": str(metrics_path),
            "selection": str(selection_path),
            "report": str(report_path),
            "notebook": str(notebook_path),
            "figures": [str(path) for path in figures],
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "metrics_path": metrics_path,
        "selection_path": selection_path,
        "report_path": report_path,
        "notebook_path": notebook_path,
        "manifest_path": output_dir / "manifest.json",
    }

