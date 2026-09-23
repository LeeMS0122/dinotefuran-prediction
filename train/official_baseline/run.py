from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from baseline_modeling.pipeline import (
    TARGET_META,
    choose_fbeta_threshold,
    evaluate_scores,
    feature_sets,
    fit_model,
    load_yaml,
    save_selected_model,
    top_k_table,
    validate_modeling_frame,
)


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["occurrence", "screening", "noncompliance"]
TABULAR_MODELS = [
    ("logistic", "Logistic Regression"),
    ("catboost", "CatBoost"),
    ("lightgbm", "LightGBM"),
    ("xgboost", "XGBoost"),
]
MODEL_ORDER = [name for _, name in TABULAR_MODELS] + ["LSTM"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(values: pd.Series) -> str:
    ordered = "\n".join(sorted(values.astype(str).tolist()))
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()


def _target_label(target: str) -> str:
    return str(TARGET_META[target]["label"])


def _validation_prediction_frame(
    validation: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    model_display: str,
    model_type: str,
    window_days: int | None = None,
) -> pd.DataFrame:
    columns = [
        "record_id",
        "duplicate_group_id",
        "event_date",
        "event_year",
        "target",
        "group_sample_weight",
    ]
    prediction = validation.loc[:, columns].copy()
    prediction["score"] = np.asarray(scores, dtype=float)
    prediction["prediction"] = (prediction["score"] >= threshold).astype(int)
    prediction["model"] = model_display
    prediction["model_type"] = model_type
    prediction["feature_set"] = "core"
    prediction["window_days"] = window_days
    prediction["threshold"] = float(threshold)
    prediction["split"] = "validation"
    return prediction


def _metric_row(
    target: str,
    model_display: str,
    model_type: str,
    metrics: dict[str, float],
    top_k: pd.DataFrame,
    fit_seconds: float,
    execution_mode: str,
    window_days: int | None = None,
) -> dict[str, Any]:
    top_10 = top_k.loc[np.isclose(top_k["top_fraction"], 0.10)].iloc[0]
    return {
        "target": target,
        "target_label": _target_label(target),
        "model": model_display,
        "model_type": model_type,
        "feature_set": "core",
        "external_variables_used": False,
        "lstm_window_days": window_days,
        "split": "validation",
        "n_rows": int(metrics["n_rows"]),
        "n_positive": int(metrics["n_positive"]),
        "unweighted_positive_rate": int(metrics["n_positive"]) / int(metrics["n_rows"]),
        "weighted_positive_rate": float(metrics["prevalence"]),
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "f2": float(metrics["f2"]),
        "pr_auc": float(metrics["average_precision"]),
        "roc_auc": float(metrics["roc_auc"]),
        "top_10pct_capture": float(top_10["capture_rate"]),
        "top_10pct_precision": float(top_10["precision"]),
        "top_10pct_lift": float(top_10["lift"]),
        "threshold": float(metrics["threshold"]),
        "fit_seconds": float(fit_seconds),
        "execution_mode": execution_mode,
    }


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Malgun Gothic", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#4A5568",
            "axes.labelcolor": "#2D3748",
            "xtick.color": "#4A5568",
            "ytick.color": "#4A5568",
            "grid.color": "#D9E0E8",
            "font.size": 10,
        }
    )


def _render_figures(metrics: pd.DataFrame, figure_dir: Path) -> list[Path]:
    _configure_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    colors = {
        "Logistic Regression": "#4C78A8",
        "CatBoost": "#F58518",
        "LightGBM": "#54A24B",
        "XGBoost": "#E45756",
        "LSTM": "#8C6BB1",
    }
    for target in TARGETS:
        part = metrics.loc[metrics["target"].eq(target)].copy()
        part["_order"] = part["model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
        part = part.sort_values("_order")

        fig, axis = plt.subplots(figsize=(9.6, 5.6))
        bars = axis.barh(
            part["model"],
            part["recall"],
            color=[colors[name] for name in part["model"]],
        )
        axis.set_title(f"{_target_label(target)} · 공식 Core-only Recall")
        axis.set_xlabel("2025년 검증 가중 Recall · F2 최적 임계값")
        axis.set_xlim(0, 1.05)
        axis.grid(axis="x", alpha=0.6)
        axis.bar_label(bars, fmt="%.4f", padding=3)
        fig.tight_layout()
        path = figure_dir / f"{target}_official_core_recall.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        metric_names = ["accuracy", "precision", "recall", "f1", "f2", "pr_auc", "top_10pct_capture"]
        labels = ["Accuracy", "Precision", "Recall", "F1", "F2", "PR-AUC", "Top 10% 포착률"]
        positions = np.arange(len(part))
        width = 0.11
        fig, axis = plt.subplots(figsize=(13.8, 6.2))
        for index, (metric, label) in enumerate(zip(metric_names, labels)):
            axis.bar(
                positions + (index - 3) * width,
                part[metric].astype(float),
                width=width,
                label=label,
            )
        axis.set_title(f"{_target_label(target)} · 공식 Core-only 성능 비교")
        axis.set_ylabel("2025년 검증 가중 성능")
        axis.set_xticks(positions, part["model"], rotation=15, ha="right")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.6)
        axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17))
        fig.tight_layout()
        path = figure_dir / f"{target}_official_core_all_metrics.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def _build_report(
    metrics: pd.DataFrame,
    population: pd.DataFrame,
    core_features: pd.DataFrame,
    reproducibility: pd.DataFrame,
    report_path: Path,
) -> None:
    display = metrics[
        [
            "target_label",
            "model",
            "lstm_window_days",
            "n_rows",
            "n_positive",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f2",
            "pr_auc",
            "top_10pct_capture",
            "threshold",
        ]
    ].copy()
    best = (
        metrics.sort_values(["target", "recall", "f2", "pr_auc"], ascending=[True, False, False, False])
        .groupby("target", observed=True)
        .head(1)
    )
    lines = [
        "# 공식 Core-only 베이스라인 v1",
        "",
        "## 기준",
        "",
        "- 목적: 이후 변수군·외부변수·튜닝 효과의 공식 비교 기준",
        "- 학습: 2015~2024년",
        "- 검증: 2025년",
        "- 테스트: 2026년 1~6월 잠금",
        "- 입력군: Core-only",
        "- 외부변수: 미사용",
        "- 표본가중치: group_sample_weight",
        "- 임계값: 2025년 검증 F2 최대",
        "- 1순위 지표: Recall",
        "- 비교 단위: 목표별 동일 validation record_id",
        "",
        "## 모집단",
        "",
        population.to_markdown(index=False),
        "",
        "## Core 변수",
        "",
        core_features.to_markdown(index=False),
        "",
        "## 5개 모델 검증 성능",
        "",
        display.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Recall 1위",
        "",
    ]
    for _, row in best.iterrows():
        window = "" if pd.isna(row["lstm_window_days"]) else f" · window {int(row['lstm_window_days'])}일"
        lines.append(
            f"- {row['target_label']}: {row['model']}{window} · Recall {row['recall']:.4f} · Precision {row['precision']:.4f} · PR-AUC {row['pr_auc']:.4f}"
        )
    lines.extend(
        [
            "",
            "## 재현성 점검",
            "",
            "- 기존 Core형 수치와 새 공식 실행 수치를 비교함",
            "- 차이는 동일 설정 재실행의 허용 오차로 확인함",
            "",
            reproducibility.to_markdown(index=False, floatfmt=".10f"),
            "",
            "## 해석 주의",
            "",
            "- 세 목표의 모집단이 다르므로 목표 간 절대 성능을 직접 비교하지 않음",
            "- 기상·국가기후 실험은 연결 가능 표본의 paired Core-only와 비교함",
            "- Recall 단독 상승은 전체 양성 예측으로도 가능하므로 Precision·F2·PR-AUC·Top 10% 포착률을 함께 확인함",
            "- 2026년 테스트 성능은 최종 설정 확정 전까지 산출하지 않음",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _reproducibility_check(metrics: pd.DataFrame, previous_path: Path) -> pd.DataFrame:
    previous = pd.read_csv(previous_path, encoding="utf-8-sig")
    previous = previous.loc[
        previous["feature_set"].eq("core")
        & previous["model_type"].isin([item[0] for item in TABULAR_MODELS])
    ].copy()
    previous["model"] = previous["model_type"].map(dict(TABULAR_MODELS))
    renamed = previous.rename(columns={"average_precision": "pr_auc"})
    compare_columns = ["recall", "precision", "f1", "f2", "pr_auc", "accuracy", "threshold"]
    merged = metrics.loc[metrics["model_type"].isin([item[0] for item in TABULAR_MODELS])].merge(
        renamed[["target", "model", *compare_columns]],
        on=["target", "model"],
        how="left",
        suffixes=("_official", "_previous"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        differences = {
            column: abs(float(row[f"{column}_official"]) - float(row[f"{column}_previous"]))
            for column in compare_columns
        }
        rows.append(
            {
                "target": row["target"],
                "model": row["model"],
                "max_absolute_difference": max(differences.values()),
                "recall_absolute_difference": differences["recall"],
                "pr_auc_absolute_difference": differences["pr_auc"],
                "status": "일치" if max(differences.values()) <= 1e-8 else "재실행 차이",
            }
        )
    return pd.DataFrame(rows)


def run_official_baseline(
    feature_dir: Path,
    output_dir: Path,
    docs_dir: Path,
    config_path: Path,
    model_config_path: Path,
) -> dict[str, Any]:
    official_config = load_yaml(config_path)
    model_config = load_yaml(model_config_path)
    feature_manifest_path = feature_dir / "feature_engineering_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"
    prediction_dir = output_dir / "predictions"
    figure_dir = output_dir / "figures"
    docs_table_dir = docs_dir / "table"
    docs_figure_dir = docs_dir / "figure"
    for directory in [table_dir, model_dir, prediction_dir, figure_dir, docs_table_dir, docs_figure_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    top_k_rows: list[pd.DataFrame] = []
    population_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    prediction_digests: dict[str, dict[str, str]] = {}
    feature_hashes: dict[str, str] = {}

    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]] = {}
    for target in TARGETS:
        feature_path = feature_dir / f"{target}_features_v1.parquet"
        data = pd.read_parquet(feature_path)
        categorical, numeric = feature_sets(feature_manifest["targets"][target])["core"]
        audit = validate_modeling_frame(data, categorical, numeric)
        train = data.loc[data["split"].eq("train")].copy()
        validation = data.loc[data["split"].eq("validation")].copy()
        loaded[target] = (train, validation, categorical, numeric)
        feature_hashes[target] = _sha256(feature_path)
        for split_name, frame in [("train", train), ("validation", validation)]:
            population_rows.append(
                {
                    "target": target,
                    "target_label": _target_label(target),
                    "split": split_name,
                    "period": "2015-2024" if split_name == "train" else "2025",
                    "n_rows": int(len(frame)),
                    "n_positive": int(frame["target"].sum()),
                    "unweighted_positive_rate": float(frame["target"].mean()),
                    "weighted_positive_rate": float(
                        np.average(frame["target"], weights=frame["group_sample_weight"])
                    ),
                    "record_id_duplicates": int(frame["record_id"].duplicated().sum()),
                    "record_id_sha256": _id_digest(frame["record_id"]),
                }
            )
        for feature_type, columns in [("categorical", categorical), ("numeric", numeric)]:
            for column in columns:
                feature_rows.append(
                    {
                        "target": target,
                        "target_label": _target_label(target),
                        "feature": column,
                        "feature_type": feature_type,
                        "train_missing_rows": int(train[column].isna().sum()),
                        "validation_missing_rows": int(validation[column].isna().sum()),
                    }
                )
        if audit["record_id_duplicates"] or audit["cross_split_groups"]:
            raise AssertionError(f"{target} 모집단 감사 실패: {audit}")

    for target in TARGETS:
        train, validation, categorical, numeric = loaded[target]
        y = validation["target"].astype(int).to_numpy()
        weights = validation["group_sample_weight"].astype(float).to_numpy()
        prediction_digests[target] = {}
        for model_type, model_display in TABULAR_MODELS:
            print(f"[{target}] {model_display} Core-only 시작", flush=True)
            started = time.perf_counter()
            model, scores = fit_model(
                model_type,
                train,
                validation,
                categorical,
                numeric,
                model_config,
            )
            elapsed = time.perf_counter() - started
            threshold = choose_fbeta_threshold(
                y,
                scores,
                weights,
                float(official_config["evaluation"]["threshold_beta"]),
            )
            metrics = evaluate_scores(
                y,
                scores,
                weights,
                threshold,
                float(official_config["evaluation"]["threshold_beta"]),
            )
            top_k = top_k_table(
                y,
                scores,
                weights,
                [float(value) for value in official_config["evaluation"]["top_k_fractions"]],
            )
            save_selected_model(model, model_type, model_dir / f"{target}_{model_type}_core")
            prediction = _validation_prediction_frame(
                validation,
                scores,
                threshold,
                model_display,
                model_type,
            )
            prediction_path = prediction_dir / f"{target}_{model_type}_core_validation.parquet"
            prediction.to_parquet(prediction_path, index=False)
            prediction_digests[target][model_display] = _id_digest(prediction["record_id"])
            metric_rows.append(
                _metric_row(
                    target,
                    model_display,
                    model_type,
                    metrics,
                    top_k,
                    elapsed,
                    "retrained_official_core_v1",
                )
            )
            top_k.insert(0, "model", model_display)
            top_k.insert(0, "target_label", _target_label(target))
            top_k.insert(0, "target", target)
            top_k_rows.append(top_k)
            print(
                f"[{target}] {model_display} 완료 · Recall={metrics['recall']:.6f} "
                f"· PR-AUC={metrics['average_precision']:.6f} · {elapsed:.1f}초",
                flush=True,
            )

    lstm_root = ROOT / "output" / str(official_config["lstm"]["source_version"])
    for target in TARGETS:
        _, validation, _, _ = loaded[target]
        window = int(official_config["lstm"]["selected_windows"][target])
        source_dir = lstm_root / "training" / f"window_{window}d"
        source_prediction_path = source_dir / f"{target}_predictions_v1.parquet"
        source_prediction = pd.read_parquet(source_prediction_path)
        if set(source_prediction["split"].astype(str).unique()) != {"validation"}:
            raise AssertionError(f"{target} LSTM 예측값에 validation 외 분할이 존재합니다.")
        left = validation[
            ["record_id", "target", "group_sample_weight"]
        ].copy()
        right = source_prediction[
            ["record_id", "target", "sample_weight", "probability"]
        ].copy()
        aligned = left.merge(
            right,
            on="record_id",
            how="outer",
            suffixes=("_feature", "_lstm"),
            indicator=True,
            validate="one_to_one",
        )
        if not aligned["_merge"].eq("both").all():
            raise AssertionError(f"{target} LSTM record_id 모집단 불일치")
        if not np.array_equal(
            aligned["target_feature"].astype(int).to_numpy(),
            aligned["target_lstm"].astype(int).to_numpy(),
        ):
            raise AssertionError(f"{target} LSTM 라벨 불일치")
        if not np.allclose(
            aligned["group_sample_weight"].astype(float).to_numpy(),
            aligned["sample_weight"].astype(float).to_numpy(),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise AssertionError(f"{target} LSTM 표본가중치 불일치")

        score_map = aligned.set_index("record_id")["probability"]
        scores = validation["record_id"].map(score_map).astype(float).to_numpy()
        y = validation["target"].astype(int).to_numpy()
        weights = validation["group_sample_weight"].astype(float).to_numpy()
        threshold = choose_fbeta_threshold(
            y,
            scores,
            weights,
            float(official_config["evaluation"]["threshold_beta"]),
        )
        metrics = evaluate_scores(
            y,
            scores,
            weights,
            threshold,
            float(official_config["evaluation"]["threshold_beta"]),
        )
        top_k = top_k_table(
            y,
            scores,
            weights,
            [float(value) for value in official_config["evaluation"]["top_k_fractions"]],
        )
        prediction = _validation_prediction_frame(
            validation,
            scores,
            threshold,
            "LSTM",
            "LSTM",
            window,
        )
        destination_prediction = prediction_dir / f"{target}_lstm_core_window_{window}d_validation.parquet"
        prediction.to_parquet(destination_prediction, index=False)
        prediction_digests[target]["LSTM"] = _id_digest(prediction["record_id"])
        for suffix in ["lstm_baseline_v1.pt", "static_preprocessor_v1.json"]:
            source = source_dir / f"{target}_{suffix}"
            destination = model_dir / f"{target}_lstm_core_window_{window}d_{suffix}"
            shutil.copy2(source, destination)
        metric_rows.append(
            _metric_row(
                target,
                "LSTM",
                "LSTM",
                metrics,
                top_k,
                0.0,
                "reused_lstm_window_tuning_v1",
                window,
            )
        )
        top_k.insert(0, "model", "LSTM")
        top_k.insert(0, "target_label", _target_label(target))
        top_k.insert(0, "target", target)
        top_k_rows.append(top_k)
        print(
            f"[{target}] LSTM window {window}일 재사용 · Recall={metrics['recall']:.6f} "
            f"· PR-AUC={metrics['average_precision']:.6f}",
            flush=True,
        )

    for target, digests in prediction_digests.items():
        if len(set(digests.values())) != 1:
            raise AssertionError(f"{target} 모델 간 validation record_id 불일치: {digests}")

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame["_target"] = metrics_frame["target"].map({target: i for i, target in enumerate(TARGETS)})
    metrics_frame["_model"] = metrics_frame["model"].map({model: i for i, model in enumerate(MODEL_ORDER)})
    metrics_frame = metrics_frame.sort_values(["_target", "_model"]).drop(columns=["_target", "_model"]).reset_index(drop=True)
    top_k_frame = pd.concat(top_k_rows, ignore_index=True)
    population_frame = pd.DataFrame(population_rows)
    core_features_frame = pd.DataFrame(feature_rows)
    previous_path = ROOT / "docs" / "베이스라인_모델링_2차" / "table" / "validation_candidate_metrics.csv"
    reproducibility_frame = _reproducibility_check(metrics_frame, previous_path)

    tables = {
        "official_core_validation_metrics.csv": metrics_frame,
        "official_core_validation_top_k.csv": top_k_frame,
        "official_core_population.csv": population_frame,
        "official_core_features.csv": core_features_frame,
        "official_core_reproducibility_check.csv": reproducibility_frame,
    }
    for name, frame in tables.items():
        output_path = table_dir / name
        docs_path = docs_table_dir / name
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        shutil.copy2(output_path, docs_path)

    figure_paths = _render_figures(metrics_frame, figure_dir)
    for figure_path in figure_paths:
        shutil.copy2(figure_path, docs_figure_dir / figure_path.name)

    report_path = docs_dir / "08_공식_Core-only_베이스라인.md"
    _build_report(
        metrics_frame,
        population_frame,
        core_features_frame,
        reproducibility_frame,
        report_path,
    )
    shutil.copy2(report_path, output_dir / report_path.name)

    manifest = {
        "version": str(official_config["version"]),
        "created_with": "official_baseline.run.run_official_baseline",
        "feature_set": "core",
        "external_variables_used": False,
        "targets": TARGETS,
        "models": MODEL_ORDER,
        "train_period": str(official_config["population"]["train"]),
        "validation_period": str(official_config["population"]["validation"]),
        "test_period": str(official_config["population"]["test"]),
        "test_data_scored": False,
        "test_predictions_saved": False,
        "test_used_for_selection": False,
        "sample_weight": str(official_config["evaluation"]["sample_weight"]),
        "primary_reporting_metric": str(official_config["evaluation"]["primary_metric"]),
        "threshold_rule": "2025 validation weighted F2 maximum",
        "comparison_rule": str(official_config["population"]["comparison_rule"]),
        "lstm_selected_windows": official_config["lstm"]["selected_windows"],
        "lstm_reused": True,
        "validation_record_id_sha256": {
            target: next(iter(digests.values())) for target, digests in prediction_digests.items()
        },
        "input_sha256": {
            "feature_manifest": _sha256(feature_manifest_path),
            "model_config": _sha256(model_config_path),
            "official_config": _sha256(config_path),
            "features": feature_hashes,
        },
        "outputs": {
            "metrics": str(table_dir / "official_core_validation_metrics.csv"),
            "top_k": str(table_dir / "official_core_validation_top_k.csv"),
            "population": str(table_dir / "official_core_population.csv"),
            "features": str(table_dir / "official_core_features.csv"),
            "reproducibility": str(table_dir / "official_core_reproducibility_check.csv"),
            "report": str(report_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "metrics": metrics_frame,
        "population": population_frame,
        "reproducibility": reproducibility_frame,
        "manifest": manifest_path,
        "report": report_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=ROOT / "output" / "features_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "official_baseline_core_v1")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs" / "공식_베이스라인")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--model-config", type=Path, default=ROOT / "baseline_modeling" / "config.yaml")
    args = parser.parse_args()
    result = run_official_baseline(
        args.feature_dir,
        args.output_dir,
        args.docs_dir,
        args.config,
        args.model_config,
    )
    print(result["metrics"][["target_label", "model", "recall", "precision", "f2", "pr_auc", "top_10pct_capture"]].to_string(index=False))
    print(f"report={result['report']}")
    print(f"manifest={result['manifest']}")


if __name__ == "__main__":
    main()
