from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


TARGET_META = {
    "occurrence": {"label": "잔류 존재", "color": "#3569A8"},
    "screening": {"label": "MRL 10% 관심농도", "color": "#D88A2D"},
    "noncompliance": {"label": "기준 부적합", "color": "#8C6AAE"},
}
SPLITS = ("train", "validation", "test")
FORBIDDEN_PREFIXES = ("label_", "result_", "judge_", "mrl_")


@dataclass
class CandidateResult:
    name: str
    model_type: str
    feature_set: str
    model: Any
    categorical: list[str]
    numeric: list[str]
    validation_scores: np.ndarray
    threshold: float
    validation_metrics: dict[str, float]
    fit_seconds: float


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "grid.linewidth": 0.8,
        "font.size": 11,
    })


def feature_sets(manifest_target: dict[str, Any]) -> dict[str, tuple[list[str], list[str]]]:
    source = manifest_target["feature_sets"]
    core_cat = list(source["core_categorical"])
    core_num = list(source["core_numeric"])
    extended_cat = core_cat + list(source["extended_categorical"])
    conditional_cat = extended_cat + list(source["conditional_categorical"])
    return {
        "core": (core_cat, core_num),
        "extended": (extended_cat, core_num),
        "conditional": (conditional_cat, core_num),
    }


def validate_modeling_frame(
    data: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> dict[str, Any]:
    required = {
        "record_id", "duplicate_group_id", "split", "target",
        "group_sample_weight", *categorical, *numeric,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    if not data["record_id"].is_unique:
        raise ValueError("record_id 중복이 존재합니다.")
    split_per_group = data.groupby("duplicate_group_id", observed=True)["split"].nunique()
    cross_split_groups = int((split_per_group > 1).sum())
    if cross_split_groups:
        raise ValueError(f"duplicate_group_id 분할 교차: {cross_split_groups}")
    target_values = set(data["target"].dropna().astype(int).unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"이진 라벨 이외 값 존재: {target_values}")
    forbidden = [
        col for col in categorical + numeric
        if col.startswith(FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"누수 위험 변수 포함: {forbidden}")
    split_counts = data.groupby("split", observed=True)["target"].agg(["size", "sum"])
    for split in SPLITS:
        if split not in split_counts.index:
            raise ValueError(f"필수 분할 누락: {split}")
        if int(split_counts.loc[split, "sum"]) == 0:
            raise ValueError(f"양성값이 없는 분할: {split}")
    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "record_id_duplicates": int(data["record_id"].duplicated().sum()),
        "cross_split_groups": cross_split_groups,
        "split_counts": {
            str(idx): {"rows": int(row["size"]), "positive": int(row["sum"])}
            for idx, row in split_counts.iterrows()
        },
    }


def prepare_categorical(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    for column in columns:
        out[column] = out[column].astype("string").fillna("__MISSING__").astype(str)
    return out


def prepare_features(
    frame: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> pd.DataFrame:
    categorical_frame = prepare_categorical(frame, categorical)
    numeric_frame = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce")
    return pd.concat([categorical_frame, numeric_frame], axis=1)


def make_logistic_pipeline(
    categorical: list[str], numeric: list[str], config: dict[str, Any], seed: int
) -> Pipeline:
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=int(config["min_frequency"]),
                sparse_output=True,
            ),
        ),
    ])
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipe, categorical),
        ("numeric", numeric_pipe, numeric),
    ])
    classifier = LogisticRegression(
        solver=str(config["solver"]),
        C=float(config["c"]),
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        class_weight=str(config["class_weight"]),
        random_state=seed,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def make_catboost(config: dict[str, Any], seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=int(config["iterations"]),
        depth=int(config["depth"]),
        learning_rate=float(config["learning_rate"]),
        l2_leaf_reg=float(config["l2_leaf_reg"]),
        random_strength=float(config["random_strength"]),
        loss_function="Logloss",
        eval_metric=str(config["eval_metric"]),
        auto_class_weights=str(config["auto_class_weights"]),
        random_seed=seed,
        thread_count=int(config["thread_count"]),
        allow_writing_files=False,
        verbose=False,
    )


def make_tree_preprocessor(
    categorical: list[str], numeric: list[str], min_frequency: int
) -> ColumnTransformer:
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=int(min_frequency),
            sparse_output=True,
        )),
    ])
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer([
        ("categorical", categorical_pipe, categorical),
        ("numeric", numeric_pipe, numeric),
    ])


def make_lightgbm(config: dict[str, Any], seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=int(config["n_estimators"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=float(config["subsample"]),
        subsample_freq=1,
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        class_weight=str(config["class_weight"]),
        random_state=seed,
        n_jobs=int(config["thread_count"]),
        verbosity=-1,
    )


def make_xgboost(
    config: dict[str, Any], seed: int, scale_pos_weight: float
) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        scale_pos_weight=float(scale_pos_weight),
        tree_method="hist",
        device=str(config.get("device", "cpu")),
        eval_metric="aucpr",
        early_stopping_rounds=int(config["early_stopping_rounds"]),
        random_state=seed,
        n_jobs=int(config["thread_count"]),
        verbosity=0,
    )


def fit_model(
    model_type: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    config: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    seed = int(config["random_seed"])
    train_x = prepare_features(train, categorical, numeric)
    validation_x = prepare_features(validation, categorical, numeric)
    train_y = train["target"].astype(int).to_numpy()
    validation_y = validation["target"].astype(int).to_numpy()
    train_weight = train["group_sample_weight"].astype(float).to_numpy()
    validation_weight = validation["group_sample_weight"].astype(float).to_numpy()

    if model_type == "logistic":
        model = make_logistic_pipeline(
            categorical, numeric, config["logistic_regression"], seed
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(train_x, train_y, classifier__sample_weight=train_weight)
    elif model_type == "catboost":
        model = make_catboost(config["catboost"], seed)
        model.fit(
            train_x,
            train_y,
            cat_features=categorical,
            sample_weight=train_weight,
            eval_set=Pool(
                validation_x,
                validation_y,
                cat_features=categorical,
                weight=validation_weight,
            ),
            early_stopping_rounds=int(config["catboost"]["early_stopping_rounds"]),
            use_best_model=True,
            verbose=False,
        )
    elif model_type in {"lightgbm", "xgboost"}:
        preprocessor = make_tree_preprocessor(
            categorical, numeric, int(config["tree_encoding"]["min_frequency"])
        )
        encoded_train = preprocessor.fit_transform(train_x)
        encoded_validation = preprocessor.transform(validation_x)
        if model_type == "lightgbm":
            classifier = make_lightgbm(config["lightgbm"], seed)
            classifier.fit(
                encoded_train,
                train_y,
                sample_weight=train_weight,
                eval_X=encoded_validation,
                eval_y=validation_y,
                eval_sample_weight=[validation_weight],
                eval_metric="average_precision",
                callbacks=[
                    lgb.early_stopping(
                        int(config["lightgbm"]["early_stopping_rounds"]), verbose=False
                    ),
                    lgb.log_evaluation(period=0),
                ],
            )
        else:
            positive_weight = float(train_weight[train_y == 1].sum())
            negative_weight = float(train_weight[train_y == 0].sum())
            classifier = make_xgboost(
                config["xgboost"], seed, negative_weight / positive_weight
            )
            classifier.fit(
                encoded_train,
                train_y,
                sample_weight=train_weight,
                eval_set=[(encoded_validation, validation_y)],
                sample_weight_eval_set=[validation_weight],
                verbose=False,
            )
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ])
    else:
        raise ValueError(f"지원하지 않는 모델: {model_type}")
    scores = model.predict_proba(validation_x)[:, 1]
    return model, np.asarray(scores, dtype=float)


def predict_scores(
    model: Any,
    model_type: str,
    frame: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
) -> np.ndarray:
    features = prepare_features(frame, categorical, numeric)
    return np.asarray(model.predict_proba(features)[:, 1], dtype=float)


def choose_fbeta_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
    beta: float = 2.0,
) -> float:
    precision, recall, thresholds = precision_recall_curve(
        y_true, scores, sample_weight=sample_weight
    )
    if len(thresholds) == 0:
        return 0.5
    beta2 = beta * beta
    denom = beta2 * precision[:-1] + recall[:-1]
    fbeta = np.divide(
        (1 + beta2) * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )
    return float(thresholds[int(np.nanargmax(fbeta))])


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
    beta: float = 2.0,
) -> dict[str, float]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    has_both = len(np.unique(y_true)) == 2
    return {
        "n_rows": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "prevalence": float(np.average(y_true, weights=sample_weight)),
        "average_precision": float(
            average_precision_score(y_true, scores, sample_weight=sample_weight)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, scores, sample_weight=sample_weight)
        ) if has_both else math.nan,
        "brier_score": float(
            brier_score_loss(y_true, scores, sample_weight=sample_weight)
        ),
        "log_loss": float(
            log_loss(y_true, scores, labels=[0, 1], sample_weight=sample_weight)
        ),
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predicted, sample_weight=sample_weight)
        ),
        "precision": float(
            precision_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predicted, sample_weight=sample_weight, zero_division=0)
        ),
        "f2": float(
            fbeta_score(
                y_true, predicted, beta=beta, sample_weight=sample_weight, zero_division=0
            )
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def top_k_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
    fractions: list[float],
) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")
    total_positive_weight = float(np.sum(sample_weight * y_true))
    prevalence = float(np.average(y_true, weights=sample_weight))
    rows: list[dict[str, float]] = []
    for fraction in fractions:
        selected_n = max(1, int(math.ceil(len(order) * fraction)))
        selected = order[:selected_n]
        selected_weight = sample_weight[selected]
        selected_positive_weight = float(np.sum(selected_weight * y_true[selected]))
        precision = float(np.average(y_true[selected], weights=selected_weight))
        capture = (
            selected_positive_weight / total_positive_weight
            if total_positive_weight > 0 else math.nan
        )
        rows.append({
            "top_fraction": float(fraction),
            "selected_n": int(selected_n),
            "capture_rate": capture,
            "precision": precision,
            "lift": precision / prevalence if prevalence > 0 else math.nan,
        })
    return pd.DataFrame(rows)


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    scores: np.ndarray,
    sample_weight: np.ndarray,
    threshold: float,
    fractions: list[float],
    repeats: int,
    confidence: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_true == 1)
    negative = np.flatnonzero(y_true == 0)
    values: dict[str, list[float]] = {
        "average_precision": [], "roc_auc": [], "recall": [],
        "precision": [], "f2": [], "top_10pct_capture": [],
    }
    for _ in range(repeats):
        indices = np.concatenate([
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ])
        rng.shuffle(indices)
        metrics = evaluate_scores(
            y_true[indices], scores[indices], sample_weight[indices], threshold
        )
        top_10 = top_k_table(
            y_true[indices], scores[indices], sample_weight[indices], [0.10]
        ).iloc[0]
        for key in ["average_precision", "roc_auc", "recall", "precision", "f2"]:
            values[key].append(float(metrics[key]))
        values["top_10pct_capture"].append(float(top_10["capture_rate"]))
    alpha = (1 - confidence) / 2
    return pd.DataFrame([
        {
            "metric": metric,
            "estimate": float(np.quantile(observed, 0.50)),
            "ci_lower": float(np.quantile(observed, alpha)),
            "ci_upper": float(np.quantile(observed, 1 - alpha)),
            "confidence": float(confidence),
            "bootstrap_repeats": int(repeats),
        }
        for metric, observed in values.items()
    ])


def save_selected_model(model: Any, model_type: str, path_without_suffix: Path) -> Path:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    if model_type == "catboost":
        path = path_without_suffix.with_suffix(".cbm")
        model.save_model(path)
    else:
        path = path_without_suffix.with_suffix(".joblib")
        joblib.dump(model, path)
    return path


def feature_importance_table(
    result: CandidateResult,
    sample: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    if result.model_type == "catboost":
        features = prepare_features(sample, result.categorical, result.numeric)
        pool = Pool(features, cat_features=result.categorical)
        importance = result.model.get_feature_importance(pool, type="FeatureImportance")
        table = pd.DataFrame({
            "feature": result.categorical + result.numeric,
            "importance": importance,
            "method": "catboost_feature_importance",
        })
        shap_sample = sample.sample(min(2000, len(sample)), random_state=seed)
        shap_features = prepare_features(shap_sample, result.categorical, result.numeric)
        shap_values = result.model.get_feature_importance(
            Pool(shap_features, cat_features=result.categorical), type="ShapValues"
        )[:, :-1]
        shap_table = pd.DataFrame({
            "feature": result.categorical + result.numeric,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        })
        return table.merge(shap_table, on="feature", how="left").sort_values(
            "mean_abs_shap", ascending=False
        )

    preprocessor = result.model.named_steps["preprocessor"]
    classifier = result.model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if result.model_type in {"lightgbm", "xgboost"}:
        importance = np.asarray(classifier.feature_importances_, dtype=float)
        return pd.DataFrame({
            "feature": names,
            "importance": importance,
            "signed_coefficient": np.nan,
            "mean_abs_shap": np.nan,
            "method": f"{result.model_type}_feature_importance",
        }).sort_values("importance", ascending=False)
    coefs = classifier.coef_[0]
    return pd.DataFrame({
        "feature": names,
        "importance": np.abs(coefs),
        "signed_coefficient": coefs,
        "mean_abs_shap": np.nan,
        "method": "absolute_logistic_coefficient",
    }).sort_values("importance", ascending=False)


def write_report(
    report_path: Path,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    top_k: pd.DataFrame,
    bootstrap: pd.DataFrame,
    selected: dict[str, CandidateResult],
    quality: dict[str, Any],
    backtest: pd.DataFrame,
    tuning_candidates: pd.DataFrame,
) -> None:
    selected_lines = []
    for target, result in selected.items():
        test_row = test_metrics.loc[test_metrics["target"] == target].iloc[0]
        top10 = top_k.loc[
            (top_k["target"] == target) & np.isclose(top_k["top_fraction"], 0.10)
        ].iloc[0]
        selected_lines.append(
            f"| {TARGET_META[target]['label']} | {result.name} | "
            f"{test_row['average_precision']:.4f} | {test_row['roc_auc']:.4f} | "
            f"{test_row['recall']:.4f} | {test_row['precision']:.4f} | "
            f"{top10['capture_rate']:.4f} |"
        )
    quality_lines = [
        f"- {TARGET_META[target]['label']}: record_id 중복 {item['record_id_duplicates']}건·그룹 교차 {item['cross_split_groups']}건"
        for target, item in quality.items()
    ]
    candidate_table = validation_metrics[[
        "target_label", "candidate", "feature_set", "average_precision", "roc_auc", "f2"
    ]].sort_values(["target_label", "average_precision"], ascending=[True, False])
    backtest_text = backtest.to_markdown(index=False, floatfmt=".4f") if not backtest.empty else "- 결과 없음"
    tuning_text = tuning_candidates[[
        "target_label", "tuning_rank", "model_type", "candidate",
        "feature_set", "average_precision", "roc_auc", "fit_seconds",
    ]].to_markdown(index=False, floatfmt=".4f")
    text = f"""# 디노테푸란 베이스라인 모델링 2차 · 4개 모델 비교

## 한눈에 보기

- 모델 선택 구간: 2025년 검증셋
- 최종 평가 구간: 2026년 1~6월 테스트셋
- 비교 모델: 사전확률·Logistic Regression·CatBoost·LightGBM·XGBoost
- 비교 변수군: 핵심·핵심+확장·조건부 포함
- 모델 선택 기준: 검증 PR-AUC
- 분류 임계값: 검증 F2 최대화
- DB 변경: 없음

## 선택 모델의 테스트 결과

| 목표 | 선택 모델 | PR-AUC | ROC-AUC | Recall | Precision | 상위 10% 포착률 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(selected_lines)}

## 검증 후보 비교

{candidate_table.to_markdown(index=False, floatfmt='.4f')}

## 목표라벨별 튜닝 후보

{tuning_text}

- 목표별 검증 PR-AUC가 높은 서로 다른 모델 유형 2개를 선정함
- 테스트셋 성능은 튜닝 후보 선정에 사용하지 않음

## 데이터 품질 확인

{chr(10).join(quality_lines)}

- 결과값·판정값·MRL·목표라벨 파생값은 입력 변수에서 제외함
- 동일 duplicate_group_id는 한 데이터 구간에만 존재함
- facility_type은 2025년 이후 코드체계 변경으로 제외함
- 테스트셋은 후보 선택과 임계값 선택에 사용하지 않음

## 불확실성 해석

- 기준 부적합 테스트 양성: 24건
- 작은 양성 표본으로 점추정 변동성이 큼
- 테스트 지표에 층화 부트스트랩 95% 신뢰구간을 함께 제공함
- 2024년을 검증·2025년을 평가로 둔 보조 백테스트를 함께 수행함

## 기준 부적합 보조 백테스트

{backtest_text}

## 산출물

- 성능표: `docs/베이스라인_모델링/table/`
- 발표용 그림: `docs/베이스라인_모델링/figure/`
- 선택 모델: `output/baseline_v1/models/`
- 선택 모델 예측값: `output/baseline_v1/predictions/`
- 전체 실행정보: `output/baseline_v1/baseline_manifest.json`
- 재실행 노트북: `Baseline_Modeling.ipynb`

## 다음 작업

- 선택 모델 오차 분석
- 품목군·연도·출처별 성능 편차 확인
- 임계값별 검사량·미검출 위험 비교
- 외부변수 추가 전후 성능 비교
- 위 표의 목표별 상위 2개 모델을 대상으로 튜닝
- 튜닝 후 XAI·오차분석·운영 임계값 시뮬레이션 수행
"""
    report_path.write_text(text, encoding="utf-8")


def render_figures(
    figure_dir: Path,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    top_k: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    importances: dict[str, pd.DataFrame],
) -> None:
    configure_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=False)
    for axis, target in zip(axes, TARGET_META):
        subset = validation.loc[validation["target"] == target].sort_values("average_precision")
        colors = ["#D3DAE3" if m == "prior" else TARGET_META[target]["color"] for m in subset["model_type"]]
        axis.barh(subset["candidate"], subset["average_precision"], color=colors)
        for i, value in enumerate(subset["average_precision"]):
            axis.text(value, i, f" {value:.3f}", va="center", fontsize=9)
        axis.set_title(TARGET_META[target]["label"])
        axis.set_xlabel("검증 PR-AUC")
        axis.grid(axis="x", alpha=0.7)
        axis.set_xlim(left=0)
    fig.suptitle("베이스라인 후보모델 검증 성능 · 2025년", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = ["average_precision", "roc_auc", "recall", "precision", "f2"]
    labels = ["PR-AUC", "ROC-AUC", "Recall", "Precision", "F2"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), sharey=True)
    for axis, target in zip(axes, TARGET_META):
        row = test.loc[test["target"] == target].iloc[0]
        values = [float(row[m]) for m in metrics]
        axis.bar(labels, values, color=TARGET_META[target]["color"], alpha=0.9)
        for i, value in enumerate(values):
            axis.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        axis.set_title(TARGET_META[target]["label"])
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.7)
        axis.tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("테스트 성능")
    fig.suptitle("선택 모델 테스트 성능 · 2026년 1~6월", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "selected_test_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9.5, 6.0))
    for target, meta in TARGET_META.items():
        subset = top_k.loc[top_k["target"] == target].sort_values("top_fraction")
        axis.plot(
            subset["top_fraction"] * 100,
            subset["capture_rate"] * 100,
            marker="o",
            linewidth=2.2,
            label=meta["label"],
            color=meta["color"],
        )
    axis.plot([0, 20], [0, 20], linestyle="--", color="#6B7280", label="무작위 기준")
    axis.set_xlabel("우선검사 비율 (%)")
    axis.set_ylabel("양성 포착률 (%)")
    axis.set_title("선택 모델의 검사 우선순위 효과 · 테스트셋")
    axis.set_xlim(0, 20.5)
    axis.set_ylim(0, 100)
    axis.grid(alpha=0.7)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_dir / "top_k_capture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for axis, target in zip(axes, TARGET_META):
        frame = predictions[target]
        precision, recall, _ = precision_recall_curve(
            frame["target"],
            frame["score"],
            sample_weight=frame["group_sample_weight"],
        )
        axis.plot(recall, precision, color=TARGET_META[target]["color"], linewidth=2.2)
        weighted_prevalence = np.average(
            frame["target"], weights=frame["group_sample_weight"]
        )
        axis.axhline(weighted_prevalence, color="#6B7280", linestyle="--", linewidth=1.3)
        axis.set_title(TARGET_META[target]["label"])
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.7)
    fig.suptitle("선택 모델 Precision–Recall 곡선 · 테스트셋", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "precision_recall_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0))
    for axis, target in zip(axes, TARGET_META):
        table = importances[target].head(12).sort_values("mean_abs_shap" if importances[target]["mean_abs_shap"].notna().any() else "importance")
        value_column = "mean_abs_shap" if table["mean_abs_shap"].notna().any() else "importance"
        axis.barh(table["feature"], table[value_column], color=TARGET_META[target]["color"])
        axis.set_title(TARGET_META[target]["label"])
        axis.set_xlabel("평균 |SHAP|" if value_column == "mean_abs_shap" else "중요도")
        axis.grid(axis="x", alpha=0.7)
    fig.suptitle("선택 모델 주요 변수", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "selected_feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_noncompliance_backtest(
    data: pd.DataFrame,
    selected: CandidateResult,
    config: dict[str, Any],
) -> pd.DataFrame:
    train = data.loc[(data["event_year"] >= 2015) & (data["event_year"] <= 2023)].copy()
    validation = data.loc[data["event_year"] == 2024].copy()
    test = data.loc[data["event_year"] == 2025].copy()
    if min(train["target"].sum(), validation["target"].sum(), test["target"].sum()) == 0:
        return pd.DataFrame()
    model, validation_scores = fit_model(
        selected.model_type,
        train,
        validation,
        selected.categorical,
        selected.numeric,
        config,
    )
    validation_weight = validation["group_sample_weight"].astype(float).to_numpy()
    threshold = choose_fbeta_threshold(
        validation["target"].astype(int).to_numpy(),
        validation_scores,
        validation_weight,
        float(config["selection"]["threshold_beta"]),
    )
    test_scores = predict_scores(
        model, selected.model_type, test, selected.categorical, selected.numeric
    )
    metrics = evaluate_scores(
        test["target"].astype(int).to_numpy(),
        test_scores,
        test["group_sample_weight"].astype(float).to_numpy(),
        threshold,
        float(config["selection"]["threshold_beta"]),
    )
    metrics.update({
        "train_period": "2015-2023",
        "validation_period": "2024",
        "test_period": "2025",
        "candidate": selected.name,
    })
    return pd.DataFrame([metrics])


def run_baseline_modeling(
    input_dir: Path,
    output_dir: Path,
    docs_dir: Path,
    config_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table_dir = docs_dir / "table"
    figure_dir = docs_dir / "figure"
    model_dir = output_dir / "models"
    prediction_dir = output_dir / "predictions"
    for directory in [table_dir, figure_dir, model_dir, prediction_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    top_k_rows: list[pd.DataFrame] = []
    bootstrap_rows: list[pd.DataFrame] = []
    selected_models: dict[str, CandidateResult] = {}
    selected_predictions: dict[str, pd.DataFrame] = {}
    importance_tables: dict[str, pd.DataFrame] = {}
    quality: dict[str, Any] = {}
    input_checksums: dict[str, str] = {}
    noncompliance_data: pd.DataFrame | None = None

    for target, target_meta in TARGET_META.items():
        input_path = input_dir / f"{target}_features_v1.parquet"
        data = pd.read_parquet(input_path)
        input_checksums[target] = sha256_file(input_path)
        sets = feature_sets(manifest["targets"][target])
        all_cat, all_num = sets["conditional"]
        quality[target] = validate_modeling_frame(data, all_cat, all_num)
        if target == "noncompliance":
            noncompliance_data = data.copy()

        train = data.loc[data["split"] == "train"].copy()
        validation = data.loc[data["split"] == "validation"].copy()
        test = data.loc[data["split"] == "test"].copy()
        y_validation = validation["target"].astype(int).to_numpy()
        validation_weight = validation["group_sample_weight"].astype(float).to_numpy()
        results: list[CandidateResult] = []

        for candidate in config["candidates"]:
            name = str(candidate["name"])
            model_type = str(candidate["model_type"])
            set_name = str(candidate["feature_set"])
            categorical, numeric = sets[set_name]
            started_at = time.perf_counter()
            print(f"[{target}] {name} 시작", flush=True)
            if model_type == "prior":
                prior = float(np.average(train["target"], weights=train["group_sample_weight"]))
                scores = np.full(len(validation), prior, dtype=float)
                model = None
            else:
                model, scores = fit_model(
                    model_type, train, validation, categorical, numeric, config
                )
            fit_seconds = time.perf_counter() - started_at
            threshold = choose_fbeta_threshold(
                y_validation,
                scores,
                validation_weight,
                float(config["selection"]["threshold_beta"]),
            )
            metrics = evaluate_scores(
                y_validation,
                scores,
                validation_weight,
                threshold,
                float(config["selection"]["threshold_beta"]),
            )
            result = CandidateResult(
                name=name,
                model_type=model_type,
                feature_set=set_name,
                model=model,
                categorical=categorical,
                numeric=numeric,
                validation_scores=scores,
                threshold=threshold,
                validation_metrics=metrics,
                fit_seconds=fit_seconds,
            )
            results.append(result)
            validation_rows.append({
                "target": target,
                "target_label": target_meta["label"],
                "candidate": name,
                "model_type": model_type,
                "feature_set": set_name,
                "fit_seconds": fit_seconds,
                **metrics,
            })
            print(
                f"[{target}] {name} 완료 · PR-AUC={metrics['average_precision']:.6f} "
                f"· {fit_seconds:.1f}초",
                flush=True,
            )

        eligible = [result for result in results if result.model_type != "prior"]
        selected = max(
            eligible,
            key=lambda item: (
                item.validation_metrics["average_precision"],
                item.validation_metrics["roc_auc"],
            ),
        )
        selected_models[target] = selected
        model_path = save_selected_model(
            selected.model, selected.model_type, model_dir / f"{target}_selected"
        )

        test_scores = predict_scores(
            selected.model,
            selected.model_type,
            test,
            selected.categorical,
            selected.numeric,
        )
        y_test = test["target"].astype(int).to_numpy()
        test_weight = test["group_sample_weight"].astype(float).to_numpy()
        metrics = evaluate_scores(
            y_test,
            test_scores,
            test_weight,
            selected.threshold,
            float(config["selection"]["threshold_beta"]),
        )
        test_rows.append({
            "target": target,
            "target_label": target_meta["label"],
            "candidate": selected.name,
            "model_type": selected.model_type,
            "feature_set": selected.feature_set,
            "threshold_source": "2025 validation F2 maximum",
            "model_path": str(model_path),
            **metrics,
        })
        target_top_k = top_k_table(
            y_test,
            test_scores,
            test_weight,
            [float(value) for value in config["evaluation"]["top_k_fractions"]],
        )
        target_top_k.insert(0, "target_label", target_meta["label"])
        target_top_k.insert(0, "target", target)
        top_k_rows.append(target_top_k)

        target_bootstrap = stratified_bootstrap_intervals(
            y_test,
            test_scores,
            test_weight,
            selected.threshold,
            [float(value) for value in config["evaluation"]["top_k_fractions"]],
            int(config["evaluation"]["bootstrap_repeats"]),
            float(config["evaluation"]["bootstrap_confidence"]),
            int(config["random_seed"]),
        )
        target_bootstrap.insert(0, "target_label", target_meta["label"])
        target_bootstrap.insert(0, "target", target)
        bootstrap_rows.append(target_bootstrap)

        prediction = test[[
            "record_id", "duplicate_group_id", "event_date", "event_year",
            "target", "group_sample_weight",
        ]].copy()
        prediction["score"] = test_scores
        prediction["prediction"] = (test_scores >= selected.threshold).astype(int)
        prediction["candidate"] = selected.name
        prediction["threshold"] = selected.threshold
        prediction.to_parquet(prediction_dir / f"{target}_test_predictions.parquet", index=False)
        selected_predictions[target] = prediction

        importance = feature_importance_table(
            selected,
            train.sample(min(20000, len(train)), random_state=int(config["random_seed"])),
            int(config["random_seed"]),
        )
        importance.insert(0, "target_label", target_meta["label"])
        importance.insert(0, "target", target)
        importance.to_csv(
            table_dir / f"{target}_selected_feature_importance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        importance_tables[target] = importance

    validation_metrics = pd.DataFrame(validation_rows)
    test_metrics = pd.DataFrame(test_rows)
    top_k = pd.concat(top_k_rows, ignore_index=True)
    bootstrap = pd.concat(bootstrap_rows, ignore_index=True)

    best_by_model = (
        validation_metrics.loc[validation_metrics["model_type"] != "prior"]
        .sort_values(
            ["target", "model_type", "average_precision", "roc_auc"],
            ascending=[True, True, False, False],
        )
        .groupby(["target", "model_type"], observed=True, as_index=False)
        .first()
    )
    tuning_candidates = (
        best_by_model.sort_values(
            ["target", "average_precision", "roc_auc"],
            ascending=[True, False, False],
        )
        .groupby("target", observed=True, group_keys=False)
        .head(int(config["selection"]["tuning_candidates_per_target"]))
        .copy()
    )
    tuning_candidates["tuning_rank"] = (
        tuning_candidates.groupby("target", observed=True).cumcount() + 1
    )

    backtest = pd.DataFrame()
    if noncompliance_data is not None:
        backtest = run_noncompliance_backtest(
            noncompliance_data, selected_models["noncompliance"], config
        )

    validation_metrics.to_csv(
        table_dir / "validation_candidate_metrics.csv", index=False, encoding="utf-8-sig"
    )
    tuning_candidates.to_csv(
        table_dir / "tuning_candidates.csv", index=False, encoding="utf-8-sig"
    )
    test_metrics.to_csv(
        table_dir / "selected_test_metrics.csv", index=False, encoding="utf-8-sig"
    )
    top_k.to_csv(table_dir / "selected_test_top_k.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(
        table_dir / "selected_test_bootstrap_ci.csv", index=False, encoding="utf-8-sig"
    )
    backtest.to_csv(
        table_dir / "noncompliance_2024_2025_backtest.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([
        {
            "target": target,
            "target_label": TARGET_META[target]["label"],
            "candidate": result.name,
            "model_type": result.model_type,
            "feature_set": result.feature_set,
            "threshold": result.threshold,
            "n_categorical": len(result.categorical),
            "n_numeric": len(result.numeric),
        }
        for target, result in selected_models.items()
    ]).to_csv(table_dir / "selected_models.csv", index=False, encoding="utf-8-sig")

    render_figures(
        figure_dir,
        validation_metrics,
        test_metrics,
        top_k,
        selected_predictions,
        importance_tables,
    )
    report_path = docs_dir / "02_베이스라인_모델링_4모델_비교.md"
    write_report(
        report_path,
        validation_metrics,
        test_metrics,
        top_k,
        bootstrap,
        selected_models,
        quality,
        backtest,
        tuning_candidates,
    )

    output_manifest = {
        "version": config["version"],
        "created_with": "baseline_modeling.pipeline.run_baseline_modeling",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "feature_manifest": str(manifest_path),
        "feature_manifest_sha256": sha256_file(manifest_path),
        "inputs": {
            target: {
                "path": str(input_dir / f"{target}_features_v1.parquet"),
                "sha256": input_checksums[target],
            }
            for target in TARGET_META
        },
        "quality": quality,
        "selected_models": {
            target: {
                "candidate": result.name,
                "model_type": result.model_type,
                "feature_set": result.feature_set,
                "threshold": result.threshold,
                "validation_average_precision": result.validation_metrics["average_precision"],
            }
            for target, result in selected_models.items()
        },
        "tuning_candidates": {
            target: tuning_candidates.loc[tuning_candidates["target"] == target, [
                "tuning_rank", "model_type", "candidate", "feature_set",
                "average_precision", "roc_auc",
            ]].to_dict(orient="records")
            for target in TARGET_META
        },
        "selection_rule": "2025 validation weighted average precision maximum",
        "threshold_rule": "2025 validation weighted F2 maximum",
        "test_period": "2026-01-01 through 2026-06-30",
        "report": str(report_path),
    }
    manifest_output_path = output_dir / "baseline_manifest.json"
    manifest_output_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "top_k": top_k,
        "bootstrap": bootstrap,
        "backtest": backtest,
        "selected_models": selected_models,
        "report_path": report_path,
        "manifest_path": manifest_output_path,
    }
