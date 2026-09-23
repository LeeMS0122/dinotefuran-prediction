from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .pipeline import (
    TARGET_META,
    choose_fbeta_threshold,
    evaluate_scores,
    feature_sets,
    fit_model,
    load_yaml,
    save_selected_model,
    validate_modeling_frame,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).parent / "config.yaml"
FEATURE_DIR = ROOT / "output" / "features_v1"
MANIFEST = FEATURE_DIR / "feature_engineering_manifest.json"
CENTRAL = ROOT / "docs" / "베이스라인_모델링_2차" / "table" / "validation_candidate_metrics.csv"
OUTPUT = ROOT / "output" / "baseline_tree_core_v1"
DOCS = ROOT / "docs" / "베이스라인_모델링_코어형_보완"
TABLE = DOCS / "table"
MODELS = OUTPUT / "models"
PREDICTIONS = OUTPUT / "predictions"
TARGETS = ["occurrence", "screening", "noncompliance"]
SPECS = [("lightgbm", "lightgbm_core"), ("xgboost", "xgboost_core")]


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    denominator = frame["precision"] + frame["recall"]
    derived_f1 = (
        2 * frame["precision"] * frame["recall"] / denominator.where(denominator > 0)
    ).fillna(0.0)
    if "f1" not in frame:
        frame["f1"] = derived_f1
    else:
        frame["f1"] = frame["f1"].fillna(derived_f1)

    prevalence = frame["prevalence"].astype(float)
    true_positive_share = frame["recall"].astype(float) * prevalence
    false_positive_share = pd.Series(0.0, index=frame.index)
    mask = frame["precision"].astype(float) > 0
    false_positive_share.loc[mask] = (
        true_positive_share.loc[mask]
        * (1 / frame.loc[mask, "precision"].astype(float) - 1)
    )
    derived_accuracy = (
        true_positive_share + 1 - prevalence - false_positive_share
    ).clip(0, 1)
    if "accuracy" not in frame:
        frame["accuracy"] = derived_accuracy
    else:
        frame["accuracy"] = frame["accuracy"].fillna(derived_accuracy)
    return frame


def update_central(supplement: pd.DataFrame, config: dict) -> pd.DataFrame:
    existing = add_derived_metrics(pd.read_csv(CENTRAL, encoding="utf-8-sig"))
    combined = pd.concat([existing, supplement], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(["target", "candidate"], keep="last")
    target_order = {value: index for index, value in enumerate(TARGETS)}
    candidate_order = {
        str(value["name"]): index for index, value in enumerate(config["candidates"])
    }
    combined["_target"] = combined["target"].map(target_order)
    combined["_candidate"] = combined["candidate"].map(candidate_order)
    combined = (
        combined.sort_values(["_target", "_candidate", "candidate"])
        .drop(columns=["_target", "_candidate"])
        .reset_index(drop=True)
    )
    combined.to_csv(CENTRAL, index=False, encoding="utf-8-sig")
    return combined


def run() -> pd.DataFrame:
    config = load_yaml(CONFIG)
    registry = {
        (str(row["name"]), str(row["model_type"]), str(row["feature_set"]))
        for row in config["candidates"]
    }
    expected_registry = {
        (candidate, model_type, "core") for model_type, candidate in SPECS
    }
    if not expected_registry.issubset(registry):
        raise ValueError("config.yaml에 코어형 후보가 등록되지 않았습니다.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for directory in [TABLE, MODELS, PREDICTIONS]:
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for target in TARGETS:
        data = pd.read_parquet(FEATURE_DIR / f"{target}_features_v1.parquet")
        sets = feature_sets(manifest["targets"][target])
        validate_modeling_frame(data, *sets["conditional"])
        train = data.loc[data["split"].eq("train")].copy()
        validation = data.loc[data["split"].eq("validation")].copy()
        categorical, numeric = sets["core"]
        y_validation = validation["target"].astype(int).to_numpy()
        weights = validation["group_sample_weight"].astype(float).to_numpy()

        for model_type, candidate in SPECS:
            print(f"[{target}] {candidate} 시작", flush=True)
            started = time.perf_counter()
            model, scores = fit_model(
                model_type, train, validation, categorical, numeric, config
            )
            elapsed = time.perf_counter() - started
            threshold = choose_fbeta_threshold(
                y_validation, scores, weights, config["selection"]["threshold_beta"]
            )
            metrics = evaluate_scores(
                y_validation,
                scores,
                weights,
                threshold,
                config["selection"]["threshold_beta"],
            )
            model_path = save_selected_model(
                model, model_type, MODELS / f"{target}_{candidate}"
            )
            prediction = validation[[
                "record_id", "duplicate_group_id", "event_date", "event_year",
                "target", "group_sample_weight",
            ]].copy()
            prediction["score"] = scores
            prediction["prediction"] = (scores >= threshold).astype(int)
            prediction["candidate"] = candidate
            prediction["threshold"] = threshold
            prediction.to_parquet(
                PREDICTIONS / f"{target}_{candidate}_validation.parquet", index=False
            )
            rows.append({
                "target": target,
                "target_label": TARGET_META[target]["label"],
                "candidate": candidate,
                "model_type": model_type,
                "feature_set": "core",
                "fit_seconds": elapsed,
                **metrics,
            })
            print(
                f"[{target}] {candidate} 완료 · Recall={metrics['recall']:.6f} "
                f"· PR-AUC={metrics['average_precision']:.6f} · {elapsed:.1f}초 "
                f"· model={model_path.name}",
                flush=True,
            )

    supplement = pd.DataFrame(rows)
    supplement.to_csv(
        TABLE / "tree_core_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combined = update_central(supplement, config)
    expected = {
        (target, candidate) for target in TARGETS for _, candidate in SPECS
    }
    observed = set(map(tuple, supplement[["target", "candidate"]].to_numpy()))
    if observed != expected or len(combined) != 36:
        raise AssertionError(
            f"보완 결과 불일치: pairs={len(observed)}, central_rows={len(combined)}"
        )
    run_manifest = {
        "version": "baseline_tree_core_v1",
        "reporting_primary_metric": "recall",
        "threshold": "2025 validation weighted F2 maximum",
        "test_data_used": False,
        "supplement_rows": 6,
        "central_validation_rows": 36,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return supplement


def main() -> None:
    columns = [
        "target_label", "candidate", "accuracy", "precision", "recall",
        "f1", "f2", "average_precision", "fit_seconds",
    ]
    print(run()[columns].to_string(index=False))


if __name__ == "__main__":
    main()
