from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_modeling.pipeline import evaluate_scores, top_k_table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "route_climate_effect_v1"
DATASETS = OUTPUT / "datasets"
TARGETS = ("occurrence", "screening", "noncompliance")
TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}
SEGMENTS = ("ALL", "DOMESTIC_ACTUAL", "IMPORT_CLIMATOLOGY")
SEGMENT_LABELS = {
    "ALL": "전체",
    "DOMESTIC_ACTUAL": "국내 실측기상",
    "IMPORT_CLIMATOLOGY": "수입 기후평년",
}
VARIANT_LABELS = {
    "internal_only": "Paired Core",
    "internal_plus_climate": "Core+공급경로별 혼합기후",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    text = "\n".join(
        frame[columns].astype(str).agg("|".join, axis=1).tolist()
    ).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def _safe_metrics(part: pd.DataFrame) -> dict[str, Any]:
    y = part["target"].to_numpy(dtype=int)
    score = part["score"].to_numpy(dtype=float)
    weight = part["group_sample_weight"].to_numpy(dtype=float)
    threshold = float(part["threshold"].iloc[0])
    result = evaluate_scores(y, score, weight, threshold, beta=2.0)
    top10 = top_k_table(y, score, weight, [0.10]).iloc[0]
    result.update(
        {
            "top10_capture_rate": float(top10["capture_rate"]),
            "top10_precision": float(top10["precision"]),
            "top10_lift": float(top10["lift"]),
        }
    )
    return result


def _bootstrap_segment(
    part: pd.DataFrame, repeats: int = 200, confidence: float = 0.95, seed: int = 42
) -> list[dict[str, Any]]:
    keys = ["record_id", "target", "group_sample_weight"]
    pair = part.pivot(index=keys, columns="variant", values=["score", "threshold"]).reset_index()
    required = {"internal_only", "internal_plus_climate"}
    available = set(pair["score"].columns)
    if not required.issubset(available):
        return []
    y = pair["target"].to_numpy(dtype=int)
    n_positive = int(y.sum())
    if n_positive < 5 or len(y) - n_positive < 5:
        return []
    weight = pair["group_sample_weight"].to_numpy(dtype=float)
    base_score = pair[("score", "internal_only")].to_numpy(dtype=float)
    mix_score = pair[("score", "internal_plus_climate")].to_numpy(dtype=float)
    base_threshold = float(pair[("threshold", "internal_only")].iloc[0])
    mix_threshold = float(pair[("threshold", "internal_plus_climate")].iloc[0])
    strata = [np.flatnonzero(y == value) for value in (0, 1)]
    rng = np.random.default_rng(seed)
    samples = {"recall": [], "f2": [], "average_precision": [], "top10_capture_rate": []}
    for _ in range(repeats):
        sampled = np.concatenate([rng.choice(index, len(index), replace=True) for index in strata])
        rng.shuffle(sampled)
        yi = y[sampled]
        wi = weight[sampled]
        base = evaluate_scores(yi, base_score[sampled], wi, base_threshold, beta=2.0)
        mix = evaluate_scores(yi, mix_score[sampled], wi, mix_threshold, beta=2.0)
        base_top = top_k_table(yi, base_score[sampled], wi, [0.10]).iloc[0]["capture_rate"]
        mix_top = top_k_table(yi, mix_score[sampled], wi, [0.10]).iloc[0]["capture_rate"]
        samples["recall"].append(mix["recall"] - base["recall"])
        samples["f2"].append(mix["f2"] - base["f2"])
        samples["average_precision"].append(
            mix["average_precision"] - base["average_precision"]
        )
        samples["top10_capture_rate"].append(mix_top - base_top)
    alpha = (1 - confidence) / 2
    rows = []
    for metric, values in samples.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "delta_mean": float(np.mean(array)),
                "ci_low": float(np.quantile(array, alpha)),
                "ci_high": float(np.quantile(array, 1 - alpha)),
                "bootstrap_repeats": repeats,
                "confidence": confidence,
            }
        )
    return rows


def build_segment_outputs() -> dict[str, Any]:
    predictions = pd.read_parquet(OUTPUT / "validation_predictions.parquet")
    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {}

    for target in TARGETS:
        paired_path = DATASETS / f"{target}_paired_high_v1.parquet"
        route_path = DATASETS / f"{target}_route_climate_features_v1.parquet"
        paired = pd.read_parquet(paired_path)
        route = pd.read_parquet(route_path)
        pred = predictions.loc[predictions["target_name"].eq(target)].copy()
        validation = paired.loc[paired["split"].eq("validation")].copy()
        if not validation["record_id"].is_unique:
            raise ValueError(f"{target}: validation record_id duplicate")
        expected_rows = len(validation) * pred[["model", "variant"]].drop_duplicates().shape[0]
        if len(pred) != expected_rows:
            raise ValueError(f"{target}: prediction row count mismatch")
        meta_columns = [
            "record_id",
            "route_climate_type",
            "source_system",
            "supply_route_v1",
            "event_date",
            "split",
            "target",
            "group_sample_weight",
        ]
        joined = pred.merge(
            validation[meta_columns],
            on="record_id",
            how="left",
            validate="many_to_one",
            suffixes=("", "_dataset"),
        )
        if joined["route_climate_type"].isna().any():
            raise ValueError(f"{target}: unmatched prediction record_id")
        if not joined["target"].eq(joined["target_dataset"]).all():
            raise ValueError(f"{target}: target changed after join")
        if not np.allclose(
            joined["group_sample_weight"], joined["group_sample_weight_dataset"]
        ):
            raise ValueError(f"{target}: sample weight changed after join")
        joined = joined.drop(columns=["target_dataset", "group_sample_weight_dataset"])

        for segment in SEGMENTS:
            segment_part = joined if segment == "ALL" else joined.loc[joined["route_climate_type"].eq(segment)]
            for (model, variant), part in segment_part.groupby(["model", "variant"], observed=True):
                if part.empty:
                    continue
                result = _safe_metrics(part)
                metric_rows.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "route_segment": segment,
                        "route_segment_label": SEGMENT_LABELS[segment],
                        "model": model,
                        "variant": variant,
                        "variant_label": VARIANT_LABELS[variant],
                        **result,
                    }
                )
            for model, model_part in segment_part.groupby("model", observed=True):
                n_pos = int(model_part.drop_duplicates("record_id")["target"].sum())
                status = "ESTIMABLE" if n_pos >= 5 else "INSUFFICIENT_POSITIVES"
                for row in _bootstrap_segment(
                    model_part,
                    seed=42 + TARGETS.index(target) * 100 + list(model_part["model"].unique()).index(model),
                ):
                    bootstrap_rows.append(
                        {
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "route_segment": segment,
                            "route_segment_label": SEGMENT_LABELS[segment],
                            "model": model,
                            "n_rows": int(model_part["record_id"].nunique()),
                            "n_positive": n_pos,
                            "status": status,
                            **row,
                        }
                    )
                if status != "ESTIMABLE":
                    bootstrap_rows.append(
                        {
                            "target": target,
                            "target_label": TARGET_LABELS[target],
                            "route_segment": segment,
                            "route_segment_label": SEGMENT_LABELS[segment],
                            "model": model,
                            "n_rows": int(model_part["record_id"].nunique()),
                            "n_positive": n_pos,
                            "status": status,
                            "metric": "recall",
                            "delta_mean": math.nan,
                            "ci_low": math.nan,
                            "ci_high": math.nan,
                            "bootstrap_repeats": 0,
                            "confidence": 0.95,
                        }
                    )

        for split_name, split_part in route.groupby("split", observed=True):
            for route_name in ("국내", "수입", "미상"):
                route_part = split_part.loc[split_part["supply_route_v1"].fillna("미상").eq(route_name)]
                if route_part.empty:
                    continue
                high = route_part["route_climate_high_confidence_flag"].eq(1)
                quality_rows.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "split": split_name,
                        "supply_route": route_name,
                        "n_rows": int(len(route_part)),
                        "n_positive": int(route_part["target"].sum()),
                        "positive_rate": float(route_part["target"].mean()),
                        "high_confidence_linked_rows": int(high.sum()),
                        "high_confidence_link_rate": float(high.mean()),
                        "proxy_rows": int(route_part["route_climate_proxy_flag"].eq(1).sum()),
                        "proxy_rate": float(route_part["route_climate_proxy_flag"].eq(1).mean()),
                    }
                )
            for source, source_part in split_part.groupby("source_system", observed=True):
                high = source_part["route_climate_high_confidence_flag"].eq(1)
                source_rows.append(
                    {
                        "target": target,
                        "target_label": TARGET_LABELS[target],
                        "split": split_name,
                        "source_system": source,
                        "n_rows": int(len(source_part)),
                        "n_positive": int(source_part["target"].sum()),
                        "high_confidence_linked_rows": int(high.sum()),
                        "high_confidence_link_rate": float(high.mean()),
                    }
                )

        feature_source = pd.read_parquet(ROOT / "output" / "features_v1" / f"{target}_features_v1.parquet")
        route_key = route[["record_id", "target", "split"]].copy()
        feature_key = feature_source[["record_id", "target", "split"]].copy()
        integrity[target] = {
            "feature_rows": int(len(feature_source)),
            "route_rows": int(len(route)),
            "row_count_preserved": bool(len(feature_source) == len(route)),
            "record_id_order_preserved": bool(
                feature_source["record_id"].astype(str).equals(route["record_id"].astype(str))
            ),
            "feature_key_digest": _stable_digest(feature_key, ["record_id", "target", "split"]),
            "route_key_digest": _stable_digest(route_key, ["record_id", "target", "split"]),
            "label_and_split_digest_match": bool(
                _stable_digest(feature_key, ["record_id", "target", "split"])
                == _stable_digest(route_key, ["record_id", "target", "split"])
            ),
            "feature_record_id_duplicates": int(feature_source["record_id"].duplicated().sum()),
            "route_record_id_duplicates": int(route["record_id"].duplicated().sum()),
            "paired_validation_rows": int(len(validation)),
            "prediction_pairs_per_record": int(pred[["model", "variant"]].drop_duplicates().shape[0]),
            "prediction_join_complete": bool(len(joined) == len(pred)),
        }

    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    quality = pd.DataFrame(quality_rows)
    source_quality = pd.DataFrame(source_rows)
    metrics.to_csv(OUTPUT / "validation_metrics_by_route.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUTPUT / "route_segment_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(OUTPUT / "route_linkage_quality_by_split.csv", index=False, encoding="utf-8-sig")
    source_quality.to_csv(
        OUTPUT / "route_linkage_quality_by_source.csv", index=False, encoding="utf-8-sig"
    )

    confusion_rows = []
    operational_rows = []
    for _, row in metrics.iterrows():
        for actual, negative, positive in (
            ("음성", row["tn"], row["fp"]),
            ("양성", row["fn"], row["tp"]),
        ):
            total = int(negative + positive)
            confusion_rows.append(
                {
                    **{key: row[key] for key in ["target", "target_label", "route_segment", "route_segment_label", "model", "variant", "variant_label"]},
                    "actual_class": actual,
                    "predicted_negative": int(negative),
                    "predicted_positive": int(positive),
                    "actual_total": total,
                    "predicted_negative_row_pct": float(negative / total) if total else math.nan,
                    "predicted_positive_row_pct": float(positive / total) if total else math.nan,
                }
            )
        selected = int(row["tp"] + row["fp"])
        operational_rows.append(
            {
                **{key: row[key] for key in ["target", "target_label", "route_segment", "route_segment_label", "model", "variant", "variant_label"]},
                "n_rows": int(row["n_rows"]),
                "n_positive": int(row["n_positive"]),
                "threshold": float(row["threshold"]),
                "selected_rows": selected,
                "selected_rate": float(selected / row["n_rows"]),
                "true_positive_rows": int(row["tp"]),
                "missed_positive_rows": int(row["fn"]),
                "recall_weighted": float(row["recall"]),
                "precision_weighted": float(row["precision"]),
                "tests_per_true_positive_raw": float(selected / row["tp"]) if row["tp"] else math.nan,
                "top10_capture_rate": float(row["top10_capture_rate"]),
                "top10_precision": float(row["top10_precision"]),
                "top10_lift": float(row["top10_lift"]),
            }
        )
    pd.DataFrame(confusion_rows).to_csv(
        OUTPUT / "confusion_matrices_by_route.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(operational_rows).to_csv(
        OUTPUT / "operational_metrics_by_route.csv", index=False, encoding="utf-8-sig"
    )

    pilot_code = ROOT / "external_variables" / "pilot_weather_join.py"
    code_text = pilot_code.read_text(encoding="utf-8")
    shift_position = code_text.find(".shift(1)")
    rolling_position = code_text.find(".rolling(", shift_position)
    climate_periods = set()
    for target in TARGETS:
        route = pd.read_parquet(DATASETS / f"{target}_route_climate_features_v1.parquet", columns=["climate_reference_period"])
        climate_periods.update(route["climate_reference_period"].dropna().astype(str).unique().tolist())
    leakage_audit = {
        "status": "PASS_WITH_TRACEABILITY_LIMITATION",
        "domestic_weather": {
            "event_day_and_future_excluded_by_code": bool(
                shift_position >= 0 and rolling_position > shift_position
            ),
            "rule": "시도-일자 집계 후 shift(1), 이후 과거 window rolling",
            "source_code": str(pilot_code),
            "source_code_sha256": _sha256(pilot_code),
            "row_level_max_measurement_date_persisted": False,
            "limitation": "집계 산출물에 window별 최종 meas_date가 없어 행별 날짜 역추적은 불가",
            "recommended_rebuild_field": "wx_max_meas_date_{window}d",
        },
        "import_climatology": {
            "reference_periods": sorted(climate_periods),
            "type": "고정 월별 기후평년 대리변수",
            "year_specific_future_weather_used": False,
            "strict_historical_availability_note": "1991-2020 평년은 2015-2019 표본 시점 이후 연도를 일부 포함하므로 회고적 고정 대리변수로 표시",
        },
        "test_data_used_for_selection": False,
    }
    (OUTPUT / "route_integrity_audit.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "future_weather_leakage_audit.json").write_text(
        json.dumps(leakage_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "metrics_rows": int(len(metrics)),
        "bootstrap_rows": int(len(bootstrap)),
        "quality_rows": int(len(quality)),
        "source_quality_rows": int(len(source_quality)),
        "integrity_all_pass": bool(
            all(
                item["row_count_preserved"]
                and item["record_id_order_preserved"]
                and item["label_and_split_digest_match"]
                and item["feature_record_id_duplicates"] == 0
                and item["route_record_id_duplicates"] == 0
                and item["prediction_join_complete"]
                for item in integrity.values()
            )
        ),
        "test_data_used": False,
        "leakage_status": leakage_audit["status"],
    }
    (OUTPUT / "segment_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    print(json.dumps(build_segment_outputs(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
