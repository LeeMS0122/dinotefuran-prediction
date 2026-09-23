from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_config(root: Path = ROOT) -> dict[str, Any]:
    return yaml.safe_load(
        (root / "tuning_v2_weather" / "config.yaml").read_text(
            encoding="utf-8-sig"
        )
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weather_columns(config: dict[str, Any], window: int) -> list[str]:
    return [
        f"wx_{field}_{window}d"
        for field in config["weather"]["numeric_base_fields"]
    ]


def lineage_column(config: dict[str, Any], window: int) -> str:
    return str(config["weather"]["required_lineage_column_pattern"]).format(
        window=window
    )


def build_registry(config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    index = 1
    for model in config["experiment"]["models"]:
        for window in config["weather"]["windows"]:
            rows.append(
                {
                    "config_id": f"WV2-{index:03d}",
                    "target": config["scope"]["target"],
                    "model": model,
                    "window_days": int(window),
                    "internal_feature_set": config["experiment"][
                        "internal_feature_set"
                    ],
                    "variant": "internal_plus_weather",
                    "comparison": "paired_internal_only",
                    "target_trials": int(
                        config["experiment"]["trials_per_configuration"]
                    ),
                    "test_data_used": False,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _range_violations(frame: pd.DataFrame, window: int) -> dict[str, int]:
    prefix = f"_{window}d"
    return {
        "temperature_order": int(
            (
                (frame[f"wx_hghst_artmp_mean{prefix}"] < frame[f"wx_temp_mean{prefix}"])
                | (frame[f"wx_temp_mean{prefix}"] < frame[f"wx_lowst_artmp_mean{prefix}"])
            ).sum()
        ),
        "humidity_outside_0_100": int(
            (
                (frame[f"wx_hum_mean{prefix}"] < 0)
                | (frame[f"wx_hum_mean{prefix}"] > 100)
            ).sum()
        ),
        "negative_wind": int(
            (
                (frame[f"wx_wind_mean{prefix}"] < 0)
                | (frame[f"wx_max_wind_mean{prefix}"] < 0)
            ).sum()
        ),
        "negative_rain": int((frame[f"wx_rn_sum{prefix}"] < 0).sum()),
        "negative_solar": int((frame[f"wx_srqty_mean{prefix}"] < 0).sum()),
    }


def run_audit(root: Path = ROOT) -> dict[str, Any]:
    config = load_config(root)
    feature_path = root / config["inputs"]["features"]
    preferred_weather_path = root / config["inputs"]["weather_v2"]
    fallback_weather_path = root / config["inputs"]["weather_fallback"]
    weather_path = (
        preferred_weather_path
        if preferred_weather_path.exists()
        else fallback_weather_path
    )
    builder_path = root / config["inputs"]["weather_builder"]
    output_root = root / config["output"]["root"]
    output_root.mkdir(parents=True, exist_ok=True)

    allowed_splits = list(config["splits"]["allowed_for_development"])
    feature_columns = [
        "record_id",
        "duplicate_group_id",
        "event_date",
        "event_year",
        "split",
        "target",
        "source_system",
        "domestic_province_std",
        "domestic_location_basis",
    ]
    features = pd.read_parquet(
        feature_path,
        columns=feature_columns,
        filters=[("split", "in", allowed_splits)],
    )
    weather = pd.read_parquet(
        weather_path,
        filters=[("split", "in", allowed_splits)],
    )
    features["event_date"] = pd.to_datetime(features["event_date"], errors="raise")
    weather["event_date"] = pd.to_datetime(weather["event_date"], errors="raise")

    key_checks = {
        "development_feature_rows": int(len(features)),
        "development_weather_rows": int(len(weather)),
        "full_feature_rows_metadata": int(pq.ParquetFile(feature_path).metadata.num_rows),
        "full_weather_rows_metadata": int(pq.ParquetFile(weather_path).metadata.num_rows),
        "feature_record_id_duplicates": int(features["record_id"].duplicated().sum()),
        "weather_record_id_duplicates": int(weather["record_id"].duplicated().sum()),
        "development_record_id_sets_equal": bool(
            set(features["record_id"].astype(str))
            == set(weather["record_id"].astype(str))
        ),
        "test_rows_read": 0,
    }
    weather_payload = weather.drop(columns=["split", "event_date"])
    joined = features.merge(
        weather_payload,
        on="record_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    weather_keys = weather[["record_id", "split", "event_date"]].rename(
        columns={
            "split": "weather_split",
            "event_date": "weather_event_date",
        }
    )
    key_compare = features[["record_id", "split", "event_date"]].merge(
        weather_keys,
        on="record_id",
        validate="one_to_one",
    )
    key_checks["split_values_equal"] = bool(
        key_compare["split"].astype(str).equals(
            key_compare["weather_split"].astype(str)
        )
    )
    key_checks["event_dates_equal"] = bool(
        key_compare["event_date"].equals(key_compare["weather_event_date"])
    )

    coverage_rows = []
    missing_rows = []
    range_rows = []
    lineage_rows = []
    common_mask = joined["split"].isin(allowed_splits)
    builder_text = builder_path.read_text(encoding="utf-8")
    shift_before_rolling = ".shift(1)" in builder_text
    for raw_window in config["weather"]["windows"]:
        window = int(raw_window)
        observed = pd.to_numeric(
            joined[f"wx_observed_days_{window}d"], errors="coerce"
        )
        coverage = pd.to_numeric(
            joined[f"wx_coverage_pct_{window}d"], errors="coerce"
        )
        eligible = coverage.ge(float(config["weather"]["minimum_coverage_pct"]))
        common_mask &= eligible
        consistency_error = (
            coverage.sub(100 * observed / window).abs().fillna(0).gt(1e-8)
        )
        for split_name, part_index in joined.groupby("split", observed=True).groups.items():
            part_eligible = eligible.loc[part_index]
            coverage_rows.append(
                {
                    "window_days": window,
                    "split": str(split_name),
                    "rows": int(len(part_index)),
                    "eligible_rows": int(part_eligible.sum()),
                    "eligible_pct": float(100 * part_eligible.mean()),
                    "positive": int(joined.loc[part_index, "target"].sum()),
                    "eligible_positive": int(
                        joined.loc[part_index].loc[part_eligible, "target"].sum()
                    ),
                    "coverage_consistency_errors": int(
                        consistency_error.loc[part_index].sum()
                    ),
                }
            )
        selected = joined.loc[eligible]
        for column in weather_columns(config, window):
            missing_rows.append(
                {
                    "window_days": window,
                    "column": column,
                    "eligible_rows": int(len(selected)),
                    "missing_rows": int(selected[column].isna().sum()),
                    "missing_pct": float(100 * selected[column].isna().mean()),
                }
            )
        range_rows.append(
            {"window_days": window, **_range_violations(selected, window)}
        )
        trace_column = lineage_column(config, window)
        trace_present = trace_column in joined.columns
        same_or_future = None
        before_window = None
        if trace_present:
            trace = pd.to_datetime(joined[trace_column], errors="coerce")
            same_or_future = int((trace >= joined["event_date"]).sum())
            before_window = int(
                (trace < joined["event_date"] - pd.to_timedelta(window, unit="D")).sum()
            )
        lineage_rows.append(
            {
                "window_days": window,
                "lineage_column": trace_column,
                "lineage_column_present": trace_present,
                "same_day_or_future_rows": same_or_future,
                "before_window_rows": before_window,
                "builder_shift_before_rolling": shift_before_rolling,
            }
        )

    common = joined.loc[common_mask].copy()
    group_split_counts = common.groupby("duplicate_group_id", observed=True)[
        "split"
    ].nunique()
    cohort = {
        "common_rows": int(len(common)),
        "common_record_id_duplicates": int(common["record_id"].duplicated().sum()),
        "cross_split_duplicate_groups": int(group_split_counts.gt(1).sum()),
        "split_rows": {
            str(split_name): {
                "rows": int(len(part)),
                "positive": int(part["target"].sum()),
                "positive_rate": float(part["target"].mean()),
            }
            for split_name, part in common.groupby("split", observed=True)
        },
        "source_rows": {
            str(key): int(value)
            for key, value in common["source_system"].value_counts(dropna=False).items()
        },
        "location_basis_rows": {
            str(key): int(value)
            for key, value in common["domestic_location_basis"]
            .value_counts(dropna=False)
            .items()
        },
    }

    coverage_frame = pd.DataFrame(coverage_rows)
    missing_frame = pd.DataFrame(missing_rows)
    range_frame = pd.DataFrame(range_rows)
    lineage_frame = pd.DataFrame(lineage_rows)
    registry = build_registry(config)
    coverage_frame.to_csv(
        output_root / "coverage_by_split_window.csv",
        index=False,
        encoding="utf-8-sig",
    )
    missing_frame.to_csv(
        output_root / "weather_feature_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )
    range_frame.to_csv(
        output_root / "weather_range_checks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lineage_frame.to_csv(
        output_root / "weather_time_lineage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    registry.to_csv(
        output_root / "experiment_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lineage_complete = bool(lineage_frame["lineage_column_present"].all())
    hard_checks = {
        "record_ids_unique": bool(
            key_checks["feature_record_id_duplicates"] == 0
            and key_checks["weather_record_id_duplicates"] == 0
        ),
        "record_id_sets_equal": key_checks["development_record_id_sets_equal"],
        "split_values_equal": key_checks["split_values_equal"],
        "event_dates_equal": key_checks["event_dates_equal"],
        "coverage_math_valid": bool(
            coverage_frame["coverage_consistency_errors"].sum() == 0
        ),
        "weather_ranges_valid": bool(
            range_frame[
                [
                    "humidity_outside_0_100",
                    "negative_wind",
                    "negative_rain",
                    "negative_solar",
                ]
            ]
            .to_numpy()
            .sum()
            == 0
        ),
        "common_record_ids_unique": cohort["common_record_id_duplicates"] == 0,
        "no_cross_split_duplicate_groups": cohort[
            "cross_split_duplicate_groups"
        ]
        == 0,
        "test_data_not_read": key_checks["test_rows_read"] == 0,
        "builder_excludes_event_day": shift_before_rolling,
    }
    if not all(hard_checks.values()):
        failed = [name for name, passed in hard_checks.items() if not passed]
        raise RuntimeError(f"weather v2 data quality audit failed: {failed}")

    status = (
        "ready_for_confirmatory_development"
        if lineage_complete
        else "blocked_pending_weather_lineage_regeneration"
    )
    findings = []
    temperature_order_violations = int(range_frame["temperature_order"].sum())
    if temperature_order_violations:
        findings.append(
            {
                "severity": "low",
                "finding": "aggregated_temperature_order_mismatch",
                "evidence": f"high/mean/low aggregate ordering mismatches={temperature_order_violations}",
                "impact": (
                    "daily fields have different missing-day patterns; the aggregate columns "
                    "are not guaranteed to share identical contributing dates"
                ),
                "remediation": (
                    "persist per-field observed-day counts or aggregate only common valid days "
                    "during the next weather regeneration"
                ),
            }
        )
    if not lineage_complete:
        findings.append(
            {
                "severity": "high",
                "finding": "row_level_weather_max_observed_date_missing",
                "evidence": (
                    "builder uses shift(1), but persisted weather rows do not contain "
                    "the actual maximum measurement date for each window"
                ),
                "impact": "confirmatory leakage evidence is not independently auditable",
                "remediation": "regenerate v2 weather parquet with wx_max_observed_date_* columns",
            }
        )
    validation_coverage = coverage_frame.loc[
        coverage_frame["split"].eq("validation")
    ]
    if float(validation_coverage["eligible_pct"].min()) < 80:
        findings.append(
            {
                "severity": "medium",
                "finding": "validation_weather_coverage_below_80pct",
                "evidence": f"minimum validation coverage={validation_coverage['eligible_pct'].min():.4f}%",
                "impact": "results apply to the linked domestic subset, not all screening rows",
                "remediation": "report linked-cohort scope and retain paired internal control",
            }
        )
    maximum_missing_pct = float(missing_frame["missing_pct"].max())
    if maximum_missing_pct > 0:
        findings.append(
            {
                "severity": "low",
                "finding": "weather_values_missing_inside_coverage_eligible_rows",
                "evidence": f"maximum eligible-row missingness={maximum_missing_pct:.4f}%",
                "impact": "coverage is based on weather-day presence, not completeness of every field",
                "remediation": "retain model-side imputation and report per-field missingness",
            }
        )

    payload = {
        "version": config["version"],
        "status": status,
        "intended_grain": "one screening inspection record",
        "development_period": "2015-2025",
        "test_data_used": False,
        "key_checks": key_checks,
        "cohort": cohort,
        "hard_checks": hard_checks,
        "findings": findings,
        "registry_rows": int(len(registry)),
        "inputs": {
            "features": {"path": str(feature_path), "sha256": sha256_file(feature_path)},
            "weather": {
                "path": str(weather_path),
                "sha256": sha256_file(weather_path),
                "source": (
                    "v2_regenerated"
                    if weather_path == preferred_weather_path
                    else "v1_fallback"
                ),
            },
            "weather_builder": {
                "path": str(builder_path),
                "sha256": sha256_file(builder_path),
            },
        },
        "outputs": {
            "coverage": str(output_root / "coverage_by_split_window.csv"),
            "missingness": str(output_root / "weather_feature_missingness.csv"),
            "range_checks": str(output_root / "weather_range_checks.csv"),
            "time_lineage": str(output_root / "weather_time_lineage.csv"),
            "experiment_registry": str(output_root / "experiment_registry.csv"),
        },
    }
    (output_root / "data_quality_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    print(json.dumps(run_audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
