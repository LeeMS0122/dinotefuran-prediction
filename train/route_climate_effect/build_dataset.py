from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("occurrence", "screening", "noncompliance")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(values: pd.Series) -> str:
    joined = "\n".join(values.astype(str).tolist()).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for split_name, part in frame.groupby("split", observed=True):
        rows[str(split_name)] = {
            "rows": int(len(part)),
            "positive": int(part["target"].sum()),
            "positive_rate": float(part["target"].mean()),
            "weighted_positive_rate": float(
                np.average(part["target"], weights=part["group_sample_weight"])
            ),
        }
    return rows


def build_route_climate_dataset(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or Path(__file__).with_name("config.yaml")
    config = _load_yaml(config_path)
    output_dir = ROOT / "output" / "route_climate_effect_v1" / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = ROOT / "output" / "features_v1"
    weather_dir = ROOT / "output" / "external_variables_v1" / "weather_pilot"
    climate_dir = (
        ROOT
        / "output"
        / "external_variables_v1"
        / "country_month_climate_pilot_v1"
    )
    window_manifest_path = ROOT / config["lstm"]["selected_window_manifest"]
    window_manifest = json.loads(window_manifest_path.read_text(encoding="utf-8"))
    windows = {
        target: int(window_manifest["selected_windows"][target]["window_days"])
        for target in TARGETS
    }
    weather_base = list(config["domestic_weather_base_fields"])
    climate_columns = list(config["import_climate_fields"])
    minimum_coverage = float(config["minimum_weather_coverage_pct"])
    target_manifest: dict[str, Any] = {}
    quality_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        window = windows[target]
        feature_path = feature_dir / f"{target}_features_v1.parquet"
        weather_path = weather_dir / f"{target}_weather_features_v1.parquet"
        climate_path = climate_dir / f"{target}_country_climate_features_v1.parquet"
        features = pd.read_parquet(feature_path)
        weather = pd.read_parquet(weather_path)
        climate = pd.read_parquet(climate_path)
        for name, frame in (("features", features), ("weather", weather), ("climate", climate)):
            if not frame["record_id"].is_unique:
                raise ValueError(f"{target}/{name}: record_id 중복")

        weather_columns = [f"wx_{field}_{window}d" for field in weather_base]
        coverage_column = f"wx_coverage_pct_{window}d"
        weather_payload = ["record_id", "province_join_key", *weather_columns, coverage_column]
        climate_payload = [
            "record_id",
            "origin_iso2_std",
            "country_mapping_confidence",
            "country_climate_matched",
            "climate_reference_period",
            "climate_point_basis",
            *climate_columns,
        ]
        merged = features.merge(
            weather[weather_payload], on="record_id", how="left", validate="one_to_one", sort=False
        ).merge(
            climate[climate_payload],
            on="record_id",
            how="left",
            validate="one_to_one",
            sort=False,
            suffixes=("", "_climate"),
        )
        if len(merged) != len(features):
            raise ValueError(f"{target}: 결합 후 행 수 변경")
        if not merged["record_id"].astype(str).equals(features["record_id"].astype(str)):
            raise ValueError(f"{target}: 결합 후 record_id 순서 변경")

        domestic = merged["supply_route_v1"].eq("국내")
        imported = merged["supply_route_v1"].eq("수입")
        domestic_available = (
            domestic
            & merged[weather_columns].notna().all(axis=1)
            & pd.to_numeric(merged[coverage_column], errors="coerce").ge(minimum_coverage)
        )
        import_available = (
            imported
            & merged["country_climate_matched"].fillna(False).astype(bool)
            & merged[climate_columns].notna().all(axis=1)
        )
        import_high = import_available & merged["country_mapping_confidence"].eq("high")
        available_all = domestic_available | import_available
        available_high = domestic_available | import_high

        # 실제 국내 기상과 수입 기후평년은 시간 해상도와 의미가 다르므로 수치열을 분리한다.
        merged.loc[~domestic, weather_columns + [coverage_column]] = np.nan
        merged.loc[~imported, climate_columns] = np.nan
        merged["route_weather_window_days"] = float(window)
        merged["route_domestic_actual_flag"] = domestic_available.astype("int8")
        merged["route_import_climatology_flag"] = import_available.astype("int8")
        merged["route_import_high_confidence_flag"] = import_high.astype("int8")
        merged["route_climate_available_flag"] = available_all.astype("int8")
        merged["route_climate_high_confidence_flag"] = available_high.astype("int8")
        merged["route_climate_type"] = np.select(
            [domestic_available, import_available],
            ["DOMESTIC_ACTUAL", "IMPORT_CLIMATOLOGY"],
            default="UNAVAILABLE",
        )
        domestic_proxy = domestic_available & ~merged["domestic_location_basis"].eq("cultivation")
        import_proxy = import_available
        merged["route_climate_proxy_flag"] = (domestic_proxy | import_proxy).astype("int8")
        merged["route_climate_proxy_type"] = np.select(
            [domestic_proxy, domestic_available, import_proxy],
            ["DOMESTIC_LOCATION_PROXY", "DOMESTIC_CULTIVATION_REGION", "IMPORT_CAPITAL_CLIMATOLOGY_PROXY"],
            default="UNAVAILABLE",
        )

        value_columns = [*weather_columns, coverage_column, *climate_columns]
        missing_columns: list[str] = []
        for column in value_columns:
            mask_column = f"{column}_missing_flag"
            merged[mask_column] = merged[column].isna().astype("int8")
            missing_columns.append(mask_column)
        route_numeric = [
            *value_columns,
            "route_weather_window_days",
            "route_domestic_actual_flag",
            "route_import_climatology_flag",
            "route_import_high_confidence_flag",
            "route_climate_available_flag",
            "route_climate_high_confidence_flag",
            "route_climate_proxy_flag",
            *missing_columns,
        ]

        output_path = output_dir / f"{target}_route_climate_features_v1.parquet"
        merged.to_parquet(output_path, index=False)
        model_split = merged["split"].isin(["train", "validation"])
        paired_all = merged.loc[model_split & available_all].copy()
        paired_high = merged.loc[model_split & available_high].copy()
        paired_all.to_parquet(output_dir / f"{target}_paired_all_v1.parquet", index=False)
        paired_high.to_parquet(output_dir / f"{target}_paired_high_v1.parquet", index=False)

        cross_split = int(
            (paired_high.groupby("duplicate_group_id", observed=True)["split"].nunique() > 1).sum()
        )
        if cross_split:
            raise ValueError(f"{target}: paired high duplicate_group_id 분할 교차 {cross_split}")
        for split_name in ("train", "validation"):
            part = paired_high.loc[paired_high["split"].eq(split_name)]
            if part.empty or part["target"].nunique() != 2:
                raise ValueError(f"{target}/{split_name}: paired high에 두 라벨이 모두 필요")

        quality_rows.append(
            {
                "target": target,
                "window_days": window,
                "feature_rows": int(len(features)),
                "joined_rows": int(len(merged)),
                "paired_all_rows": int(len(paired_all)),
                "paired_high_rows": int(len(paired_high)),
                "domestic_actual_rows": int(domestic_available.sum()),
                "import_climatology_rows": int(import_available.sum()),
                "import_high_confidence_rows": int(import_high.sum()),
                "unavailable_rows": int((~available_all).sum()),
                "record_id_duplicates": int(merged["record_id"].duplicated().sum()),
                "cross_split_duplicate_groups": cross_split,
                "row_count_preserved": bool(len(merged) == len(features)),
                "record_id_order_preserved": bool(
                    merged["record_id"].astype(str).equals(features["record_id"].astype(str))
                ),
                "paired_high_id_sha256": _id_digest(paired_high["record_id"]),
                "paired_high_split": _split_summary(paired_high),
                "test_rows_in_paired_high_file": 0,
                "test_data_used_for_selection": False,
            }
        )
        target_manifest[target] = {
            "window_days": window,
            "feature_path": str(output_path),
            "feature_sha256": _sha256(output_path),
            "paired_all_path": str(output_dir / f"{target}_paired_all_v1.parquet"),
            "paired_high_path": str(output_dir / f"{target}_paired_high_v1.parquet"),
            "weather_numeric": weather_columns + [coverage_column],
            "climate_numeric": climate_columns,
            "route_numeric": route_numeric,
        }

    quality_path = output_dir / "route_climate_data_quality.json"
    quality_path.write_text(json.dumps(quality_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "paired_high_split"}
            for row in quality_rows
        ]
    ).to_csv(output_dir / "route_climate_population_summary.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "version": config["version"],
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "created_with": "route_climate_effect.build_dataset.build_route_climate_dataset",
        "test_data_used": False,
        "selection_splits": ["train", "validation"],
        "route_rules": {
            "domestic": "기준일 이전 국내 시도 실측기상, 목표별 선택 window, 관측률 70% 이상",
            "import": "원산국-월 기후평년, 전체 연결 및 고신뢰 매핑 플래그 병행",
            "unknown": "혼합기후 비교 모집단 제외",
            "separate_value_columns": True,
        },
        "inputs": {
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "window_manifest": {
                "path": str(window_manifest_path),
                "sha256": _sha256(window_manifest_path),
            },
        },
        "targets": target_manifest,
        "quality_path": str(quality_path),
    }
    manifest_path = output_dir / "route_climate_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest_path": manifest_path, "quality_path": quality_path, "quality": quality_rows}


def main() -> None:
    result = build_route_climate_dataset()
    print(f"manifest={result['manifest_path']}")
    for row in result["quality"]:
        print(
            f"{row['target']}: high={row['paired_high_rows']:,} "
            f"domestic={row['domestic_actual_rows']:,} import_high={row['import_high_confidence_rows']:,}"
        )


if __name__ == "__main__":
    main()

