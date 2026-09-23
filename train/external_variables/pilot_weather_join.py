from __future__ import annotations

import argparse
import getpass
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


TRAIN_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = TRAIN_ROOT / "output" / "features_v1"
DEFAULT_OUTPUT = TRAIN_ROOT / "output" / "external_variables_v1" / "weather_pilot"
DEFAULT_DOCS = TRAIN_ROOT / "docs" / "외부변수_연결"
TARGETS = ("occurrence", "screening", "noncompliance")
WINDOWS = (14, 30, 60, 90)

PROVINCE_ALIASES = {
    "서울": "서울특별시",
    "서울특별시": "서울특별시",
    "부산": "부산광역시",
    "부산광역시": "부산광역시",
    "대구": "대구광역시",
    "대구광역시": "대구광역시",
    "인천": "인천광역시",
    "인천광역시": "인천광역시",
    "광주": "광주광역시",
    "광주광역시": "광주광역시",
    "대전": "대전광역시",
    "대전광역시": "대전광역시",
    "울산": "울산광역시",
    "울산광역시": "울산광역시",
    "세종": "세종특별자치시",
    "세종특별자치시": "세종특별자치시",
    "경기": "경기도",
    "경기도": "경기도",
    "강원": "강원특별자치도",
    "강원도": "강원특별자치도",
    "강원특별자치도": "강원특별자치도",
    "충북": "충청북도",
    "충청북도": "충청북도",
    "충남": "충청남도",
    "충청남도": "충청남도",
    "전북": "전북특별자치도",
    "전라북도": "전북특별자치도",
    "전북특별자치도": "전북특별자치도",
    "전남": "전라남도",
    "전라남도": "전라남도",
    "경북": "경상북도",
    "경상북도": "경상북도",
    "경남": "경상남도",
    "경상남도": "경상남도",
    "제주": "제주특별자치도",
    "제주도": "제주특별자치도",
    "제주특별자치도": "제주특별자치도",
}

DAILY_COLUMNS = [
    "temp", "hghst_artmp", "lowst_artmp", "hum", "wind", "max_wind",
    "rn", "srqty", "gr_temp", "soil_temp", "soil_wt",
]


def normalize_province(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    compact = str(value).strip()
    if compact in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[compact]
    for alias in sorted(PROVINCE_ALIASES, key=len, reverse=True):
        if compact.startswith(alias):
            return PROVINCE_ALIASES[alias]
    return pd.NA


def load_target_frames() -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    columns = ["record_id", "split", "event_date", "domestic_province_std"]
    for target in TARGETS:
        path = FEATURE_ROOT / f"{target}_features_v1.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
        frame["province_join_key"] = frame["domestic_province_std"].map(normalize_province).astype("string")
        frames[target] = frame
    return frames


def load_weather(connection, start_date, end_date) -> pd.DataFrame:
    query = text(
        """
        SELECT
            d.obsr_spot_code,
            d.meas_date,
            d.temp,
            d.hghst_artmp,
            d.lowst_artmp,
            d.hum,
            d.wind,
            d.max_wind,
            d.rn,
            d.srqty,
            d.gr_temp,
            d.soil_temp,
            d.soil_wt,
            s.instl_adres
        FROM api_data_info.nas_agri_weather_daily_info AS d
        INNER JOIN api_data_info.nas_agri_weather_obs_station AS s
          ON s.obsr_spot_code = d.obsr_spot_code
        WHERE d.meas_date BETWEEN :start_date AND :end_date
        """
    )
    return pd.read_sql(query, connection, params={"start_date": start_date, "end_date": end_date})


def aggregate_daily(weather: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather["meas_date"] = pd.to_datetime(weather["meas_date"], errors="coerce").dt.normalize()
    weather["province_join_key"] = weather["instl_adres"].map(normalize_province).astype("string")
    station_mapping = (
        weather[["obsr_spot_code", "instl_adres", "province_join_key"]]
        .drop_duplicates()
        .sort_values("obsr_spot_code")
        .reset_index(drop=True)
    )
    valid = weather.loc[weather["province_join_key"].notna() & weather["meas_date"].notna()].copy()
    province_daily = (
        valid.groupby(["province_join_key", "meas_date"], as_index=False)
        .agg(
            station_count=("obsr_spot_code", "nunique"),
            **{column: (column, "mean") for column in DAILY_COLUMNS},
        )
        .sort_values(["province_join_key", "meas_date"])
        .reset_index(drop=True)
    )
    province_daily["weather_day_present"] = 1.0
    return province_daily, station_mapping


def build_rolling_weather(province_daily: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    full_dates = pd.date_range(start_date, end_date, freq="D")
    for province, group in province_daily.groupby("province_join_key", observed=True):
        indexed = group.set_index("meas_date").reindex(full_dates)
        indexed.index.name = "event_date"
        indexed["province_join_key"] = province
        shifted = indexed[DAILY_COLUMNS + ["weather_day_present"]].shift(1)
        observed_positions = pd.Series(
            np.arange(len(indexed), dtype=float),
            index=indexed.index,
        ).where(indexed["weather_day_present"].eq(1)).shift(1)
        result = indexed[["province_join_key"]].copy()
        for window in WINDOWS:
            rolling = shifted.rolling(window=window, min_periods=1)
            for column in DAILY_COLUMNS:
                if column == "rn":
                    result[f"wx_{column}_sum_{window}d"] = rolling[column].sum()
                else:
                    result[f"wx_{column}_mean_{window}d"] = rolling[column].mean()
            result[f"wx_observed_days_{window}d"] = rolling["weather_day_present"].sum().fillna(0)
            result[f"wx_coverage_pct_{window}d"] = (
                100 * result[f"wx_observed_days_{window}d"] / window
            )
            max_positions = observed_positions.rolling(
                window=window,
                min_periods=1,
            ).max()
            max_dates = pd.Series(pd.NaT, index=indexed.index, dtype="datetime64[ns]")
            has_observation = max_positions.notna()
            max_dates.loc[has_observation] = indexed.index[
                max_positions.loc[has_observation].astype(int)
            ]
            result[f"wx_max_observed_date_{window}d"] = max_dates
        outputs.append(result.reset_index())
    return pd.concat(outputs, ignore_index=True)


def attach_and_summarize(
    target: str,
    frame: pd.DataFrame,
    rolling: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    before = len(frame)
    feature_columns = [column for column in rolling.columns if column.startswith("wx_")]
    joined = frame.merge(
        rolling,
        how="left",
        on=["province_join_key", "event_date"],
        validate="many_to_one",
    )
    if len(joined) != before:
        raise RuntimeError(f"{target}: weather join changed row count")

    output_features = joined[["record_id", "split", "province_join_key", "event_date"] + feature_columns]
    output_features.to_parquet(output_dir / f"{target}_weather_features_v1.parquet", index=False)

    joined["join_key_ready"] = joined["province_join_key"].notna() & joined["event_date"].notna()
    summaries = []
    for split, part in joined.groupby("split", dropna=False, observed=True):
        row = {
            "target": target,
            "split": split,
            "n_rows": int(len(part)),
            "join_key_ready": int(part["join_key_ready"].sum()),
            "join_key_ready_pct": round(100 * part["join_key_ready"].mean(), 4) if len(part) else np.nan,
        }
        for window in WINDOWS:
            observed = part[f"wx_observed_days_{window}d"].fillna(0)
            row[f"weather_any_{window}d"] = int((observed > 0).sum())
            row[f"weather_any_{window}d_pct_all"] = round(100 * (observed > 0).mean(), 4)
            eligible = part["join_key_ready"]
            row[f"weather_any_{window}d_pct_ready"] = round(
                100 * (observed.loc[eligible] > 0).mean(), 4
            ) if eligible.any() else np.nan
            minimum_days = int(np.ceil(window * 0.7))
            row[f"weather_70pct_{window}d"] = int((observed >= minimum_days).sum())
            row[f"weather_70pct_{window}d_pct_all"] = round(
                100 * (observed >= minimum_days).mean(), 4
            )
        summaries.append(row)

    audit = {
        "target": target,
        "input_rows": before,
        "output_rows": len(joined),
        "record_id_duplicates": int(joined["record_id"].duplicated().sum()),
        "join_row_count_preserved": len(joined) == before,
    }
    return pd.DataFrame(summaries), audit


def write_report(
    coverage: pd.DataFrame,
    station_mapping: pd.DataFrame,
    weather_rows: int,
    province_daily_rows: int,
    audits: list[dict[str, object]],
    path: Path,
) -> None:
    display_columns = [
        "target", "split", "n_rows", "join_key_ready", "join_key_ready_pct",
        "weather_any_14d_pct_all", "weather_70pct_14d_pct_all",
        "weather_any_30d_pct_all", "weather_70pct_30d_pct_all",
        "weather_any_60d_pct_all", "weather_70pct_60d_pct_all",
        "weather_any_90d_pct_all", "weather_70pct_90d_pct_all",
    ]
    core = coverage.loc[coverage["split"].isin(["train", "validation", "test"]), display_columns]
    unmapped = station_mapping.loc[station_mapping["province_join_key"].isna()]
    report = f"""# 07-3. 국내 기상정보 파일럿 연결

- 점검일: {datetime.now().date().isoformat()}
- 기상 원천: `api_data_info.nas_agri_weather_daily_info`
- 관측소 원천: `api_data_info.nas_agri_weather_obs_station`
- 조회방식: SELECT 전용·트랜잭션 READ ONLY·종료 시 ROLLBACK
- 검사 당일·미래 기상정보 제외
- 원장 및 DB 변경: 0건

## 1. 연결 구조

- 원장 키: `domestic_province_std` + `event_date`
- 기상 키: 관측소 주소에서 파생한 시도 + `meas_date`
- 다수 관측소: 시도·일자별 평균으로 1행 집계
- 강수량: 관측소 평균 후 window 기간 합계
- 나머지 값: 관측소 평균 후 window 기간 평균
- window: 14·30·60·90일
- 최소 품질 기준: window 일수의 70% 이상 관측일 존재

## 2. 원천 처리 현황

- 조회 기상행: {weather_rows:,}
- 시도·일자 집계행: {province_daily_rows:,}
- 관측소 매핑행: {len(station_mapping):,}
- 시도 미매핑 관측소: {len(unmapped):,}

## 3. 목표·기간별 연결률

{core.to_markdown(index=False)}

## 4. 무결성 점검

{pd.DataFrame(audits).to_markdown(index=False)}

## 5. 해석

- `join_key_ready_pct`: 원장에 국내 시도와 기준일이 모두 있는 비율
- `weather_any_*_pct_all`: 전체 모집단 중 과거 기상값이 1일 이상 연결된 비율
- `weather_70pct_*_pct_all`: 전체 모집단 중 window의 70% 이상 관측일이 연결된 비율
- 외부변수 효과 비교는 동일한 연결 가능 행에서 내부변수-only와 내부+기상을 비교함
- 전체 모집단 성능과 기상 연결 모집단 성능을 함께 제시함

## 6. 다음 결정

- 연결률이 충분한 목표·분할에서 내부변수 대비 기상 추가 효과를 검증함
- 시도 평균 연결은 1차 파일럿이며 생산지 좌표 확보 시 최근접 관측소 방식과 비교함
- soil 관련 변수는 원천 결측률이 높아 핵심 기상변수와 분리함
- 외부변수 구성을 확정한 뒤 5개 모델 전체 튜닝을 수행함
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="읽기 전용 기상 DB 호스트")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument(
        "--driver",
        choices=["psycopg2", "psycopg", "pg8000"],
        default="psycopg2",
    )
    parser.add_argument(
        "--password-env",
        help="DB 비밀번호를 읽을 환경변수명. 생략하면 안전한 대화형 입력 사용",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    args = parser.parse_args()

    frames = load_target_frames()
    valid_dates = pd.concat([frame["event_date"] for frame in frames.values()]).dropna()
    start_date = (valid_dates.min() - timedelta(days=max(WINDOWS))).date()
    end_date = valid_dates.max().date()

    if args.password_env:
        password = os.environ.get(args.password_env)
        if not password:
            raise RuntimeError(
                f"환경변수 {args.password_env}가 설정되지 않았습니다."
            )
    else:
        password = getpass.getpass("DB password: ")
    engine = create_engine(
        URL.create(
            f"postgresql+{args.driver}",
            username=args.user,
            password=password,
            host=args.host,
            port=args.port,
            database=args.database,
        )
    )
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")
            if connection.exec_driver_sql("SHOW transaction_read_only").scalar_one() != "on":
                raise RuntimeError("transaction_read_only is not on")
            weather = load_weather(connection, start_date, end_date)
        finally:
            transaction.rollback()
            engine.dispose()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    province_daily, station_mapping = aggregate_daily(weather)
    rolling = build_rolling_weather(province_daily, pd.Timestamp(start_date), pd.Timestamp(end_date))
    station_mapping.to_csv(args.output_dir / "weather_station_province_mapping.csv", index=False, encoding="utf-8-sig")

    coverage_parts = []
    audits = []
    for target, frame in frames.items():
        coverage, audit = attach_and_summarize(target, frame, rolling, args.output_dir)
        coverage_parts.append(coverage)
        audits.append(audit)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    coverage.to_csv(args.output_dir / "weather_join_coverage.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "weather_join_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report_path = args.docs_dir / "07-3_국내기상정보_파일럿연결.md"
    write_report(coverage, station_mapping, len(weather), len(province_daily), audits, report_path)
    print(report_path)


if __name__ == "__main__":
    main()
