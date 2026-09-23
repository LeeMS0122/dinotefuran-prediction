from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "output" / "features_v1"
DEFAULT_OUTPUT = ROOT / "output" / "external_variables_v1" / "country_month_climate_pilot_v1"
DEFAULT_DOCS = ROOT / "docs" / "외부변수_연결"
TARGETS = ("occurrence", "screening", "noncompliance")
SOURCES = ("MFDS", "IMPORT_LIMS_ONLY")
START_YEAR = 1991
END_YEAR = 2020

WORLD_BANK_URL = "https://api.worldbank.org/v2/country?format=json&per_page=400"
NASA_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
NASA_PARAMETERS = (
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "RH2M",
    "PRECTOTCORR",
    "ALLSKY_SFC_SW_DWN",
)
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
CLIMATE_COLUMNS = {
    "T2M": "clim_t2m_c",
    "T2M_MAX": "clim_t2m_max_c",
    "T2M_MIN": "clim_t2m_min_c",
    "RH2M": "clim_rh2m_pct",
    "PRECTOTCORR": "clim_precip_mm_day",
    "ALLSKY_SFC_SW_DWN": "clim_solar_mj_m2_day",
}
HIGH_CONFIDENCE_SOURCES = {
    "production_country_code",
    "origin_country_code",
    "origin_country_name",
}


def request_json(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    retries: int = 4,
    timeout: int = 90,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = session.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"API request failed: {url}") from last_error


def load_target_frames() -> dict[str, pd.DataFrame]:
    columns = [
        "record_id",
        "source_system",
        "split",
        "target",
        "event_month",
        "origin_iso2_std",
        "origin_country_mapping_source",
    ]
    frames: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        path = FEATURE_DIR / f"{target}_features_v1.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["event_month"] = pd.to_numeric(frame["event_month"], errors="coerce").astype("Int64")
        frame["country_climate_scope"] = frame["source_system"].isin(SOURCES)
        frame["country_key_ready"] = (
            frame["country_climate_scope"]
            & frame["origin_iso2_std"].notna()
            & frame["origin_iso2_std"].ne("ZZ")
            & frame["event_month"].between(1, 12)
        )
        frame["country_mapping_confidence"] = np.where(
            frame["origin_country_mapping_source"].isin(HIGH_CONFIDENCE_SOURCES),
            "high",
            np.where(
                frame["origin_country_mapping_source"].eq("import_country_code"),
                "fallback",
                "unknown",
            ),
        )
        frames[target] = frame
    return frames


def load_world_bank_reference(
    session: requests.Session,
    output_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    raw_path = output_dir / "world_bank_country_reference.json"
    csv_path = output_dir / "world_bank_country_reference.csv"
    if raw_path.exists() and not refresh:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        payload = request_json(session, WORLD_BANK_URL)
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    countries = payload[1]
    rows = []
    for item in countries:
        region = item.get("region") or {}
        latitude = pd.to_numeric(item.get("latitude"), errors="coerce")
        longitude = pd.to_numeric(item.get("longitude"), errors="coerce")
        if region.get("id") == "NA" or pd.isna(latitude) or pd.isna(longitude):
            continue
        rows.append(
            {
                "origin_iso2_std": item.get("iso2Code"),
                "iso3_code": item.get("id"),
                "world_bank_country_name": item.get("name"),
                "capital_city": item.get("capitalCity"),
                "representative_latitude": float(latitude),
                "representative_longitude": float(longitude),
                "world_bank_region": region.get("value"),
            }
        )
    reference = pd.DataFrame(rows).drop_duplicates("origin_iso2_std")
    reference.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return reference


def fetch_country_climatology(
    session: requests.Session,
    country: pd.Series,
    cache_dir: Path,
    refresh: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    iso2 = str(country["origin_iso2_std"])
    cache_path = cache_dir / f"{iso2}.json"
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        params = {
            "parameters": ",".join(NASA_PARAMETERS),
            "community": "AG",
            "longitude": country["representative_longitude"],
            "latitude": country["representative_latitude"],
            "start": START_YEAR,
            "end": END_YEAR,
            "format": "JSON",
        }
        payload = request_json(session, NASA_URL, params=params)
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(0.15)

    parameter_values = payload["properties"]["parameter"]
    parameter_meta = payload.get("parameters", {})
    rows: list[dict[str, Any]] = []
    for month_key, month_number in MONTHS.items():
        row: dict[str, Any] = {
            "origin_iso2_std": iso2,
            "event_month": month_number,
            "world_bank_country_name": country["world_bank_country_name"],
            "capital_city": country["capital_city"],
            "representative_latitude": country["representative_latitude"],
            "representative_longitude": country["representative_longitude"],
            "climate_reference_period": f"{START_YEAR}-{END_YEAR}",
            "climate_point_basis": "World Bank capital-city coordinate",
        }
        for nasa_name, output_name in CLIMATE_COLUMNS.items():
            value = parameter_values.get(nasa_name, {}).get(month_key)
            row[output_name] = np.nan if value in (None, -999, -999.0) else float(value)
        rows.append(row)
    return rows, parameter_meta


def build_climate_reference(
    frames: dict[str, pd.DataFrame],
    world_bank: pd.DataFrame,
    session: requests.Session,
    output_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    requested = sorted(
        {
            str(value)
            for frame in frames.values()
            for value in frame.loc[frame["country_key_ready"], "origin_iso2_std"].unique()
        }
    )
    available = world_bank[world_bank["origin_iso2_std"].isin(requested)].copy()
    unavailable = sorted(set(requested) - set(available["origin_iso2_std"]))

    cache_dir = output_dir / "nasa_power_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    climate_rows: list[dict[str, Any]] = []
    units: dict[str, Any] = {}
    failures: list[dict[str, str]] = []
    for index, (_, country) in enumerate(available.iterrows(), start=1):
        iso2 = str(country["origin_iso2_std"])
        try:
            rows, meta = fetch_country_climatology(session, country, cache_dir, refresh)
            climate_rows.extend(rows)
            units.update(meta)
        except Exception as exc:  # retain a complete audit instead of hiding partial failures
            failures.append({"origin_iso2_std": iso2, "error": str(exc)})
        if index % 20 == 0 or index == len(available):
            print(f"climate countries: {index}/{len(available)}")

    climate = pd.DataFrame(climate_rows)
    if not climate.empty:
        if climate.duplicated(["origin_iso2_std", "event_month"]).any():
            raise ValueError("country-month climatology key duplicated")
        climate.to_parquet(output_dir / "country_month_climate_1991_2020.parquet", index=False)
        climate.to_csv(
            output_dir / "country_month_climate_1991_2020.csv",
            index=False,
            encoding="utf-8-sig",
        )

    unmatched = pd.DataFrame(
        [{"origin_iso2_std": code, "reason": "World Bank coordinate unavailable"} for code in unavailable]
        + failures
    )
    unmatched.to_csv(output_dir / "unmatched_country_codes.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "requested_country_codes": len(requested),
        "world_bank_coordinate_matches": len(available),
        "world_bank_unmatched_codes": unavailable,
        "nasa_successful_countries": int(climate["origin_iso2_std"].nunique()) if not climate.empty else 0,
        "nasa_failed_countries": failures,
        "nasa_parameter_metadata": units,
    }
    return climate, unmatched, metadata


def attach_climate(
    target: str,
    frame: pd.DataFrame,
    climate: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    before = len(frame)
    joined = frame.merge(
        climate,
        how="left",
        on=["origin_iso2_std", "event_month"],
        validate="many_to_one",
    )
    if len(joined) != before:
        raise RuntimeError(f"{target}: climate join changed row count")
    joined["country_climate_matched"] = joined["clim_t2m_c"].notna()

    feature_cols = [
        "record_id",
        "source_system",
        "split",
        "origin_iso2_std",
        "origin_country_mapping_source",
        "country_mapping_confidence",
        "event_month",
        "country_climate_scope",
        "country_key_ready",
        "country_climate_matched",
        "world_bank_country_name",
        "capital_city",
        "representative_latitude",
        "representative_longitude",
        "climate_reference_period",
        "climate_point_basis",
        *CLIMATE_COLUMNS.values(),
    ]
    output = joined[feature_cols]
    output.to_parquet(output_dir / f"{target}_country_climate_features_v1.parquet", index=False)

    rows = []
    for (source, split), part in joined.groupby(["source_system", "split"], dropna=False, observed=True):
        if source not in SOURCES:
            continue
        ready = part["country_key_ready"]
        matched = part["country_climate_matched"]
        positive = part["target"].eq(1)
        rows.append(
            {
                "target": target,
                "source_system": source,
                "split": split,
                "n_rows": len(part),
                "n_positive": int(positive.sum()),
                "country_key_ready": int(ready.sum()),
                "country_key_ready_pct": 100 * ready.mean() if len(part) else np.nan,
                "climate_matched": int(matched.sum()),
                "climate_matched_pct_all": 100 * matched.mean() if len(part) else np.nan,
                "climate_matched_pct_ready": 100 * matched[ready].mean() if ready.any() else np.nan,
                "positive_climate_matched": int((positive & matched).sum()),
                "positive_climate_matched_pct": (
                    100 * (positive & matched).sum() / positive.sum() if positive.any() else np.nan
                ),
                "high_confidence_matched": int((matched & joined.loc[part.index, "country_mapping_confidence"].eq("high")).sum()),
                "fallback_matched": int((matched & joined.loc[part.index, "country_mapping_confidence"].eq("fallback")).sum()),
            }
        )
    audit = {
        "target": target,
        "input_rows": before,
        "output_rows": len(joined),
        "row_count_preserved": before == len(joined),
        "record_id_duplicates": int(joined["record_id"].duplicated().sum()),
    }
    return pd.DataFrame(rows), audit


def make_figures(coverage: pd.DataFrame, output_dir: Path) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    summary = (
        coverage.groupby(["target", "source_system"], observed=True)
        .agg(n_rows=("n_rows", "sum"), climate_matched=("climate_matched", "sum"))
        .reset_index()
    )
    summary["coverage_pct"] = 100 * summary["climate_matched"] / summary["n_rows"]
    target_labels = {
        "occurrence": "잔류 존재",
        "screening": "MRL 10% 관심농도",
        "noncompliance": "기준 부적합",
    }
    source_labels = {"IMPORT_LIMS_ONLY": "수입식품 단독", "MFDS": "식약처"}
    summary["target_label"] = summary["target"].map(target_labels)
    summary["source_label"] = summary["source_system"].map(source_labels)
    pivot = summary.pivot(index="target_label", columns="source_label", values="coverage_pct")
    pivot = pivot.reindex([target_labels[target] for target in TARGETS])
    n_pivot = summary.pivot(index="target_label", columns="source_label", values="n_rows").reindex(pivot.index)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    pivot.plot(kind="bar", ax=ax, color=["#2878B5", "#F29E4C"][: len(pivot.columns)])
    ax.set_title("원산국×월 기후평년값 연결률")
    ax.set_xlabel("분석 목표")
    ax.set_ylabel("연결 행 비율(%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="출처", loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.tick_params(axis="x", rotation=0)
    for column_index, container in enumerate(ax.containers):
        labels = [
            f"{value:.1f}%\n(n={int(n_value):,})"
            for value, n_value in zip(pivot.iloc[:, column_index], n_pivot.iloc[:, column_index])
        ]
        ax.bar_label(container, labels=labels, padding=2, fontsize=9)
    fig.tight_layout()
    path = figure_dir / "country_climate_coverage_by_target_source.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [path]


def write_report(
    coverage: pd.DataFrame,
    metadata: dict[str, Any],
    audits: list[dict[str, Any]],
    output_dir: Path,
    docs_dir: Path,
) -> Path:
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_path = docs_dir / "07-9_원산국_월별_기후평년값_연결_파일럿.md"
    display = coverage.copy()
    display["목표"] = display["target"].map(
        {
            "occurrence": "잔류 존재",
            "screening": "MRL 10% 관심농도",
            "noncompliance": "기준 부적합",
        }
    )
    display["출처"] = display["source_system"].map(
        {"MFDS": "식약처", "IMPORT_LIMS_ONLY": "수입식품 단독"}
    )
    display["기간"] = display["split"].map(
        {"train": "학습", "validation": "검증", "test": "테스트"}
    )
    display["전체 n"] = display["n_rows"].map(lambda value: f"{int(value):,}")
    display["양성 n"] = display["n_positive"].map(lambda value: f"{int(value):,}")
    display["국가키 확정"] = display.apply(
        lambda row: f"{int(row['country_key_ready']):,} ({row['country_key_ready_pct']:.2f}%)",
        axis=1,
    )
    display["기후 연결"] = display.apply(
        lambda row: f"{int(row['climate_matched']):,} ({row['climate_matched_pct_all']:.2f}%)",
        axis=1,
    )
    display["확정키 중 연결률"] = display["climate_matched_pct_ready"].map(lambda value: f"{value:.2f}%")
    display["양성 연결"] = display.apply(
        lambda row: "-" if int(row["n_positive"]) == 0 else
        f"{int(row['positive_climate_matched']):,} ({row['positive_climate_matched_pct']:.2f}%)",
        axis=1,
    )
    display["고신뢰/대체"] = display.apply(
        lambda row: f"{int(row['high_confidence_matched']):,}/{int(row['fallback_matched']):,}",
        axis=1,
    )
    display = display[
        ["목표", "출처", "기간", "전체 n", "양성 n", "국가키 확정", "기후 연결", "확정키 중 연결률", "양성 연결", "고신뢰/대체"]
    ]
    report = f"""# 07-9. 원산국×월 기후평년값 연결 파일럿

- 점검일: {datetime.now().date().isoformat()}
- 적용 출처: 식약처(`MFDS`)·수입식품통합시스템 단독(`IMPORT_LIMS_ONLY`)
- 원장 및 DB 변경: 0건
- 외부자료: World Bank Country API·NASA POWER Climatology API
- 기준기간: {START_YEAR}~{END_YEAR} 30년 월별 평년값

## 1. 연결 목적

- 생산지역·생산일이 없는 수입·식약처 자료에 국가별 계절 기후 특성을 보조 설명변수로 제공함.
- 실제 검사연도의 일별 날씨가 아닌 국가 대표지점의 월별 장기평년값을 사용함.
- 검사 이후 정보가 포함되지 않아 시간 누수 가능성이 낮음.

## 2. 연결 규칙

- 원장 키: `origin_iso2_std` + `event_month`
- 좌표: World Bank Country API의 국가별 수도 좌표
- 기후: NASA POWER {START_YEAR}~{END_YEAR} 월별 climatology
- 변수: 평균·최고·최저기온, 상대습도, 보정강수량, 지표면 단파 일사량
- `ZZ` 원산국 미상은 연결 제외
- 국가 코드 근거가 생산국·원산국 코드·원산국명이면 `high`, 수입국 코드 대체이면 `fallback`으로 보존

## 3. 국가 기준표 처리

- 요청 ISO2 코드: {metadata['requested_country_codes']:,}개
- World Bank 좌표 연결: {metadata['world_bank_coordinate_matches']:,}개
- NASA POWER 성공: {metadata['nasa_successful_countries']:,}개
- 좌표 미연결 코드: {', '.join(metadata['world_bank_unmatched_codes']) if metadata['world_bank_unmatched_codes'] else '없음'}
- NASA 실패 코드: {', '.join(item['origin_iso2_std'] for item in metadata['nasa_failed_countries']) if metadata['nasa_failed_countries'] else '없음'}

## 4. 목표·출처·기간별 연결률

{display.to_markdown(index=False)}

## 5. 품질검사

```json
{json.dumps(audits, ensure_ascii=False, indent=2)}
```

## 6. 해석상 주의사항

- 수도 좌표는 농산물 생산지 좌표가 아니므로 국가 규모가 큰 경우 대표성이 낮음.
- 월별 평년값은 전형적인 계절성을 나타내며 해당 연도의 이상기상·단기 강수 사건을 설명하지 못함.
- `import_country_code` 대체값은 원산국 직접값보다 신뢰도가 낮으므로 민감도 분석에서 분리함.
- 이 파일럿은 연결 가능성·커버리지 확인 단계이며 아직 베이스라인 학습 변수로 확정하지 않음.
- 성능 비교는 동일 모집단에서 내부변수-only와 내부+기후평년값을 나란히 평가해야 함.

## 7. 산출물

- `country_month_climate_1991_2020.parquet/csv`: 국가×월 기후평년 기준표
- `*_country_climate_features_v1.parquet`: 목표별 record_id 연결 변수
- `country_climate_coverage.csv`: 목표·출처·기간별 연결률
- `figures/country_climate_coverage_by_target_source.png`: PPT용 연결률 그림
- `unmatched_country_codes.csv`: 미연결 국가 코드와 사유

## 8. 다음 단계

- 고신뢰 국가코드만 사용한 효과 비교와 fallback 포함 효과 비교를 분리함.
- 3개 목표×5개 모델에서 내부변수-only 대비 기후평년값 추가 성능을 검증함.
- Recall을 1순위로 보되 PR-AUC·Precision·F1·F2·Top 10% 포착률을 함께 기록함.
"""
    report_path.write_text(report, encoding="utf-8")
    return report_path


def build_notebook(output_dir: Path) -> Path:
    import nbformat as nbf

    notebook_path = ROOT / "Country_Month_Climate_Pilot.ipynb"
    cells = [
        nbf.v4.new_markdown_cell(
            "# Country-month climate proxy pilot\n\n"
            "식약처·수입식품 자료의 원산국 ISO2와 검사 월을 1991–2020 NASA POWER "
            "기후평년값에 연결한 결과를 재현·점검한다. 수도 좌표는 생산지 대리값일 뿐이다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "ROOT = Path.cwd()\n"
            "OUTPUT = ROOT / 'output' / 'external_variables_v1' / 'country_month_climate_pilot_v1'\n"
            "coverage = pd.read_csv(OUTPUT / 'country_climate_coverage.csv')\n"
            "coverage"
        ),
        nbf.v4.new_code_cell(
            "climate = pd.read_parquet(OUTPUT / 'country_month_climate_1991_2020.parquet')\n"
            "climate.groupby('origin_iso2_std').size().describe(), climate.head()"
        ),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n"
            "display(Image(filename=str(OUTPUT / 'figures' / 'country_climate_coverage_by_target_source.png')))"
        ),
        nbf.v4.new_markdown_cell(
            "## 판정\n\n"
            "- 이 파일럿은 국가의 전형적 월별 기후를 나타내며 실제 연도 날씨가 아니다.\n"
            "- `high`와 `fallback` 국가코드 효과를 분리해 비교한다.\n"
            "- 행 수 보존과 국가×월 기준표의 유일성을 확인한 후 모델 효과 검증으로 진행한다."
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3"}
    nbf.write(notebook, notebook_path)
    return notebook_path


def run(output_dir: Path, docs_dir: Path, refresh: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "dinotefuran-country-climate-pilot/1.0"})
    frames = load_target_frames()
    world_bank = load_world_bank_reference(session, output_dir, refresh)
    climate, unmatched, metadata = build_climate_reference(
        frames, world_bank, session, output_dir, refresh
    )
    if climate.empty:
        raise RuntimeError("No NASA POWER climate records were retrieved")

    coverage_frames = []
    audits = []
    for target, frame in frames.items():
        target_coverage, audit = attach_climate(target, frame, climate, output_dir)
        coverage_frames.append(target_coverage)
        audits.append(audit)
    coverage = pd.concat(coverage_frames, ignore_index=True)
    coverage.to_csv(output_dir / "country_climate_coverage.csv", index=False, encoding="utf-8-sig")
    make_figures(coverage, output_dir)
    report_path = write_report(coverage, metadata, audits, output_dir, docs_dir)
    notebook_path = build_notebook(output_dir)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_systems": list(SOURCES),
        "climate_period": [START_YEAR, END_YEAR],
        "world_bank_url": WORLD_BANK_URL,
        "nasa_url": NASA_URL,
        "nasa_parameters": list(NASA_PARAMETERS),
        "metadata": metadata,
        "audits": audits,
        "outputs": {
            "output_dir": str(output_dir),
            "report": str(report_path),
            "notebook": str(notebook_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    manifest = run(args.output, args.docs, refresh=args.refresh)
    print(json.dumps(manifest["metadata"], ensure_ascii=False, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
