from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd

from external_variables.pilot_weather_join import normalize_province


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "output" / "eda" / "cache" / "analysis.parquet"
FEATURE_DIR = ROOT / "output" / "features_v1"
WEATHER_DIR = ROOT / "output" / "external_variables_v1" / "weather_pilot"
OUTPUT = ROOT / "output" / "weather_scope_audit_v2"
DOCS = ROOT / "docs" / "외부변수_연결"
TARGETS = ["occurrence", "screening", "noncompliance"]
TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}
WINDOWS = [14, 30, 60, 90]


def rate(numerator: pd.Series, denominator: int) -> float:
    return float(numerator.sum() / denominator * 100) if denominator else np.nan


def load_master() -> pd.DataFrame:
    columns = [
        "record_id",
        "source_system",
        "event_date",
        "date_basis",
        "collection_province",
        "origin_country_name",
        "result_raw",
        "standard_raw",
        "result_value_numeric_enriched",
        "standard_numeric_enriched",
        "label_occurrence_v1",
        "label_screening_10pct_v1",
        "label_noncompliance_v1",
    ]
    master = pd.read_parquet(ANALYSIS, columns=columns)
    if not master["record_id"].is_unique:
        raise ValueError("통합원장 record_id 중복")
    master["event_date"] = pd.to_datetime(master["event_date"], errors="coerce")
    master["province_join_key"] = master["collection_province"].map(normalize_province).astype("string")
    master["domestic_key_ready"] = master["province_join_key"].notna() & master["event_date"].notna()
    master["strict_collection_weather_ready"] = master["domestic_key_ready"] & master["date_basis"].eq("수거일")
    return master


def master_source_profile(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, part in master.groupby("source_system", observed=True):
        n = len(part)
        row = {
            "source_system": source,
            "n_rows": n,
            "event_date_known": int(part["event_date"].notna().sum()),
            "collection_province_known": int(part["collection_province"].notna().sum()),
            "province_normalized": int(part["province_join_key"].notna().sum()),
            "strict_collection_weather_ready": int(part["strict_collection_weather_ready"].sum()),
            "strict_collection_weather_ready_pct": rate(part["strict_collection_weather_ready"], n),
            "origin_country_known": int(part["origin_country_name"].notna().sum()),
            "result_raw_known": int(part["result_raw"].notna().sum()),
            "result_numeric_known": int(part["result_value_numeric_enriched"].notna().sum()),
            "standard_numeric_known": int(part["standard_numeric_enriched"].notna().sum()),
        }
        for target in TARGETS:
            column = {
                "occurrence": "label_occurrence_v1",
                "screening": "label_screening_10pct_v1",
                "noncompliance": "label_noncompliance_v1",
            }[target]
            row[f"{target}_label_known"] = int(part[column].notna().sum())
            row[f"{target}_positive"] = int(part[column].eq(1).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_rows", ascending=False).reset_index(drop=True)


def date_basis_profile(master: pd.DataFrame) -> pd.DataFrame:
    return (
        master.groupby(["source_system", "date_basis"], observed=True)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values(["source_system", "n_rows"], ascending=[True, False])
    )


def target_weather_profile() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        features = pd.read_parquet(FEATURE_DIR / f"{target}_features_v1.parquet")
        weather = pd.read_parquet(WEATHER_DIR / f"{target}_weather_features_v1.parquet")
        weather_columns = ["record_id", *[f"wx_coverage_pct_{window}d" for window in WINDOWS]]
        joined = features.merge(
            weather[weather_columns], on="record_id", how="left", validate="one_to_one"
        )
        common_weather = joined["split"].isin(["train", "validation"])
        for window in WINDOWS:
            common_weather &= joined[f"wx_coverage_pct_{window}d"].ge(70)
        for source, part in joined.groupby("source_system", observed=True):
            source_mask = common_weather.loc[part.index]
            common = part.loc[source_mask]
            rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "source_system": source,
                    "n_rows": int(len(part)),
                    "n_positive": int(part["target"].sum()),
                    "domestic_province_known": int(part["domestic_province_std"].notna().sum()),
                    "cultivation_basis": int(part["domestic_location_basis"].eq("cultivation_province").sum()),
                    "collection_fallback_basis": int(part["domestic_location_basis"].eq("collection_province").sum()),
                    "origin_iso2_known": int(part["origin_iso2_std"].ne("ZZ").sum()),
                    "weather_common_rows": int(len(common)),
                    "weather_common_positive": int(common["target"].sum()),
                }
            )
    return pd.DataFrame(rows).sort_values(["target", "n_rows"], ascending=[True, False])


def label_overlap_audit() -> dict:
    frames = {}
    for target in ["occurrence", "noncompliance"]:
        features = pd.read_parquet(
            FEATURE_DIR / f"{target}_features_v1.parquet",
            columns=["record_id", "split", "source_system", "target"],
        )
        weather = pd.read_parquet(
            WEATHER_DIR / f"{target}_weather_features_v1.parquet",
            columns=["record_id", *[f"wx_coverage_pct_{window}d" for window in WINDOWS]],
        )
        joined = features.merge(weather, on="record_id", validate="one_to_one")
        mask = joined["split"].isin(["train", "validation"])
        for window in WINDOWS:
            mask &= joined[f"wx_coverage_pct_{window}d"].ge(70)
        frames[target] = joined.loc[mask, ["record_id", "source_system", "target"]]
    paired = frames["occurrence"].merge(
        frames["noncompliance"],
        on="record_id",
        suffixes=("_occurrence", "_noncompliance"),
        validate="one_to_one",
    )
    return {
        "occurrence_rows": int(len(frames["occurrence"])),
        "noncompliance_rows": int(len(frames["noncompliance"])),
        "paired_rows": int(len(paired)),
        "target_disagreements": int((paired["target_occurrence"] != paired["target_noncompliance"]).sum()),
        "positive_rows": int(paired["target_occurrence"].sum()),
        "source_counts": {
            str(key): int(value)
            for key, value in paired["source_system_occurrence"].value_counts().items()
        },
    }


def linkage_policy(profile: pd.DataFrame) -> pd.DataFrame:
    by_source = profile.set_index("source_system")
    records = []
    policy = {
        "SAFEQ": (
            "국내 수거일 이전 단기 기상",
            "조건부 사용",
            "재배지 우선·수거지 보조. 수거일 기준이므로 생산환경 전체가 아닌 수거 전 노출 대리변수로 해석",
        ),
        "INTEGRATED_DIST_ONLY": (
            "국내 수거지역+수거일 단기 기상",
            "라벨 보강 후 사용",
            "지역·수거일은 있으나 세 목표라벨과 정량 결과가 모두 미확정",
        ),
        "MFDS": (
            "원산국×월 기후평년값",
            "외부자료 필요",
            "국내 생산지역 없음. 검사·의뢰일 주변 국내 기상을 생산환경으로 연결하면 안 됨",
        ),
        "IMPORT_LIMS_ONLY": (
            "원산국×월 기후평년값",
            "외부자료 필요",
            "원산국은 있으나 생산지역·생산일 없음. 단기 일별 기상 연결 부적합",
        ),
    }
    for source, (method, status, blocker) in policy.items():
        row = by_source.loc[source]
        records.append(
            {
                "source_system": source,
                "n_rows": int(row["n_rows"]),
                "strict_domestic_ready": int(row["strict_collection_weather_ready"]),
                "occurrence_label_known": int(row["occurrence_label_known"]),
                "screening_label_known": int(row["screening_label_known"]),
                "noncompliance_label_known": int(row["noncompliance_label_known"]),
                "recommended_method": method,
                "status": status,
                "blocker_or_caveat": blocker,
            }
        )
    return pd.DataFrame(records)


def configure_plotting() -> None:
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
            "font.size": 11,
        }
    )


def render_figure(profile: pd.DataFrame, figure_dir: Path) -> Path:
    configure_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)
    display = profile.sort_values("n_rows")
    y = np.arange(len(display))
    fig, axis = plt.subplots(figsize=(11.5, 6.2))
    axis.barh(y - 0.18, display["n_rows"], height=0.34, color="#B8C2CC", label="전체 행")
    axis.barh(y + 0.18, display["strict_collection_weather_ready"], height=0.34, color="#3569A8", label="수거지역·수거일 기상 연결 가능")
    axis.set_yticks(y, display["source_system"])
    axis.set_xlabel("건수")
    axis.set_title("출처별 국내 단기 기상 연결 가능 범위")
    axis.grid(axis="x", alpha=0.55)
    axis.legend(loc="lower right")
    for index, value in enumerate(display["strict_collection_weather_ready"]):
        axis.text(value, index + 0.18, f" {value:,}", va="center", fontsize=9)
    fig.tight_layout()
    path = figure_dir / "weather_linkage_scope_by_source.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    source_profile: pd.DataFrame,
    dates: pd.DataFrame,
    target_profile: pd.DataFrame,
    overlap: dict,
    policy: pd.DataFrame,
    figure: Path,
) -> Path:
    source_view = source_profile[
        [
            "source_system",
            "n_rows",
            "collection_province_known",
            "province_normalized",
            "strict_collection_weather_ready",
            "origin_country_known",
            "occurrence_label_known",
            "screening_label_known",
            "noncompliance_label_known",
        ]
    ]
    target_view = target_profile[
        [
            "target_label",
            "source_system",
            "n_rows",
            "n_positive",
            "domestic_province_known",
            "cultivation_basis",
            "collection_fallback_basis",
            "weather_common_rows",
            "weather_common_positive",
        ]
    ]
    text = f"""# 07-7. 출처별 기상정보 연결 범위 및 목표라벨 독립성 보강

## 한눈에 보기

- 점검 단위: 통합원장 검사결과 1건
- 점검 대상: MFDS·수입식품 단독·통합망 단독·SafeQ
- 국내 단기 기상 연결 기준: 표준 시도 + 수거일
- 기상 window: 14·30·60·90일
- DB 변경: 없음
- 2026년 테스트 사용: 없음

## 핵심 결론

- 현재 학습 가능한 국내 단기 기상 집단은 SafeQ뿐임
- 통합망 단독 63,175건은 수거지역이 있지만 목표라벨·정량 결과가 0건 확정임
- MFDS와 수입식품 단독은 국내 생산지역이 없어 국내 단기 기상 연결 대상이 아님
- 수입 자료는 원산국×월 기후평년값을 별도 외부변수로 검토하는 것이 안전함
- 현재 SafeQ 기상 집단에서는 잔류 존재와 기준 부적합 라벨이 완전히 동일함

## 출처별 연결 가능성

{source_view.to_markdown(index=False)}

## 날짜 기준

{dates.to_markdown(index=False)}

## 목표별 실제 기상 공통집단

{target_view.to_markdown(index=False)}

## 라벨 독립성 점검

- 공통 record_id: {overlap['paired_rows']:,}건
- 공통집단 양성: {overlap['positive_rows']:,}건
- 잔류 존재·기준 부적합 라벨 불일치: {overlap['target_disagreements']:,}건
- 출처 구성: {overlap['source_counts']}
- 판정: 두 목표는 현재 기상 집단에서 독립적인 목표가 아님

## 출처별 적용 정책

{policy.to_markdown(index=False)}

## 데이터 품질 이슈

- High: 통합망 63,175건은 기상 연결 키가 있으나 결과·기준·라벨이 없어 지도학습 불가
- High: SafeQ 한정 기상집단에서 occurrence와 noncompliance가 동일 라벨로 축약됨
- Medium: SafeQ의 국내지역은 재배지 우선·수거지 보조이므로 공간 정확도가 혼재함
- Medium: event_date는 수거일 또는 의뢰일이며 생산일·살포일이 아님
- Low: 시도 평균 기상은 세부 생산지의 국지 기상을 반영하지 못함

## 다음 작업

1. 통합망 63,175건의 성분 결과·판정값 원천 재점검
2. 연결 가능하면 occurrence·noncompliance 라벨 재생성
3. 통합망 국내 기상 14·30·60·90일 연결
4. 목표별 LSTM window 재비교
5. MFDS·수입식품은 원산국×월 기후평년값 파일럿을 별도 설계

## 시각자료

- `{figure.name}`
"""
    path = DOCS / "07-7_출처별_기상연결범위_목표라벨_독립성.md"
    path.write_text(text, encoding="utf-8")
    return path


def build_notebook(
    source_profile: pd.DataFrame,
    target_profile: pd.DataFrame,
    policy: pd.DataFrame,
    overlap: dict,
) -> Path:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3 (toxin)",
        "language": "python",
        "name": "python3",
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# 출처별 기상 연결 범위 및 목표라벨 독립성 점검\n\n"
            "## tl;dr\n\n"
            "- 현재 학습 가능한 국내 단기 기상 집단은 SafeQ뿐\n"
            "- 통합망 63,175건은 지역·수거일이 있으나 목표라벨이 없음\n"
            "- MFDS·수입식품은 원산국 기후평년값을 별도 검토\n"
            "- SafeQ 기상집단의 occurrence·noncompliance 라벨은 완전히 동일"
        ),
        nbf.v4.new_markdown_cell(
            "## Context & Methods\n\n"
            "검사결과 1건을 단위로 출처별 날짜·지역·원산지·목표라벨 가용성을 점검합니다. "
            "국내 단기 기상은 표준 시도와 수거일이 모두 있는 경우만 연결 가능으로 계산합니다."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "ROOT = Path.cwd()\n"
            "OUT = ROOT / 'output' / 'weather_scope_audit_v2'\n"
            "source_profile = pd.read_csv(OUT / 'source_linkage_profile.csv', encoding='utf-8-sig')\n"
            "target_profile = pd.read_csv(OUT / 'target_weather_profile.csv', encoding='utf-8-sig')\n"
            "policy = pd.read_csv(OUT / 'linkage_policy.csv', encoding='utf-8-sig')\n"
            "overlap = json.loads((OUT / 'label_overlap_audit.json').read_text(encoding='utf-8'))\n"
            "source_profile.shape, target_profile.shape"
        ),
        nbf.v4.new_markdown_cell("## Data\n\n출처별 전체 규모와 국내 단기 기상 연결 가능 건수입니다."),
        nbf.v4.new_code_cell(
            "source_profile[['source_system','n_rows','strict_collection_weather_ready','origin_country_known','occurrence_label_known','screening_label_known','noncompliance_label_known']]"
        ),
        nbf.v4.new_markdown_cell("## Results\n\n목표별 현재 실제 기상 공통집단을 확인합니다."),
        nbf.v4.new_code_cell(
            "target_profile[['target_label','source_system','n_rows','n_positive','weather_common_rows','weather_common_positive']]"
        ),
        nbf.v4.new_code_cell(
            "display = source_profile.sort_values('n_rows')\n"
            "ax = display.plot.barh(x='source_system', y=['n_rows','strict_collection_weather_ready'], figsize=(10,5), color=['#B8C2CC','#3569A8'])\n"
            "ax.set_title('Source coverage for domestic short-window weather linkage')\n"
            "ax.set_xlabel('Rows'); ax.set_ylabel(''); ax.grid(axis='x', alpha=.3); plt.tight_layout();"
        ),
        nbf.v4.new_markdown_cell(
            "## Data-quality audit\n\n"
            f"- occurrence·noncompliance 공통 기상행: {overlap['paired_rows']:,}건\n"
            f"- 라벨 불일치: {overlap['target_disagreements']:,}건\n"
            f"- 출처: {overlap['source_counts']}"
        ),
        nbf.v4.new_markdown_cell("## Takeaways\n\n출처별 권장 연결 방식과 차단 사유입니다."),
        nbf.v4.new_code_cell("policy"),
    ]
    path = ROOT / "Weather_Linkage_Scope_Audit.ipynb"
    nbf.write(notebook, path)
    return path


def validate(
    master: pd.DataFrame,
    source_profile: pd.DataFrame,
    target_profile: pd.DataFrame,
    overlap: dict,
) -> dict:
    checks = {
        "master_record_id_unique": bool(master["record_id"].is_unique),
        "source_rows_reconcile": bool(source_profile["n_rows"].sum() == len(master)),
        "four_sources_present": bool(source_profile["source_system"].nunique() == 4),
        "three_targets_profiled": bool(target_profile["target"].nunique() == 3),
        "weather_common_only_safeq": bool(
            set(target_profile.loc[target_profile["weather_common_rows"].gt(0), "source_system"]) == {"SAFEQ"}
        ),
        "integrated_labels_all_missing": bool(
            source_profile.loc[
                source_profile["source_system"].eq("INTEGRATED_DIST_ONLY"),
                ["occurrence_label_known", "screening_label_known", "noncompliance_label_known"],
            ].sum(axis=1).eq(0).all()
        ),
        "occurrence_noncompliance_labels_identical_in_weather_subset": bool(
            overlap["target_disagreements"] == 0
        ),
        "no_database_write": True,
        "test_data_not_scored": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"검증 실패: {[key for key, value in checks.items() if not value]}")
    return checks


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    master = load_master()
    source_profile = master_source_profile(master)
    dates = date_basis_profile(master)
    target_profile = target_weather_profile()
    overlap = label_overlap_audit()
    policy = linkage_policy(source_profile)
    figure = render_figure(source_profile, DOCS / "figure")

    source_profile.to_csv(OUTPUT / "source_linkage_profile.csv", index=False, encoding="utf-8-sig")
    dates.to_csv(OUTPUT / "date_basis_profile.csv", index=False, encoding="utf-8-sig")
    target_profile.to_csv(OUTPUT / "target_weather_profile.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(OUTPUT / "linkage_policy.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "label_overlap_audit.json").write_text(
        json.dumps(overlap, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = write_report(source_profile, dates, target_profile, overlap, policy, figure)
    notebook = build_notebook(source_profile, target_profile, policy, overlap)
    checks = validate(master, source_profile, target_profile, overlap)
    manifest = {
        "version": "weather_scope_audit_v2",
        "grain": "검사결과 1건",
        "input_rows": int(len(master)),
        "database_write": False,
        "test_data_scored": False,
        "report": str(report),
        "notebook": str(notebook),
        "figure": str(figure),
        "checks": checks,
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

