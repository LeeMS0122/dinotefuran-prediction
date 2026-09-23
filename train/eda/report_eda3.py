from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import EDA3_DOCS_DIR
from .report import _markdown_table


def _p_value(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value == 0:
        return "<1e-300"
    return f"{value:.3e}"


def build_eda3_report(
    df: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    source_path: Path,
    config: dict,
) -> Path:
    volume = outputs["source_year_volume"].copy()
    source_year = outputs["source_year_target_summary"].copy()
    tests = outputs["dimension_association_tests"].copy()
    coverage = outputs["country_coverage"].copy()
    country_top = outputs["country_occurrence_top"].copy()
    stable = outputs["occurrence_group_season_stable"].copy()
    composition = outputs["season_group_composition"].copy()

    source_totals = (
        volume.groupby(["source_system", "source_name"], as_index=False)["rows"]
        .sum()
        .sort_values("rows", ascending=False)
    )
    source_totals["share_pct"] = source_totals["rows"] / source_totals["rows"].sum() * 100

    occurrence_source = source_year[
        source_year["target_key"].eq("occurrence")
    ].groupby(["source_system", "source_name"], as_index=False).agg(
        labeled_rows=("labeled_rows", "sum"),
        positive_rows=("positive_rows", "sum"),
    )
    occurrence_source["positive_rate_pct"] = (
        occurrence_source["positive_rows"] / occurrence_source["labeled_rows"] * 100
    )
    occurrence_source = occurrence_source.sort_values(
        "positive_rate_pct", ascending=False
    )

    test_view = tests[
        tests["dimension"].isin(
            ["source_system", "event_year_2016_2025", "origin_country_name"]
        )
    ][["target_name", "dimension", "labeled_rows", "p_value", "cramers_v"]].copy()
    test_view["p_value"] = test_view["p_value"].map(_p_value)
    test_view["cramers_v"] = test_view["cramers_v"].round(4)
    test_view["dimension"] = test_view["dimension"].map(
        {
            "source_system": "출처",
            "event_year_2016_2025": "연도(2016~2025)",
            "origin_country_name": "원산국 정보 보유 행",
        }
    )

    overall_country = coverage[coverage["source_system"].eq("전체")].iloc[0]
    source_country = coverage[
        ~coverage["source_system"].eq("전체")
    ][
        [
            "source_name",
            "total_rows",
            "country_present_rows",
            "country_missing_rows",
            "country_availability_pct",
            "country_unique_count",
        ]
    ]

    seasonal_top = stable.sort_values(
        ["positive_rate_pct", "labeled_rows"], ascending=False
    ).head(15)
    seasonal_shift = stable.reindex(
        stable["rate_difference_pct_point"].abs().sort_values(ascending=False).index
    ).head(15)

    comp_matrix = composition.pivot(
        index="group_for_composition",
        columns="season",
        values="share_within_season_pct",
    )
    comp_matrix["계절간_구성비차이_pct_point"] = (
        comp_matrix.max(axis=1) - comp_matrix.min(axis=1)
    )
    composition_shift = comp_matrix.sort_values(
        "계절간_구성비차이_pct_point", ascending=False
    ).reset_index().head(10)
    composition_shift = composition_shift.rename(
        columns={
            "group_for_composition": "원천 품목군",
            "봄": "봄(%)",
            "여름": "여름(%)",
            "가을": "가을(%)",
            "겨울": "겨울(%)",
            "계절간_구성비차이_pct_point": "계절 간 최대차이(%p)",
        }
    )

    early_rows = int(volume.loc[volume["event_year"].lt(2016), "rows"].sum())
    partial_rows = int(
        volume.loc[
            volume["event_year"].eq(config["eda"]["latest_year"]), "rows"
        ].sum()
    )
    core_rows = int(
        volume.loc[volume["event_year"].between(2016, 2025), "rows"].sum()
    )

    text = f"""# 08. 디노테푸란 EDA 3차 출처·연도·원산국·품목계절 분석

## 1. 분석 범위

- 원천: {source_path}
- 전체: {len(df):,}건
- 분석 기간: {df['event_date_dt'].min().date()} ~ {df['event_date_dt'].max().date()}
- 주 비교 기간: 2016~2025년 {core_rows:,}건
- 2016년 이전: {early_rows:,}건
- 2026년: 부분연도 {partial_rows:,}건
- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 2. 출처별 데이터 구성

{_markdown_table(
    source_totals,
    ['source_name', 'rows', 'share_pct'],
    ['출처', '행 수', '전체 비중(%)'],
)}

![연도별 출처 구성](fig/step6_source_composition_by_year.png)

- 연도별 출처 구성 변화가 커서 전체 연도 추세에는 출처 효과가 섞일 수 있음
- 2026년은 부분연도이므로 검사 건수의 연간 단순 비교 금지
- 2016년 이전은 표본이 적어 주 비교 구간과 분리

## 3. 출처·연도별 잔류 발생

{_markdown_table(
    occurrence_source,
    ['source_name', 'labeled_rows', 'positive_rows', 'positive_rate_pct'],
    ['출처', '판정 확정', '검출', '검출률(%)'],
)}

![출처별 연도 잔류 발생률](fig/step6_occurrence_rate_by_source_year.png)

![출처·연도별 라벨 가용률](fig/step6_occurrence_label_coverage_heatmap.png)

- 출처별 검사 목적과 판정 가용성이 다름
- 출처를 합산한 검출률만으로 연도 변화 원인을 판단하지 않음
- 모델 평가 시 출처별 성능과 연도별 성능을 별도로 확인할 필요

## 4. 출처·연도·원산국 연관성

{_markdown_table(
    test_view,
    ['target_name', 'dimension', 'labeled_rows', 'p_value', 'cramers_v'],
    ['목표', '구분', '판정 확정', '카이제곱 p값', 'Cramer V'],
)}

- 표본이 커서 p값만으로 중요성을 판단하지 않음
- Cramer V는 연관성 크기이며 인과관계를 의미하지 않음
- 원산국 결과는 원산국 정보가 존재하는 행에만 적용
- 원산국별 관심농도 라벨은 판정 확정 451건뿐이므로 Cramer V 0.53을 일반화하지 않음

## 5. 원산국 정보 가용성

{_markdown_table(
    source_country,
    ['source_name', 'total_rows', 'country_present_rows', 'country_missing_rows', 'country_availability_pct', 'country_unique_count'],
    ['출처', '전체', '원산국 있음', '결측', '가용률(%)', '원산국 수'],
)}

![출처별 원산국 가용성](fig/step6_country_coverage_by_source.png)

- 전체 원산국 가용률: {overall_country['country_availability_pct']:.2f}%
- 원산국 결측: {int(overall_country['country_missing_rows']):,}건
- 통합망 단독과 SafeQ는 원산국 정보가 없어 전체 모집단의 국가별 분석 불가
- 결측을 국내산으로 간주하지 않음
- 국내산 대 수입산 이진 비교는 별도 원산지 구분값을 확보한 뒤 수행

## 6. 원산국별 잔류 발생

판정 확정 100건 이상이며 검출 건수 상위 원산국:

{_markdown_table(
    country_top.head(15),
    ['origin_country_name', 'labeled_rows', 'positive_rows', 'positive_rate_pct', 'ci95_low_pct', 'ci95_high_pct'],
    ['원산국', '판정 확정', '검출', '검출률(%)', '95% 하한', '95% 상한'],
)}

![원산국별 검출 특성](fig/step6_country_occurrence_top.png)

![원산국·연도별 발생률](fig/step6_country_year_occurrence_heatmap.png)

- 원산국별 순위는 검사 대상 품목 구성의 영향을 받음
- 검사 건수와 검출률을 함께 확인
- 국가별 셀의 표본이 30건 미만이면 연도별 히트맵에서 제외

## 7. 원천 품목군×계절 발생 특성

표본 조건: 품목군 전체 판정 확정 1,000건 이상, 계절 셀 100건 이상

{_markdown_table(
    seasonal_top,
    ['product_group_raw', 'season', 'labeled_rows', 'positive_rows', 'positive_rate_pct', 'ci95_low_pct', 'ci95_high_pct'],
    ['원천 품목군', '계절', '판정 확정', '검출', '검출률(%)', '95% 하한', '95% 상한'],
)}

![품목군×계절 발생률](fig/step7_group_season_occurrence_heatmap.png)

![주요 품목군의 계절별 발생률](fig/step7_top_groups_seasonal_profiles.png)

- 전체 계절 평균보다 품목군별 계절 차이를 우선 확인
- 높은 발생률은 품목군의 계절별 검사 대상 구성 차이를 포함할 수 있음
- 식품군 표준화 전까지 원천 품목군으로 표기

## 8. 계절별 품목군 구성 변화

{_markdown_table(
    composition_shift,
    list(composition_shift.columns),
    list(composition_shift.columns),
)}

![계절별 품목군 구성](fig/step7_season_group_composition.png)

- 계절별 검사 품목 구성 변화가 전체 계절 발생률 차이를 만들 수 있음
- 품목군 구성 차이를 확인하지 않은 단순 계절 해석은 제한

## 9. 모델링 전 결론

- 변수 확장·고도화는 이번 EDA 결과를 확인한 다음 단계에서 수행
- 현재 단계에서는 통합원장에 이미 있는 변수만 사용
- 우선 후보: 출처, 연도·월·계절, 원산국, 원천 품목군, 표준 품목명
- 모델 투입 전 결과값·MRL·판정 관련 사후 변수의 데이터 누수 여부 점검 필요
- 시간 분할을 기본으로 하고 출처별·연도별·품목군별 성능을 별도 평가

## 10. 데이터 품질 판정

- 높음: 원산국 가용률이 낮아 전체 모집단 국가별 모델 변수로 즉시 사용하기 어려움
- 중간: 품목군 결측과 출처별 분류 차이가 품목계절 분석에 영향을 줄 수 있음
- 중간: 2026년 부분연도와 연도별 출처 구성 변화가 단순 추세 비교를 왜곡할 수 있음
- 낮음: 2016년 이전 소량 자료는 주 분석기간에서 분리하면 관리 가능

## 11. 생성 파일

- table/step6_*.csv: 출처·연도·원산국·품질 점검
- table/step7_*.csv: 품목군×계절과 계절별 품목군 구성
- fig/step6_*.png: 출처·연도·원산국 그림
- fig/step7_*.png: 품목군×계절 그림
"""
    EDA3_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA3_DOCS_DIR / "EDA_3차_출처_연도_원산국_품목계절.md"
    path.write_text(text, encoding="utf-8")
    return path
