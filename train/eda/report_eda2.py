from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import EDA2_DOCS_DIR
from .report import _markdown_table


SEASON_ORDER = ["봄", "여름", "가을", "겨울"]


def _test_view(tests: pd.DataFrame) -> pd.DataFrame:
    view = tests[["target_name", "labeled_rows", "p_value", "cramers_v"]].copy()
    view["p_value"] = view["p_value"].map(
        lambda value: (
            "<1e-300"
            if pd.notna(value) and value == 0
            else f"{value:.3e}"
            if pd.notna(value)
            else "-"
        )
    )
    view["cramers_v"] = view["cramers_v"].round(4)
    return view


def build_eda2_report(
    df: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    source_path: Path,
    config: dict,
) -> Path:
    season = outputs["season_target_summary"].copy()
    season_tests = outputs["season_association_tests"].copy()
    date_basis = outputs["date_basis_by_source"].copy()
    coverage = outputs["food_group_coverage"].copy()
    group_summary = outputs["food_group_target_summary"].copy()
    group_tests = outputs["food_group_association_tests"].copy()

    occurrence_season = (
        season[season["target_key"].eq("occurrence")]
        .set_index("season")
        .reindex(SEASON_ORDER)
        .reset_index()
    )
    season_view = occurrence_season[
        [
            "season",
            "total_rows",
            "labeled_rows",
            "missing_rows",
            "positive_rows",
            "positive_rate_pct",
            "ci95_low_pct",
            "ci95_high_pct",
        ]
    ]

    target_season_view = season[
        [
            "target_name",
            "season",
            "labeled_rows",
            "positive_rows",
            "positive_rate_pct",
        ]
    ].copy()
    target_season_view["season"] = pd.Categorical(
        target_season_view["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )
    target_season_view = target_season_view.sort_values(["target_name", "season"])

    occurrence_groups = group_summary[
        group_summary["target_key"].eq("occurrence")
    ].copy()
    count_top = occurrence_groups.sort_values(
        ["positive_rows", "labeled_rows"],
        ascending=False,
    ).head(10)
    stable_top = occurrence_groups[
        occurrence_groups["labeled_rows"] >= 1000
    ].sort_values("positive_rate_pct", ascending=False).head(10)

    overall_coverage = coverage[coverage["source_system"].eq("전체")].iloc[0]
    mfds_coverage = coverage[coverage["source_system"].eq("MFDS")].iloc[0]
    season_max = occurrence_season.loc[
        occurrence_season["positive_rate_pct"].idxmax()
    ]
    screening = season[season["target_key"].eq("screening")]
    screening_max = screening.loc[screening["positive_rate_pct"].idxmax()]
    noncompliance = season[season["target_key"].eq("noncompliance")]
    noncompliance_max = noncompliance.loc[
        noncompliance["positive_rate_pct"].idxmax()
    ]

    text = f"""# 07. 디노테푸란 EDA 2차 계절·식품군 분석

> 아플라톡신 통합 발표자료 14~15쪽의 분석 구조를 디노테푸란 데이터에 적용

## 1. 분석 범위

- 원천: {source_path}
- 전체 행: {len(df):,}건
- 분석 기간: {df['event_date_dt'].min().date()} ~ {df['event_date_dt'].max().date()}
- 2026년: 부분연도
- 계절: 봄 3~5월, 여름 6~8월, 가을 9~11월, 겨울 12~2월
- 주 계절 지표: label_occurrence_v1 확정 행 중 검출 비율
- 품목군: product_group_raw 원천 분류
- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 2. 계절별 잔류 발생 특성

{_markdown_table(
    season_view,
    ['season', 'total_rows', 'labeled_rows', 'missing_rows', 'positive_rows', 'positive_rate_pct', 'ci95_low_pct', 'ci95_high_pct'],
    ['계절', '전체', '판정 확정', '미확정', '검출', '검출률(%)', '95% 하한', '95% 상한'],
)}

![계절별 검사 건수 및 잔류 발생률](fig/step4_occurrence_season_ppt_style.png)

- 잔류 발생률 최고 계절: {season_max['season']} {season_max['positive_rate_pct']:.2f}%
- 검출 건수 최고 계절: {occurrence_season.loc[occurrence_season['positive_rows'].idxmax(), 'season']} {int(occurrence_season['positive_rows'].max()):,}건
- 계절별 검사 규모가 다르므로 검출 건수와 검출률을 함께 해석
- 미확정 행은 불검출로 대체하지 않고 막대에서 별도 표시

## 3. 목표 라벨별 계절 차이

{_markdown_table(
    target_season_view,
    ['target_name', 'season', 'labeled_rows', 'positive_rows', 'positive_rate_pct'],
    ['목표', '계절', '확정', '양성', '양성률(%)'],
)}

![목표 라벨별 계절 양성률](fig/step4_three_targets_by_season.png)

- 잔류 발생률 최고: {season_max['season']} {season_max['positive_rate_pct']:.2f}%
- MRL 10% 이상 관심농도 최고: {screening_max['season']} {screening_max['positive_rate_pct']:.2f}%
- 기준 부적합률 최고: {noncompliance_max['season']} {noncompliance_max['positive_rate_pct']:.2f}%
- 목표 라벨에 따라 계절 순위가 달라 하나의 계절 결론으로 합치지 않음
- screening 0.1 경계 305건과 SafeQ 부적합 라벨 163건은 1차 EDA 검토사항 유지

계절과 라벨의 독립성 검정:

{_markdown_table(
    _test_view(season_tests),
    ['target_name', 'labeled_rows', 'p_value', 'cramers_v'],
    ['목표', '확정 행', '카이제곱 p값', 'Cramer V'],
)}

- 표본이 커서 p값만으로 중요성을 판단하지 않음
- Cramer V를 함께 보고 계절 효과의 크기를 평가
- 계절 Cramer V 범위는 {season_tests['cramers_v'].min():.3f}~{season_tests['cramers_v'].max():.3f}로 통계적으로 유의하지만 효과 크기는 작음

## 4. 날짜 기준 민감도

{_markdown_table(
    date_basis.head(10),
    ['source_system', 'date_basis', 'rows', 'share_within_source_pct'],
    ['출처', '날짜 기준', '건수', '출처 내 비중(%)'],
)}

![출처·날짜 기준별 계절 발생률](fig/step4_occurrence_season_sensitivity.png)

- MFDS는 의뢰일과 수거일이 혼재
- SafeQ와 통합망은 대부분 수거일 기준
- 계절 효과는 출처별 수집 목적과 날짜 기준 차이의 영향을 받을 수 있음
- 모델링 단계에서는 date_basis를 변수 또는 층화 기준으로 보존

## 5. 연도·월별 발생 특성

![연도·월별 잔류 발생률](fig/step4_occurrence_year_month_heatmap.png)

- 월별 판정 확정 30건 이상인 셀만 표시
- 최근 연도는 부분연도이므로 동일 기간 비교가 필요
- 연도별 검사 품목 구성 변화가 월별 패턴에 섞일 수 있음

## 6. 식품군 분류 가용성

{_markdown_table(
    coverage,
    ['source_system', 'total_rows', 'group_present_rows', 'group_missing_rows', 'group_availability_pct', 'group_unique_count', 'product_availability_pct', 'product_unique_count'],
    ['출처', '전체', '품목군 있음', '품목군 결측', '품목군 가용률(%)', '품목군 수', '품목명 가용률(%)', '품목명 수'],
)}

- 전체 품목군 결측: {int(overall_coverage['group_missing_rows']):,}건 ({100 - overall_coverage['group_availability_pct']:.2f}%)
- MFDS 품목군 결측: {int(mfds_coverage['group_missing_rows']):,}건 ({100 - mfds_coverage['group_availability_pct']:.2f}%)
- product_group_raw는 출처별 원천 분류이며 공식 식품군 1차 분류로 확정하지 않음
- product_name_std는 세부 품목 탐색에 사용하되 공식 2차 분류로 표현하지 않음

## 7. 원천 품목군별 검사·검출 규모

{_markdown_table(
    count_top,
    ['product_group_raw', 'labeled_rows', 'positive_rows', 'positive_rate_pct'],
    ['원천 품목군', '판정 확정', '검출', '검출률(%)'],
)}

![원천 품목군별 검사 및 검출 건수](fig/step5_occurrence_group_counts.png)

- 검출 건수는 검사 규모의 영향을 크게 받음
- 관리 우선순위 검토 시 검출 건수와 검출률을 함께 사용

## 8. 원천 품목군별 잔류 발생률

판정 확정 1,000건 이상 집단:

{_markdown_table(
    stable_top,
    ['product_group_raw', 'labeled_rows', 'positive_rows', 'positive_rate_pct', 'ci95_low_pct', 'ci95_high_pct'],
    ['원천 품목군', '판정 확정', '검출', '검출률(%)', '95% 하한', '95% 상한'],
)}

![원천 품목군별 잔류 발생률](fig/step5_occurrence_group_rates_stable.png)

![목표 라벨별 품목군 양성률](fig/step5_three_targets_group_heatmap.png)

품목군과 라벨의 독립성 검정:

{_markdown_table(
    _test_view(group_tests),
    ['target_name', 'labeled_rows', 'p_value', 'cramers_v'],
    ['목표', '확정 행', '카이제곱 p값', 'Cramer V'],
)}

- 소표본 고발생률 집단은 순위에서 과도하게 강조하지 않음
- 표본 1,000건 이상 결과를 발표용 주요 비교로 사용
- 품목군 Cramer V 범위는 {group_tests['cramers_v'].min():.3f}~{group_tests['cramers_v'].max():.3f}로 계절보다 연관성이 크게 나타남
- 품목군 분류 결측과 출처 구성 차이를 보정하기 전 인과 해석 금지

## 9. 주요 품목군의 세부 품목

![주요 품목군의 세부 품목](fig/step5_occurrence_detail_products.png)

- 검출 건수 상위 4개 원천 품목군 내부의 세부 품목을 비교
- 세부 품목은 최소 판정 확정 5건 조건
- 품목명 표기 통합을 추가 수행하면 2차 분류 분석의 신뢰도가 높아짐

## 10. PPT 반영 권고

- 아플라톡신 EDA 14쪽 대응: step4_occurrence_season_ppt_style.png
- 아플라톡신 EDA 15쪽 대응: step5_occurrence_group_counts.png
- 보조 근거: 계절별 3개 라벨, 날짜 기준 민감도, 품목군 신뢰구간
- 발표 문구에서는 식품군 대신 원천 품목군으로 표기
- 공식 식품군 1·2차 매핑 완료 후 명칭과 차트를 최종 교체

## 11. 생성 파일

- table/step4_*.csv: 계절·월·날짜 기준·통계 검정
- table/step5_*.csv: 품목군 가용성·발생 특성·통계 검정
- fig/step4_*.png: 계절별 발표용 그림
- fig/step5_*.png: 품목군별 발표용 그림
"""
    EDA2_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA2_DOCS_DIR / "EDA_2차_계절_식품군.md"
    path.write_text(text, encoding="utf-8")
    return path
