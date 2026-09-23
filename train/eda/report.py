from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .common import DOCS_DIR


def _cell(value) -> str:
    if pd.isna(value):
        return "-"
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value).replace("|", "/")


def _markdown_table(df: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for row in df[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    df: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    source_path: Path,
    config: dict,
) -> Path:
    total = len(df)
    summary = outputs["target_summary"].copy()
    sources = outputs["source_counts"].copy()
    judge = outputs["judge_distribution"].copy()
    checks = outputs["quality_checks"].copy()
    mismatches = outputs["label_rule_mismatches"].copy()
    completeness = outputs["core_completeness"].copy()
    date_coverage = outputs["date_coverage"].iloc[0]
    numeric_coverage = outputs["numeric_coverage"].copy()
    ratio_bands = outputs["ratio_bands"].copy()

    target_view = summary[
        [
            "target_name",
            "labeled_rows",
            "missing_rows",
            "positive_rows",
            "negative_rows",
            "positive_rate_pct",
        ]
    ]
    source_view = sources[["source_system", "rows", "share_pct"]]
    judge_view = judge[["judge_result", "rows", "share_pct"]]
    check_view = checks[["check", "value", "status", "note"]]

    result_available = int(df["result_value_numeric_enriched"].notna().sum())
    standard_available = int(df["standard_numeric_enriched"].notna().sum())
    ratio_available = int(df["result_to_mrl_ratio"].notna().sum())
    unresolved = int(df["judge_result"].eq("미확정").sum())
    review_count = int(checks["status"].eq("REVIEW").sum())
    screening_mismatches = mismatches[
        mismatches["mismatch_type"].eq("screening_boundary")
    ]
    noncompliance_mismatches = mismatches[
        mismatches["mismatch_type"].eq("mrl_exceeded_but_label_zero")
    ]

    top_lines = []
    for target_key, target_name in zip(summary["target_key"], summary["target_name"]):
        table = outputs.get(f"{target_key}_product_name_std")
        if table is None or table.empty:
            continue
        row = table.iloc[0]
        top_lines.append(
            f"- {target_name}: {row['product_name_std']} "
            f"({row['positive_rate_pct']:.2f}%, n={int(row['labeled_rows']):,})"
        )

    ratio_total = ratio_bands.groupby("ratio_band", observed=False)["rows"].sum().reset_index()
    ratio_total["share_pct"] = ratio_total["rows"] / ratio_total["rows"].sum() * 100

    text = f"""# 06. 디노테푸란 EDA 1차 기초분석

> 통합원장 v2.3의 구조·품질·목표 라벨을 점검한 모델링 전 탐색 결과

## 1. 분석 범위

- 분석 단위: 검사 결과 1건
- 전체 행: {total:,}건
- 기본키: record_id
- 분석 기간: {_cell(date_coverage['min_date'])} ~ {_cell(date_coverage['max_date'])}
- 원천 파일: {source_path}
- 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- 2026년: 부분연도 자료

## 2. 출처 구성

{_markdown_table(source_view, ['source_system', 'rows', 'share_pct'], ['출처', '건수', '구성비(%)'])}

![출처별 핵심 변수 가용률](fig/step0_source_completeness_heatmap.png)

## 3. 최종 판정 분포

{_markdown_table(judge_view, ['judge_result', 'rows', 'share_pct'], ['판정', '건수', '구성비(%)'])}

- 미확정 {unresolved:,}건 유지
- 미확정·결측을 불검출로 임의 대체하지 않음
- 통합망 단독 자료는 성분 결과가 없어 목표별 분석 모집단에서 제외될 수 있음

## 4. 목표 라벨 현황

{_markdown_table(target_view, ['target_name', 'labeled_rows', 'missing_rows', 'positive_rows', 'negative_rows', 'positive_rate_pct'], ['목표', '확정', '미확정', '양성', '음성', '양성률(%)'])}

![목표 라벨별 모집단과 양성률](fig/step1_target_overview.png)

- 잔류 발생: 검출 여부 기반
- MRL 10% 이상 관심농도 정의안: 결과값/MRL 비율 0.1 이상
- 현재 선별 라벨은 경계값 0.1인 305건이 0으로 생성되어 정의안과 불일치
- 기준 부적합 정의안: 원천 부적합 판정 또는 결과값/MRL 비율 1 초과
- 현재 부적합 라벨은 SafeQ 163건이 수치 규칙과 불일치
- 세 라벨은 목적이 달라 독립적으로 생성·보존

## 5. 데이터 품질 점검

{_markdown_table(check_view, ['check', 'value', 'status', 'note'], ['점검', '건수', '상태', '설명'])}

- REVIEW 항목: {review_count:,}개
- REVIEW는 자동 수정하지 않고 원천 규칙·업무 정의 확인 대상으로 유지
- 기본키 결측·중복 여부와 라벨 허용값을 실행 시 검증
- 10% 선별 불일치 {len(screening_mismatches):,}건: 모두 비율이 정확히 0.1인데 현재 라벨은 0
- 선별 경계 규칙을 0.1 초과로 둘지 0.1 이상으로 둘지 모델링 전에 확정 필요
- MRL 초과인데 부적합=0인 {len(noncompliance_mismatches):,}건: 모두 SafeQ 원천 판정은 불검출
- 해당 {len(noncompliance_mismatches):,}건의 비율 범위: {noncompliance_mismatches['result_to_mrl_ratio'].min():.3f} ~ {noncompliance_mismatches['result_to_mrl_ratio'].max():.3f}
- SafeQ 검사 당시 기준과 현재 연결한 품목 MRL의 기준시점·적용대상 확인 필요
- 세부 행: table/step0_label_rule_mismatches.csv

## 6. 수치 결과값·MRL 가용성

- 수치 결과값: {result_available:,}건 ({result_available / total * 100:.2f}%)
- 수치 기준값(MRL): {standard_available:,}건 ({standard_available / total * 100:.2f}%)
- 결과값/MRL 비율: {ratio_available:,}건 ({ratio_available / total * 100:.2f}%)

{_markdown_table(ratio_total, ['ratio_band', 'rows', 'share_pct'], ['MRL 대비 구간', '건수', '구성비(%)'])}

![출처별 수치형 변수 가용률](fig/step3_numeric_coverage_by_source.png)

![MRL 대비 농도 구간](fig/step3_ratio_band_by_source.png)

## 7. 출처·연도 변화

![출처별 목표 양성률](fig/step1_target_rate_by_source.png)

![연도별 목표 양성률](fig/step1_target_rate_by_year.png)

- 출처별 수집 목적·판정 규칙·결측 구조가 다름
- 단순 양성률 차이를 위해도 차이로 바로 해석하지 않음
- 2026년 수치는 부분연도이므로 전년도와 직접 비교 시 주의

## 8. 품목별 탐색

분석 가능 30건 이상 표준 품목 중 단순 양성률 상위:

{chr(10).join(top_lines) if top_lines else '- 조건을 충족한 품목 없음'}

- 상위 집단은 가설 생성용
- 표본 수와 Wilson 95% 신뢰구간을 함께 확인
- 다중 비교·수입국·연도·출처 구성 차이를 보정하기 전 인과 해석 금지

## 9. 모델링 전 권고

- 주 모델 후보: label_screening_10pct_v1
- 비교 목표: label_occurrence_v1, label_noncompliance_v1
- 목표별 확정 라벨 행만 각 모델의 모집단으로 사용
- 통합망 단독 63,175건은 목표값 연결 전 지도학습 대상에서 제외
- 수입식품 단독 미확정 자료는 추가 연결 전 음성 처리 금지
- 무작위 분할보다 연도 기반 시간 분할 우선
- 다음 EDA에서 품목·원산국·계절·출처 결합 효과와 누출 후보 점검

## 10. 생성 파일

- table/step0_*.csv: 구조·품질·가용성
- table/step1_*.csv: 라벨·출처·연도
- table/step2_*.csv: 품목·품목군·원산국·수거지역
- table/step3_*.csv: 결과값·MRL·비율
- fig/*.png: 보고서·PPT 재사용 그림
"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    path = DOCS_DIR / "EDA_1차_기초분석.md"
    path.write_text(text, encoding="utf-8")
    return path
