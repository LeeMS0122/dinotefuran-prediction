from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .common import save_figure, save_table
from .targets import target_specs, validate_binary


CORE_COLUMNS = [
    "record_id",
    "source_system",
    "event_date",
    "product_name_std",
    "origin_country_name",
    "judge_result",
    "result_value_numeric_enriched",
    "standard_numeric_enriched",
    "result_to_mrl_ratio",
    "label_occurrence_v1",
    "label_screening_10pct_v1",
    "label_noncompliance_v1",
]


def _check(name: str, value: int, expected: str, note: str = "") -> dict:
    return {
        "check": name,
        "value": int(value),
        "expected": expected,
        "status": "PASS" if value == 0 else "REVIEW",
        "note": note,
    }


def _expected_screening_label(ratio: pd.Series) -> pd.Series:
    """Return the confirmed screening label: strictly greater than 10% of MRL."""
    return pd.to_numeric(ratio, errors="coerce").gt(0.1).astype(float)


def run(df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    total = len(df)
    source_counts = (
        df["source_system"]
        .fillna("결측")
        .value_counts(dropna=False)
        .rename_axis("source_system")
        .reset_index(name="rows")
    )
    source_counts["share_pct"] = source_counts["rows"] / total * 100
    save_table(source_counts, "step0_source_counts.csv")

    judge = (
        df["judge_result"]
        .fillna("결측")
        .value_counts(dropna=False)
        .rename_axis("judge_result")
        .reset_index(name="rows")
    )
    judge["share_pct"] = judge["rows"] / total * 100
    save_table(judge, "step0_judge_distribution.csv")

    completeness = pd.DataFrame(
        {
            "column": CORE_COLUMNS,
            "non_missing_rows": [int(df[column].notna().sum()) for column in CORE_COLUMNS],
        }
    )
    completeness["missing_rows"] = total - completeness["non_missing_rows"]
    completeness["availability_pct"] = completeness["non_missing_rows"] / total * 100
    save_table(completeness, "step0_core_completeness.csv")

    dates = df["event_date_dt"]
    date_coverage = pd.DataFrame(
        [
            {
                "total_rows": total,
                "valid_date_rows": int(dates.notna().sum()),
                "missing_or_invalid_date_rows": int(dates.isna().sum()),
                "min_date": dates.min(),
                "max_date": dates.max(),
                "min_year": df["event_year"].min(),
                "max_year": df["event_year"].max(),
            }
        ]
    )
    save_table(date_coverage, "step0_date_coverage.csv")

    for spec in target_specs(config):
        validate_binary(df[spec.column], spec.column)

    expected_occurrence = df["judge_result"].map({"검출": 1.0, "불검출": 0.0})
    comparable_occurrence = expected_occurrence.notna() & df["label_occurrence_v1"].notna()
    occurrence_mismatch = (
        pd.to_numeric(df.loc[comparable_occurrence, "label_occurrence_v1"], errors="coerce")
        != expected_occurrence.loc[comparable_occurrence]
    ).sum()

    result = pd.to_numeric(df["result_value_numeric_enriched"], errors="coerce")
    standard = pd.to_numeric(df["standard_numeric_enriched"], errors="coerce")
    stored_ratio = pd.to_numeric(df["result_to_mrl_ratio"], errors="coerce")
    ratio_valid = result.notna() & standard.gt(0) & stored_ratio.notna()
    calculated_ratio = result.loc[ratio_valid] / standard.loc[ratio_valid]
    ratio_mismatch = (~np.isclose(
        stored_ratio.loc[ratio_valid],
        calculated_ratio,
        rtol=1e-8,
        atol=1e-12,
        equal_nan=True,
    )).sum()

    screening = pd.to_numeric(df["label_screening_10pct_v1"], errors="coerce")
    comparable_screening = stored_ratio.notna() & screening.notna()
    expected_screening = _expected_screening_label(
        stored_ratio.loc[comparable_screening]
    )
    screening_mismatch = (
        screening.loc[comparable_screening] != expected_screening
    ).sum()

    noncompliance = pd.to_numeric(df["label_noncompliance_v1"], errors="coerce")
    ratio_exceeded = stored_ratio.gt(1)
    noncompliance_false_negative = (ratio_exceeded & noncompliance.eq(0)).sum()

    detail_columns = [
        "record_id",
        "source_system",
        "source_dataset",
        "event_date",
        "product_name_std",
        "judge_result",
        "result_value_numeric_enriched",
        "standard_numeric_enriched",
        "standard_source",
        "result_to_mrl_ratio",
        "label_screening_10pct_v1",
        "label_noncompliance_v1",
    ]
    expected_screening_all = _expected_screening_label(stored_ratio)
    screening_mask = comparable_screening & screening.ne(expected_screening_all)
    screening_detail = df.loc[screening_mask, detail_columns].copy()
    screening_detail.insert(0, "mismatch_type", "screening_boundary")
    screening_detail["expected_label_from_numeric_rule"] = (
        expected_screening_all.loc[screening_mask].astype("int8")
    )

    noncompliance_detail = df.loc[
        ratio_exceeded & noncompliance.eq(0),
        detail_columns,
    ].copy()
    noncompliance_detail.insert(0, "mismatch_type", "mrl_exceeded_but_label_zero")
    noncompliance_detail["expected_label_from_numeric_rule"] = 1
    mismatch_details = pd.concat(
        [screening_detail, noncompliance_detail],
        ignore_index=True,
    )
    save_table(mismatch_details, "step0_label_rule_mismatches.csv")

    checks = pd.DataFrame(
        [
            _check("record_id 결측", df["record_id"].isna().sum(), "0"),
            _check("record_id 중복", df["record_id"].duplicated().sum(), "0"),
            _check("음수 결과값", (result < 0).sum(), "0"),
            _check("0 이하 기준값", (standard <= 0).sum(), "0", "비율 계산 불가"),
            _check(
                "판정-잔류발생 라벨 불일치",
                occurrence_mismatch,
                "0",
                "검출=1, 불검출=0인 비교 가능 행",
            ),
            _check(
                "저장 비율-재계산 비율 불일치",
                ratio_mismatch,
                "0",
                "결과값/기준값 비교",
            ),
            _check(
                "10% 선별 라벨 불일치",
                screening_mismatch,
                "0",
                "확정 규칙: result_to_mrl_ratio > 0.1 (정확히 0.1은 음성)",
            ),
            _check(
                "MRL 초과인데 부적합=0",
                noncompliance_false_negative,
                "0",
                "SafeQ 원천 적합 판정과 현재 연결 MRL의 기준시점 확인 필요",
            ),
        ]
    )
    save_table(checks, "step0_quality_checks.csv")

    availability = (
        df.groupby("source_system", dropna=False)[CORE_COLUMNS[2:]]
        .agg(lambda series: series.notna().mean() * 100)
        .round(1)
    )
    availability = availability.rename(
        columns={
            "event_date": "검사일",
            "product_name_std": "표준 품목",
            "origin_country_name": "원산국",
            "judge_result": "최종 판정",
            "result_value_numeric_enriched": "수치 결과값",
            "standard_numeric_enriched": "수치 MRL",
            "result_to_mrl_ratio": "결과/MRL",
            "label_occurrence_v1": "잔류 발생",
            "label_screening_10pct_v1": "MRL 10% 초과",
            "label_noncompliance_v1": "기준 부적합",
        }
    )
    fig, ax = plt.subplots(figsize=(13, max(3.5, 0.7 * len(availability))))
    sns.heatmap(
        availability,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "가용률 (%)"},
        ax=ax,
    )
    ax.set_title("출처별 핵심 변수 가용률")
    ax.set_xlabel("변수")
    ax.set_ylabel("출처")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, "step0_source_completeness_heatmap.png")

    return {
        "source_counts": source_counts,
        "judge_distribution": judge,
        "core_completeness": completeness,
        "date_coverage": date_coverage,
        "quality_checks": checks,
        "label_rule_mismatches": mismatch_details,
    }
