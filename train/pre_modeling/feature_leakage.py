from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eda.common import data_path_from_env, load_config


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs" / "변수_누수_점검"
TABLE_DIR = DOCS_DIR / "table"
FIG_DIR = DOCS_DIR / "fig"
REPORT_PATH = DOCS_DIR / "01_변수후보_데이터누수점검.md"
SUMMARY_PATH = DOCS_DIR / "summary.json"

TARGET_COLUMNS = {
    "label_occurrence_v1",
    "label_screening_10pct_v1",
    "label_noncompliance_v1",
}

HARD_LEAKAGE = {
    "judge_result", "judge_confidence", "result_raw", "result_value_numeric",
    "result_unit", "compliance_raw", "exceedance_flag", "judge_note",
    "ready_data_judge_result", "ready_data_judge_match",
    "result_value_numeric_enriched", "judge_result_source", "judge_result_db",
    "result_to_mrl_ratio", "label_occurrence_basis", "label_screening_basis",
    "label_noncompliance_basis", "label_occurrence_confidence",
    "label_screening_confidence", "label_noncompliance_confidence",
    "label_basis", "label_confidence", "label_rule_version",
    "ready_alt_judge_results", "ready_alt_judge_agreement",
    "integrated_detected_component_raw", "import_db_source_result_raw",
    "import_db_source_test_result", "import_db_judge_result_raw",
    "import_db_result_numeric", "import_db_result_evidence_status",
}

REBUILD_REQUIRED = {
    "standard_raw", "standard_numeric", "standard_raw_enriched",
    "standard_numeric_enriched", "standard_source", "standard_recovered_flag",
    "ready_standard_recovered_flag", "standard_corrected_flag",
    "standard_update_type", "import_db_standard_numeric",
}

POST_OUTCOME = {"analysis_date", "completion_date"}

GROUP_ONLY = {
    "lims_data_key", "record_id", "duplicate_group_id", "source_file",
    "source_row_number", "source_receipt_id", "source_sample_id",
    "source_request_id", "source_link_key", "import_db_lims_seq",
    "import_db_exact_key",
}

AUDIT_ONLY = {
    "cross_source_match_status", "matched_source", "matched_row_count",
    "is_canonical", "multirow_flag", "unmatched_flag", "partial_year_flag",
    "model_eligible_flag", "product_name_match", "result_value_match",
    "completion_date_match", "country_code_match", "critical_missing_count",
    "integration_note", "has_mfds", "has_import_food_system", "has_safeq",
    "has_integrated_food_network", "has_ready_data", "ready_data_match_status",
    "ready_alt_key_count", "ready_alt_keys", "ready_alt_source_types",
    "ready_alt_match_methods", "ready_alt_match_confidence",
    "05_6_component_audit_applied", "integrated_component_audit_status",
    "dinotefuran_test_coverage_status", "dinotefuran_component_evidence_flag",
    "food_residue_pesticide_analysis_flag", "detected_component_list_present_flag",
    "integrated_analysis_division_raw", "integrated_work_division_raw",
    "integrated_registration_status_raw", "integrated_component_audit_note",
    "05_7_import_db_audit_applied", "import_db_link_status",
    "import_db_source_table", "import_db_row_reuse_count", "import_db_audit_note",
    "model_eligible_occurrence", "model_eligible_screening",
    "model_eligible_noncompliance",
}

CORE_CANDIDATES = {
    "product_code", "product_name_std", "product_group_raw", "event_date",
    "origin_country_code", "origin_country_name", "import_country_code",
    "production_country_code", "collection_province", "collection_city",
    "cultivation_province", "cultivation_city", "work_type", "collection_stage",
    "cultivation_method_1", "cultivation_method_2", "facility_type",
    "survey_area", "survey_volume", "export_type",
}

CONDITIONAL_CANDIDATES = {
    "request_date", "receipt_date", "collection_date", "manufacture_date",
    "distribution_date", "shipment_expected_date", "analysis_type",
}

DIAGNOSTIC_ONLY = {
    "source_system", "source_dataset", "canonical_source_name",
    "source_membership", "analysis_agency", "request_agency", "importer_name",
    "manufacturer_name",
}

PREPROCESS_ONLY = {
    "product_name_raw", "product_name_en", "sample_name", "date_basis",
    "cultivation_address", "manufacturer_address",
}

NONINFORMATIVE = {"hazard_label", "test_item_name"}

POLICY_ORDER = [
    "사용 후보", "조건부 후보", "진단 전용", "전처리 전용", "그룹·감사 전용",
    "감사 메타데이터", "외부 기준으로 재구축", "사후 시점", "직접 누수",
    "목표 라벨", "비정보성", "수동 검토",
]

RATIONALES = {
    "사용 후보": "검사 결과 확인 전에 확보 가능한 설명 변수",
    "조건부 후보": "예측 시점의 실제 가용성을 확정한 뒤 사용",
    "진단 전용": "출처·기관·업체 효과 점검용이며 기본 모델에서는 제외",
    "전처리 전용": "표준화·파생변수 생성에만 사용하고 원문은 학습에서 제외",
    "그룹·감사 전용": "중복 방지·그룹 분할·추적용 식별자",
    "감사 메타데이터": "통합·연결·보강 과정에서 생성된 사후 메타데이터",
    "외부 기준으로 재구축": "현재 원장 값은 라벨 생성·사후 보강과 연결되어 별도 기준표로 재구축",
    "사후 시점": "분석 또는 판정 이후에만 생성되는 날짜",
    "직접 누수": "결과·판정·라벨 생성에 직접 사용된 사후정보",
    "목표 라벨": "예측 대상",
    "비정보성": "디노테퓨란 단일 위해요소 원장에서 상수 또는 준상수",
    "수동 검토": "현재 정책표에 없는 열로 사용 전 의미·시점 확인 필요",
}


def ensure_dirs() -> None:
    for path in (DOCS_DIR, TABLE_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def classify_column(column: str) -> str:
    if column in TARGET_COLUMNS:
        return "목표 라벨"
    if column in HARD_LEAKAGE:
        return "직접 누수"
    if column in REBUILD_REQUIRED:
        return "외부 기준으로 재구축"
    if column in POST_OUTCOME:
        return "사후 시점"
    if column in GROUP_ONLY:
        return "그룹·감사 전용"
    if column in AUDIT_ONLY:
        return "감사 메타데이터"
    if column in CORE_CANDIDATES:
        return "사용 후보"
    if column in CONDITIONAL_CANDIDATES:
        return "조건부 후보"
    if column in DIAGNOSTIC_ONLY:
        return "진단 전용"
    if column in PREPROCESS_ONLY:
        return "전처리 전용"
    if column in NONINFORMATIVE:
        return "비정보성"
    return "수동 검토"


def analysis_columns(header: list[str]) -> list[str]:
    wanted = (
        TARGET_COLUMNS | HARD_LEAKAGE | REBUILD_REQUIRED | POST_OUTCOME |
        GROUP_ONLY | CORE_CANDIDATES | CONDITIONAL_CANDIDATES |
        DIAGNOSTIC_ONLY | PREPROCESS_ONLY |
        {"date_basis", "source_system", "event_date"}
    )
    return [column for column in header if column in wanted]


def load_profiled_data(
    source_path: Path,
    chunksize: int = 50_000,
    distinct_cap: int = 20_000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    header = pd.read_csv(source_path, nrows=0, encoding="utf-8-sig").columns.tolist()
    selected = analysis_columns(header)
    non_null = pd.Series(0, index=header, dtype="int64")
    distinct_values: dict[str, set[str]] = {column: set() for column in header}
    distinct_capped = {column: False for column in header}
    parts: list[pd.DataFrame] = []
    total_rows = 0

    for chunk in pd.read_csv(
        source_path,
        encoding="utf-8-sig",
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    ):
        total_rows += len(chunk)
        non_null = non_null.add(chunk.notna().sum(), fill_value=0).astype("int64")
        for column in header:
            if distinct_capped[column]:
                continue
            remaining = distinct_cap + 1 - len(distinct_values[column])
            values = chunk[column].dropna().astype(str).unique()
            distinct_values[column].update(values[:remaining])
            if len(distinct_values[column]) > distinct_cap:
                distinct_capped[column] = True
        part = chunk[selected].copy()
        for column in part.columns:
            if pd.api.types.is_string_dtype(part[column]):
                part[column] = part[column].str.strip().replace("", pd.NA)
        parts.append(part)

    data = pd.concat(parts, ignore_index=True)
    inventory_rows = []
    for column in header:
        policy = classify_column(column)
        unique_count = len(distinct_values[column])
        capped = distinct_capped[column]
        coverage = float(non_null[column]) / total_rows * 100 if total_rows else np.nan
        if non_null[column] == 0:
            data_note = "전건 결측"
        elif unique_count <= 1:
            data_note = "상수"
        elif capped:
            data_note = f"고유값 {distinct_cap:,}개 초과"
        elif unique_count / max(int(non_null[column]), 1) >= 0.5:
            data_note = "고카디널리티"
        else:
            data_note = ""
        inventory_rows.append(
            {
                "column": column,
                "policy_group": policy,
                "rationale": RATIONALES[policy],
                "non_null_rows": int(non_null[column]),
                "coverage_pct": coverage,
                "distinct_values": f">{distinct_cap}" if capped else unique_count,
                "distinct_capped": capped,
                "data_note": data_note,
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    inventory["policy_group"] = pd.Categorical(
        inventory["policy_group"], categories=POLICY_ORDER, ordered=True
    )
    inventory = inventory.sort_values(["policy_group", "column"]).reset_index(drop=True)
    return data, inventory


def cramers_v(table: pd.DataFrame) -> float:
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0 or min(observed.shape) < 2:
        return 0.0
    expected = observed.sum(axis=1, keepdims=True) @ observed.sum(axis=0, keepdims=True) / n
    valid = expected > 0
    chi2 = np.sum(((observed - expected) ** 2)[valid] / expected[valid])
    phi2 = chi2 / n
    rows, cols = observed.shape
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
    rows_corrected = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    cols_corrected = cols - ((cols - 1) ** 2) / max(n - 1, 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else 0.0


def source_feature_coverage(data: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    candidates = inventory[
        inventory["policy_group"].isin(["사용 후보", "조건부 후보", "진단 전용"])
    ]["column"].tolist()
    rows = []
    for source, part in data.groupby("source_system", dropna=False):
        for column in candidates:
            if column not in part:
                continue
            rows.append(
                {
                    "source_system": str(source),
                    "column": column,
                    "rows": len(part),
                    "non_null_rows": int(part[column].notna().sum()),
                    "coverage_pct": float(part[column].notna().mean() * 100),
                    "policy_group": classify_column(column),
                }
            )
    return pd.DataFrame(rows)


def source_proxy_risk(data: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    candidates = inventory[
        inventory["policy_group"].isin(["사용 후보", "조건부 후보", "진단 전용"])
    ]["column"].tolist()
    rows = []
    source = data["source_system"].fillna("(결측)").astype(str)
    for column in candidates:
        if column == "source_system" or column not in data:
            continue
        missing = data[column].isna().map({True: "결측", False: "보유"})
        table = pd.crosstab(source, missing)
        value = cramers_v(table)
        rows.append(
            {
                "column": column,
                "policy_group": classify_column(column),
                "overall_coverage_pct": float(data[column].notna().mean() * 100),
                "missingness_source_cramers_v": value,
                "source_proxy_risk": "높음" if value >= 0.5 else "중간" if value >= 0.3 else "낮음",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["missingness_source_cramers_v", "overall_coverage_pct"], ascending=[False, False]
    ).reset_index(drop=True)


def weighted_purity(feature: pd.Series, target: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame({"feature": feature, "target": pd.to_numeric(target, errors="coerce")}).dropna()
    frame = frame[frame["target"].isin([0, 1])]
    if frame.empty:
        return {"rows": 0, "categories": 0, "purity_pct": np.nan, "baseline_pct": np.nan}
    frame["feature"] = frame["feature"].astype(str)
    table = pd.crosstab(frame["feature"], frame["target"])
    purity = table.max(axis=1).sum() / table.to_numpy().sum() * 100
    baseline = frame["target"].value_counts(normalize=True).max() * 100
    return {
        "rows": len(frame),
        "categories": len(table),
        "purity_pct": float(purity),
        "baseline_pct": float(baseline),
    }


def direct_leakage_checks(data: pd.DataFrame) -> pd.DataFrame:
    targets = {
        "잔류 발생": "label_occurrence_v1",
        "MRL 10% 관심농도": "label_screening_10pct_v1",
        "기준 부적합": "label_noncompliance_v1",
    }
    categorical_fields = [
        "judge_result", "result_raw", "compliance_raw", "ready_data_judge_result",
        "judge_result_source", "judge_result_db", "import_db_source_test_result",
        "import_db_judge_result_raw", "integrated_detected_component_raw",
    ]
    rows: list[dict] = []
    for target_name, target_col in targets.items():
        for field in categorical_fields:
            if field not in data:
                continue
            metrics = weighted_purity(data[field], data[target_col])
            rows.append(
                {
                    "target_name": target_name,
                    "field": field,
                    "check_type": "범주별 라벨 순도",
                    **metrics,
                    "score_pct": metrics["purity_pct"],
                    "lift_vs_majority_pp": metrics["purity_pct"] - metrics["baseline_pct"],
                    "interpretation": "결과·판정 사후정보이므로 학습 금지",
                }
            )

    rules = [
        ("잔류 발생", "label_occurrence_v1", "result_value_numeric_enriched", 0.0, "gt"),
        ("MRL 10% 관심농도", "label_screening_10pct_v1", "result_to_mrl_ratio", 0.1, "gt"),
        ("기준 부적합", "label_noncompliance_v1", "result_to_mrl_ratio", 1.0, "gt"),
        ("기준 부적합", "label_noncompliance_v1", "exceedance_flag", 0.5, "ge"),
    ]
    for target_name, target_col, field, threshold, operator in rules:
        if field not in data:
            continue
        value = pd.to_numeric(data[field], errors="coerce")
        target = pd.to_numeric(data[target_col], errors="coerce")
        mask = value.notna() & target.isin([0, 1])
        prediction = value[mask].ge(threshold) if operator == "ge" else value[mask].gt(threshold)
        agreement = float(prediction.eq(target[mask].astype(bool)).mean() * 100) if mask.any() else np.nan
        baseline = float(target[mask].value_counts(normalize=True).max() * 100) if mask.any() else np.nan
        rows.append(
            {
                "target_name": target_name,
                "field": field,
                "check_type": f"규칙 일치({operator} {threshold:g})",
                "rows": int(mask.sum()),
                "categories": np.nan,
                "purity_pct": np.nan,
                "baseline_pct": baseline,
                "score_pct": agreement,
                "lift_vs_majority_pp": agreement - baseline,
                "interpretation": "라벨 생성 규칙과 직접 연결되어 학습 금지",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lift_vs_majority_pp", "rows"], ascending=[False, False]
    ).reset_index(drop=True)


def stable_random_split(n_rows: int, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.choice(["train", "valid", "test"], size=n_rows, p=[0.7, 0.15, 0.15])
    return pd.Series(values)


def summarize_group_key(
    data: pd.DataFrame,
    key: pd.Series,
    key_name: str,
    random_split: pd.Series,
) -> dict:
    frame = pd.DataFrame({"key": key, "split": random_split})
    frame["year"] = pd.to_datetime(data["event_date"], errors="coerce").dt.year
    for target in TARGET_COLUMNS:
        frame[target] = pd.to_numeric(data[target], errors="coerce")
    frame = frame[frame["key"].notna() & frame["key"].astype(str).str.strip().ne("")]
    if frame.empty:
        return {
            "group_key": key_name, "available_rows": 0, "coverage_pct": 0.0,
            "unique_groups": 0, "duplicated_groups": 0, "rows_in_duplicated_groups": 0,
            "max_group_size": 0, "random_split_cross_groups": 0,
            "random_split_cross_rows": 0, "multi_year_groups": 0,
            "occurrence_conflict_groups": 0, "screening_conflict_groups": 0,
            "noncompliance_conflict_groups": 0,
        }
    sizes = frame.groupby("key").size()
    duplicated = sizes[sizes.gt(1)]
    split_span = frame.groupby("key")["split"].nunique()
    cross_groups = split_span[split_span.gt(1)].index
    year_span = frame.groupby("key")["year"].nunique(dropna=True)
    row = {
        "group_key": key_name,
        "available_rows": len(frame),
        "coverage_pct": len(frame) / len(data) * 100,
        "unique_groups": int(len(sizes)),
        "duplicated_groups": int(len(duplicated)),
        "rows_in_duplicated_groups": int(duplicated.sum()),
        "max_group_size": int(sizes.max()),
        "random_split_cross_groups": int(len(cross_groups)),
        "random_split_cross_rows": int(sizes.reindex(cross_groups).sum()),
        "multi_year_groups": int(year_span.gt(1).sum()),
    }
    for target, prefix in [
        ("label_occurrence_v1", "occurrence"),
        ("label_screening_10pct_v1", "screening"),
        ("label_noncompliance_v1", "noncompliance"),
    ]:
        conflicts = frame.groupby("key")[target].nunique(dropna=True).gt(1).sum()
        row[f"{prefix}_conflict_groups"] = int(conflicts)
    return row


def group_leakage_checks(data: pd.DataFrame) -> pd.DataFrame:
    random_split = stable_random_split(len(data))
    keys: list[tuple[str, pd.Series]] = []
    for column in ["duplicate_group_id", "lims_data_key", "source_link_key"]:
        if column in data:
            keys.append((column, data[column]))
    if {"source_system", "source_receipt_id", "source_sample_id"}.issubset(data.columns):
        present = data[["source_system", "source_receipt_id", "source_sample_id"]].notna().all(axis=1)
        composite = (
            data["source_system"].astype("string") + "|" +
            data["source_receipt_id"].astype("string") + "|" +
            data["source_sample_id"].astype("string")
        ).where(present)
        keys.append(("source+receipt+sample", composite))
    rows = [summarize_group_key(data, key, name, random_split) for name, key in keys]
    return pd.DataFrame(rows).sort_values(
        ["random_split_cross_rows", "rows_in_duplicated_groups"], ascending=False
    ).reset_index(drop=True)


def date_availability_checks(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    basis = (
        data.groupby(["source_system", "date_basis"], dropna=False)
        .size().rename("rows").reset_index()
    )
    basis["share_within_source_pct"] = basis["rows"] / basis.groupby("source_system")["rows"].transform("sum") * 100
    event = pd.to_datetime(data["event_date"], errors="coerce")
    rows = []
    for column in [
        "request_date", "receipt_date", "collection_date", "analysis_date",
        "completion_date", "manufacture_date", "distribution_date",
        "shipment_expected_date",
    ]:
        if column not in data:
            continue
        other = pd.to_datetime(data[column], errors="coerce")
        comparable = event.notna() & other.notna()
        matches = event[comparable].eq(other[comparable])
        rows.append(
            {
                "date_column": column,
                "policy_group": classify_column(column),
                "non_null_rows": int(other.notna().sum()),
                "coverage_pct": float(other.notna().mean() * 100),
                "comparable_rows": int(comparable.sum()),
                "event_date_equal_rows": int(matches.sum()),
                "event_date_equal_pct": float(matches.mean() * 100) if comparable.any() else np.nan,
            }
        )
    return basis, pd.DataFrame(rows)


def candidate_priority(inventory: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    candidates = inventory[inventory["policy_group"].isin(["사용 후보", "조건부 후보"])].copy()
    source_counts = (
        coverage[coverage["coverage_pct"].ge(5)]
        .groupby("column")["source_system"].nunique().rename("sources_with_5pct")
    )
    candidates = candidates.merge(source_counts, on="column", how="left")
    candidates["sources_with_5pct"] = candidates["sources_with_5pct"].fillna(0).astype(int)

    def priority(row: pd.Series) -> str:
        if row["policy_group"] == "사용 후보" and row["coverage_pct"] >= 50 and row["sources_with_5pct"] >= 2:
            return "우선"
        if row["coverage_pct"] >= 10:
            return "보조"
        return "보류"

    candidates["priority"] = candidates.apply(priority, axis=1)
    candidates["use_rule"] = np.select(
        [
            candidates["column"].eq("event_date"),
            candidates["column"].str.endswith("_date"),
            candidates["data_note"].eq("고카디널리티"),
        ],
        [
            "연·월·분기·계절만 파생하고 원 날짜는 분할 기준으로 보존",
            "예측 시점 이전에 존재하는지 확인 후 기간 파생",
            "희소범주 통합 또는 해시·임베딩 검토",
        ],
        default="결측 범주를 명시하고 학습 구간에서만 전처리 적합",
    )
    event_row = inventory[inventory["column"].eq("event_date")].iloc[0]
    event_sources = int(
        coverage[coverage["column"].eq("event_date") & coverage["coverage_pct"].ge(5)]["source_system"].nunique()
    )
    derived = pd.DataFrame(
        [
            {
                "column": name, "policy_group": "사용 후보", "rationale": "event_date 사전 파생",
                "non_null_rows": int(event_row["non_null_rows"]),
                "coverage_pct": float(event_row["coverage_pct"]),
                "distinct_values": distinct, "distinct_capped": False, "data_note": "파생 변수",
                "sources_with_5pct": event_sources, "priority": "우선",
                "use_rule": "event_date에서 결과 확인 전 생성",
            }
            for name, distinct in [("event_year", 11), ("event_month", 12), ("event_quarter", 4), ("season", 4)]
        ]
    )
    candidates = pd.concat([candidates, derived], ignore_index=True)
    order = pd.Categorical(candidates["priority"], categories=["우선", "보조", "보류"], ordered=True)
    return (
        candidates.assign(_order=order)
        .sort_values(["_order", "coverage_pct", "column"], ascending=[True, False, True])
        .drop(columns="_order").reset_index(drop=True)
    )


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    ensure_dirs()
    for name, table in tables.items():
        table.to_csv(TABLE_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def plot_policy_counts(inventory: pd.DataFrame) -> None:
    counts = inventory["policy_group"].value_counts().reindex(POLICY_ORDER).dropna()
    colors = [
        "#2E8B57" if label in {"사용 후보", "조건부 후보"}
        else "#C44E52" if label in {"직접 누수", "사후 시점"}
        else "#4C72B0" if label in {"진단 전용", "전처리 전용", "그룹·감사 전용"}
        else "#8C8C8C" for label in counts.index
    ]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.barh(counts.index, counts.values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("열 수")
    ax.set_title("통합원장 143개 열의 모델링 사용 정책")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, f"{int(value)}개", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_policy_counts.png", bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)


def plot_candidate_coverage(coverage: pd.DataFrame, priority: pd.DataFrame) -> None:
    ordered = priority[priority["priority"].isin(["우선", "보조"])]["column"].head(24).tolist()
    matrix = coverage[coverage["column"].isin(ordered)].pivot(
        index="column", columns="source_system", values="coverage_pct"
    ).reindex(ordered)
    fig, ax = plt.subplots(figsize=(11, max(7, len(ordered) * 0.38)))
    sns.heatmap(
        matrix, annot=True, fmt=".0f", cmap="Blues", vmin=0, vmax=100,
        cbar_kws={"label": "비결측률 (%)"}, linewidths=0.4, linecolor="white", ax=ax,
    )
    ax.set_xlabel("출처")
    ax.set_ylabel("변수 후보")
    ax.set_title("출처별 주요 변수 후보 가용률")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "candidate_coverage_by_source.png", bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)


def plot_group_leakage(groups: pd.DataFrame) -> None:
    chart = groups.sort_values("random_split_cross_rows", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars = ax.barh(chart["group_key"], chart["random_split_cross_rows"], color="#C44E52")
    ax.set_xlabel("무작위 행 분할 시 서로 다른 세트에 걸치는 행 수")
    ax.set_title("그룹 키를 무시한 70·15·15 분할의 데이터 누수 위험")
    ax.grid(axis="x", alpha=0.2)
    offset = max(float(chart["random_split_cross_rows"].max()) * 0.01, 1)
    for bar, value in zip(bars, chart["random_split_cross_rows"]):
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2, f"{int(value):,}", va="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "random_split_group_leakage.png", bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)


def plot_direct_leakage(leakage: pd.DataFrame) -> None:
    chart = leakage[leakage["rows"].ge(100)].nlargest(18, "lift_vs_majority_pp").copy()
    chart["label"] = chart["target_name"] + " · " + chart["field"]
    chart = chart.sort_values("lift_vs_majority_pp")
    fig, ax = plt.subplots(figsize=(11, 8))
    bars = ax.barh(chart["label"], chart["lift_vs_majority_pp"], color="#C44E52")
    ax.set_xlabel("다수 클래스 기준 대비 결정력 증가 (%p)")
    ax.set_title("결과·판정 사후정보의 직접 누수 강도")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, chart["lift_vs_majority_pp"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "direct_leakage_strength.png", bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)


def write_report(
    source_path: Path,
    inventory: pd.DataFrame,
    priority: pd.DataFrame,
    proxy: pd.DataFrame,
    leakage: pd.DataFrame,
    groups: pd.DataFrame,
    date_basis: pd.DataFrame,
    date_checks: pd.DataFrame,
) -> None:
    counts = inventory["policy_group"].value_counts()
    total_rows = int(inventory["non_null_rows"].max())
    manual = inventory[inventory["policy_group"].eq("수동 검토")]["column"].tolist()
    high_priority = priority[priority["priority"].eq("우선")]["column"].tolist()
    high_proxy = proxy[proxy["source_proxy_risk"].eq("높음")]["column"].tolist()
    duplicate_row = groups[groups["group_key"].eq("duplicate_group_id")]
    duplicate_summary = duplicate_row.iloc[0].to_dict() if not duplicate_row.empty else {}
    post_basis = date_basis[
        date_basis["date_basis"].fillna("").astype(str).str.contains("분석|완료|analysis|completion", case=False, regex=True)
    ]
    strongest = leakage.nlargest(6, "lift_vs_majority_pp")

    lines = [
        "# 변수 후보 및 데이터 누수 점검", "",
        "- 기준일: 2026-09-14", "- 입력: 통합원장 v2.3",
        f"- 원천 파일: `{source_path.name}`", f"- 분석 단위: 검사 결과 {total_rows:,}건",
        f"- 원천 열: {len(inventory):,}개", "- 예측 시점 가정: 검사 결과 확인 전 의뢰·접수·수거 단계", "",
        "## 결론", "",
        "- 결과값·판정값·라벨 생성근거는 직접 누수로 분류",
        "- 분석일·완료일은 사후 시점으로 분류",
        "- 현재 원장의 MRL 열은 사후 보강 이력과 결합되어 기본 모델에서 제외",
        "- MRL은 별도 규정 기준표에서 검사일 현재값으로 재구축할 때만 조건부 사용",
        "- record_id·접수번호·연결키는 피처가 아닌 그룹 분할·추적에만 사용",
        "- source_system·기관·업체명은 출처·기관 효과 진단에만 사용",
        "- 무작위 행 분할 금지", "- 시간 분할 후 동일 검사 그룹을 하나의 세트에 고정", "",
        "## 열 정책 요약", "",
    ]
    for label in POLICY_ORDER:
        if int(counts.get(label, 0)):
            lines.append(f"- {label}: {int(counts.get(label, 0))}개")
    lines.extend(["", "## 우선 변수 후보", ""])
    lines.extend([f"- `{column}`" for column in high_priority])
    lines.extend([
        "", "## 변수별 사용 규칙", "",
        "- event_date: 원 날짜는 분할 기준으로 보존",
        "- event_date: 연·월·분기·계절만 파생해 학습",
        "- 품목명·품목군: 학습 구간에서만 희소범주 기준 적합",
        "- 결측치: 출처별 결측을 확인하고 결측 범주 또는 지시자 사용",
        "- 업체·주소: 고카디널리티·식별 위험으로 기본 모델 제외",
        "- source_system: 성능 민감도 비교용 진단 모델에만 포함", "",
        "## 직접 누수 근거", "",
    ])
    for row in strongest.itertuples():
        lines.append(
            f"- {row.target_name} · `{row.field}`: n={int(row.rows):,}, "
            f"점수 {row.score_pct:.2f}%, 다수 클래스 대비 +{row.lift_vs_majority_pp:.2f}%p"
        )
    lines.extend([
        "", "## 그룹 분할 누수", "",
        f"- duplicate_group_id 보유: {int(duplicate_summary.get('available_rows', 0)):,}건",
        f"- 2건 이상 그룹: {int(duplicate_summary.get('duplicated_groups', 0)):,}개",
        f"- 중복 그룹 소속 행: {int(duplicate_summary.get('rows_in_duplicated_groups', 0)):,}건",
        f"- 무작위 분할 시 세트가 갈리는 그룹: {int(duplicate_summary.get('random_split_cross_groups', 0)):,}개",
        f"- 무작위 분할 시 세트가 갈리는 행: {int(duplicate_summary.get('random_split_cross_rows', 0)):,}건",
        f"- 여러 연도에 걸친 그룹: {int(duplicate_summary.get('multi_year_groups', 0)):,}개",
        "- 권고 그룹키: duplicate_group_id 우선",
        "- 보조키: source_system + source_receipt_id + source_sample_id", "",
        "## 출처 프록시 위험", "",
        f"- 결측 여부와 출처의 Cramér V 0.5 이상 변수: {len(high_proxy)}개",
    ])
    lines.extend([f"- `{column}`" for column in high_proxy[:15]])
    if len(high_proxy) > 15:
        lines.append(f"- 그 외 {len(high_proxy) - 15}개")
    lines.extend([
        "", "## 날짜 누수 점검", "",
        f"- 분석일·완료일 기반 event_date 행: {int(post_basis['rows'].sum()) if not post_basis.empty else 0:,}건",
    ])
    for row in date_checks.itertuples():
        lines.append(
            f"- `{row.date_column}`: 보유 {int(row.non_null_rows):,}건 "
            f"({row.coverage_pct:.2f}%), event_date 일치 {int(row.event_date_equal_rows):,}건"
        )
    lines.extend([
        "", "## 위험도", "",
        "- 높음: 결과·판정·비율·초과 여부를 학습 피처로 포함할 위험",
        "- 높음: 동일 검사 그룹을 무작위로 나눌 위험",
        "- 중간: 출처별 결측 패턴을 일반 위험 신호로 오해할 위험",
        "- 중간: 업체·주소·상세 식별자의 암기 위험",
        "- 낮음: 전건 결측·상수 열의 불필요한 포함", "",
        "## 확정 정책", "",
        "- 기본 피처셋: 사용 후보 + event_date 파생 변수",
        "- 조건부 피처셋: 예측 시점 가용성이 확인된 날짜·업무 변수",
        "- 진단 피처셋: source_system·기관·업체 변수",
        "- 금지 피처셋: 직접 누수·사후 시점·목표 라벨·감사 메타데이터",
        "- 그룹키: 학습 변수에서 제외하고 분할 무결성 검증에만 사용", "",
        "## 다음 단계", "",
        "- 목표라벨별 확정 모집단 생성",
        "- 검사 결과 확인 전 시점의 변수 스냅샷 생성",
        "- 연도 기반 학습·검증·시험 구간 확정",
        "- 동일 duplicate_group_id의 세트 단일성 검증",
        "- 기본·조건부·진단 피처셋별 베이스라인 비교", "",
        "## 산출물", "",
        "- `table/column_policy_inventory.csv`", "- `table/candidate_priority.csv`",
        "- `table/source_feature_coverage.csv`", "- `table/source_proxy_risk.csv`",
        "- `table/direct_leakage_checks.csv`", "- `table/group_leakage_checks.csv`",
        "- `table/date_basis_distribution.csv`", "- `table/date_availability_checks.csv`",
        "- `fig/feature_policy_counts.png`", "- `fig/candidate_coverage_by_source.png`",
        "- `fig/direct_leakage_strength.png`", "- `fig/random_split_group_leakage.png`",
    ])
    if manual:
        lines.extend(["", "## 수동 검토 잔여 열", ""])
        lines.extend([f"- `{column}`" for column in manual])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> dict[str, Path]:
    ensure_dirs()
    config = load_config()
    source_path = data_path_from_env(config)
    data, inventory = load_profiled_data(source_path)
    coverage = source_feature_coverage(data, inventory)
    proxy = source_proxy_risk(data, inventory)
    leakage = direct_leakage_checks(data)
    groups = group_leakage_checks(data)
    date_basis, date_checks = date_availability_checks(data)
    priority = candidate_priority(inventory, coverage)

    tables = {
        "column_policy_inventory": inventory,
        "candidate_priority": priority,
        "source_feature_coverage": coverage,
        "source_proxy_risk": proxy,
        "direct_leakage_checks": leakage,
        "group_leakage_checks": groups,
        "date_basis_distribution": date_basis,
        "date_availability_checks": date_checks,
    }
    save_tables(tables)
    plot_policy_counts(inventory)
    plot_candidate_coverage(coverage, priority)
    plot_group_leakage(groups)
    plot_direct_leakage(leakage)
    write_report(source_path, inventory, priority, proxy, leakage, groups, date_basis, date_checks)

    payload = {
        "source_file": source_path.name,
        "source_sha256": file_sha256(source_path),
        "rows": int(len(data)),
        "columns": int(len(inventory)),
        "policy_counts": {str(k): int(v) for k, v in inventory["policy_group"].value_counts().items()},
        "report": str(REPORT_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report": REPORT_PATH, "summary": SUMMARY_PATH}
