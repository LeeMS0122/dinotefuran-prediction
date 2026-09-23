from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda.common import data_path_from_env, load_config
from .feature_leakage import DOCS_DIR, TABLE_DIR, TARGET_COLUMNS, ensure_dirs


REPORT_PATH = DOCS_DIR / "02_그룹충돌_라벨규칙_추가점검.md"


def load_data() -> pd.DataFrame:
    path = data_path_from_env(load_config())
    columns = [
        "source_system", "duplicate_group_id", "label_occurrence_v1",
        "label_screening_10pct_v1", "label_noncompliance_v1",
        "result_to_mrl_ratio", "label_screening_basis",
        "standard_corrected_flag", "standard_update_type",
    ]
    data = pd.read_csv(path, usecols=columns, encoding="utf-8-sig", low_memory=False)
    for column in [
        "label_occurrence_v1", "label_screening_10pct_v1",
        "label_noncompliance_v1", "result_to_mrl_ratio",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def screening_mismatches(data: pd.DataFrame) -> pd.DataFrame:
    mask = data["result_to_mrl_ratio"].notna() & data["label_screening_10pct_v1"].notna()
    work = data.loc[mask, [
        "source_system", "label_screening_basis", "standard_corrected_flag",
        "standard_update_type", "result_to_mrl_ratio", "label_screening_10pct_v1",
    ]].copy()
    work["ratio_rule"] = work["result_to_mrl_ratio"].gt(0.1).astype(int)
    work["label"] = work["label_screening_10pct_v1"].astype(int)
    work = work[work["ratio_rule"].ne(work["label"])]
    if work.empty:
        return pd.DataFrame(columns=[
            "source_system", "label_screening_basis", "standard_corrected_flag",
            "standard_update_type", "ratio_rule", "label", "rows",
        ])
    return (
        work.groupby([
            "source_system", "label_screening_basis", "standard_corrected_flag",
            "standard_update_type", "ratio_rule", "label",
        ], dropna=False)
        .size().rename("rows").reset_index().sort_values("rows", ascending=False)
    )


def conflicts_by_source(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in sorted(TARGET_COLUMNS):
        grouped = data.groupby(
            ["source_system", "duplicate_group_id"], dropna=False
        )[target].nunique(dropna=True)
        conflicts = grouped.gt(1).groupby(level=0).sum()
        totals = grouped.groupby(level=0).size()
        for source in totals.index:
            rows.append(
                {
                    "target_column": target,
                    "source_system": source,
                    "groups": int(totals[source]),
                    "conflict_groups": int(conflicts.get(source, 0)),
                    "conflict_group_pct": float(conflicts.get(source, 0) / totals[source] * 100),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target_column", "conflict_groups"], ascending=[True, False]
    ).reset_index(drop=True)


def run() -> None:
    ensure_dirs()
    data = load_data()
    mismatch = screening_mismatches(data)
    conflicts = conflicts_by_source(data)
    mismatch.to_csv(TABLE_DIR / "screening_rule_mismatch_detail.csv", index=False, encoding="utf-8-sig")
    conflicts.to_csv(TABLE_DIR / "group_conflicts_by_source.csv", index=False, encoding="utf-8-sig")

    occurrence = conflicts[conflicts["target_column"].eq("label_occurrence_v1")]
    screening = conflicts[conflicts["target_column"].eq("label_screening_10pct_v1")]
    noncompliance = conflicts[conflicts["target_column"].eq("label_noncompliance_v1")]
    lines = [
        "# 그룹 충돌 및 관심농도 규칙 추가 점검", "",
        "- 기준일: 2026-09-14", "- 입력: 통합원장 v2.3", "",
        "## 관심농도 규칙", "",
        f"- result_to_mrl_ratio > 0.1 규칙과 라벨 불일치: {int(mismatch['rows'].sum()) if not mismatch.empty else 0:,}건",
        "- 불일치 행 자동 수정 금지",
        "- 표준값 보강·교정 이력과 label_screening_basis를 함께 확인",
        "- result_to_mrl_ratio는 검증용으로만 사용",
        "- 모델 피처에서는 제외", "",
        "## duplicate_group_id 내부 라벨 충돌", "",
        f"- 잔류 발생 충돌 그룹: {int(occurrence['conflict_groups'].sum()):,}개",
        f"- 관심농도 충돌 그룹: {int(screening['conflict_groups'].sum()):,}개",
        f"- 기준 부적합 충돌 그룹: {int(noncompliance['conflict_groups'].sum()):,}개",
        "- 충돌 그룹은 단순 중복 제거 금지",
        "- 동일 그룹을 한 분할에 고정하는 원칙은 유지",
        "- 다음 학습 데이터 설계에서 대표행·그룹 집계·충돌 제외안을 비교", "",
        "## 출처별 충돌", "",
    ]
    for row in conflicts[conflicts["conflict_groups"].gt(0)].itertuples():
        lines.append(
            f"- `{row.target_column}` · {row.source_system}: "
            f"{int(row.conflict_groups):,}/{int(row.groups):,}그룹 ({row.conflict_group_pct:.2f}%)"
        )
    lines.extend([
        "", "## 판단", "",
        "- duplicate_group_id는 삭제용 중복키가 아님",
        "- duplicate_group_id는 분할 무결성용 연결 그룹키로 사용",
        "- 라벨 충돌은 오류로 단정하지 않음",
        "- 원천 다중행·반복검사·연결 단위 차이를 다음 단계에서 표본 확인", "",
        "## 산출물", "",
        "- `table/screening_rule_mismatch_detail.csv`",
        "- `table/group_conflicts_by_source.csv`",
    ])
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[완료] 추가 진단 · {REPORT_PATH}")


if __name__ == "__main__":
    run()
