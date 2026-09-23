"""Target-specific training datasets and leakage-safe temporal/group splits."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SPLIT_ORDER = ["historical_excluded", "train", "validation", "test", "oot_monitor"]
SEASON_MAP = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
              6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


@dataclass(frozen=True)
class DesignPaths:
    input_csv: Path
    policy_yaml: Path
    output_dir: Path
    docs_dir: Path


def load_policy(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_from_event_date(value: object) -> str:
    if value is None or pd.isna(value):
        return "unknown_excluded"
    dt = pd.Timestamp(value)
    if dt.year <= 2014:
        return "historical_excluded"
    if dt.year <= 2024:
        return "train"
    if dt.year == 2025:
        return "validation"
    if dt.year == 2026 and dt.month <= 6:
        return "test"
    return "oot_monitor"


def split_from_anchor_year(year: float | int | None) -> str:
    """Year-only compatibility helper; exact 2026 split uses split_from_event_date."""
    if year is None or pd.isna(year):
        return "unknown_excluded"
    year = int(year)
    if year <= 2014:
        return "historical_excluded"
    if year <= 2024:
        return "train"
    if year == 2025:
        return "validation"
    return "test"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out["event_date"], errors="coerce")
    out["event_date"] = dt
    out["event_year"] = dt.dt.year.astype("Int16")
    out["event_month"] = dt.dt.month.astype("Int8").astype("string")
    out["event_quarter"] = ("Q" + dt.dt.quarter.astype("Int8").astype("string")).astype("string")
    out["event_season"] = dt.dt.month.map(SEASON_MAP).astype("string")
    angle = 2 * math.pi * dt.dt.month.astype(float) / 12.0
    out["event_month_sin"] = np.sin(angle).astype("float32")
    out["event_month_cos"] = np.cos(angle).astype("float32")
    out["row_calendar_split"] = dt.map(split_from_event_date).astype("string")
    return out


def assign_group_split(df: pd.DataFrame, group_col: str = "duplicate_group_id") -> pd.DataFrame:
    out = df.copy()
    fallback = "__record__" + out["record_id"].astype("string")
    out[group_col] = out[group_col].astype("string").fillna(fallback)
    out["group_anchor_date"] = out.groupby(group_col, observed=True)["event_date"].transform("max")
    out["group_anchor_year"] = out["group_anchor_date"].dt.year.astype("Int16")
    out["split"] = out["group_anchor_date"].map(split_from_event_date).astype("string")
    out["split_moved_for_group"] = (out["split"] != out["row_calendar_split"]).astype("int8")
    return out


def feature_columns(policy: dict[str, Any]) -> tuple[list[str], list[str]]:
    fs = policy["feature_sets"]
    categorical = list(fs["core_categorical"])
    numeric = list(fs["core_numeric"])
    all_features = categorical + numeric
    forbidden = tuple(fs["forbidden_patterns"])
    bad = [c for c in all_features if c.startswith(forbidden)]
    if bad:
        raise ValueError(f"Forbidden feature names: {bad}")
    return categorical, numeric


def required_columns(policy: dict[str, Any]) -> list[str]:
    categorical, _ = feature_columns(policy)
    raw_features = [c for c in categorical if not c.startswith("event_")]
    targets = [v["column"] for v in policy["targets"].values()]
    return sorted(set(["record_id", "duplicate_group_id", "source_system", "event_date", *raw_features, *targets]))


def read_master(path: Path, policy: dict[str, Any]) -> pd.DataFrame:
    usecols = required_columns(policy)
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def build_target_frame(base: pd.DataFrame, target_col: str, policy: dict[str, Any]) -> pd.DataFrame:
    categorical, numeric = feature_columns(policy)
    metadata = [
        "record_id", "duplicate_group_id", "source_system", "event_date", "event_year",
        "row_calendar_split", "group_anchor_date", "group_anchor_year", "split", "split_moved_for_group",
    ]
    frame = base.loc[base[target_col].notna()].copy()
    frame["target"] = pd.to_numeric(frame[target_col], errors="raise").astype("int8")
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError(f"Non-binary values found in {target_col}")
    grouped = frame.groupby("duplicate_group_id", observed=True)["target"]
    frame["group_size"] = grouped.transform("size").astype("int32")
    frame["group_target_conflict"] = (grouped.transform("nunique") > 1).astype("int8")
    frame["group_sample_weight"] = (1.0 / frame["group_size"]).astype("float32")
    frame = frame[metadata + ["group_size", "group_sample_weight", "group_target_conflict"] + categorical + numeric + ["target"]]
    if frame["record_id"].duplicated().any():
        raise ValueError(f"record_id is not unique for {target_col}")
    if frame.groupby("duplicate_group_id", observed=True)["split"].nunique().max() != 1:
        raise ValueError(f"Group leakage detected for {target_col}")
    return frame


def split_summary(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        part = frame.loc[frame["split"] == split]
        n = len(part)
        pos = int(part["target"].sum())
        rows.append({
            "target": target_name, "split": split, "n_rows": n, "n_positive": pos,
            "n_negative": n - pos, "positive_rate": pos / n if n else np.nan,
            "n_groups": int(part["duplicate_group_id"].nunique()),
            "n_conflict_groups": int(part.loc[part["group_target_conflict"] == 1, "duplicate_group_id"].nunique()),
            "effective_group_weight": float(part["group_sample_weight"].sum()),
        })
    return pd.DataFrame(rows)


def source_summary(frame: pd.DataFrame, target_name: str) -> pd.DataFrame:
    out = (frame.groupby(["split", "source_system"], dropna=False, observed=True)
           .agg(n_rows=("target", "size"), n_positive=("target", "sum"))
           .reset_index())
    out["target"] = target_name
    out["positive_rate"] = out["n_positive"] / out["n_rows"]
    out["share_within_split"] = out["n_rows"] / out.groupby("split", observed=True)["n_rows"].transform("sum")
    return out[["target", "split", "source_system", "n_rows", "n_positive", "positive_rate", "share_within_split"]]


def coverage_summary(frame: pd.DataFrame, target_name: str, features: list[str]) -> pd.DataFrame:
    rows = []
    for split in SPLIT_ORDER:
        part = frame.loc[frame["split"] == split]
        for col in features:
            rows.append({
                "target": target_name, "split": split, "feature": col,
                "coverage": float(part[col].notna().mean()) if len(part) else np.nan,
                "n_nonnull": int(part[col].notna().sum()), "n_rows": len(part),
            })
    return pd.DataFrame(rows)


def moved_summary(base: pd.DataFrame) -> pd.DataFrame:
    moved = base.loc[base["split_moved_for_group"] == 1]
    if moved.empty:
        return pd.DataFrame(columns=["row_calendar_split", "split", "n_rows", "n_groups"])
    return (moved.groupby(["row_calendar_split", "split"], observed=True)
            .agg(n_rows=("record_id", "size"), n_groups=("duplicate_group_id", "nunique"))
            .reset_index())


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _save_figures(summary: pd.DataFrame, source: pd.DataFrame, coverage: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary = summary.loc[summary["split"].isin(["train", "validation", "test", "oot_monitor"])].copy()
    targets = list(primary["target"].drop_duplicates())

    fig, axes = plt.subplots(1, len(targets), figsize=(16, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        part = primary.loc[primary["target"] == target]
        ax.bar(part["split"], part["n_rows"], color="#4C78A8")
        ax.set_title(target)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Rows")
        for i, value in enumerate(part["n_rows"]):
            ax.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Target population by leakage-safe split")
    fig.savefig(figure_dir / "target_population_by_split.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(targets), figsize=(16, 5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, target in zip(axes, targets):
        part = primary.loc[primary["target"] == target]
        rate = 100 * part["positive_rate"]
        ax.bar(part["split"], rate, color="#F58518")
        ax.set_title(target)
        ax.tick_params(axis="x", rotation=25)
        ax.set_ylabel("Positive rate (%)")
        for i, value in enumerate(rate):
            ax.text(i, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Positive rate by split")
    fig.savefig(figure_dir / "positive_rate_by_split.png", dpi=180)
    plt.close(fig)

    focus = source.loc[(source["target"] == "occurrence") & source["split"].isin(["train", "validation", "test", "oot_monitor"])]
    pivot = focus.pivot_table(index="split", columns="source_system", values="share_within_split", fill_value=0)
    pivot = pivot.reindex([s for s in SPLIT_ORDER if s in pivot.index])
    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="tab20")
    ax.set_title("Occurrence dataset: source composition by split")
    ax.set_ylabel("Share")
    ax.legend(title="source_system", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.figure.tight_layout()
    ax.figure.savefig(figure_dir / "source_composition_by_split.png", dpi=180)
    plt.close(ax.figure)

    cov = coverage.loc[coverage["split"].isin(["train", "test"])].copy()
    p = cov.pivot_table(index=["target", "feature"], columns="split", values="coverage").reset_index()
    p["gap_abs"] = (p.get("test", np.nan) - p.get("train", np.nan)).abs()
    top = p.sort_values("gap_abs", ascending=False).head(20).sort_values("gap_abs")
    fig, ax = plt.subplots(figsize=(11, 7))
    labels = top["target"] + " · " + top["feature"]
    ax.barh(labels, 100 * top["gap_abs"], color="#54A24B")
    ax.set_xlabel("Absolute coverage gap: train vs test (percentage points)")
    ax.set_title("Largest feature-availability shifts")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_coverage_drift.png", dpi=180)
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "- 해당 없음."
    rendered = df.copy()
    for col in rendered.select_dtypes(include=["float", "float32", "float64"]).columns:
        rendered[col] = rendered[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    headers = [str(col) for col in rendered.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rendered.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, moved: pd.DataFrame, coverage: pd.DataFrame,
                 paths: DesignPaths, input_hash: str) -> Path:
    paths.docs_dir.mkdir(parents=True, exist_ok=True)
    primary = summary.loc[summary["split"].isin(["train", "validation", "test", "oot_monitor"])].copy()
    display = primary[["target", "split", "n_rows", "n_positive", "n_negative", "positive_rate", "n_groups", "n_conflict_groups"]]
    test_noncomp = display.loc[(display["target"] == "noncompliance") & (display["split"] == "test")]
    noncomp_note = ""
    if not test_noncomp.empty:
        row = test_noncomp.iloc[0]
        noncomp_note = f"- 부적합 최종평가군: 양성 {int(row.n_positive):,}건 / 전체 {int(row.n_rows):,}건. 정확도보다 PR-AUC·재현율·업무량 대비 정밀도 중심 평가 필요.\n"
    moved_rows = int(moved["n_rows"].sum()) if len(moved) else 0
    moved_groups = int(moved["n_groups"].sum()) if len(moved) else 0
    worst = (coverage.loc[coverage["split"].isin(["train", "test"])]
             .pivot_table(index=["target", "feature"], columns="split", values="coverage")
             .reset_index())
    worst["gap_pp"] = 100 * (worst.get("test", np.nan) - worst.get("train", np.nan)).abs()
    worst = worst.nlargest(10, "gap_pp")[["target", "feature", "train", "test", "gap_pp"]]
    content = f"""# 목표라벨별 학습 데이터 설계

## 결론

- 3개 목표라벨을 서로 섞지 않고 독립 학습 데이터로 생성.
- 학습: 2015–2024년.
- 검증: 2025년.
- 최종평가: 2026년 1·2분기(1–6월).
- 2026년 7월 이후: 수집 기간이 불완전하므로 OOT 모니터링 전용. 최종 성능값 산정에서 제외.
- 2014년 이전: 희소·이질 가능성이 있어 기본 학습에서 제외하되 데이터에는 보존.
- 동일 `duplicate_group_id`: 가장 최근 관측일을 기준으로 한 구간에 통째로 배치.
- 그룹 중복 영향 완화용 `group_sample_weight = 1 / 그룹 내 라벨 보유 행 수` 생성.
- 목표 내부 판정충돌 그룹: 삭제하지 않고 `group_target_conflict`로 표시. 제외 민감도 분석 가능.

## 분할 결과

{_markdown_table(display)}

## 누수 방지 확인

- 분할 후 그룹 교차: 0개가 필수 검증 조건.
- 자체 연도구간과 다른 구간으로 이동한 행: {moved_rows:,}건.
- 이동에 관여한 그룹 합계: {moved_groups:,}개.
- 이동 사유: 여러 기간에 걸친 같은 중복그룹을 쪼개지 않기 위해 가장 최근 관측일 기준으로 보수적 배치.

{_markdown_table(moved) if len(moved) else '- 이동 행 없음.'}

## 모델 입력 원칙

- 기본 입력: 제품·식품군·원산지·수거/재배 지역·업무/수거단계·재배방식·시설·조사·수출유형·월/분기/계절.
- `event_year`, `event_date`, `source_system`: 진단 및 분할 확인용. 기본 모델 입력에서 제외.
- 결과값, 판정값, MRL 관련 값, 세 라벨 및 라벨 파생값: 입력 금지.
- 현재 통합원장의 MRL 계열은 사후 보강·라벨 생성과 얽혀 있으므로 독립적인 시점 기준 MRL을 재구축하기 전까지 입력 금지.
- 문자열 결측 대체와 범주 인코딩은 반드시 학습 구간에만 적합한 전처리기로 수행.

## 불균형 및 평가 유의사항

{noncomp_note}- occurrence·screening: PR-AUC, ROC-AUC, calibration, 목표 업무량에서의 precision/recall 병행.
- noncompliance: 희귀 양성 문제. class weight/scale_pos_weight는 학습 구간에서만 산출.
- 임계값은 2025 검증군에서 선택하고 2026년 1·2분기 최종평가군은 마지막 1회 평가에 사용.
- 2026년 7월 이후 자료는 운영 드리프트 확인용으로만 제시.
- 부적합 테스트 양성이 적으므로 bootstrap 신뢰구간과 2024→2025 보조 시계열 백테스트를 병행.

## 가용성 변화 상위 변수

{_markdown_table(worst)}

## 산출물

- 목표별 데이터: `train/output/modeling_data/{{occurrence,screening,noncompliance}}.parquet`
- 목표별 manifest: `train/output/modeling_data/*_manifest.json`
- 전체 manifest: `train/output/modeling_data/design_manifest.json`
- 요약표: `train/docs/학습데이터_설계/table/`
- 그래프: `train/docs/학습데이터_설계/figure/`

## 다음 단계

- 전처리 파이프라인: 학습군 기준 결측·희귀범주·범주인코딩 적합.
- 베이스라인: Dummy → Logistic Regression → CatBoost 순서.
- 세 목표 각각 동일 분할로 비교.
- 원본행 기준과 그룹가중 기준, 충돌그룹 제외 기준을 민감도 분석.

## 재현 정보

- 입력 파일: `{paths.input_csv}`
- 입력 SHA-256: `{input_hash}`
- 정책 파일: `{paths.policy_yaml}`
"""
    report_path = paths.docs_dir / "01_목표라벨별_학습데이터_설계.md"
    report_path.write_text(content, encoding="utf-8")
    return report_path


def run_design(paths: DesignPaths) -> dict[str, Any]:
    policy = load_policy(paths.policy_yaml)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = paths.docs_dir / "table"
    figure_dir = paths.docs_dir / "figure"
    table_dir.mkdir(parents=True, exist_ok=True)

    base = assign_group_split(add_calendar_features(read_master(paths.input_csv, policy)))
    categorical, numeric = feature_columns(policy)
    summaries, sources, coverages = [], [], []
    target_manifests: dict[str, Any] = {}

    for target_name, target_spec in policy["targets"].items():
        frame = build_target_frame(base, target_spec["column"], policy)
        out_path = paths.output_dir / f"{target_name}.parquet"
        frame.to_parquet(out_path, index=False)
        one_summary = split_summary(frame, target_name)
        summaries.append(one_summary)
        sources.append(source_summary(frame, target_name))
        coverages.append(coverage_summary(frame, target_name, categorical + numeric))
        manifest = {
            "target_name": target_name,
            "source_target_column": target_spec["column"],
            "description": target_spec["description"],
            "rows": len(frame),
            "features_categorical": categorical,
            "features_numeric": numeric,
            "metadata_columns": [c for c in frame.columns if c not in categorical + numeric + ["target"]],
            "target_column": "target",
            "parquet": str(out_path),
            "split_counts": one_summary.set_index("split")["n_rows"].to_dict(),
        }
        manifest_path = paths.output_dir / f"{target_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        target_manifests[target_name] = manifest

    summary = pd.concat(summaries, ignore_index=True)
    source = pd.concat(sources, ignore_index=True)
    coverage = pd.concat(coverages, ignore_index=True)
    moved = moved_summary(base)
    summary.to_csv(table_dir / "split_summary_by_target.csv", index=False, encoding="utf-8-sig")
    source.to_csv(table_dir / "split_source_composition.csv", index=False, encoding="utf-8-sig")
    coverage.to_csv(table_dir / "feature_coverage_by_target_split.csv", index=False, encoding="utf-8-sig")
    moved.to_csv(table_dir / "group_split_moved_rows.csv", index=False, encoding="utf-8-sig")

    integrity = []
    for target_name in policy["targets"]:
        manifest = target_manifests[target_name]
        frame = pd.read_parquet(manifest["parquet"], columns=["duplicate_group_id", "split", "record_id"])
        violations = int((frame.groupby("duplicate_group_id", observed=True)["split"].nunique() > 1).sum())
        integrity.append({"target": target_name, "group_cross_split_violations": violations,
                          "record_id_duplicates": int(frame["record_id"].duplicated().sum())})
    integrity_df = pd.DataFrame(integrity)
    integrity_df.to_csv(table_dir / "group_assignment_integrity.csv", index=False, encoding="utf-8-sig")

    _save_figures(summary, source, coverage, figure_dir)
    input_hash = sha256(paths.input_csv)
    report_path = write_report(summary, moved, coverage, paths, input_hash)
    design_manifest = {
        "input_csv": str(paths.input_csv), "input_sha256": input_hash,
        "policy_yaml": str(paths.policy_yaml), "report": str(report_path),
        "targets": target_manifests,
        "group_cross_split_violations": integrity_df.set_index("target")["group_cross_split_violations"].to_dict(),
    }
    (paths.output_dir / "design_manifest.json").write_text(
        json.dumps(design_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"summary": summary, "source": source, "coverage": coverage, "moved": moved,
            "integrity": integrity_df, "report_path": report_path, "manifest": design_manifest}




