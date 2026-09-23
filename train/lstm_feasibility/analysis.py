from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from feature_engineering.engineering import clean_product_name, normalize_group, normalize_series


TARGET_META = {
    "occurrence": {"label": "잔류 존재", "color": "#3569A8"},
    "screening": {"label": "MRL 10% 관심농도", "color": "#D88A2D"},
    "noncompliance": {"label": "기준 부적합", "color": "#8C6AAE"},
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_plotting() -> None:
    plt.rcParams.update({
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
    })


def normalize_key(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.Series, pd.Series]:
    valid = pd.Series(True, index=frame.index)
    parts: list[pd.Series] = []
    for column in columns:
        values = frame[column].astype("string").str.strip()
        column_valid = values.notna() & values.ne("") & ~values.isin(
            ["미상", "불명", "미분류", "unknown", "__MISSING__"]
        )
        valid &= column_valid
        parts.append(values.fillna("__MISSING__"))
    key = parts[0]
    for part in parts[1:]:
        key = key.str.cat(part, sep="||")
    return key, valid


def prior_active_day_counts(
    frame: pd.DataFrame,
    columns: list[str],
    windows: list[int],
    query_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Count history-frame active dates in [query_date-window, query_date)."""
    dates = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    key, valid_key = normalize_key(frame, columns)
    valid = valid_key & dates.notna()
    pairs = pd.DataFrame({"sequence_key": key[valid], "event_date": dates[valid]})
    pairs = pairs.drop_duplicates().sort_values(["sequence_key", "event_date"])
    history_by_key = {
        group_key: group["event_date"].to_numpy(dtype="datetime64[D]").astype("int64")
        for group_key, group in pairs.groupby("sequence_key", sort=False, observed=True)
    }
    query = frame if query_frame is None else query_frame
    query_dates = pd.to_datetime(query["event_date"], errors="coerce").dt.normalize()
    query_key, query_key_valid = normalize_key(query, columns)
    row_map = pd.DataFrame({
        "row_index": query.index,
        "sequence_key": query_key,
        "event_date": query_dates,
        "key_valid": query_key_valid & query_dates.notna(),
    })
    for window in windows:
        result = np.zeros(len(row_map), dtype=np.int32)
        for group_key, positions in row_map.groupby("sequence_key", sort=False, observed=True).groups.items():
            history_dates = history_by_key.get(group_key)
            if history_dates is None:
                continue
            query_values = row_map.loc[positions, "event_date"].to_numpy(dtype="datetime64[D]").astype("int64")
            right = np.searchsorted(history_dates, query_values, side="left")
            left = np.searchsorted(history_dates, query_values - int(window), side="left")
            result[np.asarray(positions, dtype=int)] = right - left
        row_map[f"prior_active_days_{window}"] = result
    return row_map


def prepare_sequence_frame(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    food_mapping: dict[str, Any],
) -> pd.DataFrame:
    """Build outcome-independent keys shared by the full ledger and target rows."""
    out = pd.DataFrame(index=frame.index)
    out["record_id"] = frame["record_id"].astype("string")
    out["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce")
    product_name, _ = clean_product_name(frame["product_name_std"])
    raw_group = normalize_group(
        frame["product_group_raw"], feature_config["food_group_mapping"]["l2_aliases"]
    )
    mapped_group = product_name.map(food_mapping.get("name_l2", {})).astype("string")
    out["sequence_food_group_l2"] = raw_group.fillna(mapped_group).fillna("미분류")
    out["sequence_product_name"] = product_name.fillna("미분류")
    province = normalize_series(frame["collection_province"])
    out["sequence_collection_province"] = province.replace(
        feature_config["province_aliases"]
    ).fillna("미분류")
    return out

def summarize_history(
    frame: pd.DataFrame,
    history_pool: pd.DataFrame,
    key_spec: dict[str, Any],
    windows: list[int],
    minimum_history_days: list[int],
    analysis_splits: list[str],
) -> pd.DataFrame:
    counts = prior_active_day_counts(
        history_pool, list(key_spec["columns"]), windows, query_frame=frame
    )
    context = frame[["split", "target"]].copy()
    context["row_index"] = frame.index
    counts = counts.merge(context, on="row_index", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for split in analysis_splits:
        subset = counts.loc[counts["split"].eq(split)]
        total = len(subset)
        valid_key = int(subset["key_valid"].sum())
        for window in windows:
            observed = subset[f"prior_active_days_{window}"].fillna(0).astype(int)
            for minimum in minimum_history_days:
                eligible = subset["key_valid"] & observed.ge(minimum)
                rows.append({
                    "sequence_key": key_spec["name"],
                    "sequence_key_label": key_spec["label"],
                    "split": split,
                    "window_days": int(window),
                    "minimum_active_days": int(minimum),
                    "n_rows": int(total),
                    "n_key_valid": valid_key,
                    "n_eligible": int(eligible.sum()),
                    "coverage_all_pct": float(eligible.mean() * 100) if total else np.nan,
                    "coverage_key_valid_pct": (
                        float(eligible.sum() / valid_key * 100) if valid_key else np.nan
                    ),
                    "median_prior_active_days": float(observed[subset["key_valid"]].median())
                    if valid_key else np.nan,
                })
    return pd.DataFrame(rows)


def target_split_summary(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    table = (
        frame.groupby("split", observed=True)["target"]
        .agg(n_rows="size", n_positive="sum")
        .reset_index()
    )
    table["n_negative"] = table["n_rows"] - table["n_positive"]
    table["positive_rate_pct"] = table["n_positive"] / table["n_rows"] * 100
    table.insert(0, "target_label", TARGET_META[target]["label"])
    table.insert(0, "target", target)
    return table


def key_coverage_summary(
    frame: pd.DataFrame,
    target: str,
    sequence_keys: list[dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in sequence_keys:
        key, valid = normalize_key(frame, list(spec["columns"]))
        for split, subset_index in frame.groupby("split", observed=True).groups.items():
            valid_subset = valid.loc[subset_index]
            rows.append({
                "target": target,
                "target_label": TARGET_META[target]["label"],
                "sequence_key": spec["name"],
                "sequence_key_label": spec["label"],
                "split": split,
                "n_rows": int(len(subset_index)),
                "n_key_valid": int(valid_subset.sum()),
                "key_coverage_pct": float(valid_subset.mean() * 100),
                "n_unique_groups": int(key.loc[subset_index][valid_subset].nunique()),
            })
    return pd.DataFrame(rows)


def weather_readiness_summary(
    frame: pd.DataFrame,
    target: str,
    readiness_column: str,
) -> pd.DataFrame:
    rows = []
    ready = pd.to_numeric(frame[readiness_column], errors="coerce").fillna(0).eq(1)
    for split, index in frame.groupby("split", observed=True).groups.items():
        subset = ready.loc[index]
        rows.append({
            "target": target,
            "target_label": TARGET_META[target]["label"],
            "split": split,
            "n_rows": int(len(index)),
            "n_weather_join_ready": int(subset.sum()),
            "weather_join_ready_pct": float(subset.mean() * 100),
        })
    return pd.DataFrame(rows)


def render_figures(
    figure_dir: Path,
    judge_distribution: pd.DataFrame,
    key_coverage: pd.DataFrame,
    history: pd.DataFrame,
    weather: pd.DataFrame,
) -> None:
    configure_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)

    order = ["불검출", "검출", "미확정"]
    colors = {"불검출": "#9AA7B8", "검출": "#D88A2D", "미확정": "#D8DEE8"}
    status = judge_distribution.set_index("judge_result").reindex(order).reset_index()
    fig, axis = plt.subplots(figsize=(9.5, 5.8))
    bars = axis.bar(status["judge_result"], status["rows"], color=[colors[x] for x in order])
    for bar, (_, row) in zip(bars, status.iterrows()):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                  f"{int(row['rows']):,}\n({row['share_pct']:.2f}%)",
                  ha="center", va="bottom")
    axis.set_ylabel("전체 원장 건수")
    axis.set_title("디노테푸란 전체 원장의 검출·불검출·미확정 현황")
    axis.grid(axis="y", alpha=0.65)
    axis.set_ylim(0, status["rows"].max() * 1.16)
    fig.tight_layout()
    fig.savefig(figure_dir / "overall_detection_status.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    subset = key_coverage.loc[key_coverage["split"].eq("train")].copy()
    pivot = subset.pivot(index="target_label", columns="sequence_key_label", values="key_coverage_pct")
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    pivot.plot(kind="bar", ax=axis, color=["#3569A8", "#D88A2D", "#8C6AAE"])
    axis.set_ylabel("학습군 연결키 보유율 (%)")
    axis.set_xlabel("")
    axis.set_ylim(0, 105)
    axis.set_title("목표라벨별 LSTM 시퀀스 연결키 가용성")
    axis.grid(axis="y", alpha=0.65)
    axis.legend(title="시퀀스 키", loc="upper right")
    axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figure_dir / "sequence_key_coverage.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    subset = history.loc[
        history["split"].eq("validation") & history["minimum_active_days"].eq(3)
    ].copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)
    for axis, target in zip(axes, TARGET_META):
        current = subset.loc[subset["target"].eq(target)]
        for key_name, group in current.groupby("sequence_key_label", observed=True):
            axis.plot(group["window_days"], group["coverage_all_pct"], marker="o", label=key_name)
        axis.set_title(TARGET_META[target]["label"])
        axis.set_xlabel("Window (일)")
        axis.set_xticks(sorted(current["window_days"].unique()))
        axis.grid(alpha=0.65)
    axes[0].set_ylabel("직전 관측일 3일 이상 확보 행 비율 (%)")
    axes[-1].legend(loc="best", fontsize=9)
    fig.suptitle("2025년 검증군의 과거 시퀀스 가용률", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figure_dir / "history_coverage_by_window.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    subset = weather.loc[weather["split"].isin(["train", "validation", "test"])].copy()
    pivot = subset.pivot(index="target_label", columns="split", values="weather_join_ready_pct")
    pivot = pivot.reindex(columns=["train", "validation", "test"])
    fig, axis = plt.subplots(figsize=(10.5, 5.8))
    pivot.plot(kind="bar", ax=axis, color=["#3569A8", "#D88A2D", "#8C6AAE"])
    axis.set_ylabel("기상 연결 준비 행 비율 (%)")
    axis.set_xlabel("")
    axis.set_ylim(0, 105)
    axis.set_title("목표라벨·분할별 기상 시퀀스 연결 준비도")
    axis.grid(axis="y", alpha=0.65)
    axis.legend(title="분할")
    axis.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(figure_dir / "weather_join_readiness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    history_pool_quality: dict[str, Any],
    target_summary: pd.DataFrame,
    judge_distribution: pd.DataFrame,
    split_summary: pd.DataFrame,
    key_coverage: pd.DataFrame,
    history: pd.DataFrame,
    weather: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    target_table = target_summary[[
        "target_name", "total_rows", "labeled_rows", "missing_rows",
        "positive_rows", "negative_rows", "positive_rate_pct",
    ]]
    key_table = key_coverage.loc[
        key_coverage["split"].eq("train"),
        ["target_label", "sequence_key_label", "n_rows", "n_key_valid", "key_coverage_pct", "n_unique_groups"],
    ]
    baseline_window = int(config["baseline_window_days"])
    baseline_history = history.loc[
        history["split"].eq("validation")
        & history["window_days"].eq(baseline_window)
        & history["minimum_active_days"].eq(3),
        ["target_label", "sequence_key_label", "n_rows", "n_eligible", "coverage_all_pct", "median_prior_active_days"],
    ]
    weather_table = weather.loc[
        weather["split"].isin(["train", "validation", "test"]),
        ["target_label", "split", "n_rows", "n_weather_join_ready", "weather_join_ready_pct"],
    ]
    occurrence = judge_distribution.set_index("judge_result")
    detected = int(occurrence.loc["검출", "rows"])
    nondetected = int(occurrence.loc["불검출", "rows"])
    undetermined = int(occurrence.loc["미확정", "rows"])
    total = detected + nondetected + undetermined
    text = f"""# 06-1. LSTM 시퀀스 설계 및 가용성 점검

## 한눈에 보기

- 초기 LSTM 베이스라인 window: {baseline_window}일
- window 비교 후보: {', '.join(str(x) for x in config['candidate_windows_days'])}일
- 최종 window 선택: 2025년 검증 PR-AUC
- 실제 기상 일별값 파일: 현재 없음
- 딥러닝 런타임: PyTorch 사용 가능
- 검사이력 기준 원장: 전체 통합원장 {history_pool_quality['rows']:,}건
- 검사이력 날짜 결측: {history_pool_quality['event_date_missing']:,}건
- DB 변경: 없음

## 중요사항 1 · 전체 검출/불검출 현황

- 전체 원장: {total:,}건
- 검출: {detected:,}건({detected / total * 100:.2f}%)
- 불검출: {nondetected:,}건({nondetected / total * 100:.2f}%)
- 미확정: {undetermined:,}건({undetermined / total * 100:.2f}%)
- 판정 확정분 검출률: {detected / (detected + nondetected) * 100:.2f}%
- 미확정은 불검출로 대체하지 않음

## 목표라벨별 모델링 모집단

{target_table.to_markdown(index=False, floatfmt='.2f')}

## 시계열 연결키 가용성 · 학습군

{key_table.to_markdown(index=False, floatfmt='.2f')}

## 30일 window의 검증군 과거이력 가용성

- 기준: 현재 검사일을 제외한 직전 30일에 서로 다른 관측일 3일 이상

{baseline_history.to_markdown(index=False, floatfmt='.2f')}

## 기상 시퀀스 연결 준비도

{weather_table.to_markdown(index=False, floatfmt='.2f')}

## 시퀀스 설계 결정

- 1차 LSTM은 window 30일로 시작함
- 튜닝에서는 14·30·60·90일을 2025년 검증 PR-AUC로 비교함
- 검사이력은 목표라벨별 확정 행이 아닌 전체 통합원장 {history_pool_quality['rows']:,}건에서 계산함
- 식품군 소분류는 원천 식품군을 우선하고, 결측 시 기존 2015~2024년 학습군의 품목명 매핑으로 보완함
- 결과값·판정값·목표라벨은 식품군 매핑과 검사이력 집계에 사용하지 않음
- 정제 품목명은 정적 임베딩 변수로 사용하고 시퀀스 키로는 보조 비교함
- 국내 시도 결측이 많아 기상정보 보유 행만 학습하면 목표별 모집단이 크게 달라질 수 있음
- 기상정보는 결측 마스크와 연결 가능 플래그를 함께 사용하고 공통 모집단 성능을 별도 보고함

## 데이터 누수 정책

- 시퀀스 기간은 `[검사일-window, 검사일)`로 고정함
- 같은 검사일 자료와 미래 자료는 시퀀스에 포함하지 않음
- 현재 검사 결과값·판정값·MRL·목표라벨은 입력하지 않음
- 과거 판정값도 결과 확정시각을 알 수 없으므로 1차 LSTM에서는 입력하지 않음
- 과거 검사 건수·관측 유무 등 결과 비의존 정보만 우선 사용함

## 판단

- LSTM 구조 실험은 가능함
- 현재 자료만으로는 식품군별 과거 검사활동 시퀀스를 구성할 수 있음
- 실제 기상 시퀀스 모델은 일별 기상값 확보·지역키 연결 이후 가능함
- LSTM과 ML의 공정 비교를 위해 전체 가용 모집단과 공통 모집단 성능을 함께 산출해야 함

## 다음 작업

- `06-2. LSTM 학습 데이터 생성`
- 30일 일별 검사활동 시퀀스와 마스크 생성
- 정적 범주형 변수 인코딩 설계
- 목표별 전체·시퀀스 생성 가능·제외 건수 저장
- PyTorch Dataset/DataLoader와 누수 방지 테스트 작성
"""
    report_path.write_text(text, encoding="utf-8")


def run_feasibility(
    input_dir: Path,
    history_pool_path: Path,
    feature_config_path: Path,
    food_mapping_path: Path,
    eda_table_dir: Path,
    output_dir: Path,
    docs_dir: Path,
    config_path: Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    table_dir = docs_dir / "table"
    figure_dir = docs_dir / "figure"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    target_summary = pd.read_csv(eda_table_dir / "step1_target_summary.csv", encoding="utf-8-sig")
    judge_distribution = pd.read_csv(eda_table_dir / "step0_judge_distribution.csv", encoding="utf-8-sig")
    feature_config = load_yaml(feature_config_path)
    food_mapping = json.loads(food_mapping_path.read_text(encoding="utf-8-sig"))
    raw_history_pool = pd.read_parquet(history_pool_path)
    history_pool = prepare_sequence_frame(raw_history_pool, feature_config, food_mapping)
    history_pool_quality = {
        "rows": int(len(history_pool)),
        "event_date_missing": int(history_pool["event_date"].isna().sum()),
        "record_id_duplicates": int(history_pool["record_id"].duplicated().sum()),
    }
    split_tables = []
    key_tables = []
    history_tables = []
    weather_tables = []
    input_checksums: dict[str, str] = {}
    quality: dict[str, Any] = {}

    for target in TARGET_META:
        input_path = input_dir / f"{target}_features_v1.parquet"
        frame = pd.read_parquet(input_path)
        input_checksums[target] = sha256_file(input_path)
        dates = pd.to_datetime(frame["event_date"], errors="coerce")
        quality[target] = {
            "rows": int(len(frame)),
            "event_date_missing": int(dates.isna().sum()),
            "record_id_duplicates": int(frame["record_id"].duplicated().sum()),
            "target_missing": int(frame["target"].isna().sum()),
        }
        split_tables.append(target_split_summary(frame, target))
        sequence_frame = prepare_sequence_frame(frame, feature_config, food_mapping)
        sequence_frame["split"] = frame["split"].values
        sequence_frame["target"] = frame["target"].values
        key_tables.append(key_coverage_summary(sequence_frame, target, config["sequence_keys"]))
        weather_tables.append(
            weather_readiness_summary(frame, target, config["weather"]["readiness_column"])
        )
        for spec in config["sequence_keys"]:
            current = summarize_history(
                sequence_frame,
                history_pool,
                spec,
                [int(x) for x in config["candidate_windows_days"]],
                [int(x) for x in config["minimum_history_days"]],
                list(config["analysis_splits"]),
            )
            current.insert(0, "target_label", TARGET_META[target]["label"])
            current.insert(0, "target", target)
            history_tables.append(current)

    split_summary = pd.concat(split_tables, ignore_index=True)
    key_coverage = pd.concat(key_tables, ignore_index=True)
    history = pd.concat(history_tables, ignore_index=True)
    weather = pd.concat(weather_tables, ignore_index=True)

    target_summary.to_csv(table_dir / "target_population_summary.csv", index=False, encoding="utf-8-sig")
    judge_distribution.to_csv(table_dir / "overall_judge_distribution.csv", index=False, encoding="utf-8-sig")
    split_summary.to_csv(table_dir / "target_split_summary.csv", index=False, encoding="utf-8-sig")
    key_coverage.to_csv(table_dir / "sequence_key_coverage.csv", index=False, encoding="utf-8-sig")
    history.to_csv(table_dir / "history_coverage_by_window.csv", index=False, encoding="utf-8-sig")
    weather.to_csv(table_dir / "weather_join_readiness.csv", index=False, encoding="utf-8-sig")

    render_figures(figure_dir, judge_distribution, key_coverage, history, weather)
    report_path = docs_dir / "06-1_LSTM_시퀀스_설계_가용성점검.md"
    write_report(
        report_path,
        history_pool_quality,
        target_summary,
        judge_distribution,
        split_summary,
        key_coverage,
        history,
        weather,
        config,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": config["version"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "inputs": {
            "history_pool": {
                "path": str(history_pool_path),
                "sha256": sha256_file(history_pool_path),
            },
            "feature_config": {
                "path": str(feature_config_path),
                "sha256": sha256_file(feature_config_path),
            },
            "food_mapping": {
                "path": str(food_mapping_path),
                "sha256": sha256_file(food_mapping_path),
            },
            "targets": {
            target: {
                "path": str(input_dir / f"{target}_features_v1.parquet"),
                "sha256": checksum,
            }
            for target, checksum in input_checksums.items()
            },
        },
        "quality": {"history_pool": history_pool_quality, "targets": quality},
        "decision": {
            "baseline_window_days": int(config["baseline_window_days"]),
            "candidate_windows_days": [int(x) for x in config["candidate_windows_days"]],
            "primary_sequence_key": "sequence_food_group_l2",
            "history_pool_scope": "full_master_ledger",
            "weather_daily_values_present": False,
            "deep_learning_backend": "pytorch",
        },
        "report": str(report_path),
    }
    manifest_path = output_dir / "lstm_feasibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report_path": report_path,
        "manifest_path": manifest_path,
        "target_summary": target_summary,
        "key_coverage": key_coverage,
        "history": history,
        "weather": weather,
    }
