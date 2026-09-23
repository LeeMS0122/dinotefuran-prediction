from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lstm_feasibility.analysis import configure_plotting, prepare_sequence_frame

from .builder import (
    TARGET_LABELS,
    build_daily_history,
    build_sequence_arrays,
    build_sequence_index,
    load_yaml,
    save_target_artifacts,
    sha256_file,
    summarize_index,
    validate_target_artifacts,
)
from .dataset import make_dataloader


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(TARGET_LABELS)


def render_figures(summary: pd.DataFrame, figure_dir: Path) -> None:
    configure_plotting()
    figure_dir.mkdir(parents=True, exist_ok=True)
    target_order = [TARGET_LABELS[x] for x in TARGETS]
    split_order = ["train", "validation", "test"]
    colors = {"train": "#3569A8", "validation": "#D88A2D", "test": "#8C6AAE"}

    subset = summary.loc[summary["split"].isin(split_order)].copy()
    pivot = subset.pivot(index="target_label", columns="split", values="min3_history_pct")
    pivot = pivot.reindex(index=target_order, columns=split_order)
    fig, axis = plt.subplots(figsize=(11, 6.2))
    pivot.plot(kind="bar", ax=axis, color=[colors[x] for x in split_order])
    axis.set_title("목표라벨·기간분할별 30일 검사이력 시퀀스 가용률")
    axis.set_ylabel("직전 30일 관측일 3일 이상 행 비율 (%)")
    axis.set_xlabel("")
    axis.set_ylim(0, 105)
    axis.grid(axis="y", alpha=0.65)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="기간 분할")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f", padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "sequence_min3_coverage_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = subset.pivot(index="target_label", columns="split", values="median_active_days")
    pivot = pivot.reindex(index=target_order, columns=split_order)
    fig, axis = plt.subplots(figsize=(11, 6.2))
    pivot.plot(kind="bar", ax=axis, color=[colors[x] for x in split_order])
    axis.set_title("목표라벨·기간분할별 30일 내 관측일 중앙값")
    axis.set_ylabel("과거 활성 관측일 중앙값 (일)")
    axis.set_xlabel("")
    axis.grid(axis="y", alpha=0.65)
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="기간 분할")
    for container in axis.containers:
        axis.bar_label(container, fmt="%.0f", padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figure_dir / "median_active_days_by_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    report_path: Path,
    summary: pd.DataFrame,
    judge: pd.DataFrame,
    history_quality: dict[str, Any],
    artifacts: dict[str, Any],
    config: dict[str, Any],
) -> None:
    validation = summary.loc[
        summary["split"].eq("validation"),
        [
            "target_label", "n_rows", "n_positive", "positive_rate_pct",
            "n_key_valid", "key_valid_pct", "n_any_history", "any_history_pct",
            "n_min3_history", "min3_history_pct", "median_active_days",
            "n_excluded_from_storage",
        ],
    ]
    files = pd.DataFrame([
        {
            "목표라벨": TARGET_LABELS[target],
            "NPZ": Path(info["array_path"]).name,
            "NPZ_MB": info["array_bytes"] / 1024**2,
            "인덱스": Path(info["index_path"]).name,
            "인덱스_MB": info["index_bytes"] / 1024**2,
        }
        for target, info in artifacts.items()
    ])
    judge_index = judge.set_index("judge_result")
    detected = int(judge_index.loc["검출", "rows"])
    nondetected = int(judge_index.loc["불검출", "rows"])
    undetermined = int(judge_index.loc["미확정", "rows"])
    total = detected + nondetected + undetermined
    window = int(config["window_days"])
    report = f"""# 06-2. LSTM 학습 데이터 생성

## 한눈에 보기

- 시퀀스 길이: {window}일
- 배열 방향: 검사일 -{window}일 → -1일
- 입력 채널: 일별 검사 건수, 관측 여부 마스크
- 검사 건수 모델 입력 변환: `log1p`
- 검사이력 모집단: 전체 통합원장 {history_quality['rows']:,}건
- 목표별 라벨 확정 행: 모두 저장
- 과거이력·식품군 키 결측 행: 삭제하지 않고 0 시퀀스와 명시적 마스크 저장
- DB 변경: 없음

## 중요사항 1 · 전체 검출/불검출 현황

- 전체 원장: {total:,}건
- 검출: {detected:,}건({detected / total * 100:.2f}%)
- 불검출: {nondetected:,}건({nondetected / total * 100:.2f}%)
- 미확정: {undetermined:,}건({undetermined / total * 100:.2f}%)
- 판정 확정분 검출률: {detected / (detected + nondetected) * 100:.2f}%
- 미확정은 불검출로 변환하지 않음

## 시퀀스 구성 규칙

- 식품군 연결키: `sequence_food_group_l2`
- 식품군 원천값을 우선 사용함
- 원천 식품군 결측 시 2015~2024년 학습군에서 적합한 품목명→식품군 매핑을 적용함
- 현재 검사일과 같은 날의 모든 검사자료를 제외함
- 검사일 이후 자료를 제외함
- 결과값·판정값·MRL·현재 목표라벨을 입력 채널로 사용하지 않음
- 과거 결과의 확정시각이 없으므로 과거 목표라벨도 입력하지 않음

## 2025년 검증군 생성 결과

{validation.to_markdown(index=False, floatfmt='.2f')}

## 저장 파일

{files.to_markdown(index=False, floatfmt='.2f')}

## 배열 명세

| 배열 | 형태 | 자료형 | 의미 |
|---|---|---|---|
| `inspection_count` | `(N, 30)` | int32 | 동일 식품군의 일별 전체 검사 건수 |
| `active_mask` | `(N, 30)` | uint8 | 검사 건수가 1건 이상인 날 |
| `target` | `(N,)` | int8 | 목표라벨 0/1 |
| `group_sample_weight` | `(N,)` | float32 | 중복그룹 크기 보정 가중치 |
| `key_valid` | `(N,)` | uint8 | 식품군 연결키 유효 여부 |
| `date_valid` | `(N,)` | uint8 | 검사일 유효 여부 |

## 품질 판단

- 전체 원장의 검사일 결측: {history_quality['event_date_missing']:,}건
- 전체 원장의 `record_id` 중복: {history_quality['record_id_duplicates']:,}건
- 목표별 저장 제외: 0건
- 이력 부족 행을 삭제하지 않아 원래 목표별 모델링 모집단을 보존함
- NPZ 배열과 Parquet 인덱스는 `array_row`로 1:1 연결함
- PyTorch Dataset은 검사 건수에 `log1p`를 적용하고 관측 마스크를 두 번째 채널로 결합함

## 현재 한계

- 실제 일별 기상값이 없어 기온·강수·습도 채널은 아직 포함하지 않음
- 식품군 키가 없는 행은 0 시퀀스이므로 `key_valid`를 반드시 함께 사용해야 함
- 30일은 초기 기준값이며 14·30·60·90일 성능 비교 후 최종 확정함

## 다음 작업

- `06-3. PyTorch LSTM 베이스라인 학습`
- 목표라벨 3종을 각각 학습함
- 동일 train·validation·test 분할을 유지함
- 2025년 검증 PR-AUC로 epoch·early stopping을 선택함
- 2026년 테스트셋은 최종 1회 평가함
- ML 4개 모델과 동일하게 전체 검출/불검출 건수와 목표별 성능표를 함께 기록함
"""
    report_path.write_text(report, encoding="utf-8")


def run_pipeline(
    input_dir: Path,
    history_pool_path: Path,
    feature_config_path: Path,
    food_mapping_path: Path,
    judge_table_path: Path,
    config_path: Path,
    output_dir: Path,
    docs_dir: Path,
    window_days_override: int | None = None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    if window_days_override is not None:
        config = dict(config)
        config["window_days"] = int(window_days_override)
    feature_config = load_yaml(feature_config_path)
    food_mapping = json.loads(food_mapping_path.read_text(encoding="utf-8-sig"))
    window = int(config["window_days"])
    minimum = int(config["minimum_active_days_reference"])

    raw_history = pd.read_parquet(history_pool_path)
    sequence_history = prepare_sequence_frame(raw_history, feature_config, food_mapping)
    daily_history = build_daily_history(sequence_history, config["sequence_key"])
    output_dir.mkdir(parents=True, exist_ok=True)
    daily_history_path = output_dir / "daily_history_food_group_l2_v1.parquet"
    daily_history.to_parquet(daily_history_path, index=False)
    history_quality = {
        "rows": int(len(sequence_history)),
        "event_date_missing": int(sequence_history["event_date"].isna().sum()),
        "record_id_duplicates": int(sequence_history["record_id"].duplicated().sum()),
        "daily_rows": int(len(daily_history)),
        "unique_sequence_keys": int(daily_history["sequence_key"].nunique()),
        "date_min": str(daily_history["event_date"].min().date()),
        "date_max": str(daily_history["event_date"].max().date()),
        "max_daily_inspection_count": int(daily_history["inspection_count"].max()),
    }

    summaries = []
    artifacts: dict[str, Any] = {}
    quality: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    for target in TARGETS:
        input_path = input_dir / f"{target}_features_v1.parquet"
        frame = pd.read_parquet(input_path).reset_index(drop=True)
        input_hashes[target] = sha256_file(input_path)
        sequence_frame = prepare_sequence_frame(frame, feature_config, food_mapping)
        arrays = build_sequence_arrays(
            daily_history,
            sequence_frame,
            config["sequence_key"],
            window,
        )
        index = build_sequence_index(frame, sequence_frame, arrays, minimum)
        quality[target] = validate_target_artifacts(frame, index, arrays, window)
        paths = save_target_artifacts(output_dir, target, index, arrays)
        batch = next(iter(make_dataloader(paths["array"], batch_size=8)))
        quality[target]["dataloader_batch_shape"] = list(batch["sequence"].shape)
        quality[target]["dataloader_target_shape"] = list(batch["target"].shape)
        summaries.append(summarize_index(index, target))
        artifacts[target] = {
            "array_path": str(paths["array"]),
            "array_bytes": paths["array"].stat().st_size,
            "array_sha256": sha256_file(paths["array"]),
            "index_path": str(paths["index"]),
            "index_bytes": paths["index"].stat().st_size,
            "index_sha256": sha256_file(paths["index"]),
        }

    summary = pd.concat(summaries, ignore_index=True)
    table_dir = docs_dir / "table"
    figure_dir = docs_dir / "figure"
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_dir / "sequence_generation_summary.csv", index=False, encoding="utf-8-sig")
    history_table = pd.DataFrame([history_quality])
    history_table.to_csv(table_dir / "history_pool_quality.csv", index=False, encoding="utf-8-sig")
    judge = pd.read_csv(judge_table_path, encoding="utf-8-sig")
    judge.to_csv(table_dir / "overall_judge_distribution.csv", index=False, encoding="utf-8-sig")
    render_figures(summary, figure_dir)

    schema = {
        "version": config["version"],
        "sequence_shape": ["N", window, 2],
        "sequence_order": config["sequence_order"],
        "channels": config["channels"],
        "array_index_join_key": "array_row",
        "stored_rows": "all target-labeled rows",
    }
    schema_path = output_dir / "sequence_schema_v1.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = docs_dir / "06-2_LSTM_학습데이터_생성.md"
    write_report(report_path, summary, judge, history_quality, artifacts, config)

    manifest = {
        "version": config["version"],
        "effective_window_days": window,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "inputs": {
            "history_pool": {"path": str(history_pool_path), "sha256": sha256_file(history_pool_path)},
            "feature_config": {"path": str(feature_config_path), "sha256": sha256_file(feature_config_path)},
            "food_mapping": {"path": str(food_mapping_path), "sha256": sha256_file(food_mapping_path)},
            "targets": {
                target: {"path": str(input_dir / f"{target}_features_v1.parquet"), "sha256": digest}
                for target, digest in input_hashes.items()
            },
        },
        "history_pool_quality": history_quality,
        "quality": quality,
        "artifacts": artifacts,
        "daily_history": {
            "path": str(daily_history_path),
            "sha256": sha256_file(daily_history_path),
        },
        "schema": {"path": str(schema_path), "sha256": sha256_file(schema_path)},
        "report": str(report_path),
    }
    manifest_path = output_dir / "lstm_sequence_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report_path": report_path,
        "manifest_path": manifest_path,
        "summary": summary,
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output" / "features_v1")
    parser.add_argument("--history-pool", type=Path, default=ROOT / "output" / "eda" / "cache" / "analysis.parquet")
    parser.add_argument("--feature-config", type=Path, default=ROOT / "feature_engineering" / "reference_config.yaml")
    parser.add_argument("--food-mapping", type=Path, default=ROOT / "output" / "features_v1" / "occurrence_food_mappings_v1.json")
    parser.add_argument("--judge-table", type=Path, default=ROOT / "docs" / "eda_1차" / "table" / "step0_judge_distribution.csv")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "lstm_sequences_v1")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs" / "LSTM_학습데이터_생성")
    parser.add_argument("--window-days", type=int, default=None)
    args = parser.parse_args()
    result = run_pipeline(
        args.input_dir,
        args.history_pool,
        args.feature_config,
        args.food_mapping,
        args.judge_table,
        args.config,
        args.output_dir,
        args.docs_dir,
        args.window_days,
    )
    print(f"report={result['report_path']}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
