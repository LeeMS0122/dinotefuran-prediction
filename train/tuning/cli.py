from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_tabular_bundle
from .final_evaluation import build_final_candidate_lock, evaluate_final_test
from .paired_sensitivity import evaluate_external_sensitivity
from .registry import (
    load_config,
    load_registry,
    resolved_registry,
    write_validation_manifest,
)
from .runner import run_config
from .spaces import load_search_spaces


ROOT = Path(__file__).resolve().parents[1]


def _output_root(config: dict[str, Any]) -> Path:
    return ROOT / config["output"]["root"]


def _select_rows(
    registry: pd.DataFrame,
    config_ids: list[str] | None,
    tier: str | None,
    run_all: bool,
) -> pd.DataFrame:
    if config_ids:
        unknown = sorted(set(config_ids).difference(registry["config_id"]))
        if unknown:
            raise ValueError(f"알 수 없는 config_id: {unknown}")
        return registry.loc[registry["config_id"].isin(config_ids)].copy()
    if tier:
        return registry.loc[registry["tuning_tier"].eq(tier)].copy()
    if run_all:
        return registry.copy()
    raise ValueError("--config-id, --tier 또는 --all 중 하나가 필요함")


def _validate(
    config: dict[str, Any],
    registry: pd.DataFrame,
    deep: bool = False,
) -> dict[str, Any]:
    output = _output_root(config)
    manifest = write_validation_manifest(ROOT, config, registry, output)
    spaces = load_search_spaces(ROOT / config["inputs"]["search_spaces"])
    missing: list[str] = []
    for target in sorted(registry["target"].unique()):
        feature = ROOT / "output" / "features_v1" / f"{target}_features_v1.parquet"
        if not feature.exists():
            missing.append(str(feature))
    for row in registry.loc[registry["model"].eq("lstm")].itertuples():
        window = int(float(row.window_days))
        sequence_root = (
            ROOT
            / "output"
            / "lstm_window_tuning_v1"
            / "sequences"
            / f"window_{window}d"
        )
        for suffix in ("sequences_v1.npz", "sequence_index_v1.parquet"):
            path = sequence_root / f"{row.target}_{suffix}"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"튜닝 입력 누락: {missing}")
    result = {
        "version": config["version"],
        "registry_rows": int(len(registry)),
        "tier_counts": registry["tuning_tier"].value_counts().to_dict(),
        "models": sorted(spaces["models"]),
        "validation_manifest": str(manifest),
        "test_data_used": False,
    }
    if deep:
        route_rows: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for _, row in registry.sort_values("config_id").iterrows():
            key = (
                row["target"],
                row["model"] == "lstm",
                row["evidence_family"],
                row["feature_set"],
                row["window_days"] if pd.notna(row["window_days"]) else None,
            )
            if key in seen or (
                row["model"] == "lstm"
                and row["evidence_family"] == "internal_full_population"
            ):
                continue
            seen.add(key)
            bundle = load_tabular_bundle(
                ROOT,
                row,
                config,
                quick_cap_per_split=200,
            )
            route_rows.append(
                {
                    "target": row["target"],
                    "model_scope": "lstm" if row["model"] == "lstm" else "tabular",
                    "evidence_family": row["evidence_family"],
                    "feature_set": row["feature_set"],
                    "window_days": row["window_days"],
                    "population": bundle.population,
                    "train_rows_checked": int(bundle.frame["split"].eq("train").sum()),
                    "validation_rows_checked": int(
                        bundle.frame["split"].eq("validation").sum()
                    ),
                    "train_hash": bundle.train_hash,
                    "validation_hash": bundle.validation_hash,
                    "test_data_used": False,
                }
            )
        route_path = output / "data_route_validation.csv"
        pd.DataFrame(route_rows).to_csv(
            route_path,
            index=False,
            encoding="utf-8-sig",
        )
        result["deep_data_routes"] = len(route_rows)
        result["data_route_validation"] = str(route_path)
    return result


def _run_rows(
    rows: pd.DataFrame,
    config: dict[str, Any],
    spaces: dict[str, Any],
    n_trials: int | None,
    device: str,
    quick_cap: int | None,
    smoke: bool,
) -> list[dict[str, Any]]:
    output = _output_root(config)
    summaries: list[dict[str, Any]] = []
    defaults = config["optimization"]["default_trials"]
    for _, row in rows.sort_values("config_id").iterrows():
        target_trials = int(n_trials or defaults[str(row["model"])])
        print(
            f"[{row['config_id']}][{row['target']}][{row['model']}] "
            f"target_trials={target_trials} smoke={smoke}",
            flush=True,
        )
        summaries.append(
            run_config(
                ROOT,
                row,
                config,
                spaces,
                output,
                target_trials=target_trials,
                device_name=device,
                quick_cap_per_split=quick_cap,
                smoke=smoke,
            )
        )
    return summaries


def _status(config: dict[str, Any], registry: pd.DataFrame) -> pd.DataFrame:
    output = _output_root(config)
    rows: list[dict[str, Any]] = []
    for mode in ("smoke", "runs"):
        for row in resolved_registry(registry, config).itertuples():
            summary_path = output / mode / row.config_id / "run_summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "mode": mode,
                    "config_id": row.config_id,
                    "target": row.target,
                    "model": row.model,
                    "completed_trials": summary["completed_after"],
                    "invalid_trials": summary.get("invalid_trials", 0),
                    "failed_trials": summary.get("failed_trials", 0),
                    "target_trials": summary["target_trials"],
                    "best_recall": summary["best_trial"]["recall"],
                    "best_f2": summary["best_trial"]["f2"],
                    "test_data_used": summary["test_data_used"],
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="디노테푸란 38개 구성 공통 하이퍼파라미터 튜닝"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser(
        "validate", help="실험대장·탐색공간·입력 파일 검증"
    )
    validate.add_argument(
        "--deep",
        action="store_true",
        help="모든 고유 데이터 라우팅을 실제로 열어 검증",
    )

    smoke = subparsers.add_parser("smoke", help="저장·재시작 경로 smoke run")
    smoke.add_argument("--config-id")
    smoke.add_argument("--n-trials", type=int)
    smoke.add_argument("--quick-cap", type=int)
    smoke.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    run = subparsers.add_parser("run", help="선택 구성 튜닝 실행 또는 재시작")
    run.add_argument("--config-id", action="append")
    run.add_argument(
        "--tier",
        choices=[
            "A1_required_core",
            "A2_internal_enhanced",
            "B_external_sensitivity",
        ],
    )
    run.add_argument("--all", action="store_true")
    run.add_argument("--n-trials", type=int)
    run.add_argument("--quick-cap", type=int)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    compare = subparsers.add_parser(
        "compare-external",
        help="B 외부 민감도 후보를 동일 모집단 internal-only 대조군과 비교",
    )
    compare.add_argument("--config-id", action="append")
    compare.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    compare.add_argument("--bootstrap-repeats", type=int, default=500)
    compare.add_argument("--force", action="store_true")

    subparsers.add_parser(
        "lock-final",
        help="검증 데이터로 확정한 내부 최종 후보와 체크포인트를 잠금",
    )
    subparsers.add_parser(
        "evaluate-final",
        help="잠근 최종 후보를 2026 test split에서 한 번 평가",
    )
    subparsers.add_parser("status", help="완료 trial과 선택 결과 확인")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(ROOT)
    registry = load_registry(ROOT, config)
    spaces = load_search_spaces(ROOT / config["inputs"]["search_spaces"])

    if args.command == "validate":
        print(
            json.dumps(
                _validate(config, registry, deep=bool(args.deep)),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "status":
        status = _status(config, registry)
        print("아직 실행 결과가 없습니다." if status.empty else status.to_string(index=False))
        return
    if args.command == "smoke":
        smoke_config = config["smoke"]
        config_id = args.config_id or smoke_config["config_id"]
        rows = _select_rows(registry, [config_id], None, False)
        summaries = _run_rows(
            rows,
            config,
            spaces,
            n_trials=args.n_trials or int(smoke_config["n_trials"]),
            device=args.device,
            quick_cap=args.quick_cap or int(smoke_config["quick_cap_per_split"]),
            smoke=True,
        )
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return
    if args.command == "compare-external":
        summary = evaluate_external_sensitivity(
            ROOT,
            config,
            registry,
            config_ids=args.config_id,
            device_name=args.device,
            bootstrap_repeats=int(args.bootstrap_repeats),
            force=bool(args.force),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "lock-final":
        print(
            json.dumps(
                build_final_candidate_lock(ROOT, config),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "evaluate-final":
        print(
            json.dumps(
                evaluate_final_test(ROOT, config),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    rows = _select_rows(registry, args.config_id, args.tier, args.all)
    summaries = _run_rows(
        rows,
        config,
        spaces,
        n_trials=args.n_trials,
        device=args.device,
        quick_cap=args.quick_cap,
        smoke=False,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
