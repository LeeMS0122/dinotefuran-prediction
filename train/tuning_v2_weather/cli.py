from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from lstm_baseline.training import resolve_device
from tuning.data import load_tabular_bundle
from tuning.paired_sensitivity import (
    METRICS,
    _prediction_metrics,
    _run_tabular_internal_control,
    paired_bootstrap_deltas,
)
from tuning.runner import _write_json, run_config
from tuning.spaces import load_search_spaces

from .audit import build_registry, load_config, run_audit


ROOT = Path(__file__).resolve().parents[1]


def runtime_config(root: Path = ROOT) -> dict[str, Any]:
    weather = load_config(root)
    base = yaml.safe_load(
        (root / "tuning" / "config.yaml").read_text(encoding="utf-8-sig")
    )
    result = copy.deepcopy(base)
    result["version"] = weather["version"]
    result["seed"] = int(weather["seed"])
    result["output"]["root"] = weather["output"]["root"]
    result["optimization"]["objective"] = (
        "validation_recall_at_weighted_f2_threshold_weather_v2"
    )
    for model in weather["experiment"]["models"]:
        result["optimization"]["default_trials"][model] = int(
            weather["experiment"]["trials_per_configuration"]
        )
    result["evaluation"]["threshold_metric"] = weather["experiment"][
        "threshold_metric"
    ]
    result["evaluation"]["threshold_beta"] = float(
        weather["experiment"]["threshold_beta"]
    )
    domestic = result["external"]["domestic_weather"]
    domestic["weather_root"] = str(Path(weather["inputs"]["weather_v2"]).parent)
    domestic["minimum_coverage_pct"] = float(
        weather["weather"]["minimum_coverage_pct"]
    )
    domestic["common_population_windows"] = [
        int(value) for value in weather["weather"]["windows"]
    ]
    domestic["weather_base_fields"] = list(
        weather["weather"]["numeric_base_fields"]
    )
    result["smoke"] = {
        "config_id": "WV2-001",
        "n_trials": 1,
        "quick_cap_per_split": 2000,
    }
    return result


def resolved_registry(root: Path = ROOT) -> pd.DataFrame:
    weather = load_config(root)
    frame = build_registry(weather).copy()
    frame["target_label"] = "MRL 10% 초과"
    frame["model_label"] = frame["model"]
    frame["tuning_tier"] = "weather_v2_challenger"
    frame["candidate"] = frame["window_days"].map(
        lambda value: f"weather_{int(value)}d"
    )
    frame["feature_set"] = frame["internal_feature_set"]
    frame["selection_status"] = "planned"
    frame["selection_reason"] = "paired same-population weather challenger"
    frame["evidence_family"] = "domestic_weather_paired"
    return frame


def _ensure_ready(root: Path = ROOT) -> dict[str, Any]:
    audit = run_audit(root)
    if audit["status"] != "ready_for_confirmatory_development":
        raise RuntimeError(f"weather v2 data-quality gate is not ready: {audit['status']}")
    if audit["key_checks"]["test_rows_read"] != 0:
        raise RuntimeError("development audit read test rows")
    return audit


def validate(deep: bool = False, root: Path = ROOT) -> dict[str, Any]:
    audit = _ensure_ready(root)
    config = runtime_config(root)
    registry = resolved_registry(root)
    spaces = load_search_spaces(root / config["inputs"]["search_spaces"])
    expected_models = set(load_config(root)["experiment"]["models"])
    if len(registry) != 12 or registry["config_id"].nunique() != 12:
        raise ValueError("weather v2 registry must contain 12 unique configurations")
    if set(registry["model"]) != expected_models:
        raise ValueError("weather v2 registry model set mismatch")
    missing_spaces = sorted(expected_models.difference(spaces["models"]))
    if missing_spaces:
        raise ValueError(f"missing search spaces: {missing_spaces}")
    output = root / config["output"]["root"]
    output.mkdir(parents=True, exist_ok=True)
    registry_path = output / "experiment_registry_resolved.csv"
    registry.to_csv(registry_path, index=False, encoding="utf-8-sig")
    result: dict[str, Any] = {
        "version": config["version"],
        "status": audit["status"],
        "registry_rows": int(len(registry)),
        "models": sorted(expected_models),
        "windows": sorted(int(value) for value in registry["window_days"].unique()),
        "common_population_across_windows": True,
        "registry": str(registry_path),
        "test_data_used": False,
    }
    if deep:
        hashes = set()
        routes = []
        for _, row in registry.groupby("window_days", sort=True).first().reset_index().iterrows():
            bundle = load_tabular_bundle(root, row, config, quick_cap_per_split=200)
            hashes.add((bundle.train_hash, bundle.validation_hash))
            routes.append(
                {
                    "window_days": int(row["window_days"]),
                    "train_rows_checked": int(bundle.frame["split"].eq("train").sum()),
                    "validation_rows_checked": int(
                        bundle.frame["split"].eq("validation").sum()
                    ),
                    "population": bundle.population,
                    "train_hash": bundle.train_hash,
                    "validation_hash": bundle.validation_hash,
                }
            )
        if len(hashes) != 1:
            raise ValueError("weather windows do not use the same sampled population")
        route_path = output / "data_route_validation.csv"
        pd.DataFrame(routes).to_csv(route_path, index=False, encoding="utf-8-sig")
        result["deep_data_routes"] = len(routes)
        result["data_route_validation"] = str(route_path)
    return result


def _select_rows(registry: pd.DataFrame, ids: list[str] | None, run_all: bool) -> pd.DataFrame:
    if ids:
        unknown = sorted(set(ids).difference(registry["config_id"]))
        if unknown:
            raise ValueError(f"unknown config_id: {unknown}")
        return registry.loc[registry["config_id"].isin(ids)].copy()
    if run_all:
        return registry.copy()
    raise ValueError("--config-id or --all is required")


def run_rows(
    rows: pd.DataFrame,
    n_trials: int | None,
    device: str,
    quick_cap: int | None,
    smoke: bool,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    _ensure_ready(root)
    config = runtime_config(root)
    spaces = load_search_spaces(root / config["inputs"]["search_spaces"])
    output = root / config["output"]["root"]
    summaries = []
    for _, row in rows.sort_values("config_id").iterrows():
        target_trials = int(
            n_trials or config["optimization"]["default_trials"][str(row["model"])]
        )
        print(
            f"[{row['config_id']}][{row['model']}][{int(row['window_days'])}d] "
            f"target_trials={target_trials} smoke={smoke}",
            flush=True,
        )
        summaries.append(
            run_config(
                root,
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


def classify_result(
    deltas: dict[str, float],
    bootstrap: pd.DataFrame,
    selection: dict[str, Any],
) -> str:
    intervals = bootstrap.set_index("metric")
    passed = (
        deltas["recall"] >= float(selection["minimum_recall_improvement_abs"])
        and (
            not selection["require_recall_ci_low_above_zero"]
            or float(intervals.loc["recall", "ci_low"]) > 0
        )
        and deltas["f2"] >= float(selection["minimum_f2_delta_abs"])
        and deltas["average_precision"]
        >= float(selection["minimum_average_precision_delta_abs"])
        and deltas["top10_capture_rate"]
        >= float(selection["minimum_top10_capture_delta_abs"])
    )
    return "promote_weather_candidate" if passed else "retain_internal_candidate"


def compare(
    ids: list[str] | None,
    device_name: str,
    repeats: int | None,
    force: bool,
    root: Path = ROOT,
) -> dict[str, Any]:
    _ensure_ready(root)
    weather = load_config(root)
    config = runtime_config(root)
    registry = resolved_registry(root)
    rows = _select_rows(registry, ids, run_all=not ids)
    output = root / config["output"]["root"]
    paired_root = output / "paired_sensitivity"
    paired_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    repeat_count = int(repeats or weather["experiment"]["bootstrap_repeats"])
    confidence = float(weather["experiment"]["bootstrap_confidence"])
    for offset, (_, row) in enumerate(rows.sort_values("config_id").iterrows(), 1):
        config_id = str(row["config_id"])
        best_path = output / "runs" / config_id / "best_trial.json"
        if not best_path.exists():
            raise FileNotFoundError(f"{config_id}: best_trial.json missing")
        best = json.loads(best_path.read_text(encoding="utf-8"))
        trial_dir = output / "runs" / config_id / f"trial_{int(best['trial_number']):04d}"
        external_path = trial_dir / "validation_predictions.parquet"
        control_dir = paired_root / config_id / "internal_only"
        control_result = control_dir / "control_result.json"
        control_predictions = control_dir / "validation_predictions.parquet"
        if force or not (control_result.exists() and control_predictions.exists()):
            control_dir.mkdir(parents=True, exist_ok=True)
            control = _run_tabular_internal_control(
                root, row, config, dict(best["params"]), control_dir, device
            )
        else:
            control = json.loads(control_result.read_text(encoding="utf-8"))
        if (
            control["train_hash"] != best["train_hash"]
            or control["validation_hash"] != best["validation_hash"]
        ):
            raise ValueError(f"{config_id}: internal/weather population hash mismatch")
        internal_predictions = pd.read_parquet(control_predictions)
        external_predictions = pd.read_parquet(external_path)
        beta = float(config["evaluation"]["threshold_beta"])
        internal_metrics = _prediction_metrics(internal_predictions, beta)
        external_metrics = _prediction_metrics(external_predictions, beta)
        deltas = {
            metric: external_metrics[metric] - internal_metrics[metric]
            for metric in METRICS
        }
        bootstrap = paired_bootstrap_deltas(
            internal_predictions,
            external_predictions,
            repeat_count,
            confidence,
            int(config["seed"]) + offset,
            beta,
        )
        decision = classify_result(deltas, bootstrap, weather["selection"])
        record = {
            "config_id": config_id,
            "run_id": best["run_id"],
            "model": str(row["model"]),
            "window_days": int(row["window_days"]),
            "population": control["population"],
            "train_hash": control["train_hash"],
            "validation_hash": control["validation_hash"],
            "internal_metrics": internal_metrics,
            "weather_metrics": external_metrics,
            "deltas": deltas,
            "bootstrap": bootstrap.to_dict(orient="records"),
            "decision": decision,
            "test_data_used": False,
        }
        _write_json(paired_root / config_id / "paired_result.json", record)
        print(f"[{config_id}] decision={decision}", flush=True)
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(paired_root.glob("WV2-*/paired_result.json"))
    ]
    summary = {
        "version": weather["version"],
        "evaluated_configs": len(results),
        "bootstrap_repeats": repeat_count,
        "confidence": confidence,
        "decision_counts": pd.Series(
            [item["decision"] for item in results], dtype="object"
        ).value_counts().to_dict(),
        "test_data_used": False,
        "results": results,
    }
    _write_json(paired_root / "summary.json", summary)
    return summary


def select_challenger(root: Path = ROOT) -> dict[str, Any]:
    weather = load_config(root)
    paired_root = root / weather["output"]["root"] / "paired_sensitivity"
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(paired_root.glob("WV2-*/paired_result.json"))
    ]
    if len(results) != 12:
        raise RuntimeError(f"12 paired results required before lock; found {len(results)}")
    eligible = [item for item in results if item["decision"] == "promote_weather_candidate"]
    selected = max(
        eligible,
        key=lambda item: (
            item["weather_metrics"]["recall"],
            item["weather_metrics"]["f2"],
            item["weather_metrics"]["average_precision"],
        ),
        default=None,
    )
    lock = {
        "version": weather["version"],
        "decision": "weather_challenger_locked" if selected else "retain_CFG-018",
        "selected": selected,
        "champion": weather["scope"]["champion"],
        "selection_rules": weather["selection"],
        "test_data_used": False,
    }
    _write_json(root / weather["output"]["root"] / "challenger_lock.json", lock)
    return lock


def status(root: Path = ROOT) -> pd.DataFrame:
    config = runtime_config(root)
    output = root / config["output"]["root"]
    rows = []
    for mode in ("smoke", "runs"):
        for row in resolved_registry(root).itertuples():
            path = output / mode / row.config_id / "run_summary.json"
            if path.exists():
                summary = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "mode": mode,
                        "config_id": row.config_id,
                        "model": row.model,
                        "window_days": int(row.window_days),
                        "completed_trials": summary["completed_after"],
                        "target_trials": summary["target_trials"],
                        "best_recall": summary["best_trial"]["recall"],
                        "best_f2": summary["best_trial"]["f2"],
                        "test_data_used": summary["test_data_used"],
                    }
                )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dinotefuran weather v2 paired tuning")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate")
    check.add_argument("--deep", action="store_true")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config-id", default="WV2-001")
    smoke.add_argument("--n-trials", type=int, default=1)
    smoke.add_argument("--quick-cap", type=int, default=2000)
    smoke.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run = commands.add_parser("run")
    run.add_argument("--config-id", action="append")
    run.add_argument("--all", action="store_true")
    run.add_argument("--n-trials", type=int)
    run.add_argument("--quick-cap", type=int)
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    paired = commands.add_parser("compare")
    paired.add_argument("--config-id", action="append")
    paired.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    paired.add_argument("--bootstrap-repeats", type=int)
    paired.add_argument("--force", action="store_true")
    commands.add_parser("select")
    final_test = commands.add_parser("evaluate-shared-test")
    final_test.add_argument("--bootstrap-repeats", type=int)
    commands.add_parser("status")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(json.dumps(validate(args.deep), ensure_ascii=False, indent=2))
        return
    if args.command == "status":
        frame = status()
        print("no runs yet" if frame.empty else frame.to_string(index=False))
        return
    if args.command == "select":
        print(json.dumps(select_challenger(), ensure_ascii=False, indent=2))
        return
    if args.command == "evaluate-shared-test":
        from .final_evaluation import evaluate_shared_test

        print(
            json.dumps(
                evaluate_shared_test(bootstrap_repeats=args.bootstrap_repeats),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    registry = resolved_registry()
    if args.command == "smoke":
        rows = _select_rows(registry, [args.config_id], False)
        result = run_rows(
            rows, args.n_trials, args.device, args.quick_cap, smoke=True
        )
    elif args.command == "compare":
        result = compare(
            args.config_id, args.device, args.bootstrap_repeats, args.force
        )
    else:
        rows = _select_rows(registry, args.config_id, args.all)
        result = run_rows(
            rows, args.n_trials, args.device, args.quick_cap, smoke=False
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
