from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from lstm_baseline.training import resolve_device, run_target
from lstm_data.run import run_pipeline as run_sequence_pipeline

from .analysis import finalize_window_comparison

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).parent / "config.yaml"
BASE_LSTM_CONFIG = ROOT / "lstm_baseline" / "config.yaml"
OUTPUT_ROOT = ROOT / "output" / "lstm_window_tuning_v1"
DOCS_ROOT = ROOT / "docs" / "LSTM_window_비교"
FEATURE_DIR = ROOT / "output" / "features_v1"
HISTORY_POOL = ROOT / "output" / "eda" / "cache" / "analysis.parquet"
FEATURE_CONFIG = ROOT / "feature_engineering" / "reference_config.yaml"
FOOD_MAPPING = FEATURE_DIR / "occurrence_food_mappings_v1.json"
JUDGE_TABLE = ROOT / "docs" / "eda_1차" / "table" / "step0_judge_distribution.csv"
SEQUENCE_CONFIG = ROOT / "lstm_data" / "config.yaml"


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def sequence_dir(window: int) -> Path:
    return OUTPUT_ROOT / "sequences" / f"window_{window}d"


def training_dir(window: int) -> Path:
    return OUTPUT_ROOT / "training" / f"window_{window}d"


def sequence_is_ready(window: int) -> bool:
    directory = sequence_dir(window)
    manifest_path = directory / "lstm_sequence_manifest_v1.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("effective_window_days", -1)) != window:
        return False
    return all(
        (directory / f"{target}_sequences_v1.npz").exists()
        and (directory / f"{target}_sequence_index_v1.parquet").exists()
        for target in ["occurrence", "screening", "noncompliance"]
    )


def training_is_ready(window: int) -> bool:
    directory = training_dir(window)
    manifest_path = directory / "window_training_manifest.json"
    metrics_path = directory / "window_validation_metrics.csv"
    if not manifest_path.exists() or not metrics_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        int(manifest.get("window_days", -1)) == window
        and manifest.get("test_data_used") is False
        and len(pd.read_csv(metrics_path, encoding="utf-8-sig")) == 3
    )


def prepare_sequences(window: int, force: bool) -> None:
    if sequence_is_ready(window) and not force:
        print(f"[window={window}] 시퀀스 재사용", flush=True)
        return
    output = sequence_dir(window)
    docs = DOCS_ROOT / "sequences" / f"window_{window}d"
    print(f"[window={window}] 시퀀스 생성 시작", flush=True)
    run_sequence_pipeline(
        FEATURE_DIR,
        HISTORY_POOL,
        FEATURE_CONFIG,
        FOOD_MAPPING,
        JUDGE_TABLE,
        SEQUENCE_CONFIG,
        output,
        docs,
        window_days_override=window,
    )
    print(f"[window={window}] 시퀀스 생성 완료", flush=True)


def train_window(window: int, device: torch.device, force: bool) -> None:
    if training_is_ready(window) and not force:
        print(f"[window={window}] 학습 결과 재사용", flush=True)
        return
    config = copy.deepcopy(load_config(BASE_LSTM_CONFIG))
    config["version"] = "lstm_window_tuning_v1"
    config["sequence"]["window_days"] = window
    config["evaluation"]["test_usage"] = "not_used_for_window_selection"
    output = training_dir(window)
    output.mkdir(parents=True, exist_ok=True)
    runs = []
    targets = ["occurrence", "screening", "noncompliance"]
    for target in targets:
        result = run_target(
            target_name=target,
            sequence_dir=sequence_dir(window),
            feature_dir=FEATURE_DIR,
            output_dir=output,
            config=config,
            device=device,
            evaluate_test=False,
        )
        preprocessor = result.pop("preprocessor")
        preprocessor_path = output / f"{target}_static_preprocessor_v1.json"
        preprocessor_path.write_text(
            json.dumps(preprocessor, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["preprocessor_path"] = str(preprocessor_path)
        runs.append(result)

    metrics = pd.concat(
        [
            pd.read_csv(output / f"{target}_metrics_v1.csv", encoding="utf-8-sig")
            for target in targets
        ],
        ignore_index=True,
    )
    metrics.insert(0, "window_days", window)
    metrics.to_csv(
        output / "window_validation_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_k = pd.concat(
        [
            pd.read_csv(output / f"{target}_top_k_v1.csv", encoding="utf-8-sig")
            for target in targets
        ],
        ignore_index=True,
    )
    top_k.insert(0, "window_days", window)
    top_k.to_csv(
        output / "window_validation_top_k.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manifest = {
        "version": "lstm_window_tuning_v1",
        "window_days": window,
        "device": str(device),
        "seed": int(config["seed"]),
        "test_data_used": False,
        "selection_split": "validation",
        "runs": runs,
    }
    (output / "window_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_windows(value: str, allowed: list[int]) -> list[int]:
    if value == "all":
        return allowed
    selected = int(value)
    if selected not in allowed:
        raise ValueError(f"지원하지 않는 window: {selected}")
    return [selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["all", "sequences", "train", "finalize"],
        default="all",
    )
    parser.add_argument("--window", default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    experiment = load_config(CONFIG_PATH)
    allowed = [int(value) for value in experiment["windows"]]
    windows = parse_windows(args.window, allowed)
    device = resolve_device(args.device)

    if args.stage in {"all", "sequences"}:
        for window in windows:
            prepare_sequences(window, args.force)
    if args.stage in {"all", "train"}:
        for window in windows:
            if not sequence_is_ready(window):
                raise FileNotFoundError(
                    f"window {window} 시퀀스가 없습니다. --stage sequences를 먼저 실행하세요."
                )
            train_window(window, device, args.force)
    if args.stage in {"all", "finalize"}:
        missing = [window for window in allowed if not training_is_ready(window)]
        if missing:
            raise FileNotFoundError(f"학습 결과 누락 window: {missing}")
        finalize_window_comparison(experiment, OUTPUT_ROOT, DOCS_ROOT, ROOT)
    print(f"device={device}")
    print(f"output={OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
