from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .training import TARGET_LABELS, resolve_device, run_target


ROOT = Path(__file__).resolve().parents[1]
TARGETS = tuple(TARGET_LABELS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["all", *TARGETS], default="all")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-cap", type=int, default=5000)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="window 선택용: 테스트셋 예측·평가를 수행하지 않음",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=ROOT / "output" / "lstm_sequences_v1",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=ROOT / "output" / "features_v1",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    targets = list(TARGETS) if args.target == "all" else [args.target]
    device = resolve_device(args.device)

    output_dir = args.output_dir or ROOT / "output" / (
        "lstm_baseline_smoke" if args.quick else "lstm_baseline_v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []

    for target in targets:
        result = run_target(
            target_name=target,
            sequence_dir=args.sequence_dir,
            feature_dir=args.feature_dir,
            output_dir=output_dir,
            config=config,
            device=device,
            max_epochs_override=args.max_epochs or (2 if args.quick else None),
            quick_cap=args.quick_cap if args.quick else None,
            evaluate_test=not args.validation_only,
        )

        preprocessor = result.pop("preprocessor")
        preprocessor_path = (
            output_dir / f"{target}_static_preprocessor_v1.json"
        )
        preprocessor_path.write_text(
            json.dumps(preprocessor, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        result["preprocessor_path"] = str(preprocessor_path)
        runs.append(result)

    metrics = pd.concat(
        [
            pd.read_csv(
                output_dir / f"{target}_metrics_v1.csv",
                encoding="utf-8-sig",
            )
            for target in targets
        ],
        ignore_index=True,
    )

    top_k = pd.concat(
        [
            pd.read_csv(
                output_dir / f"{target}_top_k_v1.csv",
                encoding="utf-8-sig",
            )
            for target in targets
        ],
        ignore_index=True,
    )

    metrics.to_csv(
        output_dir / "lstm_baseline_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_k.to_csv(
        output_dir / "lstm_top_k_capture.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "version": config["version"],
        "mode": "smoke" if args.quick else "full",
        "device": str(device),
        "targets": targets,
        "test_data_used": not args.validation_only,
        "policy": config["policy"],
        "runs": runs,
    }
    (output_dir / "lstm_baseline_manifest_v1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"device={device}")
    print(f"output={output_dir}")


if __name__ == "__main__":
    main()
