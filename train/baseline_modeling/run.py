from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_baseline_modeling


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output" / "features_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "baseline_v2")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs" / "베이스라인_모델링_2차")
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    parser.add_argument(
        "--feature-manifest",
        type=Path,
        default=ROOT / "output" / "features_v1" / "feature_engineering_manifest.json",
    )
    args = parser.parse_args()
    result = run_baseline_modeling(
        args.input_dir,
        args.output_dir,
        args.docs_dir,
        args.config,
        args.feature_manifest,
    )
    print(result["test_metrics"].to_string(index=False))
    print(f"report={result['report_path']}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
