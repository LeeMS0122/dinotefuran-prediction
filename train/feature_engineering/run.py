from __future__ import annotations

import argparse
from pathlib import Path

from .engineering import run_feature_engineering


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output" / "modeling_data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "features_v1")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs" / "변수_확장_고도화")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "reference_config.yaml")
    args = parser.parse_args()
    result = run_feature_engineering(args.input_dir, args.output_dir, args.docs_dir, args.config)
    print(result["coverage"].to_string(index=False))
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()

