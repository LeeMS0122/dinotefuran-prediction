"""CLI entry point for the target-specific training-data design stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from training_design import DesignPaths, run_design


HERE = Path(__file__).resolve().parent
TRAIN_DIR = HERE.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Integrated master CSV")
    parser.add_argument("--policy", type=Path, default=HERE / "training_design_policy.yaml")
    parser.add_argument("--output-dir", type=Path, default=TRAIN_DIR / "output" / "modeling_data")
    parser.add_argument("--docs-dir", type=Path, default=TRAIN_DIR / "docs" / "학습데이터_설계")
    return parser.parse_args()


def main() -> None:
    load_dotenv(TRAIN_DIR / ".env")
    args = parse_args()
    input_path = args.input or Path(os.environ["DINO_EDA_DATA_PATH"])
    result = run_design(DesignPaths(input_path, args.policy, args.output_dir, args.docs_dir))
    print(result["summary"].to_string(index=False))
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()

