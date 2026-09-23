from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run_feasibility


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=ROOT / "output" / "features_v1")
    parser.add_argument(
        "--history-pool", type=Path,
        default=ROOT / "output" / "eda" / "cache" / "analysis.parquet",
    )
    parser.add_argument(
        "--feature-config", type=Path,
        default=ROOT / "feature_engineering" / "reference_config.yaml",
    )
    parser.add_argument(
        "--food-mapping", type=Path,
        default=ROOT / "output" / "features_v1" / "occurrence_food_mappings_v1.json",
    )
    parser.add_argument(
        "--eda-table-dir", type=Path, default=ROOT / "docs" / "eda_1차" / "table"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output" / "lstm_feasibility_v1"
    )
    parser.add_argument(
        "--docs-dir", type=Path, default=ROOT / "docs" / "LSTM_시퀀스_점검"
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    args = parser.parse_args()
    result = run_feasibility(
        args.input_dir,
        args.history_pool,
        args.feature_config,
        args.food_mapping,
        args.eda_table_dir,
        args.output_dir,
        args.docs_dir,
        args.config,
    )
    print(f"report={result['report_path']}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
