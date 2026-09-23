from __future__ import annotations

import argparse
from pathlib import Path

from route_climate_effect.build_dataset import build_route_climate_dataset
from route_climate_effect.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Core+공급경로별 혼합기후 paired 비교")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-lstm", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    if not args.skip_build:
        build_route_climate_dataset(args.config)
    result = run_experiment(args.config, args.device, args.skip_lstm)
    print(f"metrics={result['metrics_path']}")
    print(f"selection={result['selection_path']}")
    print(f"report={result['report_path']}")


if __name__ == "__main__":
    main()

