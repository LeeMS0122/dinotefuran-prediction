from __future__ import annotations

import argparse
from pathlib import Path

from .experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-lstm", action="store_true")
    args = parser.parse_args()
    result = run_experiment(args.config, args.device, skip_lstm=args.skip_lstm)
    print(f"device={result['device']}")
    print(f"report={result['report_path']}")
    print(f"manifest={result['manifest_path']}")


if __name__ == "__main__":
    main()
