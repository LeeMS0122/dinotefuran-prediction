from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tuning.data import load_tabular_bundle
from tuning_v2_weather.cli import (
    build_parser,
    classify_result,
    resolved_registry,
    runtime_config,
)
from tuning_v2_weather.final_evaluation import common_test_mask


class WeatherV2CliTests(unittest.TestCase):
    def test_shared_test_cli_is_explicit(self) -> None:
        args = build_parser().parse_args(
            ["evaluate-shared-test", "--bootstrap-repeats", "25"]
        )
        self.assertEqual(args.command, "evaluate-shared-test")
        self.assertEqual(args.bootstrap_repeats, 25)

    def test_common_test_mask_requires_period_and_all_window_coverage(self) -> None:
        frame = pd.DataFrame(
            {
                "split": ["test", "test", "test", "validation"],
                "event_date": [
                    "2026-01-01",
                    "2026-06-30",
                    "2026-07-01",
                    "2026-01-15",
                ],
                "wx_coverage_pct_14d": [70.0, 70.0, 100.0, 100.0],
                "wx_coverage_pct_30d": [70.0, 70.0, 100.0, 100.0],
                "wx_coverage_pct_60d": [70.0, 69.9, 100.0, 100.0],
                "wx_coverage_pct_90d": [70.0, 70.0, 100.0, 100.0],
            }
        )
        mask = common_test_mask(
            frame, [14, 30, 60, 90], 70.0, "2026-01-01", "2026-06-30"
        )
        self.assertEqual(mask.tolist(), [True, False, False, False])

    def test_runtime_registry_has_runner_fields_and_no_test_use(self) -> None:
        registry = resolved_registry(ROOT)
        required = {
            "config_id",
            "target",
            "model",
            "window_days",
            "feature_set",
            "tuning_tier",
            "candidate",
            "evidence_family",
        }
        self.assertFalse(required.difference(registry.columns))
        self.assertEqual(set(registry["feature_set"]), {"conditional"})
        self.assertEqual(set(registry["evidence_family"]), {"domestic_weather_paired"})
        self.assertFalse(registry["test_data_used"].any())

    def test_all_windows_use_the_same_common_population(self) -> None:
        config = runtime_config(ROOT)
        if not (ROOT / config["inputs"]["feature_manifest"]).exists():
            self.skipTest("기상 연결 원천 특성자료는 공개 저장소에 포함되지 않음")
        registry = resolved_registry(ROOT)
        hashes = set()
        for _, row in registry.groupby("window_days", sort=True).first().reset_index().iterrows():
            bundle = load_tabular_bundle(
                ROOT, row, config, quick_cap_per_split=100
            )
            hashes.add((bundle.train_hash, bundle.validation_hash))
            self.assertEqual(
                bundle.population,
                "domestic_weather_common_14d_30d_60d_90d_linked_train_validation",
            )
        self.assertEqual(len(hashes), 1)

    def test_selection_boundaries_are_applied_exactly(self) -> None:
        selection = {
            "minimum_recall_improvement_abs": 0.03,
            "require_recall_ci_low_above_zero": True,
            "minimum_f2_delta_abs": -0.01,
            "minimum_average_precision_delta_abs": -0.01,
            "minimum_top10_capture_delta_abs": 0.0,
        }
        deltas = {
            "recall": 0.03,
            "f2": -0.01,
            "average_precision": -0.01,
            "top10_capture_rate": 0.0,
        }
        bootstrap = pd.DataFrame(
            [
                {"metric": "recall", "ci_low": 0.0001, "ci_high": 0.05},
                {"metric": "f2", "ci_low": -0.02, "ci_high": 0.01},
                {"metric": "average_precision", "ci_low": -0.02, "ci_high": 0.01},
                {"metric": "top10_capture_rate", "ci_low": -0.01, "ci_high": 0.01},
            ]
        )
        self.assertEqual(
            classify_result(deltas, bootstrap, selection),
            "promote_weather_candidate",
        )
        bootstrap.loc[bootstrap["metric"].eq("recall"), "ci_low"] = 0.0
        self.assertEqual(
            classify_result(deltas, bootstrap, selection),
            "retain_internal_candidate",
        )


if __name__ == "__main__":
    unittest.main()
