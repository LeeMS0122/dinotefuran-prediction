from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_modeling.pipeline import load_yaml
from baseline_modeling.run_tree_core_supplement import CONFIG, SPECS, TARGETS


class TreeCoreSupplementTests(unittest.TestCase):
    def test_config_registers_core_candidates(self) -> None:
        config = load_yaml(CONFIG)
        registry = {
            (str(row["name"]), str(row["model_type"]), str(row["feature_set"]))
            for row in config["candidates"]
        }
        for model_type, candidate in SPECS:
            self.assertIn((candidate, model_type, "core"), registry)
        self.assertEqual(config["selection"]["reporting_primary_metric"], "recall")

    def test_saved_supplement_has_six_valid_rows(self) -> None:
        path = (
            ROOT
            / "docs"
            / "베이스라인_모델링_코어형_보완"
            / "table"
            / "tree_core_validation_metrics.csv"
        )
        if not path.exists():
            self.skipTest("원본 실행의 집계 산출물은 공개 저장소에 포함되지 않음")
        result = pd.read_csv(path, encoding="utf-8-sig")
        expected = {
            (target, candidate) for target in TARGETS for _, candidate in SPECS
        }
        observed = set(map(tuple, result[["target", "candidate"]].to_numpy()))
        self.assertEqual(observed, expected)
        self.assertTrue(
            result[["accuracy", "precision", "recall", "f1"]]
            .apply(lambda column: column.between(0, 1).all())
            .all()
        )


if __name__ == "__main__":
    unittest.main()
