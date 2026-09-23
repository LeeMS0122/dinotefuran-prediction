from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from eda.targets import TargetSpec, label_summary, ratio_band, validate_binary, wilson_interval
from eda.step0_profile import _expected_screening_label
from eda.step4_season import season_from_month, summarize_target
from eda.step6_sources_countries import source_display
from eda.step7_group_season import collapse_categories
from pre_modeling.deep_diagnostics import screening_mismatches


class TargetRuleTests(unittest.TestCase):
    def test_binary_validation_accepts_missing(self):
        validate_binary(pd.Series([0, 1, np.nan]), "label")

    def test_binary_validation_rejects_other_values(self):
        with self.assertRaises(ValueError):
            validate_binary(pd.Series([0, 1, 2]), "label")

    def test_missing_label_is_not_counted_as_negative(self):
        spec = TargetSpec("test", "label", "테스트", "테스트 목적")
        result = label_summary(pd.DataFrame({"label": [0, 1, np.nan]}), spec)
        self.assertEqual(result["labeled_rows"], 2)
        self.assertEqual(result["missing_rows"], 1)
        self.assertEqual(result["negative_rows"], 1)

    def test_ratio_band_boundaries(self):
        bands = ratio_band(pd.Series([0.0999, 0.1, 1.0, 1.0001, np.nan]))
        values = list(bands)
        self.assertEqual(values[:4], [
            "MRL 10% 미만",
            "MRL 10~100%",
            "MRL 10~100%",
            "MRL 초과",
        ])
        self.assertTrue(pd.isna(values[4]))

    def test_screening_threshold_is_strictly_greater_than_point_one(self):
        labels = _expected_screening_label(pd.Series([0.1, 0.1000001]))
        self.assertEqual(labels.tolist(), [0.0, 1.0])

    def test_deep_diagnostic_uses_same_strict_screening_boundary(self):
        data = pd.DataFrame(
            {
                "source_system": ["exact", "above"],
                "label_screening_basis": ["test", "test"],
                "standard_corrected_flag": [0, 0],
                "standard_update_type": ["none", "none"],
                "result_to_mrl_ratio": [0.1, 0.1000001],
                "label_screening_10pct_v1": [0, 1],
            }
        )
        self.assertTrue(screening_mismatches(data).empty)

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(10, 100)
        self.assertLess(low, 0.1)
        self.assertGreater(high, 0.1)

    def test_season_boundaries(self):
        expected = {
            1: "겨울",
            2: "겨울",
            3: "봄",
            5: "봄",
            6: "여름",
            8: "여름",
            9: "가을",
            11: "가을",
            12: "겨울",
        }
        self.assertEqual(
            {month: season_from_month(month) for month in expected},
            expected,
        )

    def test_group_summary_preserves_missing_labels(self):
        data = pd.DataFrame(
            {
                "season": ["봄", "봄", "여름"],
                "label": [1, np.nan, 0],
            }
        )
        result = summarize_target(data, "label", ["season"]).set_index("season")
        self.assertEqual(result.loc["봄", "total_rows"], 2)
        self.assertEqual(result.loc["봄", "labeled_rows"], 1)
        self.assertEqual(result.loc["봄", "missing_rows"], 1)
        self.assertEqual(result.loc["봄", "positive_rows"], 1)

    def test_source_display_uses_korean_label(self):
        self.assertEqual(source_display("MFDS"), "식약처")
        self.assertEqual(source_display("UNKNOWN"), "UNKNOWN")

    def test_category_collapse_keeps_only_leading_values(self):
        result = collapse_categories(
            pd.Series(["엽채류", "과실류", np.nan]),
            ["엽채류"],
        )
        self.assertEqual(result.tolist(), ["엽채류", "기타", "기타"])


if __name__ == "__main__":
    unittest.main()
