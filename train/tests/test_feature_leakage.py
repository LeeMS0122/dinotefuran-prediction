from __future__ import annotations

import unittest

import pandas as pd

from pre_modeling.feature_leakage import (
    classify_column,
    direct_leakage_checks,
    summarize_group_key,
)


class FeatureLeakageTests(unittest.TestCase):
    def test_result_fields_are_hard_leakage(self):
        self.assertEqual(classify_column("judge_result"), "직접 누수")
        self.assertEqual(classify_column("result_to_mrl_ratio"), "직접 누수")

    def test_post_outcome_dates_are_excluded(self):
        self.assertEqual(classify_column("analysis_date"), "사후 시점")
        self.assertEqual(classify_column("completion_date"), "사후 시점")

    def test_product_and_calendar_fields_are_candidates(self):
        self.assertEqual(classify_column("product_name_std"), "사용 후보")
        self.assertEqual(classify_column("event_date"), "사용 후보")

    def test_screening_rule_is_exact_leakage(self):
        data = pd.DataFrame(
            {
                "label_occurrence_v1": [0, 1, 1],
                "label_screening_10pct_v1": [0, 0, 1],
                "label_noncompliance_v1": [0, 0, 1],
                "result_to_mrl_ratio": [0.01, 0.1, 1.2],
                "result_value_numeric_enriched": [0, 0.01, 0.5],
                "exceedance_flag": [0, 0, 1],
            }
        )
        result = direct_leakage_checks(data)
        row = result[
            result["target_name"].eq("MRL 10% 관심농도")
            & result["field"].eq("result_to_mrl_ratio")
        ].iloc[0]
        self.assertEqual(row["score_pct"], 100.0)

    def test_group_summary_detects_split_crossing(self):
        data = pd.DataFrame(
            {
                "event_date": ["2024-01-01", "2024-01-01", "2025-01-01"],
                "label_occurrence_v1": [0, 0, 1],
                "label_screening_10pct_v1": [0, 0, 1],
                "label_noncompliance_v1": [0, 0, 1],
            }
        )
        result = summarize_group_key(
            data,
            pd.Series(["a", "a", "b"]),
            "test_key",
            pd.Series(["train", "test", "train"]),
        )
        self.assertEqual(result["random_split_cross_groups"], 1)
        self.assertEqual(result["random_split_cross_rows"], 2)


if __name__ == "__main__":
    unittest.main()
