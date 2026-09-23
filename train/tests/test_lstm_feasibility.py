from pathlib import Path
import sys
import unittest

import pandas as pd


TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))

from lstm_feasibility.analysis import normalize_key, prior_active_day_counts


class LstmFeasibilityTests(unittest.TestCase):
    def test_same_day_and_future_are_excluded(self):
        frame = pd.DataFrame({
            "event_date": pd.to_datetime([
                "2025-01-01", "2025-01-01", "2025-01-10", "2025-02-20"
            ]),
            "group": ["A", "A", "A", "A"],
        })
        result = prior_active_day_counts(frame, ["group"], [14]).set_index("row_index")
        self.assertEqual(result.loc[0, "prior_active_days_14"], 0)
        self.assertEqual(result.loc[1, "prior_active_days_14"], 0)
        self.assertEqual(result.loc[2, "prior_active_days_14"], 1)
        self.assertEqual(result.loc[3, "prior_active_days_14"], 0)

    def test_missing_key_is_not_eligible(self):
        frame = pd.DataFrame({"group": ["A", None, "미상"]})
        _, valid = normalize_key(frame, ["group"])
        self.assertEqual(valid.tolist(), [True, False, False])

    def test_window_boundary_is_included(self):
        frame = pd.DataFrame({
            "event_date": pd.to_datetime(["2025-01-01", "2025-01-15"]),
            "group": ["A", "A"],
        })
        result = prior_active_day_counts(frame, ["group"], [14]).set_index("row_index")
        self.assertEqual(result.loc[1, "prior_active_days_14"], 1)

    def test_query_counts_use_full_history_pool(self):
        history = pd.DataFrame({
            "event_date": pd.to_datetime(["2025-01-01", "2025-01-03", "2025-01-10"]),
            "group": ["A", "A", "A"],
        })
        query = pd.DataFrame({
            "event_date": pd.to_datetime(["2025-01-10", "2025-01-11"]),
            "group": ["A", "A"],
        })
        result = prior_active_day_counts(history, ["group"], [14], query_frame=query)
        self.assertEqual(result.loc[0, "prior_active_days_14"], 2)
        self.assertEqual(result.loc[1, "prior_active_days_14"], 3)


if __name__ == "__main__":
    unittest.main()
