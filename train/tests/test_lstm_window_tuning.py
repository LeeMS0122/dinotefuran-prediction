from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lstm_baseline.training import classification_metrics
from lstm_window_tuning.analysis import select_windows


class LstmWindowTuningTests(unittest.TestCase):
    def test_classification_metrics_include_accuracy_and_f1(self) -> None:
        metrics = classification_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0.1, 0.7, 0.8, 0.9]),
            np.ones(4),
            threshold=0.5,
            beta=2.0,
        )
        self.assertAlmostEqual(metrics["weighted_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["weighted_precision"], 2 / 3)
        self.assertAlmostEqual(metrics["weighted_recall"], 1.0)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.8)

    def test_window_selection_applies_f1_floor_before_recall(self) -> None:
        rows = []
        for target in ["occurrence", "screening", "noncompliance"]:
            rows.extend([
                {
                    "target": target,
                    "window_days": 14,
                    "weighted_recall": 0.99,
                    "weighted_f1": 0.70,
                    "weighted_pr_auc": 0.60,
                },
                {
                    "target": target,
                    "window_days": 30,
                    "weighted_recall": 0.90,
                    "weighted_f1": 0.80,
                    "weighted_pr_auc": 0.70,
                },
            ])
        _, selected = select_windows(pd.DataFrame(rows), relative_floor=0.95)
        self.assertEqual(set(selected["window_days"]), {30})


if __name__ == "__main__":
    unittest.main()
