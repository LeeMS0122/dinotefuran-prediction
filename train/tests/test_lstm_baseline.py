from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch
import yaml


TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))

from lstm_baseline.data import StaticPreprocessor
from lstm_baseline.model import HybridSequenceLSTM
from lstm_baseline.training import classification_metrics, select_fbeta_threshold


class LstmBaselineTests(unittest.TestCase):
    def test_model_returns_one_logit_per_row(self):
        model = HybridSequenceLSTM([4, 5], numeric_size=3, hidden_size=8)
        logits = model(
            torch.zeros(6, 30, 2),
            torch.zeros(6, 2, dtype=torch.long),
            torch.zeros(6, 3),
        )
        self.assertEqual(tuple(logits.shape), (6,))

    def test_static_preprocessor_maps_unseen_category_to_zero(self):
        train = pd.DataFrame({"category": ["A", "B"], "number": [1.0, 3.0]})
        processor = StaticPreprocessor.fit(train, ["category"], ["number"])
        category, number = processor.transform(
            pd.DataFrame({"category": ["C"], "number": [2.0]})
        )
        self.assertEqual(category.tolist(), [[0]])
        self.assertAlmostEqual(float(number[0, 0]), 0.0, places=6)

    def test_validation_threshold_and_metrics_are_valid(self):
        y = np.array([0, 0, 1, 1], dtype=np.int8)
        probability = np.array([0.1, 0.4, 0.6, 0.9])
        weight = np.ones(4, dtype=np.float32)
        threshold, score = select_fbeta_threshold(y, probability, weight)
        metrics = classification_metrics(y, probability, weight, threshold, beta=2.0)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(metrics["weighted_pr_auc"], 1.0)

    def test_policy_tunes_all_five_models(self):
        config = yaml.safe_load((TRAIN / "lstm_baseline" / "config.yaml").read_text(encoding="utf-8"))
        self.assertTrue(config["policy"]["tune_all_models"])
        self.assertEqual(len(config["policy"]["models"]), 5)
        self.assertTrue(config["policy"]["baseline_rank_does_not_exclude_tuning"])


if __name__ == "__main__":
    unittest.main()
