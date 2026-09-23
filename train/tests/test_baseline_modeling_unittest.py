from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))

from baseline_modeling.pipeline import (
    choose_fbeta_threshold,
    evaluate_scores,
    feature_sets,
    top_k_table,
    validate_modeling_frame,
    make_tree_preprocessor,
    make_xgboost,
)


class BaselineModelingTests(unittest.TestCase):
    def test_feature_sets_expand_monotonically(self):
        manifest_target = {
            "feature_sets": {
                "core_categorical": ["a"],
                "core_numeric": ["n"],
                "extended_categorical": ["b"],
                "conditional_categorical": ["c"],
            }
        }
        sets = feature_sets(manifest_target)
        self.assertEqual(sets["core"], (["a"], ["n"]))
        self.assertEqual(sets["extended"], (["a", "b"], ["n"]))
        self.assertEqual(sets["conditional"], (["a", "b", "c"], ["n"]))

    def test_threshold_and_metrics_are_valid(self):
        y = np.array([0, 0, 1, 1])
        score = np.array([0.1, 0.4, 0.6, 0.9])
        weight = np.ones(4)
        threshold = choose_fbeta_threshold(y, score, weight)
        metrics = evaluate_scores(y, score, weight, threshold)
        self.assertGreaterEqual(threshold, 0)
        self.assertLessEqual(threshold, 1)
        self.assertEqual(metrics["average_precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_top_k_capture_uses_highest_scores(self):
        y = np.array([1, 0, 1, 0])
        score = np.array([0.9, 0.8, 0.7, 0.1])
        result = top_k_table(y, score, np.ones(4), [0.5]).iloc[0]
        self.assertEqual(result["selected_n"], 2)
        self.assertEqual(result["capture_rate"], 0.5)
        self.assertEqual(result["precision"], 0.5)

    def test_validation_rejects_group_cross_split(self):
        data = pd.DataFrame({
            "record_id": ["1", "2", "3", "4", "5", "6"],
            "duplicate_group_id": ["x", "x", "y", "z", "v", "w"],
            "split": ["train", "validation", "train", "validation", "test", "test"],
            "target": [0, 1, 1, 0, 1, 0],
            "group_sample_weight": [1.0] * 6,
            "cat": ["a"] * 6,
            "num": [0.0] * 6,
        })
        with self.assertRaisesRegex(ValueError, "분할 교차"):
            validate_modeling_frame(data, ["cat"], ["num"])


    def test_tree_preprocessor_handles_unseen_categories(self):
        train = pd.DataFrame({"cat": ["a", "a", "b"], "num": [1.0, np.nan, 3.0]})
        validation = pd.DataFrame({"cat": ["new"], "num": [2.0]})
        preprocessor = make_tree_preprocessor(["cat"], ["num"], min_frequency=1)
        transformed_train = preprocessor.fit_transform(train)
        transformed_validation = preprocessor.transform(validation)
        self.assertEqual(transformed_train.shape[1], transformed_validation.shape[1])
        self.assertEqual(transformed_validation.shape[0], 1)

    def test_xgboost_honors_requested_device(self):
        model = make_xgboost(
            {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_child_weight": 1.0,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 2,
                "thread_count": 1,
                "device": "cuda",
            },
            seed=42,
            scale_pos_weight=1.0,
        )

        self.assertEqual(model.get_params()["device"], "cuda")


if __name__ == "__main__":
    unittest.main()
