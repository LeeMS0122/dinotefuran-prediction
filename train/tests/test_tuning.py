from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import optuna
import pandas as pd

from tuning.paired_sensitivity import classify_paired_result
from tuning.final_evaluation import build_final_candidate_lock
from tuning.registry import load_config, load_registry, resolved_registry
from tuning.runner import select_balanced_trial
from tuning.spaces import load_search_spaces, sample_params


ROOT = Path(__file__).resolve().parents[1]


class TuningTests(unittest.TestCase):
    def _write_final_candidate_fixture(self, root: Path) -> tuple[dict, Path]:
        output = root / "output" / "tuning_v1"
        runs = output / "runs"
        output.mkdir(parents=True)
        selected = []
        best_paths = {}
        for index, target in enumerate(
            ("occurrence", "screening", "noncompliance"),
            start=1,
        ):
            config_id = f"CFG-{index:03d}"
            checkpoint = runs / config_id / "trial_0001" / "model.joblib"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"{target}-checkpoint".encode())
            best = {
                "target": target,
                "config_id": config_id,
                "run_id": f"{config_id}-T0001",
                "model": "xgboost",
                "feature_set": "conditional",
                "params": {"max_depth": 3},
                "threshold": 0.5,
                "recall": 0.8,
                "f2": 0.7,
                "average_precision": 0.6,
                "checkpoint_path": str(checkpoint),
                "test_data_used": False,
            }
            best_path = runs / config_id / "best_trial.json"
            best_path.write_text(json.dumps(best), encoding="utf-8")
            best_paths[target] = best_path
            selected.append(
                {
                    "target": target,
                    "config_id": config_id,
                    "model": "xgboost",
                }
            )
        (output / "internal_final_candidates.json").write_text(
            json.dumps({"selected": selected}),
            encoding="utf-8",
        )
        return {"version": "test", "output": {"root": "output/tuning_v1"}}, best_paths

    def test_registry_is_frozen_and_has_38_unique_configs(self):
        config = load_config(ROOT)
        registry = load_registry(ROOT, config)
        resolved = resolved_registry(registry, config)
        self.assertEqual(len(registry), 38)
        self.assertEqual(registry["config_id"].nunique(), 38)
        self.assertEqual(
            registry["tuning_tier"].value_counts().to_dict(),
            {
                "A1_required_core": 15,
                "A2_internal_enhanced": 12,
                "B_external_sensitivity": 11,
            },
        )
        self.assertFalse(resolved["test_data_used"].any())

    def test_search_spaces_cover_all_models_and_can_sample(self):
        spaces = load_search_spaces(ROOT / "tuning" / "search_spaces.yaml")
        self.assertEqual(
            set(spaces["models"]),
            {"logistic", "catboost", "lightgbm", "xgboost", "lstm"},
        )
        for model in spaces["models"]:
            trial = optuna.trial.FixedTrial(
                {
                    name: (
                        rule["choices"][0]
                        if rule["type"] == "categorical"
                        else rule["low"]
                    )
                    for name, rule in spaces["models"][model].items()
                    if name != "fixed"
                }
            )
            sampled = sample_params(trial, model, spaces)
            self.assertTrue(sampled)

    def test_balanced_selector_applies_f2_and_pr_auc_floors(self):
        records = [
            {
                "run_id": "high-recall-unsafe",
                "status": "complete",
                "recall": 0.99,
                "f2": 0.50,
                "average_precision": 0.50,
            },
            {
                "run_id": "balanced",
                "status": "complete",
                "recall": 0.90,
                "f2": 0.96,
                "average_precision": 0.95,
            },
            {
                "run_id": "max-guardrails",
                "status": "complete",
                "recall": 0.85,
                "f2": 1.00,
                "average_precision": 1.00,
            },
        ]
        selected = select_balanced_trial(
            records,
            {
                "f2_relative_floor": 0.95,
                "average_precision_relative_floor": 0.90,
            },
        )
        self.assertEqual(selected["run_id"], "balanced")

    def test_balanced_selector_returns_none_when_guardrail_intersection_is_empty(self):
        records = [
            {
                "run_id": "max-f2",
                "status": "complete",
                "recall": 0.50,
                "f2": 1.00,
                "average_precision": 0.50,
            },
            {
                "run_id": "max-average-precision",
                "status": "complete",
                "recall": 0.60,
                "f2": 0.50,
                "average_precision": 1.00,
            },
        ]

        selected = select_balanced_trial(
            records,
            {
                "f2_relative_floor": 0.95,
                "average_precision_relative_floor": 0.90,
            },
        )

        self.assertIsNone(selected)

    def test_paired_result_promotes_only_with_non_harmful_guardrails(self):
        bootstrap = pd.DataFrame(
            [
                {"metric": "recall", "ci_low": 0.01, "ci_high": 0.10},
                {"metric": "f2", "ci_low": -0.01, "ci_high": 0.05},
                {"metric": "average_precision", "ci_low": -0.01, "ci_high": 0.03},
            ]
        )
        decision = classify_paired_result(
            {"recall": 0.05, "f2": 0.02, "average_precision": 0.01},
            bootstrap,
        )
        self.assertEqual(decision, "promote_external_candidate")

    def test_paired_result_retains_internal_when_average_precision_degrades(self):
        bootstrap = pd.DataFrame(
            [
                {"metric": "recall", "ci_low": -0.01, "ci_high": 0.10},
                {"metric": "f2", "ci_low": -0.01, "ci_high": 0.05},
                {"metric": "average_precision", "ci_low": -0.05, "ci_high": 0.01},
            ]
        )
        decision = classify_paired_result(
            {"recall": 0.05, "f2": 0.01, "average_precision": -0.01},
            bootstrap,
        )
        self.assertEqual(decision, "retain_internal_candidate")

    def test_final_candidate_lock_is_idempotent_and_freezes_three_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._write_final_candidate_fixture(root)

            first = build_final_candidate_lock(root, config)
            second = build_final_candidate_lock(root, config)

            self.assertEqual(first, second)
            self.assertEqual(
                {candidate["target"] for candidate in first["candidates"]},
                {"occurrence", "screening", "noncompliance"},
            )
            self.assertFalse(first["test_data_used_for_selection"])
            self.assertEqual(first["external_candidates_promoted"], 0)

    def test_final_candidate_lock_rejects_test_tainted_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, best_paths = self._write_final_candidate_fixture(root)
            best_path = best_paths["screening"]
            best = json.loads(best_path.read_text(encoding="utf-8"))
            best["test_data_used"] = True
            best_path.write_text(json.dumps(best), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "test"):
                build_final_candidate_lock(root, config)


if __name__ == "__main__":
    unittest.main()
