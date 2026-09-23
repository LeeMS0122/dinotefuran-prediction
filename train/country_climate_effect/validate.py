from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_modeling.pipeline import evaluate_scores, top_k_table


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "country_climate_effect_v1"


def main() -> None:
    metrics = pd.read_csv(OUTPUT / "validation_metrics.csv", encoding="utf-8-sig")
    predictions = pd.read_parquet(OUTPUT / "validation_predictions.parquet")
    deltas = pd.read_csv(OUTPUT / "climate_effect_deltas.csv", encoding="utf-8-sig")
    bootstrap = pd.read_csv(OUTPUT / "climate_effect_bootstrap_ci.csv", encoding="utf-8-sig")
    quality = json.loads((OUTPUT / "data_quality.json").read_text(encoding="utf-8"))
    checks: dict[str, object] = {}
    checks["metric_rows_60"] = len(metrics) == 60
    checks["targets_3"] = metrics["target"].nunique() == 3
    checks["cohorts_2"] = metrics["cohort"].nunique() == 2
    checks["models_5"] = metrics["model"].nunique() == 5
    checks["two_variants_per_group"] = bool(
        metrics.groupby(["target", "cohort", "model"])["variant"].nunique().eq(2).all()
    )
    checks["validation_only"] = set(metrics["split"]) == {"validation"}
    checks["delta_rows_30"] = len(deltas) == 30
    checks["bootstrap_rows_120"] = len(bootstrap) == 120
    checks["prediction_pairs_complete"] = bool(
        predictions.groupby(["target_name", "cohort", "model", "record_id"])["variant"]
        .nunique()
        .eq(2)
        .all()
    )
    checks["quality_row_preserved"] = all(item["row_count_preserved"] for item in quality)
    checks["quality_record_id_unique"] = all(
        item["comparison_record_id_duplicates"] == 0 for item in quality
    )
    checks["quality_no_split_cross"] = all(
        item["cross_split_duplicate_groups"] == 0 for item in quality
    )
    checks["quality_no_climate_missing"] = all(item["climate_missing_cells"] == 0 for item in quality)

    maximum_error = 0.0
    for row in metrics.itertuples(index=False):
        subset = predictions.loc[
            predictions["target_name"].eq(row.target)
            & predictions["cohort"].eq(row.cohort)
            & predictions["model"].eq(row.model)
            & predictions["variant"].eq(row.variant)
        ]
        # Use the full-precision threshold retained with the prediction parquet.
        # CSV parsing can move a score-equal threshold by one floating-point ULP.
        prediction_thresholds = subset["threshold"].unique()
        if len(prediction_thresholds) != 1:
            raise ValueError(
                f"threshold not unique: {row.target}/{row.cohort}/{row.model}/{row.variant}"
            )
        threshold = float(prediction_thresholds[0])
        observed = evaluate_scores(
            subset["target"].to_numpy(dtype=int),
            subset["score"].to_numpy(dtype=float),
            subset["group_sample_weight"].to_numpy(dtype=float),
            threshold,
            2.0,
        )
        top10 = top_k_table(
            subset["target"].to_numpy(dtype=int),
            subset["score"].to_numpy(dtype=float),
            subset["group_sample_weight"].to_numpy(dtype=float),
            [0.10],
        ).iloc[0]
        pairs = {
            "accuracy": row.accuracy,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1,
            "f2": row.f2,
            "average_precision": row.average_precision,
            "roc_auc": row.roc_auc,
        }
        for name, expected in pairs.items():
            maximum_error = max(maximum_error, abs(float(expected) - float(observed[name])))
        maximum_error = max(
            maximum_error,
            abs(float(row.top10_capture_rate) - float(top10["capture_rate"])),
        )
    checks["metric_recompute_max_abs_error"] = maximum_error
    checks["metrics_recompute_match"] = maximum_error < 1e-10
    metric_columns = ["accuracy", "precision", "recall", "f1", "f2", "average_precision", "roc_auc", "top10_capture_rate"]
    checks["metric_ranges_valid"] = bool(
        metrics[metric_columns].apply(lambda column: column.between(0, 1).all()).all()
    )
    checks["passed"] = all(
        value is True or (key == "metric_recompute_max_abs_error" and float(value) < 1e-10)
        for key, value in checks.items()
        if key != "passed"
    )
    (OUTPUT / "validation_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["passed"]:
        raise SystemExit("country climate effect validation failed")


if __name__ == "__main__":
    main()
