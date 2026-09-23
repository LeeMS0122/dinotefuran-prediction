from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output_dir = ROOT / "output" / "route_climate_effect_v1"
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(output_dir / "validation_metrics.csv")
    predictions = pd.read_parquet(output_dir / "validation_predictions.parquet")
    selection = pd.read_csv(output_dir / "tuning_input_selection.csv")
    expected_models = 5 if not manifest["skip_lstm"] else 4
    expected_metric_rows = 3 * expected_models * 2
    if len(metrics) != expected_metric_rows:
        raise ValueError(f"성능표 행 수 오류: {len(metrics)} != {expected_metric_rows}")
    if len(selection) != 3 * expected_models:
        raise ValueError("튜닝 입력군 선정표 행 수 오류")
    if set(metrics["split"]) != {"validation"}:
        raise ValueError("validation 이외 분할 성능 포함")
    if manifest["test_data_used"]:
        raise ValueError("테스트 데이터 사용 플래그 오류")
    for (target, model), group in predictions.groupby(["target_name", "model"], observed=True):
        ids = group.groupby("variant", observed=True)["record_id"].apply(lambda s: set(s.astype(str)))
        if len(ids) != 2 or ids.iloc[0] != ids.iloc[1]:
            raise ValueError(f"{target}/{model}: paired record_id 불일치")
    if metrics[["accuracy", "precision", "recall", "f1", "f2", "average_precision"]].isna().any().any():
        raise ValueError("핵심 성능지표 결측")
    required_files = [
        output_dir / "route_climate_effect_deltas.csv",
        output_dir / "route_climate_effect_bootstrap_ci.csv",
        output_dir / "confusion_matrices_raw_and_row_pct.csv",
        output_dir / "operational_metrics.csv",
        output_dir / "population_summary.csv",
        ROOT / "Route_Climate_Effect_Comparison.ipynb",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    result = {
        "status": "PASS",
        "metric_rows": int(len(metrics)),
        "selection_rows": int(len(selection)),
        "prediction_rows": int(len(predictions)),
        "test_data_used": False,
        "paired_ids_match": True,
    }
    (output_dir / "validation_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

