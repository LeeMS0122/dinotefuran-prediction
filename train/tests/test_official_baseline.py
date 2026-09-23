from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_official_config_locks_test_and_uses_core_only() -> None:
    config = yaml.safe_load(
        (ROOT / "official_baseline" / "config.yaml").read_text(encoding="utf-8")
    )
    assert config["features"]["feature_set"] == "core"
    assert config["features"]["external_variables_used"] is False
    assert config["test_lock"] == {
        "score_test": False,
        "save_test_predictions": False,
        "select_model_on_test": False,
    }


def test_official_config_has_five_models_and_selected_lstm_windows() -> None:
    config = yaml.safe_load(
        (ROOT / "official_baseline" / "config.yaml").read_text(encoding="utf-8")
    )
    assert [item["display_name"] for item in config["models"]] == [
        "Logistic Regression",
        "CatBoost",
        "LightGBM",
        "XGBoost",
        "LSTM",
    ]
    assert config["lstm"]["selected_windows"] == {
        "occurrence": 90,
        "screening": 30,
        "noncompliance": 30,
    }
