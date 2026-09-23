from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_variables.audit_weather_scope_v2 import master_source_profile
from external_variables.pilot_weather_join import build_rolling_weather
from tuning_v2_weather.audit import build_registry, load_config


class WeatherScopeAuditTests(unittest.TestCase):
    def test_source_profile_counts_ready_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "source_system": ["A", "A", "B"],
                "event_date": pd.to_datetime(["2025-01-01", None, "2025-01-02"]),
                "collection_province": ["경기", None, "서울"],
                "province_join_key": ["경기도", pd.NA, "서울특별시"],
                "date_basis": ["수거일", "수거일", "의뢰일"],
                "strict_collection_weather_ready": [True, False, False],
                "origin_country_name": [None, None, "미국"],
                "result_raw": [None, None, None],
                "result_value_numeric_enriched": [None, None, None],
                "standard_numeric_enriched": [None, None, None],
                "label_occurrence_v1": [1, None, 0],
                "label_screening_10pct_v1": [1, None, None],
                "label_noncompliance_v1": [0, None, 0],
            }
        )
        result = master_source_profile(frame).set_index("source_system")
        self.assertEqual(result.loc["A", "n_rows"], 2)
        self.assertEqual(result.loc["A", "strict_collection_weather_ready"], 1)
        self.assertEqual(result.loc["B", "origin_country_known"], 1)

    def test_rolling_weather_persists_strictly_prior_observation_date(self) -> None:
        daily = pd.DataFrame(
            {
                "province_join_key": ["경기도", "경기도"],
                "meas_date": pd.to_datetime(["2025-01-01", "2025-01-03"]),
                "weather_day_present": [1.0, 1.0],
                "station_count": [1, 1],
                "temp": [1.0, 3.0],
                "hghst_artmp": [2.0, 4.0],
                "lowst_artmp": [0.0, 2.0],
                "hum": [50.0, 60.0],
                "wind": [1.0, 1.0],
                "max_wind": [2.0, 2.0],
                "rn": [0.0, 1.0],
                "srqty": [5.0, 6.0],
                "gr_temp": [1.0, 3.0],
                "soil_temp": [1.0, 3.0],
                "soil_wt": [10.0, 11.0],
            }
        )
        result = build_rolling_weather(
            daily,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-04"),
        )
        observed = result.set_index("event_date")["wx_max_observed_date_14d"]
        self.assertTrue(pd.isna(observed.loc[pd.Timestamp("2025-01-01")]))
        self.assertEqual(
            observed.loc[pd.Timestamp("2025-01-03")],
            pd.Timestamp("2025-01-01"),
        )
        self.assertEqual(
            observed.loc[pd.Timestamp("2025-01-04")],
            pd.Timestamp("2025-01-03"),
        )
        self.assertTrue(
            (
                result["wx_max_observed_date_14d"].dropna()
                < result.loc[
                    result["wx_max_observed_date_14d"].notna(),
                    "event_date",
                ]
            ).all()
        )

    def test_weather_v2_registry_has_twelve_unique_challengers(self) -> None:
        registry = build_registry(load_config(ROOT))
        self.assertEqual(len(registry), 12)
        self.assertEqual(registry["config_id"].nunique(), 12)
        self.assertEqual(set(registry["model"]), {"catboost", "lightgbm", "xgboost"})
        self.assertEqual(set(registry["window_days"]), {14, 30, 60, 90})
        self.assertFalse(registry["test_data_used"].any())


if __name__ == "__main__":
    unittest.main()
