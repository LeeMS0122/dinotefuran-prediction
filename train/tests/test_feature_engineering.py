from pathlib import Path
import sys

import pandas as pd


TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))

from feature_engineering.engineering import (
    NAME_RULES,
    classify_text,
    clean_product_name,
    load_config,
    transform_features,
    fit_food_mappings,
)


CONFIG = load_config(TRAIN / "feature_engineering" / "reference_config.yaml")


def _fixture() -> pd.DataFrame:
    return pd.DataFrame({
        "record_id": ["1", "2", "3", "4"],
        "split": ["train", "train", "train", "test"],
        "product_code": ["A1", "A1", "A1", "A1"],
        "product_name_std": ["사과(홍로)", "사과 1kg", "사과", "사과(부사)"],
        "product_group_raw": ["과실류", "과실류", "과실류", None],
        "collection_province": ["경북", None, None, "경상북도"],
        "cultivation_province": [None, "강원도", None, None],
        "production_country_code": ["KR", None, None, None],
        "origin_country_code": [None, "CN", None, None],
        "origin_country_name": [None, None, "베트남", None],
        "import_country_code": [None, None, None, "US"],
        "event_date": pd.to_datetime(["2020-01-01", "2021-04-01", "2022-07-01", "2026-03-01"]),
        "event_month": ["1", "4", "7", "3"],
        "event_quarter": ["Q1", "Q2", "Q3", "Q1"],
        "event_season": ["winter", "spring", "summer", "spring"],
        "event_month_sin": [0.5, 0.86, -0.5, 1.0],
        "event_month_cos": [0.86, -0.5, -0.86, 0.0],
        "work_type": [None] * 4,
        "collection_stage": [None] * 4,
        "cultivation_method_1": [None] * 4,
        "facility_type": [None] * 4,
        "target": [0, 1, 0, 1],
    })


def test_product_name_cleaning_and_composite_flag():
    key, composite = clean_product_name(pd.Series(["사과(홍로) 1kg", "사과, 배"]))
    assert key.iloc[0] == "사과"
    assert list(composite) == [0, 1]


def test_train_only_mapping_applies_to_test():
    data = _fixture()
    mappings = fit_food_mappings(data.loc[data["split"] == "train"], CONFIG)
    out = transform_features(data, mappings, CONFIG)
    assert out.loc[out["split"] == "test", "food_group_l1_std"].iat[0] == "과실류"
    assert out.loc[out["split"] == "test", "food_group_mapping_source"].iat[0] == "train_code_map"


def test_country_and_province_normalization():
    data = _fixture()
    mappings = fit_food_mappings(data.loc[data["split"] == "train"], CONFIG)
    out = transform_features(data, mappings, CONFIG)
    assert list(out["origin_iso2_std"]) == ["KR", "CN", "VN", "US"]
    assert out.loc[0, "domestic_province_std"] == "경상북도"
    assert out.loc[1, "domestic_province_std"] == "강원특별자치도"


def test_time_features_do_not_use_target():
    data = _fixture()
    mappings = fit_food_mappings(data.loc[data["split"] == "train"], CONFIG)
    out1 = transform_features(data, mappings, CONFIG)
    data["target"] = 1 - data["target"]
    out2 = transform_features(data, mappings, CONFIG)
    cols = ["event_dayofyear_sin", "event_dayofyear_cos", "event_week_sin", "event_week_cos"]
    pd.testing.assert_frame_equal(out1[cols], out2[cols])


def test_keyword_classifier():
    assert classify_text("방울토마토", NAME_RULES) == "채소류"
    assert classify_text("홍로 사과", NAME_RULES) == "과실류"

