from pathlib import Path
import sys

import pandas as pd


PRE_MODELING = Path(__file__).resolve().parents[1] / "pre_modeling"
sys.path.insert(0, str(PRE_MODELING))

from training_design import (
    add_calendar_features,
    assign_group_split,
    feature_columns,
    load_policy,
    split_from_anchor_year,
    split_from_event_date,
)


def test_split_boundaries():
    assert split_from_anchor_year(2014) == "historical_excluded"
    assert split_from_anchor_year(2015) == "train"
    assert split_from_anchor_year(2024) == "train"
    assert split_from_anchor_year(2025) == "validation"
    assert split_from_event_date("2026-01-01") == "test"
    assert split_from_event_date("2026-06-30") == "test"
    assert split_from_event_date("2026-07-01") == "oot_monitor"


def test_group_uses_latest_date_and_never_crosses_split():
    df = pd.DataFrame({
        "record_id": ["r1", "r2", "r3"],
        "duplicate_group_id": ["g1", "g1", "g2"],
        "event_date": ["2024-01-01", "2026-02-01", "2025-03-01"],
    })
    out = assign_group_split(add_calendar_features(df))
    assert set(out.loc[out.duplicate_group_id == "g1", "split"]) == {"test"}
    assert out.groupby("duplicate_group_id")["split"].nunique().max() == 1
    assert out.loc[out.record_id == "r1", "split_moved_for_group"].iat[0] == 1


def test_calendar_features():
    out = add_calendar_features(pd.DataFrame({"event_date": ["2024-01-15", "2024-07-15"]}))
    assert list(out["event_season"]) == ["winter", "summer"]
    assert list(out["event_quarter"]) == ["Q1", "Q3"]


def test_feature_policy_has_no_direct_leakage():
    policy = load_policy(PRE_MODELING / "training_design_policy.yaml")
    categorical, numeric = feature_columns(policy)
    forbidden = tuple(policy["feature_sets"]["forbidden_patterns"])
    assert not [c for c in categorical + numeric if c.startswith(forbidden)]
