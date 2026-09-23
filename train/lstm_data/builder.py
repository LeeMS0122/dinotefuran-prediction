from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lstm_feasibility.analysis import normalize_key, prepare_sequence_frame


TARGET_LABELS = {
    "occurrence": "잔류 존재",
    "screening": "MRL 10% 관심농도",
    "noncompliance": "기준 부적합",
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_daily_history(history: pd.DataFrame, key_column: str) -> pd.DataFrame:
    dates = pd.to_datetime(history["event_date"], errors="coerce").dt.normalize()
    _, key_valid = normalize_key(history, [key_column])
    valid = key_valid & dates.notna()
    daily = pd.DataFrame({
        "sequence_key": history.loc[valid, key_column].astype("string"),
        "event_date": dates.loc[valid],
    })
    daily = (
        daily.groupby(["sequence_key", "event_date"], observed=True)
        .size()
        .rename("inspection_count")
        .reset_index()
        .sort_values(["sequence_key", "event_date"])
        .reset_index(drop=True)
    )
    daily["inspection_count"] = daily["inspection_count"].astype("int32")
    return daily


def build_sequence_arrays(
    daily_history: pd.DataFrame,
    query: pd.DataFrame,
    key_column: str,
    window_days: int,
    chunk_size: int = 50_000,
) -> dict[str, np.ndarray]:
    """Return chronological daily counts for [date-window, date)."""
    query = query.reset_index(drop=True)
    query_dates = pd.to_datetime(query["event_date"], errors="coerce").dt.normalize()
    _, key_valid = normalize_key(query, [key_column])
    date_valid = query_dates.notna()
    usable = key_valid & date_valid
    counts = np.zeros((len(query), int(window_days)), dtype=np.int32)
    offsets = np.arange(int(window_days), 0, -1, dtype=np.int64)

    history_groups = {
        str(group_key): (
            group["event_date"].to_numpy(dtype="datetime64[D]").astype(np.int64),
            group["inspection_count"].to_numpy(dtype=np.int32),
        )
        for group_key, group in daily_history.groupby("sequence_key", sort=False, observed=True)
    }

    work = pd.DataFrame({
        "sequence_key": query[key_column].astype("string"),
        "event_date": query_dates,
        "usable": usable,
    })
    for group_key, positions in work.loc[usable].groupby("sequence_key", sort=False, observed=True).groups.items():
        history_values = history_groups.get(str(group_key))
        if history_values is None:
            continue
        history_dates, history_counts = history_values
        position_array = np.asarray(list(positions), dtype=np.int64)
        for start in range(0, len(position_array), chunk_size):
            current_positions = position_array[start:start + chunk_size]
            query_day = work.loc[current_positions, "event_date"].to_numpy(
                dtype="datetime64[D]"
            ).astype(np.int64)
            desired_days = query_day[:, None] - offsets[None, :]
            found_positions = np.searchsorted(history_dates, desired_days, side="left")
            in_range = found_positions < len(history_dates)
            safe_positions = np.minimum(found_positions, max(len(history_dates) - 1, 0))
            exact = in_range & (history_dates[safe_positions] == desired_days)
            block = np.zeros_like(desired_days, dtype=np.int32)
            block[exact] = history_counts[safe_positions[exact]]
            counts[current_positions] = block

    active_mask = counts.gt(0) if isinstance(counts, pd.DataFrame) else counts > 0
    return {
        "inspection_count": counts,
        "active_mask": active_mask.astype(np.uint8),
        "key_valid": key_valid.to_numpy(dtype=np.uint8),
        "date_valid": date_valid.to_numpy(dtype=np.uint8),
    }


def build_sequence_index(
    frame: pd.DataFrame,
    sequence_frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    minimum_active_days: int,
) -> pd.DataFrame:
    active_days = arrays["active_mask"].sum(axis=1).astype(np.int16)
    inspection_count = arrays["inspection_count"].sum(axis=1).astype(np.int32)
    index = pd.DataFrame({
        "array_row": np.arange(len(frame), dtype=np.int32),
        "record_id": frame["record_id"].astype("string"),
        "duplicate_group_id": frame["duplicate_group_id"].astype("string"),
        "event_date": pd.to_datetime(frame["event_date"], errors="coerce"),
        "split": frame["split"].astype("string"),
        "target": pd.to_numeric(frame["target"], errors="coerce").astype("int8"),
        "group_sample_weight": pd.to_numeric(
            frame["group_sample_weight"], errors="coerce"
        ).fillna(1.0).astype("float32"),
        "sequence_key": sequence_frame["sequence_food_group_l2"].astype("string"),
        "key_valid": arrays["key_valid"].astype(bool),
        "date_valid": arrays["date_valid"].astype(bool),
        "history_active_days": active_days,
        "history_inspection_count": inspection_count,
        "has_any_history": active_days > 0,
        "has_min3_history": active_days >= int(minimum_active_days),
    })
    return index


def summarize_index(index: pd.DataFrame, target_name: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, part in index.groupby("split", observed=True):
        n_rows = len(part)
        rows.append({
            "target": target_name,
            "target_label": TARGET_LABELS[target_name],
            "split": split,
            "n_rows": int(n_rows),
            "n_positive": int(part["target"].sum()),
            "positive_rate_pct": float(part["target"].mean() * 100),
            "n_key_valid": int(part["key_valid"].sum()),
            "key_valid_pct": float(part["key_valid"].mean() * 100),
            "n_any_history": int(part["has_any_history"].sum()),
            "any_history_pct": float(part["has_any_history"].mean() * 100),
            "n_min3_history": int(part["has_min3_history"].sum()),
            "min3_history_pct": float(part["has_min3_history"].mean() * 100),
            "median_active_days": float(part["history_active_days"].median()),
            "median_inspection_count": float(part["history_inspection_count"].median()),
            "n_excluded_from_storage": 0,
        })
    return pd.DataFrame(rows)


def validate_target_artifacts(
    frame: pd.DataFrame,
    index: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    window_days: int,
) -> dict[str, Any]:
    n_rows = len(frame)
    checks = {
        "row_count_matches": len(index) == n_rows,
        "sequence_shape_matches": arrays["inspection_count"].shape == (n_rows, window_days),
        "active_mask_shape_matches": arrays["active_mask"].shape == (n_rows, window_days),
        "nonnegative_counts": bool((arrays["inspection_count"] >= 0).all()),
        "mask_matches_counts": bool(
            np.array_equal(arrays["active_mask"], (arrays["inspection_count"] > 0).astype(np.uint8))
        ),
        "target_binary": bool(index["target"].isin([0, 1]).all()),
        "record_id_unique": bool(index["record_id"].is_unique),
        "array_row_contiguous": bool(np.array_equal(index["array_row"], np.arange(n_rows))),
        "all_rows_retained": len(index) == n_rows,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"LSTM sequence artifact validation failed: {failed}")
    return checks


def save_target_artifacts(
    output_dir: Path,
    target_name: str,
    index: pd.DataFrame,
    arrays: dict[str, np.ndarray],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / f"{target_name}_sequences_v1.npz"
    index_path = output_dir / f"{target_name}_sequence_index_v1.parquet"
    np.savez_compressed(
        array_path,
        inspection_count=arrays["inspection_count"],
        active_mask=arrays["active_mask"],
        target=index["target"].to_numpy(dtype=np.int8),
        group_sample_weight=index["group_sample_weight"].to_numpy(dtype=np.float32),
        key_valid=arrays["key_valid"],
        date_valid=arrays["date_valid"],
    )
    index.to_parquet(index_path, index=False)
    return {"array": array_path, "index": index_path}

