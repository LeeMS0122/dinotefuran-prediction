from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


MISSING_TOKEN = "__MISSING__"


@dataclass
class StaticPreprocessor:
    categorical_columns: list[str]
    numeric_columns: list[str]
    category_maps: dict[str, dict[str, int]]
    numeric_means: dict[str, float]
    numeric_scales: dict[str, float]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        categorical_columns: list[str],
        numeric_columns: list[str],
    ) -> "StaticPreprocessor":
        category_maps: dict[str, dict[str, int]] = {}
        for column in categorical_columns:
            values = frame[column].astype("string").fillna(MISSING_TOKEN)
            categories = sorted(values.unique().tolist())
            category_maps[column] = {str(value): index + 1 for index, value in enumerate(categories)}
        numeric_means: dict[str, float] = {}
        numeric_scales: dict[str, float] = {}
        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            mean = float(values.mean()) if values.notna().any() else 0.0
            scale = float(values.std(ddof=0)) if values.notna().any() else 1.0
            numeric_means[column] = mean
            numeric_scales[column] = scale if np.isfinite(scale) and scale > 0 else 1.0
        return cls(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            category_maps=category_maps,
            numeric_means=numeric_means,
            numeric_scales=numeric_scales,
        )

    @property
    def categorical_cardinalities(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_columns]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        categorical = np.zeros((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for position, column in enumerate(self.categorical_columns):
            values = frame[column].astype("string").fillna(MISSING_TOKEN)
            categorical[:, position] = values.map(self.category_maps[column]).fillna(0).to_numpy(dtype=np.int64)
        numeric = np.zeros((len(frame), len(self.numeric_columns)), dtype=np.float32)
        for position, column in enumerate(self.numeric_columns):
            values = pd.to_numeric(frame[column], errors="coerce").fillna(self.numeric_means[column])
            numeric[:, position] = (
                (values - self.numeric_means[column]) / self.numeric_scales[column]
            ).to_numpy(dtype=np.float32)
        return categorical, numeric

    def to_dict(self) -> dict[str, Any]:
        return {
            "categorical_columns": self.categorical_columns,
            "numeric_columns": self.numeric_columns,
            "category_maps": self.category_maps,
            "numeric_means": self.numeric_means,
            "numeric_scales": self.numeric_scales,
            "unknown_category_code": 0,
            "missing_token": MISSING_TOKEN,
        }


class HybridSequenceDataset(Dataset):
    def __init__(
        self,
        inspection_count: np.ndarray,
        active_mask: np.ndarray,
        categorical: np.ndarray,
        numeric: np.ndarray,
        target: np.ndarray,
        sample_weight: np.ndarray,
        rows: np.ndarray,
    ) -> None:
        self.inspection_count = inspection_count
        self.active_mask = active_mask
        self.categorical = categorical
        self.numeric = numeric
        self.target = target
        self.sample_weight = sample_weight
        self.rows = rows.astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[item])
        count = np.log1p(self.inspection_count[row]).astype(np.float32)
        active = self.active_mask[row].astype(np.float32)
        sequence = np.stack([count, active], axis=-1)
        return {
            "sequence": torch.from_numpy(sequence),
            "categorical": torch.from_numpy(self.categorical[row]),
            "numeric": torch.from_numpy(self.numeric[row]),
            "target": torch.tensor(float(self.target[row]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weight[row]), dtype=torch.float32),
            "array_row": torch.tensor(row, dtype=torch.int64),
        }


def load_aligned_target(
    sequence_dir: Path,
    feature_dir: Path,
    target_name: str,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    archive = np.load(sequence_dir / f"{target_name}_sequences_v1.npz")
    arrays = {name: archive[name] for name in archive.files}
    index = pd.read_parquet(sequence_dir / f"{target_name}_sequence_index_v1.parquet")
    features = pd.read_parquet(feature_dir / f"{target_name}_features_v1.parquet").reset_index(drop=True)
    if len(index) != len(features) or len(index) != len(arrays["target"]):
        raise ValueError("Sequence, index, and feature row counts differ")
    if not index["record_id"].astype(str).equals(features["record_id"].astype(str)):
        raise ValueError("Sequence index and feature record_id order differ")
    if not np.array_equal(index["target"].to_numpy(dtype=np.int8), arrays["target"]):
        raise ValueError("Sequence target and index target differ")
    return arrays, index, features


def make_loader(
    arrays: dict[str, np.ndarray],
    categorical: np.ndarray,
    numeric: np.ndarray,
    rows: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = HybridSequenceDataset(
        arrays["inspection_count"],
        arrays["active_mask"],
        categorical,
        numeric,
        arrays["target"],
        arrays["group_sample_weight"],
        rows,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )

