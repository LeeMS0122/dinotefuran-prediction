from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class DinotefuranSequenceDataset(Dataset):
    """Two-channel input: log1p inspection count and active-day mask."""

    def __init__(self, npz_path: Path | str, rows: np.ndarray | None = None):
        archive = np.load(npz_path)
        self.inspection_count = archive["inspection_count"]
        self.active_mask = archive["active_mask"]
        self.target = archive["target"]
        self.sample_weight = archive["group_sample_weight"]
        self.key_valid = archive["key_valid"]
        self.date_valid = archive["date_valid"]
        self.rows = np.arange(len(self.target), dtype=np.int64) if rows is None else rows.astype(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        row = int(self.rows[item])
        count = np.log1p(self.inspection_count[row]).astype(np.float32)
        active = self.active_mask[row].astype(np.float32)
        sequence = np.stack([count, active], axis=-1)
        return {
            "sequence": torch.from_numpy(sequence),
            "target": torch.tensor(float(self.target[row]), dtype=torch.float32),
            "sample_weight": torch.tensor(float(self.sample_weight[row]), dtype=torch.float32),
            "key_valid": torch.tensor(float(self.key_valid[row]), dtype=torch.float32),
            "date_valid": torch.tensor(float(self.date_valid[row]), dtype=torch.float32),
            "array_row": torch.tensor(row, dtype=torch.int64),
        }


def make_dataloader(
    npz_path: Path | str,
    rows: np.ndarray | None = None,
    batch_size: int = 256,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = DinotefuranSequenceDataset(npz_path, rows=rows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

