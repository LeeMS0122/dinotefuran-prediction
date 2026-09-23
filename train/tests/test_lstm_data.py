from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


TRAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN))

from lstm_data.builder import build_sequence_arrays, build_sequence_index
from lstm_data.dataset import DinotefuranSequenceDataset, make_dataloader


class LstmDataTests(unittest.TestCase):
    def test_sequence_is_oldest_to_recent_and_excludes_same_day(self):
        daily = pd.DataFrame({
            "sequence_key": ["A", "A", "A"],
            "event_date": pd.to_datetime(["2025-01-01", "2025-01-09", "2025-01-10"]),
            "inspection_count": [2, 3, 99],
        })
        query = pd.DataFrame({
            "event_date": pd.to_datetime(["2025-01-10"]),
            "sequence_food_group_l2": ["A"],
        })
        arrays = build_sequence_arrays(daily, query, "sequence_food_group_l2", 3)
        self.assertEqual(arrays["inspection_count"].tolist(), [[0, 0, 3]])
        self.assertEqual(arrays["active_mask"].tolist(), [[0, 0, 1]])

    def test_window_boundary_is_included_and_future_is_excluded(self):
        daily = pd.DataFrame({
            "sequence_key": ["A", "A"],
            "event_date": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "inspection_count": [4, 7],
        })
        query = pd.DataFrame({
            "event_date": pd.to_datetime(["2025-01-04"]),
            "sequence_food_group_l2": ["A"],
        })
        arrays = build_sequence_arrays(daily, query, "sequence_food_group_l2", 3)
        self.assertEqual(arrays["inspection_count"].tolist(), [[4, 0, 0]])

    def test_invalid_key_is_retained_as_zero_sequence(self):
        frame = pd.DataFrame({
            "record_id": ["r1"],
            "duplicate_group_id": ["g1"],
            "event_date": pd.to_datetime(["2025-01-04"]),
            "split": ["validation"],
            "target": [1],
            "group_sample_weight": [1.0],
        })
        sequence = pd.DataFrame({
            "event_date": frame["event_date"],
            "sequence_food_group_l2": ["미분류"],
        })
        daily = pd.DataFrame(columns=["sequence_key", "event_date", "inspection_count"])
        arrays = build_sequence_arrays(daily, sequence, "sequence_food_group_l2", 3)
        index = build_sequence_index(frame, sequence, arrays, 3)
        self.assertEqual(len(index), 1)
        self.assertFalse(index.loc[0, "key_valid"])
        self.assertEqual(arrays["inspection_count"].sum(), 0)

    def test_dataset_builds_two_channels_and_dataloader_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npz"
            np.savez_compressed(
                path,
                inspection_count=np.array([[0, 3, 1], [2, 0, 0]], dtype=np.int32),
                active_mask=np.array([[0, 1, 1], [1, 0, 0]], dtype=np.uint8),
                target=np.array([1, 0], dtype=np.int8),
                group_sample_weight=np.array([1.0, 0.5], dtype=np.float32),
                key_valid=np.array([1, 1], dtype=np.uint8),
                date_valid=np.array([1, 1], dtype=np.uint8),
            )
            dataset = DinotefuranSequenceDataset(path)
            sample = dataset[0]
            self.assertEqual(tuple(sample["sequence"].shape), (3, 2))
            self.assertAlmostEqual(float(sample["sequence"][1, 0]), float(np.log1p(3)), places=6)
            loader = make_dataloader(path, batch_size=2)
            batch = next(iter(loader))
            self.assertEqual(tuple(batch["sequence"].shape), (2, 3, 2))


if __name__ == "__main__":
    unittest.main()
