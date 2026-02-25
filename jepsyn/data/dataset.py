"""
PyTorch Dataset and collate utilities for spike-event window data.

Consumed by experiment scripts (e.g. multi_session.py) via load_and_prepare_data().
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Columns that must exist in the parquet before any processing.
REQUIRED_COLUMNS = [
    "session_id",
    "window_id",
    "window_start_ms",
    "window_end_ms",
    "events_units",
    "events_times_ms",
]


class SpikeWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping a DataFrame of spike-event windows.

    Each item yields:
        session_id (int)              : session identifier
        unit_ids   (LongTensor [E])   : 1-indexed contiguous unit index per spike (0 = PAD)
        time_ids   (LongTensor [E])   : floor(ms offset), clipped to [0, window_len_ms - 1]
        labels     (dict, optional)   : flattened stimulus metadata, only when include_labels=True

    Args:
        df               : DataFrame slice for this split.
        session_unit_map : {session_id: {raw_unit_id: contiguous_1indexed_idx}}.
                           Built from the full dataset so every session is covered.
        include_labels   : If True, parse stimulus metadata into a flat labels dict.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        session_unit_map: Dict[int, Dict[int, int]],
        include_labels: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.session_unit_map = session_unit_map
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sid = int(row["session_id"])
        unit_map = self.session_unit_map[sid]

        # Remap raw AllenSDK unit IDs → contiguous 1-indexed (0 reserved for PAD)
        unit_ids = torch.tensor(
            np.array([unit_map[u] for u in row["events_units"]], dtype=np.int64),
            dtype=torch.long,
        )

        # floor + clip: safer than round at window boundaries
        window_len_ms = int(row["window_end_ms"]) - int(row["window_start_ms"])
        time_ids = torch.tensor(
            np.floor(np.asarray(row["events_times_ms"], dtype=np.float32))
            .astype(np.int64)
            .clip(0, window_len_ms - 1),
            dtype=torch.long,
        )

        item = {
            "session_id": sid,
            "unit_ids": unit_ids,
            "time_ids": time_ids,
        }

        if self.include_labels:
            stimulus_events = row.get("stimulus") or []
            if stimulus_events:
                item["labels"] = {
                    "image_name": stimulus_events[0].get("image_name"),
                    "is_change": bool(stimulus_events[0].get("is_change", False)),
                    "stimulus_block": int(stimulus_events[0].get("stimulus_block", -1)),
                }
            else:
                item["labels"] = {
                    "image_name": None,
                    "is_change": False,
                    "stimulus_block": -1,
                }

        return item


def spike_collate_fn(batch: List[dict]) -> dict:
    """
    Collate variable-length spike sequences into padded batches.

    Pads unit_ids and time_ids with zeros to the longest sequence in the batch.
    attention_mask is True for real tokens and False for padding positions.

    Returns:
        session_ids    (LongTensor [B])
        unit_ids       (LongTensor [B, max_E])
        time_ids       (LongTensor [B, max_E])
        attention_mask (BoolTensor [B, max_E])
        labels         (list[dict], only present when include_labels=True)
    """
    session_ids = torch.tensor(
        [x["session_id"] for x in batch], dtype=torch.long
    )
    unit_ids_list = [x["unit_ids"] for x in batch]
    time_ids_list = [x["time_ids"] for x in batch]

    max_len = max((len(u) for u in unit_ids_list), default=0)
    B = len(batch)

    unit_ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    time_ids_padded = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, (u, t) in enumerate(zip(unit_ids_list, time_ids_list)):
        L = len(u)
        if L > 0:
            unit_ids_padded[i, :L] = u
            time_ids_padded[i, :L] = t
            attention_mask[i, :L] = True

    out = {
        "session_ids": session_ids,
        "unit_ids": unit_ids_padded,
        "time_ids": time_ids_padded,
        "attention_mask": attention_mask,
    }
    if "labels" in batch[0]:
        out["labels"] = [x["labels"] for x in batch]
    return out
