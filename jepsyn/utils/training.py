"""
Training utilities shared across experiment scripts.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from jepsyn.data import REQUIRED_COLUMNS, SpikeWindowDataset, spike_collate_fn


def create_context_mask(attn_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Randomly hide mask_ratio of real tokens per sample for the context encoder.

    Only tokens that are True in attn_mask (real spike events) are eligible.
    PAD tokens (False) are always left as False in the output.

    Args:
        attn_mask  : [B, E] bool — True = real token.
        mask_ratio : Fraction of real tokens to hide from the context encoder.

    Returns:
        ctx_mask : [B, E] bool — attn_mask with some real tokens hidden (set to False).
    """
    B, E = attn_mask.shape
    rand = torch.rand(B, E, device=attn_mask.device)
    rand[~attn_mask] = 2.0  # PAD: score > 1 so never selected for masking

    n_real = attn_mask.float().sum(dim=1)          # [B]
    n_to_mask = (n_real * mask_ratio).long()        # [B]

    sorted_rand, _ = rand.sort(dim=1)
    idx = n_to_mask.clamp(min=0, max=E - 1).unsqueeze(1)
    threshold = sorted_rand.gather(1, idx)          # [B, 1]

    ctx_mask = attn_mask.clone()
    ctx_mask[rand < threshold] = False
    return ctx_mask


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """
    In-place EMA update: target = momentum * target + (1 - momentum) * online.

    Called after every optimizer step. The target encoder is never touched by
    the optimizer — only updated here.
    """
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


def apply_unit_dropout(
    unit_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_ratio: float = 0.3,
    min_units: int = 30,
) -> torch.Tensor:
    """
    POYO-style unit dropout: randomly mask out all spikes from a subset of units
    per sample, forcing the model to be robust to missing neurons.

    Only applied during training (caller's responsibility to skip at eval time).

    Args:
        unit_ids      : [B, E] long — 1-indexed unit IDs (0 = PAD).
        attn_mask     : [B, E] bool — True = real token, False = PAD.
        dropout_ratio : Fraction of unique units to drop per sample.
        min_units     : Minimum number of units guaranteed to remain active.

    Returns:
        new_mask : [B, E] bool — attn_mask with dropped-unit spikes set to False.
    """
    if dropout_ratio <= 0.0:
        return attn_mask

    B        = unit_ids.shape[0]
    new_mask = attn_mask.clone()

    for i in range(B):
        real_ids     = unit_ids[i][attn_mask[i]]   # non-PAD unit IDs for this sample
        unique_units = real_ids.unique()
        n_units      = unique_units.shape[0]

        n_keep = max(min_units, int(n_units * (1.0 - dropout_ratio)))
        n_keep = min(n_keep, n_units)  # can't keep more than we have

        perm       = torch.randperm(n_units, device=unit_ids.device)
        keep_units = unique_units[perm[:n_keep]]

        # Zero out spikes from dropped units (those not in keep_units).
        is_dropped        = ~torch.isin(unit_ids[i], keep_units)
        new_mask[i]       = attn_mask[i] & ~is_dropped

    return new_mask


def load_and_prepare_data(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, Dict[int, int]], List[int]]:
    """
    Load the spike-window parquet, validate it, split by session, and return DataLoaders.

    The split is done at the **session level**: entire sessions are held out for
    val and test — no windows from test sessions appear in train or val.
    test_session_ids contains sessions the encoder has never seen during training;
    identify_units() is the only adaptation allowed on those sessions before probing.

    Each DataLoader yields batches from spike_collate_fn:
        session_ids    LongTensor [B]
        unit_ids       LongTensor [B, max_E]  — 1-indexed contiguous unit idx (0 = PAD)
        time_ids       LongTensor [B, max_E]  — floor(ms offset), clipped to window
        attention_mask BoolTensor [B, max_E]  — True = real token, False = padding
        labels         list[dict]             — test loader only, flattened stimulus fields

    Args:
        config: Validated configuration dict (from verify_config).

    Returns:
        (train_loader, val_loader, test_loader, session_unit_maps, test_session_ids)
        session_unit_maps: {session_id: {raw_unit_id: 1-indexed contiguous idx}}
            Needed by the training function to size per-session embedding tables.
    """
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError("data_path not found in configuration")

    dataset = pd.read_parquet(data_path, engine="pyarrow")

    # Column validation
    missing = [c for c in REQUIRED_COLUMNS if c not in dataset.columns]
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")

    print("Validating dataset integrity...")

    # No duplicate window_ids with conflicting timestamps
    dup = dataset.groupby("window_id").agg(
        {"window_start_ms": "nunique", "window_end_ms": "nunique"}
    )
    conflicts = dup[(dup["window_start_ms"] > 1) | (dup["window_end_ms"] > 1)]
    if not conflicts.empty:
        raise ValueError(
            f"Found {len(conflicts)} window_ids with conflicting timestamps: "
            f"{conflicts.index.tolist()[:5]}"
        )

    # events_units and events_times_ms must have equal length per row
    mismatches = dataset[
        dataset["events_units"].apply(len) != dataset["events_times_ms"].apply(len)
    ]
    if not mismatches.empty:
        raise ValueError(
            f"Found {len(mismatches)} rows where events_units and events_times_ms "
            f"have different lengths (window_ids: {mismatches['window_id'].tolist()[:5]})"
        )

    print("Passed basic validation checks.")

    # Build per-session unit maps from the full dataset before splitting.
    # Maps raw AllenSDK unit IDs → 1-indexed contiguous indices; 0 is reserved for PAD.
    session_unit_maps: Dict[int, Dict[int, int]] = {}
    for sid, grp in dataset.groupby("session_id"):
        all_units = sorted({int(u) for arr in grp["events_units"] for u in arr})
        session_unit_maps[int(sid)] = {
            raw: idx + 1 for idx, raw in enumerate(all_units)
        }
    print(
        f"Built unit maps for {len(session_unit_maps)} sessions "
        f"(sizes: {[len(m) for m in session_unit_maps.values()]})"
    )

    # Session-level train / val / test split.
    # Entire sessions are assigned to exactly one partition — no session appears
    # in more than one of {train, val, test}.
    data_cfg = config.get("data", {})
    train_size = data_cfg.get("train_split", 0.7)
    val_size = data_cfg.get("val_split", 0.15)
    test_size = data_cfg.get("test_split", 0.15)
    random_state = data_cfg.get("random_state", 42)

    unique_sessions = dataset["session_id"].unique()
    train_val_sessions, test_sessions = train_test_split(
        unique_sessions, test_size=test_size, random_state=random_state
    )
    train_sessions, val_sessions = train_test_split(
        train_val_sessions,
        test_size=val_size / (train_size + val_size),
        random_state=random_state,
    )

    train_df = dataset[dataset["session_id"].isin(train_sessions)]
    val_df = dataset[dataset["session_id"].isin(val_sessions)]
    test_df = dataset[dataset["session_id"].isin(test_sessions)]

    print(f"Train: {len(train_df)} windows ({len(train_sessions)} sessions)")
    print(f"Val:   {len(val_df)} windows ({len(val_sessions)} sessions)")
    print(f"Test:  {len(test_df)} windows ({len(test_sessions)} sessions)")

    batch_size = config.get("training_config", {}).get("batch_size", 32)
    has_stimulus = "stimulus" in dataset.columns

    train_loader = DataLoader(
        SpikeWindowDataset(train_df, session_unit_maps),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=spike_collate_fn,
    )
    val_loader = DataLoader(
        SpikeWindowDataset(val_df, session_unit_maps),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=spike_collate_fn,
    )
    test_loader = DataLoader(
        SpikeWindowDataset(test_df, session_unit_maps, include_labels=has_stimulus),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=spike_collate_fn,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        session_unit_maps,
        sorted(int(s) for s in test_sessions.tolist()),
    )
