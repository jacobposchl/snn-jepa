"""
Script to create a preprocessed, windowed dataset given a configuration.

Output is a single CSV with one row per temporal window, matching the
expected format in `experiments/multi_session/multi_session.py`.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from jepsyn.data.data_handler import VBNDataHandler
from jepsyn.data.preprocess import NeuropixelsPreprocessor
from jepsyn.utils.binning import bin_trial_aligned


def _load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config and normalize dataset settings."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    if "data_path" not in raw or not raw["data_path"]:
        raise ValueError("Config must define a non-empty 'data_path' for the output CSV.")

    dataset_cfg = raw.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Config must contain a 'dataset_config' section.")

    cache_dir = dataset_cfg.get("cache_dir")
    if not cache_dir:
        raise ValueError("dataset_config.cache_dir is required.")

    session_ids = dataset_cfg.get("session_ids")
    if not session_ids or not isinstance(session_ids, list):
        raise ValueError(
            "dataset_config.session_ids must be a non-empty list of ecephys session IDs."
        )

    # Optional brain area filter
    brain_areas = dataset_cfg.get("brain_areas") or []
    if brain_areas and not isinstance(brain_areas, list):
        raise ValueError("dataset_config.brain_areas must be a list if provided.")

    # Quality thresholds with defaults
    quality_cfg = dataset_cfg.get("quality") or {}
    min_snr = float(quality_cfg.get("min_snr", 1.0))
    min_firing_rate = float(quality_cfg.get("min_firing_rate", 0.1))
    max_isi_violations = float(quality_cfg.get("max_isi_violations", 1.0))

    # Windowing parameters
    windowing_cfg = dataset_cfg.get("windowing") or {}
    bin_size_ms = float(windowing_cfg.get("bin_size_ms", 10.0))
    window_size_ms = float(windowing_cfg.get("window_size_ms", 400.0))

    if bin_size_ms <= 0 or window_size_ms <= 0:
        raise ValueError("windowing.bin_size_ms and windowing.window_size_ms must be > 0.")

    # For now, we generate one event-aligned window per image-change event.
    # stride_ms is reserved for future sliding-window support.
    stride_ms = float(windowing_cfg.get("stride_ms", window_size_ms))

    if window_size_ms % bin_size_ms != 0:
        raise ValueError(
            "windowing.window_size_ms must be an integer multiple of windowing.bin_size_ms."
        )

    return {
        "data_path": Path(raw["data_path"]),
        "cache_dir": Path(cache_dir),
        "session_ids": [int(s) for s in session_ids],
        "brain_areas": brain_areas,
        "quality": {
            "min_snr": min_snr,
            "min_firing_rate": min_firing_rate,
            "max_isi_violations": max_isi_violations,
        },
        "windowing": {
            "bin_size_ms": bin_size_ms,
            "window_size_ms": window_size_ms,
            "stride_ms": stride_ms,
        },
    }


def _build_session_windows(
    handler: VBNDataHandler,
    session_id: int,
    cfg: Dict[str, Any],
    next_window_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Create window rows for a single session.

    Each window is aligned to an image-change event in the active behavior block.
    """
    print(f"\nProcessing session {session_id}...")

    try:
        session = handler.load_session(session_id)
    except Exception as e:
        print(f"  Skipping session {session_id}: failed to load ({e})")
        return [], next_window_id

    # Preprocess units and spikes
    preproc = (
        NeuropixelsPreprocessor(session)
        .validate_integrity()
        .clean()
        .filter_units(
            min_snr=cfg["quality"]["min_snr"],
            min_firing_rate=cfg["quality"]["min_firing_rate"],
            max_isi_violations=cfg["quality"]["max_isi_violations"],
            brain_areas=cfg["brain_areas"] or None,
            handler=handler,
        )
    )
    processed = preproc.get()
    units_df = processed["units"]
    good_unit_ids = units_df.index.tolist()

    if not good_unit_ids:
        print(f"  No units passed quality/area filters in session {session_id}; skipping.")
        return [], next_window_id

    # Get change times from active block only (as in get_or_create_dataset)
    stim = session.stimulus_presentations
    active_stim = stim[stim["active"] == True].copy()
    change_events = active_stim[active_stim["is_change"] == True].copy()

    if len(change_events) == 0:
        print(f"  No active image-change events in session {session_id}; skipping.")
        return [], next_window_id

    change_times = change_events["start_time"].values  # seconds

    bin_size_ms = cfg["windowing"]["bin_size_ms"]
    window_size_ms = cfg["windowing"]["window_size_ms"]

    # Bin spikes around each change event: [0, window_size_ms]
    trial_aligned = bin_trial_aligned(
        spike_times_dict=session.spike_times,
        event_times=change_times,
        unit_ids=good_unit_ids,
        bin_size_ms=bin_size_ms,
        pre_time_ms=0.0,
        post_time_ms=window_size_ms,
    )

    binned = trial_aligned["binned"]  # (n_events, n_units, n_time_bins)
    unit_ids = trial_aligned["unit_ids"]

    n_events, n_units, n_time_bins = binned.shape
    assert n_units == len(unit_ids)

    rows: List[Dict[str, Any]] = []

    for event_idx in range(n_events):
        event_row = change_events.iloc[event_idx]
        event_time_s = float(event_row["start_time"])

        # Absolute window times in ms
        window_start_ms = int(round(event_time_s * 1000.0))
        window_end_ms = window_start_ms + int(window_size_ms)

        # Reconstruct spike events from binned counts
        events_units: List[int] = []
        events_times_ms: List[float] = []

        counts = binned[event_idx]  # (n_units, n_time_bins)
        for u_idx, unit_id in enumerate(unit_ids):
            unit_counts = counts[u_idx]
            for t_idx in range(n_time_bins):
                c = int(unit_counts[t_idx])
                if c <= 0:
                    continue
                # Place spikes at bin center, relative to window start
                t_ms = t_idx * bin_size_ms + (bin_size_ms / 2.0)
                events_units.extend([int(unit_id)] * c)
                events_times_ms.extend([t_ms] * c)

        # Minimal stimulus JSON: image-change event at time 0
        stim_event = {
            "time_ms": 0.0,
            "image_name": event_row.get("image_name", None),
            "stimulus_block": int(event_row.get("stimulus_block", 0)),
            "is_change": bool(event_row.get("is_change", True)),
        }
        stimulus_json = json.dumps([stim_event])

        # Minimal behavior JSON placeholder (extend later if needed)
        behavior_json = json.dumps([])

        row = {
            "session_id": int(session_id),
            "window_id": int(next_window_id),
            "window_start_ms": window_start_ms,
            "window_end_ms": window_end_ms,
            "events_units": json.dumps(events_units),
            "events_times_ms": json.dumps(events_times_ms),
            "stimulus": stimulus_json,
            "behavior": behavior_json,
        }
        rows.append(row)
        next_window_id += 1

    print(
        f"  Created {len(rows)} windows "
        f"({n_units} units, {n_time_bins} bins, window={window_size_ms} ms)."
    )
    return rows, next_window_id


def main(config_path: Path) -> None:
    cfg = _load_and_validate_config(config_path)

    out_path: Path = cfg["data_path"]
    cache_dir: Path = cfg["cache_dir"]
    session_ids: List[int] = cfg["session_ids"]

    print("Creating dataset with configuration:")
    print(f"  Output CSV: {out_path}")
    print(f"  Cache dir:  {cache_dir}")
    print(f"  Sessions:   {session_ids}")
    print(f"  Brain areas: {cfg['brain_areas'] or 'ALL'}")
    print(
        f"  Quality: min_snr={cfg['quality']['min_snr']}, "
        f"min_firing_rate={cfg['quality']['min_firing_rate']}, "
        f"max_isi_violations={cfg['quality']['max_isi_violations']}"
    )
    print(
        f"  Windowing: bin_size_ms={cfg['windowing']['bin_size_ms']}, "
        f"window_size_ms={cfg['windowing']['window_size_ms']}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    handler = VBNDataHandler(cache_dir, metadata_only=False)

    all_rows: List[Dict[str, Any]] = []
    next_window_id = 0

    for sid in session_ids:
        session_rows, next_window_id = _build_session_windows(
            handler, sid, cfg, next_window_id
        )
        all_rows.extend(session_rows)

    if not all_rows:
        raise RuntimeError("No windows were created; check config and filters.")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_path, index=False)

    print(f"\nWrote {len(df)} windows to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a preprocessed windowed dataset for multi-session experiments"
    )
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to configuration YAML file for dataset settings",
    )
    args = parser.parse_args()

    print("Creating dataset")
    print("=" * 60)
    main(config_path=args.config_path)