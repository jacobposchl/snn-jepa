"""
Script to create a preprocessed, windowed dataset given a configuration.

Output is a single Parquet file with one row per temporal window, matching the
expected format in `experiments/multi_session/multi_session.py`.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow
import temporaldata as td
import yaml

from jepsyn.data.data_handler import VBNDataHandler
from jepsyn.data.preprocess import NeuropixelsPreprocessor


def _load_and_validate_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML config from path and normalize dataset settings.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Config file must contain a top-level mapping.")

    if "data_path" not in raw or not raw["data_path"]:
        raise ValueError(
            "Config must define a non-empty 'data_path' for the output file."
        )

    dataset_cfg = raw.get("dataset_config")
    if not isinstance(dataset_cfg, dict):
        raise ValueError("Config must contain a 'dataset_config' section.")

    cache_dir = dataset_cfg.get("cache_dir")
    if not cache_dir:
        raise ValueError("dataset_config.cache_dir is required.")

    session_ids = dataset_cfg.get("session_ids") or []
    if session_ids and not isinstance(session_ids, list):
        raise ValueError(
            "dataset_config.session_ids must be a list of ecephys session IDs if provided."
        )

    # filter data by provided brain areas, or grab all brain areas
    brain_areas = dataset_cfg.get("brain_areas") or []
    if brain_areas and not isinstance(brain_areas, list):
        raise ValueError("dataset_config.brain_areas must be a list if provided.")

    # filter the data by quality metrics; use defaults if not provided in config
    quality_cfg = dataset_cfg.get("quality") or {}
    min_snr = float(quality_cfg.get("min_snr", 1.0))
    min_firing_rate = float(quality_cfg.get("min_firing_rate", 0.1))
    max_isi_violations = float(quality_cfg.get("max_isi_violations", 1.0))

    # set window parameters; use defaults if not provided in config
    windowing_cfg = dataset_cfg.get("windowing") or {}
    bin_size_ms = float(windowing_cfg.get("bin_size_ms", 10.0))
    window_size_ms = float(windowing_cfg.get("window_size_ms", 400.0))

    if bin_size_ms <= 0 or window_size_ms <= 0:
        raise ValueError(
            "windowing.bin_size_ms and windowing.window_size_ms must be > 0."
        )

    stride_ms = float(windowing_cfg.get("stride_ms", window_size_ms))

    if window_size_ms % bin_size_ms != 0:
        raise ValueError(
            "windowing.window_size_ms must be an integer multiple of windowing.bin_size_ms."
        )

    # Resolve relative paths relative to the config file's directory so that
    # the config works correctly regardless of the working directory.
    config_dir = config_path.resolve().parent

    return {
        "data_path": (config_dir / raw["data_path"]).resolve(),
        "cache_dir": (config_dir / cache_dir).resolve(),
        "session_ids": [int(s) for s in session_ids],  # empty list = auto-discover
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


def _discover_sessions(handler: VBNDataHandler, brain_areas: List[str]) -> List[int]:
    """
    Auto-discover all ecephys sessions that recorded from areas matching brain_areas.

    Each element of brain_areas is treated as a substring so ["VIS"] matches
    VISp, VISl, VISrl, VISam, VISpm, etc.
    """
    sessions = handler.sessions_table

    def _has_matching_area(areas) -> bool:
        if isinstance(areas, list):
            return any(any(p in a for p in brain_areas) for a in areas)
        if isinstance(areas, str):
            # column may be stored as a string repr of a list
            return any(p in areas for p in brain_areas)
        return False

    mask = sessions["structure_acronyms"].apply(_has_matching_area)
    ids = list(sessions[mask].index)
    print(f"Auto-discovered {len(ids)} sessions with areas matching {brain_areas}")
    return ids


def _build_session_windows(
    handler: VBNDataHandler,
    session_id: int,
    cfg: Dict[str, Any],
    next_window_id: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Create window rows for a single session.

    Each window is aligned to an image-change event in the active behavior block.
    Uses temporaldata to extract spikes in each interval, preserving exact timestamps.
    """
    print(f"\nProcessing session {session_id}...")

    try:
        session = handler.load_session(session_id)
    except Exception as e:
        print(f"  Skipping session {session_id}: failed to load ({e})")
        return [], next_window_id

    # preprocess data using the prepressing script in jepsyn.
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
        print(
            f"  No units passed quality/area filters in session {session_id}; skipping."
        )
        return [], next_window_id

    stim = session.stimulus_presentations
    active_stim = stim[stim["active"] == True].copy()
    change_events = active_stim[active_stim["is_change"] == True].copy()
    # Exclude omission events (no image shown) from non-change windows.
    # Use the authoritative `omitted` boolean column; fall back to checking
    # image_name in case the column is absent in an older SDK version.
    if "omitted" in active_stim.columns:
        nonchange_events = active_stim[
            (active_stim["is_change"] == False) & (~active_stim["omitted"])
        ].copy()
    else:
        nonchange_events = active_stim[
            (active_stim["is_change"] == False)
            & active_stim["image_name"].notna()
            & (active_stim["image_name"] != "omitted")
        ].copy()

    if len(change_events) == 0:
        print(f"  No active image-change events in session {session_id}; skipping.")
        return [], next_window_id

    window_size_ms = cfg["windowing"]["window_size_ms"]

    # Restrict spikes to the active change-detection block only (excludes RF
    # mapping, gray screens, passive replay, etc.)
    active_start = float(active_stim["start_time"].min())
    active_end   = float(active_stim["end_time"].max())

    # Pool all spikes and unit IDs into sorted arrays for temporaldata
    all_spike_times = []
    all_unit_ids = []
    for unit_id in good_unit_ids:
        spikes = session.spike_times[unit_id]
        spikes = spikes[(spikes >= active_start) & (spikes <= active_end)]
        all_spike_times.append(spikes)
        all_unit_ids.append(np.full(len(spikes), unit_id, dtype=np.int32))

    all_spike_times = np.concatenate(all_spike_times)
    all_unit_ids = np.concatenate(all_unit_ids)

    sort_idx = np.argsort(all_spike_times)
    all_spike_times = all_spike_times[sort_idx]
    all_unit_ids = all_unit_ids[sort_idx]

    # use temporadata to generate a time series of spiking events
    # IrregularTimeSeries: timestamps in seconds, unit_id per spike
    spike_ts = td.IrregularTimeSeries(
        timestamps=all_spike_times,
        unit_id=all_unit_ids,
        domain="auto",
    )

    rows: List[Dict[str, Any]] = []

    def _extract_windows(events_df: pd.DataFrame) -> None:
        nonlocal next_window_id
        for event_idx in range(len(events_df)):
            event_row = events_df.iloc[event_idx]
            event_time_s = float(event_row["start_time"])
            window_start_s = event_time_s
            window_end_s = event_time_s + (window_size_ms / 1000.0)

            window_data = spike_ts.slice(window_start_s, window_end_s, reset_origin=True)

            window_start_ms = int(round(event_time_s * 1000.0))
            window_end_ms = window_start_ms + int(window_size_ms)

            rel_times_s = window_data.timestamps
            events_times_ms = np.array(rel_times_s * 1000.0, dtype=np.float32)
            events_units = np.array(window_data.unit_id, dtype=np.int32)

            stimulus = [
                {
                    "time_ms": 0.0,
                    "image_name": event_row.get("image_name", None),
                    "stimulus_block": int(event_row.get("stimulus_block", 0)),
                    "is_change": bool(event_row.get("is_change", False)),
                }
            ]

            row = {
                "session_id": int(session_id),
                "window_id": int(next_window_id),
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "events_units": events_units,
                "events_times_ms": events_times_ms,
                "stimulus": stimulus,
                "behavior": [],
            }
            rows.append(row)
            next_window_id += 1

    _extract_windows(change_events)

    # Match the number of non-change windows to change windows so classes are balanced.
    # If there are more non-change events than change events, randomly sample to match.
    n_change = len(change_events)
    if len(nonchange_events) > n_change:
        nonchange_events = nonchange_events.sample(n=n_change, random_state=42)
    _extract_windows(nonchange_events)

    n_change_rows    = len(change_events)
    n_nonchange_rows = len(nonchange_events)
    print(
        f"  Created {len(rows)} windows ({len(good_unit_ids)} units, window={window_size_ms} ms) "
        f"— {n_change_rows} change + {n_nonchange_rows} non-change."
    )
    return rows, next_window_id


def main(config_path: Path) -> None:
    cfg = _load_and_validate_config(config_path)

    out_path: Path = cfg["data_path"]
    parquet_path = out_path.with_suffix(".parquet")
    cache_dir: Path = cfg["cache_dir"]

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    handler = VBNDataHandler(cache_dir, metadata_only=False)

    # Auto-discover sessions when none are specified in the config
    session_ids: List[int] = cfg["session_ids"] or _discover_sessions(
        handler, cfg["brain_areas"] or ["VIS"]
    )

    print("Creating dataset with configuration:")
    print(f"  Output Parquet: {parquet_path}")
    print(f"  Cache dir:  {cache_dir}")
    print(f"  Sessions:   {len(session_ids)} sessions")
    print(f"  Brain areas: {cfg['brain_areas'] or 'ALL (VIS pattern)'}")
    print(
        f"  Quality: min_snr={cfg['quality']['min_snr']}, "
        f"min_firing_rate={cfg['quality']['min_firing_rate']}, "
        f"max_isi_violations={cfg['quality']['max_isi_violations']}"
    )
    print(
        f"  Windowing: bin_size_ms={cfg['windowing']['bin_size_ms']}, "
        f"window_size_ms={cfg['windowing']['window_size_ms']}"
    )

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

    # ── Attach session metadata and per-image novelty ────────────────────────
    # experience_level ('Familiar'/'Novel') and image_set ('G'/'H') come from
    # the sessions table rather than being stored per-window during extraction,
    # so we join them in post-processing.
    sessions_meta = handler.sessions_table
    if "experience_level" in sessions_meta.columns and "image_set" in sessions_meta.columns:
        df["experience_level"] = df["session_id"].map(sessions_meta["experience_level"])
        df["image_set"]        = df["session_id"].map(sessions_meta["image_set"])

        # Flatten the nested stimulus list to get the image_name per row.
        image_names = df["stimulus"].apply(
            lambda s: s[0].get("image_name") if s else None
        )

        # Shared images appear in sessions from BOTH image sets (G ∩ H).
        # These images are familiar even in a 'Novel' session because the animal
        # trained on the other set that shares them.
        combos = pd.DataFrame(
            {"image_name": image_names, "image_set": df["image_set"]}
        ).dropna()
        img_to_sets = combos.groupby("image_name")["image_set"].apply(set)
        shared_images: set = set(
            img_to_sets[img_to_sets.apply(lambda s: "G" in s and "H" in s)].index
        )
        if shared_images:
            print(f"\nShared images across G and H (familiar in novel sessions): {sorted(shared_images)}")
        else:
            print("\nNo images found in both G and H sets — single-set dataset or sets don't overlap.")

        # image_is_novel: True only when the session is Novel AND the specific
        # image was not part of the animal's training set (i.e. not shared).
        df["image_is_novel"] = (df["experience_level"] == "Novel") & (
            ~image_names.isin(shared_images)
        )
        n_novel   = int(df["image_is_novel"].sum())
        n_familiar = len(df) - n_novel
        print(f"  image_is_novel: {n_novel} novel-image windows, {n_familiar} familiar-image windows")
    else:
        print(
            "\nWarning: sessions table missing 'experience_level' or 'image_set' columns; "
            "skipping per-image novelty."
        )

    df.to_parquet(parquet_path, engine="pyarrow", index=False)

    print(f"\nWrote {len(df)} windows to {parquet_path}")


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
