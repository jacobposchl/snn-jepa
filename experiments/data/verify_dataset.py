import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_stim_field(df: pd.DataFrame, field: str) -> pd.Series:
    """Pull a scalar field out of the nested stimulus list column."""
    return df["stimulus"].apply(lambda s: s[0].get(field) if s else None)


def _pass(msg: str) -> None:
    print(f"  ✓  {msg}")


def _fail(msg: str) -> None:
    print(f"  ✗  {msg}")


def _info(msg: str) -> None:
    print(f"     {msg}")


# ── Check sections ────────────────────────────────────────────────────────────

def check_schema(df: pd.DataFrame) -> bool:
    print("\n" + "=" * 60)
    print("1. SCHEMA")
    print("=" * 60)
    required = [
        "session_id", "window_id", "window_start_ms", "window_end_ms",
        "events_units", "events_times_ms", "stimulus", "behavior",
    ]
    new_cols = ["experience_level", "image_set", "image_is_novel"]
    missing_req = [c for c in required if c not in df.columns]
    missing_new = [c for c in new_cols if c not in df.columns]

    if missing_req:
        _fail(f"Missing required columns: {missing_req}")
        return False
    _pass("All required columns present")

    if missing_new:
        _fail(
            f"Missing novelty/metadata columns: {missing_new} — "
            "re-run create_dataset.py to regenerate"
        )
    else:
        _pass("Novelty/metadata columns present: experience_level, image_set, image_is_novel")

    return not bool(missing_req)


def check_sessions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("2. SESSION COVERAGE")
    print("=" * 60)
    n = df["session_id"].nunique()
    _info(f"Total windows:   {len(df)}")
    _info(f"Unique sessions: {n}")
    if n < 3:
        _fail(f"Only {n} session(s) — need ≥3 for train/val/test split")
    else:
        _pass(f"{n} sessions → safe for train/val/test split")

    if "experience_level" in df.columns:
        _info("\nSessions by experience_level:")
        for lvl, grp in df.groupby("experience_level")["session_id"]:
            _info(f"  {lvl}: {grp.nunique()} sessions, {len(grp)} windows")

    if "image_set" in df.columns:
        _info("\nSessions by image_set:")
        for iset, grp in df.groupby("image_set")["session_id"]:
            _info(f"  Set {iset}: {grp.nunique()} sessions, {len(grp)} windows")


def check_window_integrity(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("3. WINDOW INTEGRITY")
    print("=" * 60)

    # Unique window IDs
    if df["window_id"].nunique() != len(df):
        _fail(f"Duplicate window_ids ({len(df) - df['window_id'].nunique()} duplicates)")
    else:
        _pass("All window_ids are unique")

    # Consistent window durations
    durations = df["window_end_ms"] - df["window_start_ms"]
    unique_dur = durations.unique()
    if len(unique_dur) == 1:
        _pass(f"All windows are {unique_dur[0]} ms")
    else:
        _fail(f"Inconsistent window durations: {unique_dur}")

    # Spike time bounds
    window_ms = float(durations.mode()[0])
    out_of_bounds = 0
    mismatch = 0
    for _, row in df.iterrows():
        times = row["events_times_ms"]
        units = row["events_units"]
        if len(times) != len(units):
            mismatch += 1
        if len(times) > 0 and (float(times.min()) < 0 or float(times.max()) > window_ms):
            out_of_bounds += 1

    if mismatch:
        _fail(f"{mismatch} windows have mismatched spike-time / unit-id array lengths")
    else:
        _pass("All spike-time and unit-id arrays have matching lengths")

    if out_of_bounds:
        _fail(f"{out_of_bounds} windows contain spike times outside [0, {window_ms}] ms")
    else:
        _pass(f"All spike times are within [0, {window_ms}] ms")


def check_omissions(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("4. OMISSION FILTERING")
    print("=" * 60)
    image_names = _extract_stim_field(df, "image_name")
    n_omitted = (image_names == "omitted").sum()
    n_null = image_names.isna().sum()

    if n_omitted:
        _fail(f"{n_omitted} windows have image_name == 'omitted' — omissions not filtered")
    else:
        _pass("No 'omitted' image_name values")

    if n_null:
        _fail(f"{n_null} windows have null image_name")
    else:
        _pass("No null image_name values")

    _info(f"Unique image names across full dataset: {image_names.nunique()}")


def check_images_per_session(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("5. IMAGES PER SESSION (should be ≤8 per session)")
    print("=" * 60)
    image_names = _extract_stim_field(df, "image_name")
    df = df.copy()
    df["_img"] = image_names

    per_session = df.groupby("session_id")["_img"].nunique()
    over_8 = per_session[per_session > 8]

    if over_8.empty:
        _pass(f"All sessions show ≤8 unique images (max = {per_session.max()})")
    else:
        _fail(f"{len(over_8)} session(s) show >8 images: {over_8.to_dict()}")
        _info("This likely means image sets G and H are mixed within a session — unexpected")

    _info(f"Images per session — min: {per_session.min()}, max: {per_session.max()}, "
          f"median: {per_session.median():.0f}")


def check_is_change_balance(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("6. is_change BALANCE (should be ~50/50 per session)")
    print("=" * 60)
    is_change = _extract_stim_field(df, "is_change")
    df = df.copy()
    df["_is_change"] = is_change

    overall_frac = df["_is_change"].mean()
    _info(f"Overall change fraction: {overall_frac:.3f} (target 0.500)")
    if abs(overall_frac - 0.5) > 0.05:
        _fail(f"Overall imbalance: {overall_frac:.3f} change fraction")
    else:
        _pass("Dataset-wide is_change fraction within 5 pp of 0.500")

    per_session = df.groupby("session_id")["_is_change"].mean()
    bad = per_session[abs(per_session - 0.5) > 0.1]
    if bad.empty:
        _pass(f"All sessions within 10 pp of 0.500 (range: {per_session.min():.3f}–{per_session.max():.3f})")
    else:
        _fail(f"{len(bad)} session(s) >10 pp away from 0.500: {bad.round(3).to_dict()}")


def check_novelty(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("7. PER-IMAGE NOVELTY")
    print("=" * 60)

    if "image_is_novel" not in df.columns or "experience_level" not in df.columns:
        _fail("Novelty columns missing — re-run create_dataset.py")
        return

    image_names = _extract_stim_field(df, "image_name")
    df = df.copy()
    df["_img"] = image_names

    # Rule 1: in Familiar sessions, image_is_novel must always be False
    familiar = df[df["experience_level"] == "Familiar"]
    if len(familiar):
        leak = familiar["image_is_novel"].sum()
        if leak:
            _fail(f"{leak} windows in Familiar sessions are flagged image_is_novel=True (should be 0)")
        else:
            _pass(f"Familiar sessions ({len(familiar)} windows): image_is_novel is always False")

    # Rule 2: in Novel sessions, some images are novel, some are shared
    novel = df[df["experience_level"] == "Novel"]
    if len(novel):
        shared_in_novel = novel[~novel["image_is_novel"]]["_img"].dropna().unique()
        unique_in_novel  = novel[ novel["image_is_novel"]]["_img"].dropna().unique()
        _info(f"Novel sessions ({len(novel)} windows):")
        _info(f"  Shared/familiar images:  {sorted(shared_in_novel)} ({len(shared_in_novel)} images)")
        _info(f"  Truly novel images:      {sorted(unique_in_novel)} ({len(unique_in_novel)} images)")
        if len(shared_in_novel) == 2:
            _pass("Exactly 2 shared images in novel sessions (matches G∩H = 2 images)")
        elif len(shared_in_novel) == 0:
            _fail("0 shared images — all Novel-session images marked novel; shared-image detection may have failed")
        else:
            _info(f"Note: {len(shared_in_novel)} shared images (expected 2 for G/H datasets)")

    # Summary fractions
    n_novel_wins = int(df["image_is_novel"].sum())
    n_fam_wins   = len(df) - n_novel_wins
    _info(f"\nNovelty breakdown: {n_novel_wins} novel-image windows, {n_fam_wins} familiar-image windows "
          f"({100*n_novel_wins/len(df):.1f}% novel)")


def check_spike_stats(df: pd.DataFrame) -> dict:
    print("\n" + "=" * 60)
    print("8. SPIKE STATISTICS")
    print("=" * 60)
    spike_counts = df["events_times_ms"].apply(len)
    empty = (spike_counts == 0).sum()
    all_flat = np.concatenate(df["events_times_ms"].values)

    _info(f"Total spikes:          {len(all_flat):,}")
    _info(f"Mean spikes/window:    {spike_counts.mean():.1f}")
    _info(f"Empty windows:         {empty} ({100*empty/len(df):.1f}%)")
    if empty / len(df) > 0.1:
        _fail(f">{10}% empty windows — possible spike filtering issue")
    else:
        _pass(f"Empty window rate acceptable ({100*empty/len(df):.1f}%)")
    return {"spike_counts": spike_counts, "all_flat": all_flat}


def plot_summary(df: pd.DataFrame, spike_counts, all_flat, parquet_path: Path) -> None:
    print("\nGenerating verification plots...")
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Raster of first non-empty window
    ax = axs[0, 0]
    non_empty = df[df["events_times_ms"].apply(len) > 0]
    row = non_empty.iloc[0]
    ax.scatter(row["events_times_ms"], row["events_units"], s=1, color="black", alpha=0.6)
    ax.set_title(f"Raster: session {row['session_id']} window {row['window_id']}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Unit ID")

    # 2. Spike count distribution
    ax = axs[0, 1]
    ax.hist(spike_counts, bins=40, color="teal", edgecolor="black", alpha=0.7)
    ax.set_title("Spikes per window")
    ax.set_xlabel("Spike count")
    ax.set_ylabel("Windows")

    # 3. Population PSTH
    ax = axs[0, 2]
    bins = np.linspace(0, df["window_end_ms"].iloc[0] - df["window_start_ms"].iloc[0], 81)
    ax.hist(all_flat, bins=bins, color="royalblue", edgecolor="black", alpha=0.8)
    ax.set_title("Population PSTH (all sessions)")
    ax.set_xlabel("Time from flash onset (ms)")
    ax.set_ylabel("Total spikes")

    # 4. is_change balance per session
    ax = axs[1, 0]
    is_change = _extract_stim_field(df, "is_change")
    per_sess  = is_change.groupby(df["session_id"]).mean()
    ax.bar(range(len(per_sess)), per_sess.values, color="salmon", edgecolor="black")
    ax.axhline(0.5, color="black", linestyle="--", label="0.500 target")
    ax.set_title("is_change fraction per session")
    ax.set_ylabel("Fraction change")
    ax.set_xticks([])
    ax.legend()

    # 5. Image count per session
    ax = axs[1, 1]
    image_names = _extract_stim_field(df, "image_name")
    img_per_sess = image_names.groupby(df["session_id"]).nunique()
    ax.bar(range(len(img_per_sess)), img_per_sess.values, color="mediumseagreen", edgecolor="black")
    ax.axhline(8, color="black", linestyle="--", label="8 images expected")
    ax.set_title("Unique images per session")
    ax.set_ylabel("Image count")
    ax.set_xticks([])
    ax.legend()

    # 6. Novelty breakdown (if available)
    ax = axs[1, 2]
    if "image_is_novel" in df.columns and "experience_level" in df.columns:
        labels = ["Familiar\n(all)", "Novel session\nshared img", "Novel session\nnovel img"]
        familiar_df = df[df["experience_level"] == "Familiar"]
        novel_df    = df[df["experience_level"] == "Novel"]
        counts = [
            len(familiar_df),
            int((~novel_df["image_is_novel"]).sum()),
            int(novel_df["image_is_novel"].sum()),
        ]
        ax.bar(labels, counts, color=["steelblue", "orange", "tomato"], edgecolor="black")
        ax.set_title("Per-image novelty breakdown")
        ax.set_ylabel("Windows")
    else:
        ax.set_title("Novelty breakdown (no data)")
        ax.text(0.5, 0.5, "image_is_novel not in dataset", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out = parquet_path.parent / "dataset_verification.png"
    plt.savefig(out, dpi=120)
    print(f"Saved plots to: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def verify_dataset(parquet_path: Path) -> None:
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    ok = check_schema(df)
    if not ok:
        return

    check_sessions(df)
    check_window_integrity(df)
    check_omissions(df)
    check_images_per_session(df)
    check_is_change_balance(df)
    check_novelty(df)
    stats = check_spike_stats(df)

    plot_summary(df, stats["spike_counts"], stats["all_flat"], parquet_path)

    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Parquet neural dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to the .parquet file")
    args = parser.parse_args()
    verify_dataset(args.dataset_path)
