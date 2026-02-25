import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def verify_dataset(parquet_path: Path):
    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    print("\n" + "=" * 50)
    print("1. MULTI-SESSION COMPATIBILITY CHECKS")
    print("=" * 50)

    # Check 1: Required Columns
    required_cols = [
        "session_id",
        "window_id",
        "window_start_ms",
        "window_end_ms",
        "events_units",
        "events_times_ms",
        "stimulus",
        "behavior",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ FAIL: Missing required columns: {missing_cols}")
        return
    print("✓ Schema: All required columns present.")

    # Check 2: Minimum Sessions for Train/Val/Test Split
    unique_sessions = df["session_id"].unique()
    if len(unique_sessions) < 3:
        print(
            f"❌ FAIL: Found {len(unique_sessions)} session(s). Multi-session split requires at least 3."
        )
    else:
        print(
            f"✓ Split Ready: Found {len(unique_sessions)} unique sessions (Safe for Train/Val/Test)."
        )

    # Check 3: Conflicting Window IDs (Mirrors multi_session.py)
    duplicate_check = df.groupby("window_id").agg(
        {"window_start_ms": "nunique", "window_end_ms": "nunique"}
    )
    conflicts = duplicate_check[
        (duplicate_check["window_start_ms"] > 1)
        | (duplicate_check["window_end_ms"] > 1)
    ]
    if not conflicts.empty:
        print(
            f"❌ FAIL: Found {len(conflicts)} window_ids with conflicting timestamps!"
        )
    else:
        print("✓ Integrity: No overlapping or conflicting window IDs.")

    print("\n" + "=" * 50)
    print("2. BIOLOGICAL & STATISTICAL CHECKS")
    print("=" * 50)

    total_windows = len(df)
    empty_windows = 0
    spike_counts = []
    all_times = []

    for idx, row in df.iterrows():
        times = row["events_times_ms"]
        units = row["events_units"]

        # Check 4: Array Length Matching (Mirrors multi_session.py)
        if len(times) != len(units):
            print(f"❌ FAIL: Array length mismatch in window {row['window_id']}")
            return

        count = len(times)
        spike_counts.append(count)
        if count == 0:
            empty_windows += 1
        else:
            all_times.append(times)

    print("✓ Arrays: All spike times and unit IDs match perfectly.")

    all_times_flat = np.concatenate(all_times) if all_times else np.array([])

    print(f"\nTotal Windows: {total_windows}")
    print(
        f"Empty Windows (0 spikes): {empty_windows} ({empty_windows/total_windows*100:.1f}%)"
    )

    if len(all_times_flat) > 0:
        avg_spikes = np.mean(spike_counts)
        avg_units_per_window = np.mean(
            [
                len(np.unique(row["events_units"]))
                for _, row in df.iterrows()
                if len(row["events_units"]) > 0
            ]
        )
        window_sec = 0.400  # 400ms
        approx_hz = (
            (avg_spikes / avg_units_per_window) / window_sec
            if avg_units_per_window > 0
            else 0
        )

        print(f"Total Spikes: {len(all_times_flat)}")
        print(f"Average Spikes/Window: {avg_spikes:.1f}")
        print(f"Approx Population Firing Rate: ~{approx_hz:.2f} Hz per unit")
        print(
            f"Global Bounds: [{np.min(all_times_flat):.2f} ms, {np.max(all_times_flat):.2f} ms]"
        )
    else:
        print("❌ CRITICAL WARNING: No spikes found in the entire dataset!")
        return

    print("\nGenerating advanced verification plots...")

    # --- Plotting Code (Unchanged) ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    ax_raster, ax_dist = axs[0, 0], axs[0, 1]
    ax_psth, ax_sess = axs[1, 0], axs[1, 1]

    non_empty_df = df[df["events_times_ms"].apply(len) > 0]
    sample_row = non_empty_df.iloc[0]
    ax_raster.scatter(
        sample_row["events_times_ms"],
        sample_row["events_units"],
        s=2,
        color="black",
        alpha=0.7,
    )
    ax_raster.set_title(
        f"Raster: Session {sample_row['session_id']} (Win: {sample_row['window_id']})"
    )
    ax_raster.set_xlabel("Time (ms)")
    ax_raster.set_ylabel("Unit ID")
    ax_raster.grid(True, alpha=0.3)

    ax_dist.hist(spike_counts, bins=30, color="teal", edgecolor="black", alpha=0.7)
    ax_dist.set_title("Distribution of Spikes per Window")
    ax_dist.set_xlabel("Number of Spikes")
    ax_dist.set_ylabel("Frequency")
    ax_dist.grid(True, alpha=0.3)

    bins = np.linspace(0, 400, 81)
    ax_psth.hist(
        all_times_flat, bins=bins, color="royalblue", edgecolor="black", alpha=0.8
    )
    ax_psth.set_title("Global Population PSTH")
    ax_psth.set_xlabel("Time relative to stimulus change (ms)")
    ax_psth.set_ylabel("Total Spike Count")
    ax_psth.grid(True, alpha=0.3)

    for sess_id in unique_sessions:
        sess_df = df[df["session_id"] == sess_id]
        sess_times = [
            row["events_times_ms"]
            for _, row in sess_df.iterrows()
            if len(row["events_times_ms"]) > 0
        ]
        if sess_times:
            sess_flat = np.concatenate(sess_times)
            ax_sess.hist(
                sess_flat,
                bins=bins,
                alpha=0.4,
                density=True,
                label=f"Sess {sess_id}",
                histtype="stepfilled",
            )
            ax_sess.hist(
                sess_flat,
                bins=bins,
                alpha=0.8,
                density=True,
                histtype="step",
                linewidth=1.5,
            )

    ax_sess.set_title("Normalized PSTH by Session")
    ax_sess.set_xlabel("Time (ms)")
    ax_sess.set_ylabel("Density")
    ax_sess.legend(fontsize="small")
    ax_sess.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = parquet_path.parent / "dataset_verification_advanced.png"
    plt.savefig(plot_path)
    print(f"\nSaved advanced verification plots to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Parquet neural dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to the .parquet file")
    args = parser.parse_args()
    verify_dataset(args.dataset_path)
