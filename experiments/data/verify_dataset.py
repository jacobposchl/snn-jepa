import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def verify_dataset(parquet_path: Path):
    print(f"Loading dataset from: {parquet_path}")
    df = pd.read_parquet(parquet_path, engine="pyarrow")

    print("-" * 30)
    print("SANITY CHECKS")
    print("-" * 30)

    total_windows = len(df)
    empty_windows = 0
    all_times = []

    for idx, row in df.iterrows():
        times = row["events_times_ms"]
        if len(times) == 0:
            empty_windows += 1
        else:
            all_times.append(times)

    all_times_flat = np.concatenate(all_times) if all_times else np.array([])

    print(f"Total Windows: {total_windows}")
    print(
        f"Empty Windows (0 spikes): {empty_windows} ({empty_windows/total_windows*100:.1f}%)"
    )

    if len(all_times_flat) > 0:
        print(f"Total Spikes Across Dataset: {len(all_times_flat)}")
        print(f"Global Min Timestamp: {np.min(all_times_flat):.2f} ms")
        print(f"Global Max Timestamp: {np.max(all_times_flat):.2f} ms")
    else:
        print("CRITICAL WARNING: No spikes found in the entire dataset!")
        return

    print("\nGenerating verification plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Plot 1: Raster Plot (Single Window) ---
    # Pick a random non-empty window to visualize
    non_empty_df = df[df["events_times_ms"].apply(len) > 0]
    sample_row = non_empty_df.iloc[0]  # Grab the first valid window
    sample_times = sample_row["events_times_ms"]
    sample_units = sample_row["events_units"]

    ax1.scatter(sample_times, sample_units, s=2, color="black", alpha=0.7)
    ax1.set_title(
        f"Raster Plot - Session {sample_row['session_id']} (Window ID: {sample_row['window_id']})"
    )
    ax1.set_ylabel("Unit ID")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Population PSTH (All Windows) ---
    # Histogram of all spikes across all windows to see the evoked response
    bins = np.linspace(0, 400, 81)  # 5ms bins from 0 to 400ms
    ax2.hist(all_times_flat, bins=bins, color="royalblue", edgecolor="black", alpha=0.8)
    ax2.set_title("Population PSTH (All Sessions, All Windows)")
    ax2.set_xlabel("Time relative to stimulus change (ms)")
    ax2.set_ylabel("Total Spike Count")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = parquet_path.parent / "dataset_verification.png"
    plt.savefig(plot_path)
    print(f"\nSaved verification plots to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Parquet neural dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to the .parquet file")
    args = parser.parse_args()
    verify_dataset(args.dataset_path)
