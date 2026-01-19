"""
Test script to verify binning.py works correctly.

Tests:
1. Spike count conservation - total spikes should match sum of binned counts
2. Shape sanity checks - output dimensions match expectations
3. Visual comparison - raster vs binned heatmap
4. Trial alignment verification - PSTH should show response after stimulus
5. Known bin check - manual count verification

Run from project root:
    python temp_tests/binning_test.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt

from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

from utils.binning import (
    bin_spike_times,
    bin_population,
    bin_session,
    extract_trials,
    get_time_bins,
)
from plots.raster import (
    plot_raster,
    plot_binned_heatmap,
    plot_raster_with_binned,
    plot_trial_raster,
    plot_psth,
)


def test_spike_count_conservation(spike_times_dict, binned, unit_ids):
    """Test 1: Total spikes should equal sum of binned counts."""
    print("\n" + "=" * 60)
    print("TEST 1: Spike Count Conservation")
    print("=" * 60)
    
    all_passed = True
    for i, uid in enumerate(unit_ids[:5]):  # Test first 5 units
        raw_count = len(spike_times_dict[uid])
        binned_count = int(binned[i].sum())
        match = raw_count == binned_count
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"  Unit {uid}: raw={raw_count}, binned={binned_count} {status}")
        if not match:
            all_passed = False
    
    if len(unit_ids) > 5:
        # Check remaining units silently
        for i, uid in enumerate(unit_ids[5:], start=5):
            raw_count = len(spike_times_dict[uid])
            binned_count = int(binned[i].sum())
            if raw_count != binned_count:
                all_passed = False
                print(f"  Unit {uid}: raw={raw_count}, binned={binned_count} ✗ FAIL")
    
    print(f"\n  Overall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


def test_shape_sanity(binned, bin_edges, unit_ids, bin_size_ms, start_time, end_time):
    """Test 2: Output dimensions should match expectations."""
    print("\n" + "=" * 60)
    print("TEST 2: Shape Sanity Checks")
    print("=" * 60)
    
    n_units, n_bins = binned.shape
    expected_bins = len(bin_edges) - 1
    duration_s = end_time - start_time
    expected_bins_approx = int(duration_s * 1000 / bin_size_ms)
    
    print(f"  Binned shape: {binned.shape}")
    print(f"  Number of units: {n_units} (expected: {len(unit_ids)})")
    print(f"  Number of bins: {n_bins} (from bin_edges: {expected_bins})")
    print(f"  Expected bins (approx): {expected_bins_approx} (duration={duration_s:.1f}s, bin_size={bin_size_ms}ms)")
    
    units_ok = n_units == len(unit_ids)
    bins_ok = n_bins == expected_bins
    approx_ok = abs(n_bins - expected_bins_approx) <= 2  # Allow small rounding diff
    
    print(f"\n  Units dimension: {'✓ PASS' if units_ok else '✗ FAIL'}")
    print(f"  Bins dimension: {'✓ PASS' if bins_ok else '✗ FAIL'}")
    print(f"  Bins approx check: {'✓ PASS' if approx_ok else '✗ FAIL'}")
    
    return units_ok and bins_ok and approx_ok


def test_known_bin(spike_times_dict, unit_ids, bin_size_ms=10.0):
    """Test 3: Manual count for a specific bin."""
    print("\n" + "=" * 60)
    print("TEST 3: Known Bin Check (Manual Verification)")
    print("=" * 60)
    
    # Pick first unit with spikes
    test_uid = None
    for uid in unit_ids:
        if len(spike_times_dict[uid]) > 100:
            test_uid = uid
            break
    
    if test_uid is None:
        print("  No suitable unit found for test")
        return True
    
    spikes = spike_times_dict[test_uid]
    
    # Pick a time window around the median spike time
    median_time = np.median(spikes)
    test_start = median_time
    test_end = median_time + (bin_size_ms / 1000.0)
    
    # Manual count
    manual_count = np.sum((spikes >= test_start) & (spikes < test_end))
    
    # Binned count
    bin_edges = get_time_bins(test_start, test_end, bin_size_ms)
    binned_count = int(bin_spike_times(spikes, bin_edges).sum())
    
    match = manual_count == binned_count
    print(f"  Unit {test_uid}")
    print(f"  Time window: [{test_start:.4f}, {test_end:.4f})")
    print(f"  Manual count: {manual_count}")
    print(f"  Binned count: {binned_count}")
    print(f"  Result: {'✓ PASS' if match else '✗ FAIL'}")
    
    return match


def test_trial_shapes(trials_result):
    """Test 4: Trial extraction shape check."""
    print("\n" + "=" * 60)
    print("TEST 4: Trial Extraction Shape Check")
    print("=" * 60)
    
    trials = trials_result["trials"]
    unit_ids = trials_result["unit_ids"]
    stim_df = trials_result["stimulus_df"]
    time_bins = trials_result["time_bins_ms"]
    
    n_trials, n_units, n_time_bins = trials.shape
    
    print(f"  Trials shape: {trials.shape}")
    print(f"  Expected trials: {len(stim_df)}")
    print(f"  Expected units: {len(unit_ids)}")
    print(f"  Time bins: {len(time_bins)}")
    
    trials_ok = n_trials == len(stim_df)
    units_ok = n_units == len(unit_ids)
    time_ok = n_time_bins == len(time_bins)
    
    print(f"\n  Trials dimension: {'✓ PASS' if trials_ok else '✗ FAIL'}")
    print(f"  Units dimension: {'✓ PASS' if units_ok else '✗ FAIL'}")
    print(f"  Time dimension: {'✓ PASS' if time_ok else '✗ FAIL'}")
    
    return trials_ok and units_ok and time_ok


def visual_test_raster_vs_binned(spike_times_dict, binned, bin_edges, unit_ids, save_path=None):
    """Visual test: Compare raster with binned heatmap."""
    print("\n" + "=" * 60)
    print("VISUAL TEST: Raster vs Binned Comparison")
    print("=" * 60)
    
    # Select a subset of units and time window for visibility
    n_units_to_show = min(10, len(unit_ids))
    subset_ids = unit_ids[:n_units_to_show]
    subset_binned = binned[:n_units_to_show]
    
    # Focus on a 5-second window
    mid_time = (bin_edges[0] + bin_edges[-1]) / 2
    start_vis = mid_time - 2.5
    end_vis = mid_time + 2.5
    
    # Find bin indices for this window
    start_idx = np.searchsorted(bin_edges, start_vis)
    end_idx = np.searchsorted(bin_edges, end_vis)
    
    subset_binned_windowed = subset_binned[:, start_idx:end_idx]
    bin_edges_windowed = bin_edges[start_idx:end_idx+1]
    
    fig, (ax1, ax2) = plot_raster_with_binned(
        spike_times_dict,
        subset_binned_windowed,
        bin_edges_windowed,
        subset_ids,
        start_time=start_vis,
        end_time=end_vis,
        title=f"Binning Verification ({n_units_to_show} units, 5s window)",
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {save_path}")
    else:
        plt.show()
    
    print("  Visual check: Do high spike densities in raster match bright regions in heatmap?")
    return True


def visual_test_trial_alignment(trials_result, save_path=None):
    """Visual test: PSTH should show response after stimulus onset."""
    print("\n" + "=" * 60)
    print("VISUAL TEST: Trial Alignment (PSTH)")
    print("=" * 60)
    
    trials = trials_result["trials"]
    time_bins = trials_result["time_bins_ms"]
    bin_size = trials_result["bin_size_ms"]
    
    # Find a unit with good response (high variance across time)
    mean_per_unit = trials.mean(axis=0)  # (n_units, n_time_bins)
    variance_per_unit = mean_per_unit.var(axis=1)
    best_unit_idx = np.argmax(variance_per_unit)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Trial raster for best unit
    plot_trial_raster(
        trials,
        time_bins,
        unit_idx=best_unit_idx,
        ax=ax1,
        title=f"Trial Raster (Unit idx={best_unit_idx})",
    )
    
    # PSTH for best unit
    plot_psth(
        trials,
        time_bins,
        bin_size,
        unit_idx=best_unit_idx,
        ax=ax2,
        title=f"PSTH (Unit idx={best_unit_idx})",
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to: {save_path}")
    else:
        plt.show()
    
    print("  Visual check: Is there increased activity after time=0 (stimulus onset)?")
    return True


def main():
    print("\n" + "#" * 60)
    print("# BINNING.PY VERIFICATION TESTS")
    print("#" * 60)
    
    # Configuration
    data_dir = Path(__file__).parent.parent / "visual_behavior_neuropixels_data"
    session_id = 1047969464  # The session we downloaded
    bin_size_ms = 10.0
    
    # Output directory for plots
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nLoading session {session_id}...")
    cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=data_dir)
    session = cache.get_ecephys_session(ecephys_session_id=session_id)
    print("Session loaded!")
    
    # Get raw spike times
    spike_times_dict = session.spike_times
    print(f"Total units in session: {len(spike_times_dict)}")
    
    # === Test full session binning ===
    print("\n" + "-" * 60)
    print("Testing bin_session()...")
    print("-" * 60)
    
    result = bin_session(
        session,
        bin_size_ms=bin_size_ms,
        quality_filter=None,  # Include all units for testing
    )
    
    binned = result["binned"]
    unit_ids = result["unit_ids"]
    bin_edges = result["bin_edges"]
    
    print(f"Binned shape: {binned.shape}")
    print(f"Time range: [{bin_edges[0]:.2f}, {bin_edges[-1]:.2f}] seconds")
    
    # Run tests
    results = []
    
    results.append(("Spike Count Conservation", 
                    test_spike_count_conservation(spike_times_dict, binned, unit_ids)))
    
    results.append(("Shape Sanity", 
                    test_shape_sanity(binned, bin_edges, unit_ids, bin_size_ms, 
                                     bin_edges[0], bin_edges[-1])))
    
    results.append(("Known Bin Check", 
                    test_known_bin(spike_times_dict, unit_ids, bin_size_ms)))
    
    # Visual test
    visual_test_raster_vs_binned(
        spike_times_dict, binned, bin_edges, unit_ids,
        save_path=output_dir / "raster_vs_binned.png"
    )
    
    # === Test trial extraction ===
    print("\n" + "-" * 60)
    print("Testing extract_trials()...")
    print("-" * 60)
    
    # Get available stimulus types
    stim_df = session.stimulus_presentations
    stim_types = stim_df["stimulus_name"].unique()
    print(f"Available stimulus types: {stim_types[:5]}...")  # Show first 5
    
    # Try to extract trials for a common stimulus type
    test_stimulus = None
    for stim in ["natural_scenes", "gabors", "flashes", "natural_movie_one"]:
        if stim in stim_types:
            test_stimulus = stim
            break
    
    if test_stimulus is None:
        test_stimulus = stim_types[0]
    
    print(f"Using stimulus type: {test_stimulus}")
    
    try:
        trials_result = extract_trials(
            session,
            bin_size_ms=bin_size_ms,
            pre_time_ms=200.0,
            post_time_ms=500.0,
            stimulus_name=test_stimulus,
            quality_filter=None,
        )
        
        results.append(("Trial Shapes", test_trial_shapes(trials_result)))
        
        visual_test_trial_alignment(
            trials_result,
            save_path=output_dir / "trial_alignment.png"
        )
        
    except Exception as e:
        print(f"  Trial extraction failed: {e}")
        results.append(("Trial Shapes", False))
    
    # === Summary ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("=" * 60)
    
    print(f"\nVisual outputs saved to: {output_dir}")
    print("Please review the plots to verify visual correctness.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

