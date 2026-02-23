"""
Functions for binning neural spike data.

Converts raw spike times into binned spike counts suitable for neural network input.

Core functions:
    - bin_spike_times: Bin spike times for a single unit
    - bin_population: Bin spike times for multiple units simultaneously
    - bin_trial_aligned: Bin spikes aligned to specific events

Utilities:
    - get_time_bins: Generate time bin edges for a given range
"""

from typing import Optional, Union, Sequence
import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_time_bins(
    start: float,
    end: float,
    bin_size_ms: float,
) -> np.ndarray:
    """
    Generate time bin edges for a given range.
    
    Args:
        start: Start time in seconds
        end: End time in seconds
        bin_size_ms: Bin size in milliseconds
        
    Returns:
        Array of bin edges in seconds, shape (n_bins + 1,)
    """
    bin_size_s = bin_size_ms / 1000.0
    return np.arange(start, end + bin_size_s, bin_size_s)


def bin_spike_times(
    spike_times: np.ndarray,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """
    Bin spike times for a single unit.
    
    Args:
        spike_times: Array of spike times in seconds
        bin_edges: Array of bin edges in seconds, shape (n_bins + 1,)
        
    Returns:
        Array of spike counts per bin, shape (n_bins,)
    """
    counts, _ = np.histogram(spike_times, bins=bin_edges)
    return counts.astype(np.float32)


def bin_population(
    spike_times_dict: dict[int, np.ndarray],
    bin_edges: np.ndarray,
    unit_ids: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Bin spike times for multiple units.
    
    Args:
        spike_times_dict: Dict mapping unit_id -> spike times array
        bin_edges: Array of bin edges in seconds
        unit_ids: Optional list of unit IDs to include (default: all)
        
    Returns:
        Tuple of:
            - Binned spike counts, shape (n_units, n_bins)
            - List of unit IDs in the same order as the array rows
    """
    if unit_ids is None:
        unit_ids = list(spike_times_dict.keys())
    
    n_bins = len(bin_edges) - 1
    binned = np.zeros((len(unit_ids), n_bins), dtype=np.float32)
    
    for i, uid in enumerate(unit_ids):
        if uid in spike_times_dict:
            binned[i] = bin_spike_times(spike_times_dict[uid], bin_edges)
    
    return binned, list(unit_ids)


def bin_trial_aligned(
    spike_times_dict: dict[int, np.ndarray],
    event_times: np.ndarray,
    unit_ids: Sequence[int],
    bin_size_ms: float,
    pre_time_ms: float,
    post_time_ms: float,
    as_torch: bool = False,
) -> dict:
    """
    Bin spikes aligned to specific event times (e.g., stimulus onset, reward).
    
    Creates a 3D array of shape (n_events, n_units, n_time_bins) where each
    event is aligned to time zero.
    
    Args:
        spike_times_dict: Dict mapping unit_id -> spike times array (in seconds)
        event_times: Array of event times to align to (in seconds)
        unit_ids: List of unit IDs to include in output
        bin_size_ms: Bin size in milliseconds (default: 10ms)
        pre_time_ms: Time before event to include (ms, default: 500ms)
        post_time_ms: Time after event to include (ms, default: 500ms)
        as_torch: If True, return torch.Tensor instead of numpy array
        
    Returns:
        Dict containing:
            - "binned": Event-aligned data, shape (n_events, n_units, n_time_bins)
            - "unit_ids": List of unit IDs (same order as array dimension)
            - "time_bins_ms": Time bin centers relative to event (ms)
            - "bin_size_ms": Bin size used
            - "bin_edges_relative": Bin edges relative to event (seconds)
            
    Example:
        # Get change times from trials
        change_times = trials['change_time_no_display_delay'].values
        
        # Bin spikes around changes
        result = bin_trial_aligned(
            spike_times_dict=session.spike_times,
            event_times=change_times,
            unit_ids=[123, 456, 789],
            bin_size_ms=10,
            pre_time_ms=500,
            post_time_ms=1000
        )
        # result['binned'] has shape (n_changes, 3, 150)
    """
    # Convert to seconds
    pre_time_s = pre_time_ms / 1000.0
    post_time_s = post_time_ms / 1000.0
    
    # Create template bin edges relative to event (time 0)
    template_bins = get_time_bins(-pre_time_s, post_time_s, bin_size_ms)
    n_time_bins = len(template_bins) - 1
    n_events = len(event_times)
    n_units = len(unit_ids)
    
    # Pre-allocate output
    binned = np.zeros((n_events, n_units, n_time_bins), dtype=np.float32)
    
    # Bin spikes for each event
    for event_idx, event_time in enumerate(event_times):
        # Absolute bin edges for this event
        event_bins = template_bins + event_time
        
        for unit_idx, uid in enumerate(unit_ids):
            if uid in spike_times_dict:
                spikes = spike_times_dict[uid]
                # Filter spikes within window
                mask = (spikes >= event_bins[0]) & (spikes <= event_bins[-1])
                window_spikes = spikes[mask]
                binned[event_idx, unit_idx] = bin_spike_times(window_spikes, event_bins)
    
    # Time bin centers (in ms, relative to event)
    time_bins_ms = (template_bins[:-1] + template_bins[1:]) / 2 * 1000
    
    if as_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        binned = torch.from_numpy(binned)
    
    return {
        "binned": binned,
        "unit_ids": list(unit_ids),
        "time_bins_ms": time_bins_ms,
        "bin_size_ms": bin_size_ms,
        "bin_edges_relative": template_bins,
    }
