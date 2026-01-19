"""
Functions for binning neural spike data.

Converts raw spike times into binned spike counts suitable for neural network input.

Core functions:
    - bin_spike_times: Bin spike times for a single unit
    - bin_population: Bin spike times for multiple units simultaneously

Allen SDK integration:
    - bin_session: Bin all/selected units from an Allen SDK session
    - extract_trials: Extract trial-aligned binned data around stimulus events

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


def _get_units_df(session, spike_times_dict: dict) -> pd.DataFrame:
    """
    Get units DataFrame from session, handling different Allen SDK APIs.
    
    Args:
        session: Allen SDK session object
        spike_times_dict: Dict of spike times (fallback for unit IDs)
        
    Returns:
        DataFrame with unit metadata, indexed by unit_id
    """
    # Try different ways to access units
    units_df = None
    
    # Method 1: Direct attribute (some session types)
    if hasattr(session, 'units') and session.units is not None:
        try:
            units_df = session.units.copy()
        except Exception:
            pass
    
    # Method 2: get_units() method
    if units_df is None and hasattr(session, 'get_units'):
        try:
            units_df = session.get_units().copy()
        except Exception:
            pass
    
    # Method 3: Create minimal DataFrame from spike_times keys
    if units_df is None:
        unit_ids = list(spike_times_dict.keys())
        units_df = pd.DataFrame(index=unit_ids)
        units_df.index.name = 'unit_id'
    
    return units_df


def _get_stimulus_presentations(session) -> pd.DataFrame:
    """
    Get stimulus presentations DataFrame, handling different Allen SDK APIs.
    
    Args:
        session: Allen SDK session object
        
    Returns:
        DataFrame with stimulus presentation info
    """
    # Try different ways to access stimulus presentations
    if hasattr(session, 'stimulus_presentations'):
        try:
            return session.stimulus_presentations.copy()
        except Exception:
            pass
    
    if hasattr(session, 'get_stimulus_presentations'):
        try:
            return session.get_stimulus_presentations().copy()
        except Exception:
            pass
    
    # Return empty DataFrame if not available
    return pd.DataFrame()


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


def bin_session(
    session,
    bin_size_ms: float = 10.0,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    unit_ids: Optional[Sequence[int]] = None,
    area_filter: Optional[Union[str, list[str]]] = None,
    quality_filter: Optional[str] = "good",
    as_torch: bool = False,
) -> dict:
    """
    Bin all/selected units from an Allen SDK BehaviorEcephysSession.
    
    Args:
        session: Allen SDK BehaviorEcephysSession object
        bin_size_ms: Bin size in milliseconds (default: 10ms)
        start_time: Start time in seconds (default: session start)
        end_time: End time in seconds (default: session end)
        unit_ids: Specific unit IDs to include (overrides area/quality filters)
        area_filter: Brain area(s) to include, e.g. "VISp" or ["VISp", "VISl"]
        quality_filter: Unit quality filter - "good", "all", or None
        as_torch: If True, return torch.Tensor instead of numpy array
        
    Returns:
        Dict containing:
            - "binned": Binned spike counts, shape (n_units, n_bins)
            - "unit_ids": List of unit IDs
            - "bin_edges": Array of bin edges in seconds
            - "bin_size_ms": Bin size used
            - "units_df": DataFrame with unit metadata for included units
    """
    # Get spike times and units table
    spike_times_dict = session.spike_times
    units_df = session.units.copy()
    
    # Apply filters to select units
    if unit_ids is not None:
        # Use explicitly provided unit IDs
        selected_ids = [uid for uid in unit_ids if uid in spike_times_dict]
    else:
        # Apply area filter
        if area_filter is not None:
            if isinstance(area_filter, str):
                area_filter = [area_filter]
            units_df = units_df[units_df["ecephys_structure_acronym"].isin(area_filter)]
        
        # Apply quality filter
        if quality_filter == "good":
            # Filter for good units based on common quality metrics
            if "quality" in units_df.columns:
                units_df = units_df[units_df["quality"] == "good"]
            elif "isi_violations" in units_df.columns:
                # Fallback: use ISI violations < 0.5 as quality proxy
                units_df = units_df[units_df["isi_violations"] < 0.5]
        
        selected_ids = list(units_df.index)
    
    # Filter units_df to only selected units
    units_df = units_df.loc[units_df.index.isin(selected_ids)]
    
    # Determine time range
    if start_time is None or end_time is None:
        all_spikes = np.concatenate([
            spike_times_dict[uid] for uid in selected_ids 
            if len(spike_times_dict[uid]) > 0
        ])
        if start_time is None:
            start_time = float(np.min(all_spikes))
        if end_time is None:
            end_time = float(np.max(all_spikes))
    
    # Generate bin edges and bin the data
    bin_edges = get_time_bins(start_time, end_time, bin_size_ms)
    binned, unit_ids_ordered = bin_population(spike_times_dict, bin_edges, selected_ids)
    
    # Convert to torch if requested
    if as_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        binned = torch.from_numpy(binned)
    
    return {
        "binned": binned,
        "unit_ids": unit_ids_ordered,
        "bin_edges": bin_edges,
        "bin_size_ms": bin_size_ms,
        "units_df": units_df,
    }


def extract_trials(
    session,
    bin_size_ms: float = 10.0,
    pre_time_ms: float = 500.0,
    post_time_ms: float = 500.0,
    stimulus_name: Optional[str] = None,
    unit_ids: Optional[Sequence[int]] = None,
    area_filter: Optional[Union[str, list[str]]] = None,
    quality_filter: Optional[str] = "good",
    as_torch: bool = False,
) -> dict:
    """
    Extract trial-aligned binned data around stimulus presentations.
    
    Creates a 3D tensor of shape (n_trials, n_units, n_time_bins) where each
    trial is aligned to stimulus onset.
    
    Args:
        session: Allen SDK BehaviorEcephysSession object
        bin_size_ms: Bin size in milliseconds
        pre_time_ms: Time before stimulus onset to include (ms)
        post_time_ms: Time after stimulus onset to include (ms)
        stimulus_name: Filter for specific stimulus type (e.g., "natural_movie_one")
        unit_ids: Specific unit IDs to include
        area_filter: Brain area(s) to include
        quality_filter: Unit quality filter
        as_torch: If True, return torch.Tensor
        
    Returns:
        Dict containing:
            - "trials": Trial-aligned data, shape (n_trials, n_units, n_time_bins)
            - "unit_ids": List of unit IDs
            - "stimulus_df": DataFrame with stimulus presentation info
            - "time_bins_ms": Time bin centers relative to stimulus onset (ms)
            - "bin_size_ms": Bin size used
    """
    # Get stimulus presentations
    stim_df = session.stimulus_presentations.copy()
    
    if stimulus_name is not None:
        stim_df = stim_df[stim_df["stimulus_name"] == stimulus_name]
    
    if len(stim_df) == 0:
        raise ValueError(f"No stimulus presentations found for: {stimulus_name}")
    
    # Get spike times and filter units (reuse logic from bin_session)
    spike_times_dict = session.spike_times
    units_df = session.units.copy()
    
    if unit_ids is not None:
        selected_ids = [uid for uid in unit_ids if uid in spike_times_dict]
    else:
        if area_filter is not None:
            if isinstance(area_filter, str):
                area_filter = [area_filter]
            units_df = units_df[units_df["ecephys_structure_acronym"].isin(area_filter)]
        
        if quality_filter == "good":
            if "quality" in units_df.columns:
                units_df = units_df[units_df["quality"] == "good"]
            elif "isi_violations" in units_df.columns:
                units_df = units_df[units_df["isi_violations"] < 0.5]
        
        selected_ids = list(units_df.index)
    
    # Calculate dimensions
    pre_time_s = pre_time_ms / 1000.0
    post_time_s = post_time_ms / 1000.0
    trial_duration_s = pre_time_s + post_time_s
    
    # Template bin edges for a single trial (relative to onset)
    template_bins = get_time_bins(-pre_time_s, post_time_s, bin_size_ms)
    n_time_bins = len(template_bins) - 1
    n_trials = len(stim_df)
    n_units = len(selected_ids)
    
    # Pre-allocate output array
    trials = np.zeros((n_trials, n_units, n_time_bins), dtype=np.float32)
    
    # Extract each trial
    onset_times = stim_df["start_time"].values
    
    for trial_idx, onset in enumerate(onset_times):
        # Bin edges for this trial
        trial_bins = template_bins + onset
        
        for unit_idx, uid in enumerate(selected_ids):
            spikes = spike_times_dict[uid]
            # Only consider spikes within the trial window
            mask = (spikes >= trial_bins[0]) & (spikes <= trial_bins[-1])
            trial_spikes = spikes[mask]
            trials[trial_idx, unit_idx] = bin_spike_times(trial_spikes, trial_bins)
    
    # Time bin centers relative to onset (in ms)
    time_bins_ms = (template_bins[:-1] + template_bins[1:]) / 2 * 1000
    
    if as_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        trials = torch.from_numpy(trials)
    
    return {
        "trials": trials,
        "unit_ids": selected_ids,
        "stimulus_df": stim_df,
        "time_bins_ms": time_bins_ms,
        "bin_size_ms": bin_size_ms,
    }


# Convenience aliases for common bin sizes
def bin_1ms(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin at 1ms resolution."""
    return bin_spike_times(spike_times, bin_edges)


def bin_10ms(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin at 10ms resolution."""
    return bin_spike_times(spike_times, bin_edges)


def bin_50ms(spike_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Bin at 50ms resolution."""
    return bin_spike_times(spike_times, bin_edges)
