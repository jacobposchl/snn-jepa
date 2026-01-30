"""
Windowing utilities for creating context-target pairs for JEPA training.

Implements double sliding window approach where at each time step we extract:
- Context window: Past neural activity used as input
- Target window: Future neural activity to predict

Core functions:
    - create_context_target_windows: Create sliding windows with context and target
    - create_multi_target_windows: Create windows with multiple target positions
    - window_trials: Apply windowing to trial-aligned data
"""

from typing import Optional, Union, Tuple, List
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def create_context_target_windows(
    data: np.ndarray,
    context_size: int,
    target_size: int,
    target_offset: int,
    stride: int,
    as_torch: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create context-target window pairs from binned neural data.
    
    Uses a sliding window approach where each context window is paired with
    a target window at a specified offset.
    
    Args:
        data: Binned neural data, shape (n_units, n_time_bins)
        context_size: Number of time bins in context window
        target_size: Number of time bins in target window
        target_offset: Number of bins between end of context and start of target
                      0 = target immediately follows context
                      >0 = gap between context and target
                      <0 = overlap between context and target
        stride: Step size for sliding window (default: 1)
        as_torch: If True, return torch.Tensor instead of numpy array
        
    Returns:
        Tuple of (context_windows, target_windows):
            - context_windows: shape (n_windows, n_units, context_size)
            - target_windows: shape (n_windows, n_units, target_size)
            
    Example:
        # Create windows with 100ms context, 50ms target, 20ms gap
        # (assuming 10ms bins: 10 bins context, 5 bins target, 2 bins gap)
        contexts, targets = create_context_target_windows(
            data=binned_data,  # shape (n_units, 1000)
            context_size=10,
            target_size=5,
            target_offset=2,
            stride=1
        )
        # contexts shape: (n_windows, n_units, 10)
        # targets shape: (n_windows, n_units, 5)
    """
    n_units, n_time_bins = data.shape
    
    # Calculate required total length for one window
    total_window_length = context_size + target_offset + target_size
    
    if total_window_length > n_time_bins:
        raise ValueError(
            f"Total window length ({total_window_length}) exceeds "
            f"data length ({n_time_bins})"
        )
    
    # Calculate number of windows
    n_windows = (n_time_bins - total_window_length) // stride + 1
    
    # Pre-allocate arrays
    context_windows = np.zeros((n_windows, n_units, context_size), dtype=np.float32)
    target_windows = np.zeros((n_windows, n_units, target_size), dtype=np.float32)
    
    # Extract windows
    for i in range(n_windows):
        start_idx = i * stride
        context_end = start_idx + context_size
        target_start = context_end + target_offset
        target_end = target_start + target_size
        
        context_windows[i] = data[:, start_idx:context_end]
        target_windows[i] = data[:, target_start:target_end]
    
    if as_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        context_windows = torch.from_numpy(context_windows)
        target_windows = torch.from_numpy(target_windows)
    
    return context_windows, target_windows


def window_trials(
    trial_data: np.ndarray,
    context_size: int,
    target_size: int,
    target_offset: int = 0,
    stride: int = 1,
    as_torch: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply windowing to trial-aligned data.
    
    Processes multiple trials independently, creating context-target pairs
    within each trial. Useful for trial-based JEPA training.
    
    Args:
        trial_data: Trial-aligned binned data, shape (n_trials, n_units, n_time_bins)
        context_size: Number of time bins in context window
        target_size: Number of time bins in target window
        target_offset: Bins between context and target
        stride: Step size for sliding window within each trial
        as_torch: If True, return torch.Tensor
        
    Returns:
        Tuple of (context_windows, target_windows):
            - context_windows: shape (n_total_windows, n_units, context_size)
            - target_windows: shape (n_total_windows, n_units, target_size)
            
        Note: n_total_windows = n_trials * n_windows_per_trial
        
    Example:
        # Apply windowing to trial-aligned data from bin_trial_aligned
        result = bin_trial_aligned(...)  # shape (100, 50, 200) - 100 trials, 50 units, 200 bins
        contexts, targets = window_trials(
            trial_data=result['binned'],
            context_size=10,
            target_size=5,
            target_offset=2
        )
    """
    n_trials, n_units, n_time_bins = trial_data.shape
    
    # Process each trial to collect all windows
    all_contexts = []
    all_targets = []
    
    for trial_idx in range(n_trials):
        trial = trial_data[trial_idx]  # shape (n_units, n_time_bins)
        
        try:
            contexts, targets = create_context_target_windows(
                data=trial,
                context_size=context_size,
                target_size=target_size,
                target_offset=target_offset,
                stride=stride,
                as_torch=False,  # Keep as numpy for concatenation
            )
            all_contexts.append(contexts)
            all_targets.append(targets)
        except ValueError:
            # Skip trials that are too short
            continue
    
    if not all_contexts:
        raise ValueError("No valid windows could be created from any trial")
    
    # Concatenate all windows across trials
    context_windows = np.concatenate(all_contexts, axis=0)
    target_windows = np.concatenate(all_targets, axis=0)
    
    if as_torch:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        context_windows = torch.from_numpy(context_windows)
        target_windows = torch.from_numpy(target_windows)
    
    return context_windows, target_windows


def get_window_metadata(
    n_time_bins: int,
    context_size: int,
    target_size: int,
    target_offset: int,
    stride: int,
    bin_size_ms: float,
) -> dict:
    """
    Calculate metadata for windowing configuration.
    
    Useful for understanding the temporal structure of the windows before
    creating them.
    
    Args:
        n_time_bins: Total number of time bins in data
        context_size: Number of bins in context window
        target_size: Number of bins in target window
        target_offset: Bins between context and target
        stride: Stride for sliding window
        bin_size_ms: Size of each time bin in milliseconds
        
    Returns:
        Dictionary with windowing metadata:
            - n_windows: Number of windows that will be created
            - context_duration_ms: Duration of context in milliseconds
            - target_duration_ms: Duration of target in milliseconds
            - offset_duration_ms: Gap/overlap duration in milliseconds
            - total_window_duration_ms: Total duration covered by one window
            - coverage_ms: Total time covered by all windows
    """
    total_window_length = context_size + target_offset + target_size
    n_windows = (n_time_bins - total_window_length) // stride + 1
    
    return {
        "n_windows": max(0, n_windows),
        "context_duration_ms": context_size * bin_size_ms,
        "target_duration_ms": target_size * bin_size_ms,
        "offset_duration_ms": target_offset * bin_size_ms,
        "total_window_duration_ms": total_window_length * bin_size_ms,
        "coverage_ms": (n_time_bins - total_window_length + stride) * bin_size_ms,
        "context_size_bins": context_size,
        "target_size_bins": target_size,
        "target_offset_bins": target_offset,
        "stride_bins": stride,
    }


def validate_window_params(
    data_shape: Tuple[int, ...],
    context_size: int,
    target_size: int,
    target_offset: int,
    stride: int,
) -> None:
    """
    Validate windowing parameters against data shape.
    
    Raises ValueError if parameters are invalid.
    
    Args:
        data_shape: Shape of input data (n_units, n_time_bins) or (n_trials, n_units, n_time_bins)
        context_size: Context window size
        target_size: Target window size
        target_offset: Offset between context and target
        stride: Sliding window stride
    """

    if len(data_shape) == 2:
        n_time_bins = data_shape[1]
    elif len(data_shape) == 3:
        n_time_bins = data_shape[2]
    else:
        raise ValueError(f"Expected 2D or 3D data, got shape {data_shape}")
    
    if context_size <= 0:
        raise ValueError(f"context_size must be positive, got {context_size}")
    
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    
    total_window_length = context_size + target_offset + target_size
    
    if total_window_length > n_time_bins:
        raise ValueError(
            f"Total window length ({total_window_length} bins) exceeds "
            f"data length ({n_time_bins} bins). "
            f"Breakdown: context={context_size}, offset={target_offset}, "
            f"target={target_size}"
        )
    
    n_windows = (n_time_bins - total_window_length) // stride + 1
    
    if n_windows <= 0:
        raise ValueError(
            f"Configuration produces {n_windows} windows. "
            f"Data has {n_time_bins} bins, but need {total_window_length} bins per window."
        )
