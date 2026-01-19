"""
Raster plot utilities for visualizing neural spike data.

Functions:
    - plot_raster: Classic raster plot of spike times
    - plot_binned_heatmap: Heatmap visualization of binned spike counts
    - plot_raster_with_binned: Side-by-side comparison of raw vs binned
    - plot_trial_raster: Trial-aligned raster plot
    - plot_psth: Peri-stimulus time histogram (average firing rate)
"""

from typing import Optional, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def plot_raster(
    spike_times: Union[np.ndarray, dict[int, np.ndarray]],
    unit_ids: Optional[Sequence[int]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "black",
    marker: str = "|",
    markersize: float = 2.0,
    title: Optional[str] = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Unit",
) -> plt.Axes:
    """
    Plot a classic raster plot of spike times.
    
    Args:
        spike_times: Either a single array of spike times, or a dict mapping
                     unit_id -> spike times array
        unit_ids: If spike_times is a dict, optionally specify which units to plot
                  and in what order
        start_time: Start time for x-axis (default: min spike time)
        end_time: End time for x-axis (default: max spike time)
        ax: Matplotlib axes to plot on (creates new figure if None)
        color: Color for spike markers
        marker: Marker style (default "|" for vertical lines)
        markersize: Size of markers
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Handle single unit vs population
    if isinstance(spike_times, np.ndarray):
        spike_times = {0: spike_times}
        unit_ids = [0]
    elif unit_ids is None:
        unit_ids = list(spike_times.keys())
    
    # Plot each unit
    for row_idx, uid in enumerate(unit_ids):
        spikes = spike_times.get(uid, np.array([]))
        if len(spikes) > 0:
            # Filter to time range if specified
            if start_time is not None:
                spikes = spikes[spikes >= start_time]
            if end_time is not None:
                spikes = spikes[spikes <= end_time]
            
            ax.plot(
                spikes,
                np.full_like(spikes, row_idx),
                marker,
                color=color,
                markersize=markersize,
                linestyle="none",
            )
    
    # Set axis limits
    if start_time is not None and end_time is not None:
        ax.set_xlim(start_time, end_time)
    
    ax.set_ylim(-0.5, len(unit_ids) - 0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    # Y-axis ticks
    if len(unit_ids) <= 20:
        ax.set_yticks(range(len(unit_ids)))
        ax.set_yticklabels([str(uid) for uid in unit_ids])
    
    return ax


def plot_binned_heatmap(
    binned: np.ndarray,
    bin_edges: Optional[np.ndarray] = None,
    unit_ids: Optional[Sequence[int]] = None,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Time (s)",
    ylabel: str = "Unit",
) -> plt.Axes:
    """
    Plot binned spike counts as a heatmap.
    
    Args:
        binned: Binned spike counts, shape (n_units, n_bins)
        bin_edges: Array of bin edges for x-axis labeling
        unit_ids: Unit IDs for y-axis labeling
        ax: Matplotlib axes to plot on
        cmap: Colormap name
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        colorbar: Whether to show colorbar
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine extent
    if bin_edges is not None:
        extent = [bin_edges[0], bin_edges[-1], -0.5, binned.shape[0] - 0.5]
    else:
        extent = [0, binned.shape[1], -0.5, binned.shape[0] - 0.5]
    
    im = ax.imshow(
        binned,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )
    
    if colorbar:
        plt.colorbar(im, ax=ax, label="Spike count")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    # Y-axis ticks
    if unit_ids is not None and len(unit_ids) <= 20:
        ax.set_yticks(range(len(unit_ids)))
        ax.set_yticklabels([str(uid) for uid in unit_ids])
    
    return ax


def plot_raster_with_binned(
    spike_times: dict[int, np.ndarray],
    binned: np.ndarray,
    bin_edges: np.ndarray,
    unit_ids: Sequence[int],
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    figsize: tuple = (14, 8),
    title: Optional[str] = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """
    Plot raster and binned heatmap side by side for comparison.
    
    Args:
        spike_times: Dict mapping unit_id -> spike times array
        binned: Binned spike counts, shape (n_units, n_bins)
        bin_edges: Array of bin edges
        unit_ids: List of unit IDs in order
        start_time: Start time for display
        end_time: End time for display
        figsize: Figure size
        title: Overall figure title
        
    Returns:
        Tuple of (figure, (raster_axes, heatmap_axes))
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    if start_time is None:
        start_time = bin_edges[0]
    if end_time is None:
        end_time = bin_edges[-1]
    
    # Plot raster
    plot_raster(
        spike_times,
        unit_ids=unit_ids,
        start_time=start_time,
        end_time=end_time,
        ax=ax1,
        title="Raw Spike Times (Raster)",
    )
    
    # Plot binned heatmap
    plot_binned_heatmap(
        binned,
        bin_edges=bin_edges,
        unit_ids=unit_ids,
        ax=ax2,
        title="Binned Spike Counts (Heatmap)",
    )
    
    ax2.set_xlim(start_time, end_time)
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    return fig, (ax1, ax2)


def plot_trial_raster(
    trials: np.ndarray,
    time_bins_ms: np.ndarray,
    unit_idx: int = 0,
    ax: Optional[plt.Axes] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    xlabel: str = "Time from stimulus (ms)",
    ylabel: str = "Trial",
) -> plt.Axes:
    """
    Plot trial-aligned raster for a single unit.
    
    Args:
        trials: Trial-aligned data, shape (n_trials, n_units, n_time_bins)
        time_bins_ms: Time bin centers relative to stimulus onset (ms)
        unit_idx: Index of unit to plot
        ax: Matplotlib axes
        cmap: Colormap
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    unit_data = trials[:, unit_idx, :]  # Shape: (n_trials, n_time_bins)
    
    extent = [time_bins_ms[0], time_bins_ms[-1], -0.5, unit_data.shape[0] - 0.5]
    
    im = ax.imshow(
        unit_data,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=extent,
    )
    
    # Add vertical line at stimulus onset
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Stimulus onset")
    
    plt.colorbar(im, ax=ax, label="Spike count")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    
    if title:
        ax.set_title(title)
    
    return ax


def plot_psth(
    trials: np.ndarray,
    time_bins_ms: np.ndarray,
    bin_size_ms: float,
    unit_idx: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    color: str = "blue",
    show_sem: bool = True,
    title: Optional[str] = None,
    xlabel: str = "Time from stimulus (ms)",
    ylabel: str = "Firing rate (Hz)",
) -> plt.Axes:
    """
    Plot peri-stimulus time histogram (PSTH) - average firing rate across trials.
    
    Args:
        trials: Trial-aligned data, shape (n_trials, n_units, n_time_bins)
        time_bins_ms: Time bin centers relative to stimulus onset (ms)
        bin_size_ms: Bin size in milliseconds (for Hz conversion)
        unit_idx: Index of unit to plot (if None, averages across all units)
        ax: Matplotlib axes
        color: Line color
        show_sem: Whether to show standard error of mean as shading
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Select unit or average across units
    if unit_idx is not None:
        data = trials[:, unit_idx, :]  # Shape: (n_trials, n_time_bins)
    else:
        data = trials.mean(axis=1)  # Average across units
    
    # Convert spike counts to firing rate (Hz)
    bin_size_s = bin_size_ms / 1000.0
    firing_rate = data / bin_size_s
    
    # Compute mean and SEM across trials
    mean_rate = firing_rate.mean(axis=0)
    
    ax.plot(time_bins_ms, mean_rate, color=color, linewidth=2)
    
    if show_sem and firing_rate.shape[0] > 1:
        sem = firing_rate.std(axis=0) / np.sqrt(firing_rate.shape[0])
        ax.fill_between(
            time_bins_ms,
            mean_rate - sem,
            mean_rate + sem,
            color=color,
            alpha=0.3,
        )
    
    # Add vertical line at stimulus onset
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Stimulus onset")
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    
    if title:
        ax.set_title(title)
    
    return ax
