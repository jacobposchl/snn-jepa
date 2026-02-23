"""
Neural activity visualization utilities for validating data quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict


def plot_spike_count_distribution(
    spike_counts: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Spike Count Distribution"
) -> plt.Axes:
    """
    Plot distribution of spike counts across windows.
    
    Args:
        spike_counts: Array of shape (num_windows, num_units)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    counts_flat = spike_counts.flatten()
    
    ax.hist(counts_flat, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(counts_flat), color='red', linestyle='--', 
               label=f'Mean: {np.mean(counts_flat):.2f}')
    ax.axvline(x=np.median(counts_flat), color='blue', linestyle='--', 
               label=f'Median: {np.median(counts_flat):.2f}')
    
    ax.set_xlabel('Spike Count per Window')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_unit_participation(
    spike_counts: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Unit Participation per Window"
) -> plt.Axes:
    """
    Plot number of active units per window.
    
    Args:
        spike_counts: Array of shape (num_windows, num_units)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    active_units = (spike_counts > 0).sum(axis=1)
    
    ax.hist(active_units, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(active_units), color='red', linestyle='--', 
               label=f'Mean: {np.mean(active_units):.1f}')
    
    ax.set_xlabel('Number of Active Units')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_firing_rate_stability(
    spike_counts: np.ndarray,
    window_size_ms: float,
    ax: Optional[plt.Axes] = None,
    title: str = "Firing Rate Stability Across Windows"
) -> plt.Axes:
    """
    Plot mean firing rate over time to check for stability.
    
    Args:
        spike_counts: Array of shape (num_windows, num_units)
        window_size_ms: Window duration in milliseconds
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    window_size_s = window_size_ms / 1000.0
    firing_rates = spike_counts / window_size_s
    mean_fr = firing_rates.mean(axis=1)
    
    ax.plot(mean_fr, linewidth=1, alpha=0.7)
    ax.axhline(y=np.mean(mean_fr), color='red', linestyle='--', 
               label=f'Overall Mean: {np.mean(mean_fr):.2f} Hz')
    
    ax.set_xlabel('Window Index')
    ax.set_ylabel('Mean Firing Rate (Hz)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_temporal_autocorrelation(
    spike_counts: np.ndarray,
    max_lag: int = 50,
    ax: Optional[plt.Axes] = None,
    title: str = "Temporal Autocorrelation"
) -> plt.Axes:
    """
    Plot autocorrelation of spike counts to verify temporal structure.
    
    Args:
        spike_counts: Array of shape (num_windows, num_units)
        max_lag: Maximum lag to compute
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    mean_counts = spike_counts.mean(axis=1)
    
    autocorr = np.correlate(mean_counts - mean_counts.mean(), 
                           mean_counts - mean_counts.mean(), 
                           mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr[:max_lag] / autocorr[0]
    
    ax.plot(range(len(autocorr)), autocorr, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Lag (windows)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return ax
