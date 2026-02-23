"""
Training visualization utilities for monitoring model training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


def plot_loss_curves(
    total_losses: List[float],
    pred_losses: List[float],
    sigreg_losses: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Training Loss Curves"
) -> plt.Axes:
    """
    Plot training loss curves over epochs.
    
    Args:
        total_losses: Total loss per epoch
        pred_losses: Prediction loss per epoch
        sigreg_losses: SIGReg loss per epoch
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(len(total_losses))
    
    ax.plot(epochs, total_losses, label='Total Loss', linewidth=2)
    ax.plot(epochs, pred_losses, label='Prediction Loss', linewidth=2)
    ax.plot(epochs, sigreg_losses, label='SIGReg Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_loss_ratio(
    pred_losses: List[float],
    sigreg_losses: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Prediction/SIGReg Loss Ratio"
) -> plt.Axes:
    """
    Plot ratio of prediction loss to SIGReg loss over epochs.
    
    Args:
        pred_losses: Prediction loss per epoch
        sigreg_losses: SIGReg loss per epoch
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(len(pred_losses))
    ratios = [p / s if s > 0 else 0 for p, s in zip(pred_losses, sigreg_losses)]
    
    ax.plot(epochs, ratios, linewidth=2, color='purple')
    ax.axhline(y=1.0, color='red', linestyle='--', label='Equal contribution')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Loss / SIGReg Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax
