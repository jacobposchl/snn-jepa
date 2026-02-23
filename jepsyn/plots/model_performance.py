"""
Model performance visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional


def plot_prediction_vs_actual(
    predicted: torch.Tensor,
    target: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = "Predicted vs Actual Embeddings"
) -> plt.Axes:
    """
    Scatter plot of predicted vs actual latent coordinates.
    
    Args:
        predicted: Predicted embeddings (N, latent_dim)
        target: Target embeddings (N, latent_dim)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    pred_np = predicted.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    
    ax.scatter(target_np, pred_np, alpha=0.3, s=1)
    
    min_val = min(target_np.min(), pred_np.min())
    max_val = max(target_np.max(), pred_np.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Actual Embedding Value')
    ax.set_ylabel('Predicted Embedding Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax


def plot_prediction_error_distribution(
    predicted: torch.Tensor,
    target: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = "Prediction Error Distribution"
) -> plt.Axes:
    """
    Plot distribution of prediction errors.
    
    Args:
        predicted: Predicted embeddings (N, latent_dim)
        target: Target embeddings (N, latent_dim)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    errors = (pred_np - target_np).flatten()
    
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero error')
    ax.axvline(x=np.mean(errors), color='blue', linestyle='--', 
               label=f'Mean: {np.mean(errors):.4f}')
    
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_mse_per_dimension(
    predicted: torch.Tensor,
    target: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = "MSE per Latent Dimension"
) -> plt.Axes:
    """
    Plot MSE for each latent dimension.
    
    Args:
        predicted: Predicted embeddings (N, latent_dim)
        target: Target embeddings (N, latent_dim)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    pred_np = predicted.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    mse_per_dim = ((pred_np - target_np) ** 2).mean(axis=0)
    
    ax.bar(range(len(mse_per_dim)), mse_per_dim, alpha=0.7, edgecolor='black')
    ax.axhline(y=np.mean(mse_per_dim), color='red', linestyle='--', 
               label=f'Mean MSE: {np.mean(mse_per_dim):.4f}')
    
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('MSE')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    return ax
