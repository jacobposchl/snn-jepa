"""
Latent space visualization utilities for analyzing learned representations.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import Optional
from sklearn.decomposition import PCA


def plot_latent_distribution(
    embeddings: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = "Latent Space Distribution"
) -> plt.Axes:
    """
    Plot histogram of latent representation values.
    
    Args:
        embeddings: Tensor of shape (N, latent_dim)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    embeddings_np = embeddings.detach().cpu().numpy().flatten()
    
    ax.hist(embeddings_np, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', label='Zero')
    
    ax.set_xlabel('Embedding Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_latent_dimensionality(
    embeddings: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = "Latent Space Dimensionality"
) -> plt.Axes:
    """
    Plot PCA variance explained to assess effective dimensionality.
    
    Args:
        embeddings: Tensor of shape (N, latent_dim)
        ax: Matplotlib axes
        title: Plot title
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    embeddings_np = embeddings.detach().cpu().numpy()
    
    pca = PCA()
    pca.fit(embeddings_np)
    
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    
    ax.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 
            marker='o', linewidth=2)
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% variance')
    
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Variance Explained')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def plot_prediction_accuracy(
    predicted: torch.Tensor,
    target: torch.Tensor,
    ax: Optional[plt.Axes] = None,
    title: str = "Prediction Accuracy"
) -> plt.Axes:
    """
    Plot cosine similarity between predicted and target embeddings.
    
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
    
    similarities = []
    for p, t in zip(pred_np, target_np):
        cos_sim = np.dot(p, t) / (np.linalg.norm(p) * np.linalg.norm(t))
        similarities.append(cos_sim)
    
    ax.hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(similarities), color='red', linestyle='--', 
               label=f'Mean: {np.mean(similarities):.3f}')
    
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax
