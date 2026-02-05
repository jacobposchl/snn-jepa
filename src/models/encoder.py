"""
Encoder model for creating a latent representation of binned neural data.

Uses a Transformer architecture to capture temporal dependencies in neural
population activity across time.

The encoder takes binned spike counts and outputs compact latent representations
suitable for downstream tasks like prediction (JEPA framework).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for binned neural data.
    
    Processes neural population activity across time using self-attention,
    allowing each timestep to attend to all other timesteps in the sequence.
    
    Architecture:
        Input projection -> Positional encoding -> Transformer blocks -> Output projection
        
    Args:
        n_units: Number of input units (neurons) in the population
        latent_dim: Dimensionality of the latent representation
        d_model: Internal dimensionality of transformer (default: 256)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for positional encoding (default: 5000)
    """
    
    def __init__(
        self,
        n_units: int,
        latent_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
    ):
        super().__init__()
        
        self.n_units = n_units
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Input projection: map from n_units to d_model
        self.input_projection = nn.Linear(n_units, d_model)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection: map from d_model to latent_dim
        self.output_projection = nn.Linear(d_model, latent_dim)
        
        # Layer norm for output
        self.output_norm = nn.LayerNorm(latent_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize output projection with small weights to avoid variance collapse
        nn.init.xavier_normal_(self.output_projection.weight, gain=0.01)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer encoder.
        
        Args:
            x: Input tensor of shape:
                - (batch, n_units) for single time step -> will be unsqueezed
                - (batch, n_time_bins, n_units) for multiple time steps
        
        Returns:
            Latent representations of shape:
                - (batch, latent_dim) for single time step input
                - (batch, n_time_bins, latent_dim) for multiple time steps
        """
        squeeze_output = False
        
        # Handle 2D input (batch, n_units) - single time step
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, n_units)
            squeeze_output = True
        
        # Now x is (batch, n_time_bins, n_units)
        batch_size, seq_len, n_units = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Output projection
        x = self.output_projection(x)  # (batch, seq_len, latent_dim)
        x = self.output_norm(x)
        
        # Squeeze back to 2D if input was 2D
        if squeeze_output:
            x = x.squeeze(1)  # (batch, latent_dim)
        
        return x


class NeuralEncoder(nn.Module):
    """
    High-level encoder wrapper for neural data.
    
    This is the main interface for encoding binned neural data. Wraps
    TransformerEncoder for capturing temporal dependencies.
    
    Args:
        n_units: Number of input units (neurons)
        latent_dim: Dimensionality of latent representation
        encoder_type: Type of encoder ('transformer')
        **kwargs: Additional arguments passed to the underlying encoder
    """
    
    def __init__(
        self,
        n_units: int,
        latent_dim: int,
        encoder_type: str = "transformer",
        **kwargs,
    ):
        super().__init__()
        
        self.n_units = n_units
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        
        if encoder_type.lower() == "transformer":
            self.encoder = TransformerEncoder(n_units, latent_dim, **kwargs)
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                "Currently only 'transformer' is supported."
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode binned neural data.
        
        Args:
            x: Binned spike counts, shape:
                - (batch, n_units) for single time step
                - (batch, n_time_bins, n_units) for multiple time steps
        
        Returns:
            Latent representations, shape:
                - (batch, latent_dim) for single time step
                - (batch, n_time_bins, latent_dim) for multiple time steps
        """
        return self.encoder(x)
    
    def encode_window(
        self,
        x: torch.Tensor,
        window_size: int = 1,
        stride: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode using sliding windows over time.
        
        Useful for creating overlapping context windows for prediction tasks.
        
        Args:
            x: Input of shape (batch, n_time_bins, n_units)
            window_size: Number of time bins per window
            stride: Step size between windows
        
        Returns:
            Tuple of:
                - encoded_windows: (batch, n_windows, latent_dim)
                - window_indices: (n_windows,) array of window start indices
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, n_time_bins, n_units), got {x.shape}"
            )
        
        batch_size, n_time_bins, n_units = x.shape
        
        # Create sliding windows
        windows = []
        indices = []
        
        for i in range(0, n_time_bins - window_size + 1, stride):
            window = x[:, i : i + window_size, :]  # (batch, window_size, n_units)
            windows.append(window)
            indices.append(i)
        
        if len(windows) == 0:
            raise ValueError(
                "No valid windows created. Try smaller window_size or larger input."
            )
        
        # Stack windows: (n_windows, batch, window_size, n_units)
        windows_tensor = torch.stack(windows, dim=0)
        n_windows = windows_tensor.shape[0]
        
        # Reshape to process all windows: (n_windows * batch, window_size, n_units)
        windows_flat = windows_tensor.view(-1, window_size, n_units)
        
        # Encode each window (output: n_windows * batch, latent_dim)
        # Output shape: (n_windows * batch, window_size, latent_dim)
        encoded_flat = self.encoder(windows_flat)
        
        # Take last timestep from each window: (n_windows * batch, latent_dim)
        encoded_flat = encoded_flat[:, -1, :]
        # Reshape back: (n_windows, batch, latent_dim)
        encoded_windows = encoded_flat.view(n_windows, batch_size, self.latent_dim)
        
        # Transpose to (batch, n_windows, latent_dim)
        encoded_windows = encoded_windows.transpose(0, 1)
        
        return encoded_windows, torch.tensor(indices, device=x.device)
    
    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent representation."""
        return self.latent_dim