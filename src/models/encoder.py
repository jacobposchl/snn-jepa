"""
Encoder model for creating a latent representation of binned neural data.

Currently uses an MLP (Multi-Layer Perceptron) architecture, designed to be
easily replaceable with a transformer in the future.

The encoder takes binned spike counts and outputs compact latent representations
suitable for downstream tasks like prediction (JEPA framework).
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron encoder for binned neural data.
    
    Processes neural population activity at each time step independently,
    mapping from n_units-dimensional input to latent_dim-dimensional output.
    
    Architecture:
        Input (n_units) -> Hidden layers -> Output (latent_dim)
        
    Args:
        n_units: Number of input units (neurons) in the population
        latent_dim: Dimensionality of the latent representation
        hidden_dims: List of hidden layer dimensions. If None, uses default
                    architecture: [n_units * 2, n_units, n_units // 2]
        activation: Activation function ('relu', 'gelu', 'tanh', 'elu')
        dropout: Dropout probability (0.0 = no dropout)
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        n_units: int,
        latent_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        
        self.n_units = n_units
        self.latent_dim = latent_dim
        
        # Default hidden architecture if not specified
        if hidden_dims is None:
            hidden_dims = [
                n_units * 2,
                n_units,
                n_units // 2,
            ]
        
        # Build layers
        layers = []
        dims = [n_units] + hidden_dims + [latent_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation, batch norm, and dropout (except for last layer)
            if i < len(dims) - 2:  # Not the last layer
                # Activation
                if activation.lower() == "relu":
                    layers.append(nn.ReLU())
                elif activation.lower() == "gelu":
                    layers.append(nn.GELU())
                elif activation.lower() == "tanh":
                    layers.append(nn.Tanh())
                elif activation.lower() == "elu":
                    layers.append(nn.ELU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Batch normalization
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                
                # Dropout
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape:
                - (batch, n_units) for single time step
                - (batch, n_time_bins, n_units) for multiple time steps
                - (batch, n_time_bins, n_units) will be processed per timestep
        
        Returns:
            Latent representations of shape:
                - (batch, latent_dim) for single time step input
                - (batch, n_time_bins, latent_dim) for multiple time steps
        """
        original_shape = x.shape
        
        # Handle 2D input (batch, n_units) - single time step
        if x.dim() == 2:
            return self.mlp(x)
        
        # Handle 3D input (batch, n_time_bins, n_units) - multiple time steps
        elif x.dim() == 3:
            batch_size, n_time_bins, n_units = x.shape
            
            # Reshape to (batch * n_time_bins, n_units) for processing
            x_flat = x.view(-1, n_units)
            
            # Encode
            latent_flat = self.mlp(x_flat)
            
            # Reshape back to (batch, n_time_bins, latent_dim)
            latent = latent_flat.view(batch_size, n_time_bins, self.latent_dim)
            
            return latent
        
        else:
            raise ValueError(
                f"Expected input of shape (batch, n_units) or "
                f"(batch, n_time_bins, n_units), got {x.shape}"
            )


class NeuralEncoder(nn.Module):
    """
    High-level encoder wrapper for neural data.
    
    This is the main interface for encoding binned neural data. Currently wraps
    MLPEncoder, but designed to allow easy swapping to TransformerEncoder or
    other architectures in the future.
    
    Args:
        n_units: Number of input units (neurons)
        latent_dim: Dimensionality of latent representation
        encoder_type: Type of encoder ('mlp' or future types like 'transformer')
        **kwargs: Additional arguments passed to the underlying encoder
    """
    
    def __init__(
        self,
        n_units: int,
        latent_dim: int,
        encoder_type: str = "mlp",
        **kwargs,
    ):
        super().__init__()
        
        self.n_units = n_units
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type
        
        if encoder_type.lower() == "mlp":
            self.encoder = MLPEncoder(n_units, latent_dim, **kwargs)
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}. "
                "Currently only 'mlp' is supported."
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
        # For now, we average the window before encoding
        window_means = windows_flat.mean(dim=1)  # (n_windows * batch, n_units)
        encoded_flat = self.encoder(window_means)
        
        # Reshape back: (n_windows, batch, latent_dim)
        encoded_windows = encoded_flat.view(n_windows, batch_size, self.latent_dim)
        
        # Transpose to (batch, n_windows, latent_dim)
        encoded_windows = encoded_windows.transpose(0, 1)
        
        return encoded_windows, torch.tensor(indices, device=x.device)
    
    def get_latent_dim(self) -> int:
        """Return the dimensionality of the latent representation."""
        return self.latent_dim


# Convenience function for creating encoders
def create_encoder(
    n_units: int,
    latent_dim: int,
    encoder_type: str = "mlp",
    **kwargs,
) -> NeuralEncoder:
    """
    Factory function to create a neural encoder.
    
    Args:
        n_units: Number of input units
        latent_dim: Latent dimensionality
        encoder_type: Type of encoder ('mlp' for now)
        **kwargs: Additional arguments for the encoder
    
    Returns:
        NeuralEncoder instance
    """
    return NeuralEncoder(n_units, latent_dim, encoder_type, **kwargs)
