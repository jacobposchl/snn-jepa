"""
Predictor model for predicting future latent representations of neural data.

Part of the JEPA (Joint-Embedding Predictive Architecture) framework.

The predictor takes context (past latent representations) and predicts future
latent representations, enabling self-supervised learning by predicting future
states in the learned latent space.

Currently uses MLP architecture, designed to be easily replaceable with
RNN/LSTM/Transformer in the future.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    """
    Multi-Layer Perceptron predictor for JEPA.
    
    Takes a sequence of past latent representations and predicts future ones.
    
    Architecture:
        Context (context_len * latent_dim) -> Hidden layers -> 
        Prediction (horizon * latent_dim)
    
    Args:
        latent_dim: Dimensionality of input/output latent representations
        horizon: Number of future time steps to predict
        context_len: Number of past time steps to use as context
        hidden_dims: List of hidden layer dimensions. If None, uses default:
                    [latent_dim * (context_len + horizon), latent_dim * horizon]
        activation: Activation function ('relu', 'gelu', 'tanh', 'elu')
        dropout: Dropout probability (0.0 = no dropout)
        use_batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        latent_dim: int,
        horizon: int = 1,
        context_len: int = 1,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.context_len = context_len
        
        # Default hidden architecture if not specified
        if hidden_dims is None:
            hidden_dims = [
                latent_dim * (context_len + horizon),
                latent_dim * horizon,
            ]
        
        # Input: flattened context (context_len * latent_dim)
        input_dim = context_len * latent_dim
        # Output: flattened predictions (horizon * latent_dim)
        output_dim = horizon * latent_dim
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
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
        
    def forward(
        self,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict future latent representations from context.
        
        Args:
            context: Context latents of shape:
                - (batch, context_len, latent_dim) for sequence context
                - (batch, latent_dim) if context_len=1 (will be expanded)
        
        Returns:
            Predicted future latents of shape (batch, horizon, latent_dim)
        """
        # Handle 2D input (batch, latent_dim) when context_len=1
        if context.dim() == 2:
            if self.context_len != 1:
                raise ValueError(
                    f"Expected 3D input when context_len={self.context_len}, "
                    f"got shape {context.shape}"
                )
            # Add time dimension
            context = context.unsqueeze(1)  # (batch, 1, latent_dim)
        
        batch_size = context.shape[0]
        
        # Validate input shape
        if context.shape[1] != self.context_len or context.shape[2] != self.latent_dim:
            raise ValueError(
                f"Expected context shape (batch, {self.context_len}, {self.latent_dim}), "
                f"got {context.shape}"
            )
        
        # Flatten context: (batch, context_len * latent_dim)
        context_flat = context.view(batch_size, -1)
        
        # Predict: (batch, horizon * latent_dim)
        pred_flat = self.mlp(context_flat)
        
        # Reshape to (batch, horizon, latent_dim)
        pred = pred_flat.view(batch_size, self.horizon, self.latent_dim)
        
        return pred


class RNNPredictor(nn.Module):
    """
    RNN-based predictor for JEPA.
    
    Uses recurrent layers (LSTM or GRU) to process sequential context and predict future latents.
    Better suited for long-term dependencies than MLP.
    
    Args:
        latent_dim: Dimensionality of input/output latent representations
        horizon: Number of future time steps to predict
        context_len: Number of past time steps to use as context
        hidden_dim: Hidden dimension of RNN (default: 256)
        num_layers: Number of RNN layers (default: 2)
        rnn_type: Type of RNN ('LSTM' or 'GRU', default: 'LSTM')
        dropout: Dropout probability in RNN (default: 0.0)
    """
    
    def __init__(
        self,
        latent_dim: int,
        horizon: int = 1,
        context_len: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,
        rnn_type: str = "LSTM",
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.context_len = context_len
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.upper()
        
        # RNN layer
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                input_size=latent_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Prediction head: hidden state -> sequence of future embeddings
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * latent_dim),
        )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict future latent representations from context.
        
        Args:
            context: Context latents of shape (batch, context_len, latent_dim)
        
        Returns:
            Predicted future latents of shape (batch, horizon, latent_dim)
        """
        batch_size = context.shape[0]
        
        # Process context through RNN
        # Output: (batch, context_len, hidden_dim)
        # Hidden: tuple of (num_layers, batch, hidden_dim) for LSTM, or tensor for GRU
        rnn_out, hidden = self.rnn(context)
        
        # Use last hidden state to predict future
        if self.rnn_type == "LSTM":
            # hidden is (h, c), we want h
            last_hidden = hidden[0][-1, :, :]  # (batch, hidden_dim)
        else:
            # hidden is already the state
            last_hidden = hidden[-1, :, :]  # (batch, hidden_dim)
        
        # Predict future embeddings
        pred_flat = self.prediction_head(last_hidden)  # (batch, horizon * latent_dim)
        pred = pred_flat.view(batch_size, self.horizon, self.latent_dim)
        
        return pred


class NeuralPredictor(nn.Module):
    """
    High-level predictor wrapper for JEPA.
    
    This is the main interface for predicting future latent representations.
    Currently wraps MLPPredictor, but designed to allow easy swapping to
    RNN/LSTM/Transformer predictors in the future.
    
    Args:
        latent_dim: Dimensionality of latent representations
        horizon: Number of future time steps to predict
        context_len: Number of past time steps to use as context
        predictor_type: Type of predictor ('mlp' or future types like 'rnn', 'transformer')
        **kwargs: Additional arguments passed to the underlying predictor
    """
    
    def __init__(
        self,
        latent_dim: int,
        horizon: int = 1,
        context_len: int = 1,
        predictor_type: str = "mlp",
        **kwargs,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.context_len = context_len
        self.predictor_type = predictor_type
        
        if predictor_type.lower() == "mlp":
            self.predictor = MLPPredictor(
                latent_dim, horizon, context_len, **kwargs
            )
        elif predictor_type.lower() == "rnn":
            self.predictor = RNNPredictor(
                latent_dim, horizon, context_len, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown predictor_type: {predictor_type}. "
                "Supported types: 'mlp', 'rnn'"
            )
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict future latent representations from context.
        
        Args:
            context: Context latents of shape:
                - (batch, context_len, latent_dim) for sequence context
                - (batch, latent_dim) if context_len=1 (will be expanded)
        
        Returns:
            Predicted future latents of shape (batch, horizon, latent_dim)
        """
        return self.predictor(context)
    
    def predict_next(
        self,
        context: torch.Tensor,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """
        Predict next n_steps by iteratively using predictions as context.
        
        Useful for longer-term predictions beyond the trained horizon.
        
        Args:
            context: Initial context of shape (batch, context_len, latent_dim)
            n_steps: Number of steps to predict (can be > horizon)
        
        Returns:
            Predicted latents of shape (batch, n_steps, latent_dim)
        """
        batch_size = context.shape[0]
        device = context.device
        
        # Store all predictions
        all_predictions = []
        
        # Current context window
        current_context = context
        
        for step in range(n_steps):
            # Predict next horizon steps
            next_pred = self.predictor(current_context)  # (batch, horizon, latent_dim)
            
            # Store first prediction
            all_predictions.append(next_pred[:, 0:1, :])  # (batch, 1, latent_dim)
            
            # Update context: remove oldest, add newest prediction
            if step < n_steps - 1:  # Not the last step
                # Shift context window
                current_context = torch.cat(
                    [current_context[:, 1:, :], next_pred[:, 0:1, :]], dim=1
                )
        
        # Concatenate all predictions
        predictions = torch.cat(all_predictions, dim=1)  # (batch, n_steps, latent_dim)
        
        return predictions
    
    def compute_prediction_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        criterion: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prediction loss for training.
        
        Args:
            context: Context latents of shape (batch, context_len, latent_dim)
            target: Target future latents of shape (batch, horizon, latent_dim)
            criterion: Loss function (default: MSE loss)
        
        Returns:
            Tuple of (predictions, loss)
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        predictions = self.predictor(context)
        loss = criterion(predictions, target)
        
        return predictions, loss
    
    def get_latent_dim(self) -> int:
        """Return the dimensionality of latent representations."""
        return self.latent_dim
    
    def get_horizon(self) -> int:
        """Return the prediction horizon."""
        return self.horizon
    
    def get_context_len(self) -> int:
        """Return the context length."""
        return self.context_len


# Convenience function for creating predictors
def create_predictor(
    latent_dim: int,
    horizon: int = 1,
    context_len: int = 1,
    predictor_type: str = "mlp",
    **kwargs,
) -> NeuralPredictor:
    """
    Factory function to create a neural predictor.
    
    Args:
        latent_dim: Dimensionality of latent representations
        horizon: Number of future time steps to predict
        context_len: Number of past time steps to use as context
        predictor_type: Type of predictor ('mlp' for now)
        **kwargs: Additional arguments for the predictor
    
    Returns:
        NeuralPredictor instance
    """
    return NeuralPredictor(
        latent_dim, horizon, context_len, predictor_type, **kwargs
    )
