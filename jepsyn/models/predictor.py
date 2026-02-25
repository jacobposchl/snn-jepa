"""
Predictor model for the masking JEPA framework.

The predictor maps context latents (from partially-masked input) to predicted target
latents (from fully-observed input), enabling self-supervised learning entirely in
latent space.

Both input and output have shape [B, L, D] — the same fixed latent set produced by
PerceiverEncoder. There is no temporal forecasting axis; the predictor refines the
full latent set using self-attention.

The predictor is intentionally narrow (fewer layers / smaller FFN) relative to the
encoder. If the predictor is too expressive it can trivially map any context latent
to any target, preventing the encoder from learning useful representations.
"""

import torch
import torch.nn as nn


class PerceiverPredictor(nn.Module):
    """
    Narrow Transformer predictor for masking JEPA.

    Operates over the full latent set produced by PerceiverEncoder:
        Z_ctx [B, L, D]  →  Z_pred [B, L, D]

    All L latent slots are always present (no variable-length padding here), so
    no attention mask is needed.

    Args:
        d_model         : Latent dimension — must match the encoder's d_model.
        n_layers        : Number of Transformer blocks. Keep smaller than encoder's
                          n_self_attn_layers (default 2 vs encoder's 4).
        n_heads         : Attention heads. Keep smaller than encoder (default 4 vs 8).
        dim_feedforward : FFN hidden dim. Keep smaller than encoder (default 512 vs 1024).
        dropout         : Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,   # pre-LN, matches encoder style
            ),
            num_layers=n_layers,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_ctx : [B, L, D] — context latents from PerceiverEncoder
                    (encoder ran with only the unmasked spike events visible).

        Returns:
            z_pred : [B, L, D] — predicted target latents.
        """
        return self.output_norm(self.blocks(z_ctx))


class NeuralPredictor(nn.Module):
    """
    High-level predictor wrapper.

    Wraps PerceiverPredictor and provides the stable public interface consumed
    by multi_session.py.

    Args:
        d_model        : Latent dimension — must match the encoder's d_model.
        predictor_type : Currently only 'transformer' is supported.
        **kwargs       : Forwarded to PerceiverPredictor
                         (n_layers, n_heads, dim_feedforward, dropout).
    """

    def __init__(
        self,
        d_model: int = 256,
        predictor_type: str = "transformer",
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.predictor_type = predictor_type

        if predictor_type.lower() == "transformer":
            self.predictor = PerceiverPredictor(d_model=d_model, **kwargs)
        else:
            raise ValueError(
                f"Unknown predictor_type: '{predictor_type}'. "
                "Currently only 'transformer' is supported."
            )

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        """
        Predict target latents from context latents.

        Args:
            z_ctx  : [B, L, D] — context latents (masked input encoding).

        Returns:
            z_pred : [B, L, D] — predicted target latents.
        """
        return self.predictor(z_ctx)

    def get_latent_dim(self) -> int:
        """Return the latent dimension D."""
        return self.d_model
