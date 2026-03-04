"""
mae.py
MAE (Masked Autoencoder) decoder for ablation against LeJEPA.

Instead of predicting in latent space (JEPA-style), MAE reconstructs
the original masked spike token embeddings from the latent representations.

Architecture:
    NeuralEncoder (shared, same as LeJEPA)
    → MAEDecoder: latents [B, L, D] → reconstructed spike tokens [B, E, D]
    → MSE reconstruction loss against original (unmasked) spike embeddings
"""

import torch
import torch.nn as nn
from typing import Tuple


class MAEDecoder(nn.Module):
    """
    Lightweight Transformer decoder that reconstructs masked spike token
    embeddings from the encoder's latent representations.

    Takes the latent set Z [B, L, D] from NeuralEncoder and cross-attends
    over it to reconstruct the original spike token embeddings at masked
    positions [B, E, D].

    Args:
        d_model         : Model dimension (must match encoder d_model).
        n_heads         : Number of attention heads in decoder.
        n_layers        : Number of decoder Transformer layers.
        dim_feedforward : FFN hidden dimension.
        dropout         : Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Learned mask token: replaces masked spike embeddings as decoder queries
        self.mask_token = nn.Parameter(torch.randn(d_model))

        # Decoder: cross-attends over encoder latents Z to reconstruct spike embeddings
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection: maps decoder output back to d_model spike embedding space
        self.output_proj = nn.Linear(d_model, d_model)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        Z: torch.Tensor,            # [B, L, D] encoder latents
        x_target: torch.Tensor,     # [B, E, D] original spike embeddings (reconstruction target)
        ctx_mask: torch.Tensor,     # [B, E] bool, True = visible (context), False = masked
        attn_mask: torch.Tensor,    # [B, E] bool, True = real token (not padding)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct masked spike embeddings from encoder latents.

        Args:
            Z          : [B, L, D] — encoder latent set (memory for cross-attention).
            x_target   : [B, E, D] — original spike token embeddings before masking.
            ctx_mask   : [B, E] bool — True = context token (visible to encoder),
                         False = masked token (what we need to reconstruct).
            attn_mask  : [B, E] bool — True = real spike token, False = padding.

        Returns:
            x_pred     : [B, E, D] — reconstructed spike embeddings at ALL positions.
            mask_loss  : scalar    — MSE reconstruction loss at masked positions only.
        """
        B, E, D = x_target.shape

        # Build decoder input:
        # - visible tokens keep their original embedding
        # - masked tokens are replaced with the learned mask token
        x_decoder = x_target.clone()
        masked_positions = (~ctx_mask) & attn_mask      # True = masked AND real (not padding)
        x_decoder[masked_positions] = self.mask_token   # replace masked real tokens

        # Cross-attend over encoder latents Z to reconstruct spike embeddings
        # TransformerDecoder: tgt=decoder_input, memory=encoder_latents
        x_pred = self.decoder(x_decoder, Z)             # [B, E, D]
        x_pred = self.output_proj(self.output_norm(x_pred))

        # Reconstruction loss: MSE only at masked real token positions
        if masked_positions.any():
            pred_masked   = x_pred[masked_positions]    # [N_masked, D]
            target_masked = x_target[masked_positions]  # [N_masked, D]
            mask_loss     = nn.functional.mse_loss(pred_masked, target_masked)
        else:
            mask_loss = torch.tensor(0.0, device=Z.device, requires_grad=True)

        return x_pred, mask_loss