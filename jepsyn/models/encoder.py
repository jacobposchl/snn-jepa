"""
Perceiver-style encoder for spike event token streams.

Takes variable-length event sequences (unit_ids, time_ids) and compresses them
into a fixed latent set via cross-attention, then refines via self-attention.

Architecture:
    Token embedding (unit + time) → Cross-attention → Latent self-attention → Output
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class _FFN(nn.Module):
    """Two-layer feedforward network used inside the cross-attention block."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverEncoder(nn.Module):
    """
    Perceiver-style encoder: variable-length spike events → fixed latent set.

    Pipeline per forward pass:
        1. Embed each event: unit_embed(session-specific) + time_embed(shared) → x [B, E, D]
        2. Cross-attend: learned latents attend over event tokens → z [B, L, D]
        3. Self-attend: refine latents through M Transformer blocks → z [B, L, D]
        4. Pool: h = mean(z, dim=1) → [B, D]

    Args:
        session_unit_maps : {session_id: {raw_unit_id: 1-indexed contiguous idx}}.
                            Passed directly from load_and_prepare_data().
        d_model           : Embedding and model dimension.
        n_latents         : Number of learned latent slots (L).
        max_time_ms       : Time embedding vocabulary size; must be ≥ window_size_ms.
        n_cross_attn_heads: Attention heads for the cross-attention layer.
        n_self_attn_layers: Number of latent self-attention Transformer blocks.
        n_self_attn_heads : Attention heads for self-attention.
        dim_feedforward   : FFN hidden dim in self-attention blocks.
        dropout           : Dropout probability throughout.
    """

    def __init__(
        self,
        session_unit_maps: Dict[int, Dict[int, int]],
        d_model: int = 256,
        n_latents: int = 64,
        max_time_ms: int = 400,
        n_cross_attn_heads: int = 4,
        n_self_attn_layers: int = 4,
        n_self_attn_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latents = n_latents

        # Per-session unit embedding tables.
        # Vocab size = N_s + 1; index 0 is PAD (padding_idx=0 → zero vector).
        self.unit_embeds = nn.ModuleDict({
            str(sid): nn.Embedding(len(unit_map) + 1, d_model, padding_idx=0)
            for sid, unit_map in session_unit_maps.items()
        })

        # Shared time embedding: one entry per ms in the window.
        self.time_embed = nn.Embedding(max_time_ms, d_model)

        # Learned latent array [L, D].
        self.latents = nn.Parameter(torch.randn(n_latents, d_model))

        # Input norm applied to event tokens before cross-attention.
        self.input_norm = nn.LayerNorm(d_model)

        # Cross-attention (latents as queries, event tokens as keys/values).
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_cross_attn_heads, batch_first=True, dropout=dropout
        )
        self.cross_norm_q = nn.LayerNorm(d_model)    # pre-norm on queries (latents)
        self.cross_norm_kv = nn.LayerNorm(d_model)   # pre-norm on keys/values (events)
        self.cross_ffn = _FFN(d_model, dim_feedforward, dropout)
        self.cross_norm_ff = nn.LayerNorm(d_model)

        # Latent self-attention blocks (pre-norm Transformer).
        self.self_attn_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_self_attn_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=n_self_attn_layers,
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        session_ids: torch.Tensor,  # [B] long
        unit_ids: torch.Tensor,     # [B, E] long, 1-indexed, 0 = PAD
        time_ids: torch.Tensor,     # [B, E] long, values in [0, max_time_ms)
        attn_mask: torch.Tensor,    # [B, E] bool, True = real token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            session_ids : [B] — which session each example belongs to.
            unit_ids    : [B, E] — 1-indexed unit token per spike (0 = PAD).
            time_ids    : [B, E] — integer ms offset within window.
            attn_mask   : [B, E] — True for real tokens, False for padding.

        Returns:
            Z : [B, L, D] — full latent set.
            h : [B, D]    — mean-pooled latent (for probes / loss).
        """
        B = session_ids.shape[0]

        # 1. Event token embedding
        # Unit embeddings: dispatch per session (ModuleDict requires str keys).
        x_unit = torch.zeros(
            B, unit_ids.shape[1], self.d_model,
            device=unit_ids.device, dtype=torch.float32
        )
        for sid_val in session_ids.unique():
            sid_str = str(sid_val.item())
            mask = (session_ids == sid_val)                # [B] bool
            x_unit[mask] = self.unit_embeds[sid_str](unit_ids[mask])

        # Time embeddings: shared across sessions.
        x = x_unit + self.time_embed(time_ids)             # [B, E, D]
        x = self.input_norm(x)

        # 2. Cross-attention: latents attend over event tokens.
        z0 = self.latents[None].expand(B, -1, -1)          # [B, L, D]
        q = self.cross_norm_q(z0)
        kv = self.cross_norm_kv(x)
        # PyTorch MHA key_padding_mask: True = IGNORE (inverse of our attn_mask).
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=~attn_mask
        )
        z = z0 + attn_out                                  # residual

        # Post-cross-attention FFN with residual.
        z = z + self.cross_ffn(self.cross_norm_ff(z))

        # 3. Latent self-attention.
        z = self.self_attn_blocks(z)                       # [B, L, D]

        # 4. Output norm and mean pool.
        z = self.output_norm(z)
        h = z.mean(dim=1)                                  # [B, D]

        return z, h


class NeuralEncoder(nn.Module):
    """
    High-level encoder wrapper.

    Wraps PerceiverEncoder and provides the stable public interface consumed
    by multi_session.py.

    Args:
        session_unit_maps : {session_id: {raw_unit_id: 1-indexed idx}}.
                            Passed from load_and_prepare_data().
        d_model           : Model/embedding dimension (also the latent dimension).
        n_latents         : Number of learned latent slots.
        max_time_ms       : Time embedding vocabulary size.
        encoder_type      : Currently only 'perceiver' is supported.
        **kwargs          : Forwarded to PerceiverEncoder.
    """

    def __init__(
        self,
        session_unit_maps: Dict[int, Dict[int, int]],
        d_model: int = 256,
        n_latents: int = 64,
        max_time_ms: int = 400,
        encoder_type: str = "perceiver",
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latents = n_latents
        self.encoder_type = encoder_type

        if encoder_type.lower() == "perceiver":
            self.encoder = PerceiverEncoder(
                session_unit_maps=session_unit_maps,
                d_model=d_model,
                n_latents=n_latents,
                max_time_ms=max_time_ms,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type: '{encoder_type}'. "
                "Currently only 'perceiver' is supported."
            )

    def forward(
        self,
        session_ids: torch.Tensor,
        unit_ids: torch.Tensor,
        time_ids: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of spike event windows.

        Args:
            session_ids : [B] long
            unit_ids    : [B, E] long, 1-indexed (0 = PAD)
            time_ids    : [B, E] long, ms offset in [0, max_time_ms)
            attn_mask   : [B, E] bool, True = real token

        Returns:
            Z : [B, L, D] — full latent set (use for JEPA prediction target/context)
            h : [B, D]    — mean-pooled latent (use for probes)
        """
        return self.encoder(session_ids, unit_ids, time_ids, attn_mask)

    def get_latent_dim(self) -> int:
        """Return the model dimension D."""
        return self.d_model

    def get_n_latents(self) -> int:
        """Return the number of latent slots L."""
        return self.n_latents
