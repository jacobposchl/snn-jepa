"""
POYO PerceiverIO encoder for multi-session spike event streams.

Replaces the earlier absolute-time-embedding design with:
    - RoPE (Rotary Position Embeddings) in all attention layers via torch_brain
    - Latent timestamps spread uniformly over the window
    - Per-session unit embedding tables (unchanged from before)
    - Session embedding injected into latent initialization
    - Optional [START]/[END] delimiter tokens per unit per window

Architecture:
    Unit embed (per-session)
    + Delimiter injection (optional)
    → Cross-attention with RoPE: latents attend over spike tokens    [B, L, D]
    → Latent self-attention with RoPE (L layers)                    [B, L, D]
    → Mean pool                                                      [B, D]
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_brain.nn import RotaryCrossAttention, RotarySelfAttention
from torch_brain.nn.rotary_embedding import RotaryEmbedding


class _FFN(nn.Module):
    """Two-layer feedforward network used inside Transformer blocks."""

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
    POYO PerceiverIO encoder: variable-length spike events → fixed latent set.

    Key POYO mechanisms:
        - Per-session unit embedding tables (session-specific vocabulary)
        - RoPE temporal encoding in cross- and self-attention (no additive time embed)
        - [START]/[END] delimiter tokens make every registered unit visible each window
        - Session embedding shifts the latent initialization per session

    Note on LayerNorm:
        RotaryCrossAttention and RotarySelfAttention both apply internal pre-LayerNorm
        to their inputs. We do NOT add separate input norms before them.
        The post-attention FFN (and its pre-norm) remains our responsibility.

    Note on mask convention:
        context_mask in RotaryCrossAttention: True = attend to this token.
        This is the same convention as our attn_mask (True = real token),
        so we pass attn_mask directly (NOT ~attn_mask).

    Args:
        session_unit_maps   : {session_id: {raw_unit_id: 1-indexed contiguous idx}}.
        d_model             : Embedding and model dimension.
        n_latents           : Number of learned latent slots (L).
        window_size_s       : Window duration in seconds; used for latent timestamp grid
                              and END delimiter token timestamps. Must match windowing config.
        n_cross_attn_heads  : Attention heads for cross-attention.
        n_self_attn_layers  : Number of latent self-attention layers.
        n_self_attn_heads   : Attention heads for self-attention.
        dim_feedforward     : FFN hidden dim.
        dropout             : Dropout probability.
        rope_t_min          : Minimum RoPE period in seconds (default 1ms = 1e-3).
        rope_t_max          : Maximum RoPE period in seconds (default 4s).
        use_delimiter_tokens: If True, add START/END delimiter tokens per unit per window.
    """

    def __init__(
        self,
        session_unit_maps: Dict[int, Dict[int, int]],
        d_model: int = 256,
        n_latents: int = 64,
        window_size_s: float = 0.4,
        n_cross_attn_heads: int = 4,
        n_self_attn_layers: int = 4,
        n_self_attn_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        rope_t_min: float = 1e-3,
        rope_t_max: float = 4.0,
        use_delimiter_tokens: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_latents = n_latents
        self.window_size_s = window_size_s
        self.use_delimiter_tokens = use_delimiter_tokens

        # Store for delimiter injection in forward()
        self.session_unit_maps = session_unit_maps

        # Per-session unit embedding tables.
        # Vocab size = N_s + 1; index 0 is PAD (padding_idx=0 → zero vector).
        self.unit_embeds = nn.ModuleDict({
            str(sid): nn.Embedding(len(unit_map) + 1, d_model, padding_idx=0)
            for sid, unit_map in session_unit_maps.items()
        })

        # Session embedding table: one row per session (all sessions pre-allocated,
        # including test sessions — their rows start random and are adapted via identify_units).
        all_sids = sorted(session_unit_maps.keys())
        self.session_id_to_idx: Dict[int, int] = {sid: i for i, sid in enumerate(all_sids)}
        self.session_embed = nn.Embedding(len(all_sids), d_model)

        # Delimiter embedding: [0 = START, 1 = END]
        if use_delimiter_tokens:
            self.delimiter_embed = nn.Embedding(2, d_model)

        # Learned latent array [L, D].
        self.latents = nn.Parameter(torch.randn(n_latents, d_model))

        # Latent timestamps: uniform grid over [0, window_size_s] — no gradient.
        self.register_buffer(
            "latent_timestamps",
            torch.linspace(0.0, window_size_s, n_latents),  # [L]
        )

        # RoPE: one embedding per attention type (different head dims for cross vs self).
        dim_head_cross = d_model // n_cross_attn_heads  # e.g. 256//4 = 64
        dim_head_self  = d_model // n_self_attn_heads   # e.g. 256//8 = 32
        self.rope_cross = RotaryEmbedding(dim=dim_head_cross, t_min=rope_t_min, t_max=rope_t_max)
        self.rope_self  = RotaryEmbedding(dim=dim_head_self,  t_min=rope_t_min, t_max=rope_t_max)

        # Cross-attention: latents (queries) attend over spike tokens (keys/values).
        # RotaryCrossAttention includes internal pre-LayerNorm for both query and context.
        self.cross_attn = RotaryCrossAttention(
            dim=d_model,
            heads=n_cross_attn_heads,
            dim_head=dim_head_cross,
            dropout=dropout,
        )
        # Post-cross-attention FFN with pre-norm (our responsibility).
        self.cross_ffn     = _FFN(d_model, dim_feedforward, dropout)
        self.cross_norm_ff = nn.LayerNorm(d_model)

        # Latent self-attention: L layers, each with RoPE.
        # RotarySelfAttention includes internal pre-LayerNorm.
        self.self_attn_layers = nn.ModuleList([
            RotarySelfAttention(
                dim=d_model,
                heads=n_self_attn_heads,
                dim_head=dim_head_self,
                dropout=dropout,
            )
            for _ in range(n_self_attn_layers)
        ])
        # Per-layer post-attention FFN with pre-norm.
        self.ffn_layers      = nn.ModuleList([_FFN(d_model, dim_feedforward, dropout) for _ in range(n_self_attn_layers)])
        self.ffn_norm_layers = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_self_attn_layers)])

        self.output_norm = nn.LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Delimiter injection
    # ------------------------------------------------------------------

    def _inject_delimiters(
        self,
        x_unit: torch.Tensor,      # [B, E, D] spike token embeddings
        time_ids: torch.Tensor,    # [B, E] integer ms offsets
        session_ids: torch.Tensor, # [B] long
        attn_mask: torch.Tensor,   # [B, E] bool, True=real spike
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each sample: add a START token (t=0 s) and END token (t=window_size_s)
        for every unit registered in that session, including silent units that fired
        no spikes in this window.

        All delimiter tokens are marked as real (True) in the returned mask.
        Delimiter token = delimiter_embed(0_or_1) + unit_embed(unit_idx).

        Returns:
            x_combined   : [B, E + max_delim, D]  — spike tokens + delimiter tokens
            timestamps_s : [B, E + max_delim]      — float seconds
            mask_combined: [B, E + max_delim] bool — True = real token
        """
        B      = session_ids.shape[0]
        device = session_ids.device

        delim_embeds_list: List[torch.Tensor] = []
        delim_times_list:  List[torch.Tensor] = []

        for i in range(B):
            sid      = session_ids[i].item()
            unit_map = self.session_unit_maps[sid]
            n_units  = len(unit_map)

            # 1-indexed unit embedding indices for all registered units in session.
            unit_indices = torch.arange(1, n_units + 1, dtype=torch.long, device=device)
            unit_embs    = self.unit_embeds[str(sid)](unit_indices)  # [n_units, D]

            start_type   = torch.zeros(n_units, dtype=torch.long, device=device)
            end_type     = torch.ones(n_units,  dtype=torch.long, device=device)
            start_tokens = self.delimiter_embed(start_type) + unit_embs  # [n_units, D]
            end_tokens   = self.delimiter_embed(end_type)   + unit_embs  # [n_units, D]

            delim_embeds_list.append(torch.cat([start_tokens, end_tokens], dim=0))  # [2*n_units, D]
            delim_times_list.append(torch.cat([
                torch.zeros(n_units, device=device),
                torch.full((n_units,), self.window_size_s, device=device),
            ]))  # [2*n_units]

        # Pad delimiter tokens across batch to the largest delimiter count.
        max_delim  = max(t.shape[0] for t in delim_embeds_list)
        delim_pad  = torch.zeros(B, max_delim, self.d_model, device=device)
        times_pad  = torch.zeros(B, max_delim, device=device)
        delim_mask = torch.zeros(B, max_delim, dtype=torch.bool, device=device)

        for i, (embs, times) in enumerate(zip(delim_embeds_list, delim_times_list)):
            n = embs.shape[0]
            delim_pad[i, :n]  = embs
            times_pad[i, :n]  = times
            delim_mask[i, :n] = True

        spike_times_s = time_ids.float() / 1000.0                          # [B, E] ms → s
        x_combined    = torch.cat([x_unit, delim_pad], dim=1)              # [B, E+max_delim, D]
        timestamps_s  = torch.cat([spike_times_s, times_pad], dim=1)       # [B, E+max_delim]
        mask_combined = torch.cat([attn_mask, delim_mask], dim=1)          # [B, E+max_delim]

        return x_combined, timestamps_s, mask_combined

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        session_ids: torch.Tensor,  # [B] long
        unit_ids: torch.Tensor,     # [B, E] long, 1-indexed, 0 = PAD
        time_ids: torch.Tensor,     # [B, E] long, integer ms offset within window
        attn_mask: torch.Tensor,    # [B, E] bool, True = real token
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            session_ids : [B] — session index per example.
            unit_ids    : [B, E] — 1-indexed unit token per spike (0 = PAD).
            time_ids    : [B, E] — integer ms offset within window.
            attn_mask   : [B, E] — True for real tokens, False for padding.

        Returns:
            Z : [B, L, D] — full latent set.
            h : [B, D]    — mean-pooled latent.
        """
        B = session_ids.shape[0]

        # 1. Unit embeddings (per-session dispatch).
        x_unit = torch.zeros(
            B, unit_ids.shape[1], self.d_model,
            device=unit_ids.device, dtype=torch.float32,
        )
        for sid_val in session_ids.unique():
            sid_str = str(sid_val.item())
            mask    = (session_ids == sid_val)
            x_unit[mask] = self.unit_embeds[sid_str](unit_ids[mask])

        # 2. Delimiter token injection (optional).
        if self.use_delimiter_tokens:
            x, timestamps_s, attn_mask = self._inject_delimiters(
                x_unit, time_ids, session_ids, attn_mask
            )
        else:
            x            = x_unit
            timestamps_s = time_ids.float() / 1000.0  # [B, E] ms → seconds

        # 3. RoPE angle embeddings for spike tokens and latent slots.
        spike_pos_emb    = self.rope_cross(timestamps_s)                         # [B, E', dim_head_cross]
        latent_ts        = self.latent_timestamps.unsqueeze(0).expand(B, -1)     # [B, L]
        latent_pos_cross = self.rope_cross(latent_ts)                            # [B, L, dim_head_cross]
        latent_pos_self  = self.rope_self(latent_ts)                             # [B, L, dim_head_self]

        # 4. Latent initialization: shared learned latents + session-specific offset.
        sess_idx = torch.tensor(
            [self.session_id_to_idx[s.item()] for s in session_ids],
            dtype=torch.long, device=session_ids.device,
        )
        z0 = self.latents[None].expand(B, -1, -1) + self.session_embed(sess_idx).unsqueeze(1)

        # 5. Cross-attention: latents attend over spike/delimiter tokens.
        # RotaryCrossAttention: context_mask uses True=attend convention (same as attn_mask).
        # Internal pre-norm applied to both query (z0) and context (x).
        attn_out = self.cross_attn(z0, x, latent_pos_cross, spike_pos_emb, context_mask=attn_mask)
        z = z0 + attn_out
        z = z + self.cross_ffn(self.cross_norm_ff(z))

        # 6. Latent self-attention (L layers).
        # RotarySelfAttention applies internal pre-norm; latents have no padding so x_mask=None.
        for self_attn, ffn, ffn_norm in zip(
            self.self_attn_layers, self.ffn_layers, self.ffn_norm_layers
        ):
            z = z + self_attn(z, latent_pos_self)
            z = z + ffn(ffn_norm(z))

        # 7. Output norm + mean pool.
        z = self.output_norm(z)
        h = z.mean(dim=1)  # [B, D]

        return z, h


class NeuralEncoder(nn.Module):
    """
    High-level encoder wrapper.

    Wraps PerceiverEncoder and provides the stable public interface consumed
    by multi_session.py.

    Args:
        session_unit_maps : {session_id: {raw_unit_id: 1-indexed idx}}.
        d_model           : Model/embedding dimension.
        n_latents         : Number of learned latent slots.
        window_size_s     : Window duration in seconds (replaces max_time_ms).
        encoder_type      : Currently only 'perceiver' is supported.
        **kwargs          : Forwarded to PerceiverEncoder (rope_t_min, rope_t_max,
                            use_delimiter_tokens, n_cross_attn_heads, etc.).
    """

    def __init__(
        self,
        session_unit_maps: Dict[int, Dict[int, int]],
        d_model: int = 256,
        n_latents: int = 64,
        window_size_s: float = 0.4,
        encoder_type: str = "perceiver",
        **kwargs,
    ):
        super().__init__()

        self.d_model      = d_model
        self.n_latents    = n_latents
        self.encoder_type = encoder_type

        if encoder_type.lower() == "perceiver":
            self.encoder = PerceiverEncoder(
                session_unit_maps=session_unit_maps,
                d_model=d_model,
                n_latents=n_latents,
                window_size_s=window_size_s,
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
            time_ids    : [B, E] long, integer ms offset in window
            attn_mask   : [B, E] bool, True = real token

        Returns:
            Z : [B, L, D] — full latent set
            h : [B, D]    — mean-pooled latent
        """
        return self.encoder(session_ids, unit_ids, time_ids, attn_mask)

    def get_latent_dim(self) -> int:
        """Return the model dimension D."""
        return self.d_model

    def get_n_latents(self) -> int:
        """Return the number of latent slots L."""
        return self.n_latents
