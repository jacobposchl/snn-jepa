"""
Training utilities shared across experiment scripts.
"""

import torch
import torch.nn as nn


def create_context_mask(attn_mask: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    Randomly hide mask_ratio of real tokens per sample for the context encoder.

    Only tokens that are True in attn_mask (real spike events) are eligible.
    PAD tokens (False) are always left as False in the output.

    Args:
        attn_mask  : [B, E] bool — True = real token.
        mask_ratio : Fraction of real tokens to hide from the context encoder.

    Returns:
        ctx_mask : [B, E] bool — attn_mask with some real tokens hidden (set to False).
    """
    B, E = attn_mask.shape
    rand = torch.rand(B, E, device=attn_mask.device)
    rand[~attn_mask] = 2.0  # PAD: score > 1 so never selected for masking

    n_real = attn_mask.float().sum(dim=1)          # [B]
    n_to_mask = (n_real * mask_ratio).long()        # [B]

    sorted_rand, _ = rand.sort(dim=1)
    idx = n_to_mask.clamp(min=0, max=E - 1).unsqueeze(1)
    threshold = sorted_rand.gather(1, idx)          # [B, 1]

    ctx_mask = attn_mask.clone()
    ctx_mask[rand < threshold] = False
    return ctx_mask


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float) -> None:
    """
    In-place EMA update: target = momentum * target + (1 - momentum) * online.

    Called after every optimizer step. The target encoder is never touched by
    the optimizer — only updated here.
    """
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(momentum).add_(p_online.data, alpha=1.0 - momentum)


def apply_unit_dropout(
    unit_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    dropout_ratio: float = 0.3,
    min_units: int = 30,
) -> torch.Tensor:
    """
    POYO-style unit dropout: randomly mask out all spikes from a subset of units
    per sample, forcing the model to be robust to missing neurons.

    Only applied during training (caller's responsibility to skip at eval time).

    Args:
        unit_ids      : [B, E] long — 1-indexed unit IDs (0 = PAD).
        attn_mask     : [B, E] bool — True = real token, False = PAD.
        dropout_ratio : Fraction of unique units to drop per sample.
        min_units     : Minimum number of units guaranteed to remain active.

    Returns:
        new_mask : [B, E] bool — attn_mask with dropped-unit spikes set to False.
    """
    if dropout_ratio <= 0.0:
        return attn_mask

    B        = unit_ids.shape[0]
    new_mask = attn_mask.clone()

    for i in range(B):
        real_ids     = unit_ids[i][attn_mask[i]]   # non-PAD unit IDs for this sample
        unique_units = real_ids.unique()
        n_units      = unique_units.shape[0]

        n_keep = max(min_units, int(n_units * (1.0 - dropout_ratio)))
        n_keep = min(n_keep, n_units)  # can't keep more than we have

        perm       = torch.randperm(n_units, device=unit_ids.device)
        keep_units = unique_units[perm[:n_keep]]

        # Zero out spikes from dropped units (those not in keep_units).
        is_dropped        = ~torch.isin(unit_ids[i], keep_units)
        new_mask[i]       = attn_mask[i] & ~is_dropped

    return new_mask
