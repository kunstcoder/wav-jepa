"""Mask sampling helpers for JEPA pre-training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from wav_jepa_minimal.defaults import AUDIOSET_DEFAULTS


@dataclass(frozen=True)
class MaskConfig:
    """Configuration for AudioSet-style contiguous time-block masks."""

    context_mask_prob: float = AUDIOSET_DEFAULTS.context_mask_prob
    context_mask_length: int = AUDIOSET_DEFAULTS.context_mask_length
    target_prob: float = AUDIOSET_DEFAULTS.target_prob
    target_length: int = AUDIOSET_DEFAULTS.target_length
    target_masks_per_context: int = AUDIOSET_DEFAULTS.target_masks_per_context
    ratio_cutoff: float = AUDIOSET_DEFAULTS.ratio_cutoff


def _contiguous_mask(
    batch_size: int, length: int, mask_length: int, device: torch.device
) -> Tensor:
    mask_length = min(mask_length, max(1, length - 1))
    starts = torch.randint(0, length - mask_length + 1, (batch_size,), device=device)
    offsets = torch.arange(length, device=device).unsqueeze(0)
    return (offsets >= starts.unsqueeze(1)) & (offsets < (starts + mask_length).unsqueeze(1))


def sample_context_target_masks(
    batch_size: int,
    length: int,
    config: MaskConfig,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample boolean context and target masks using upstream AudioSet defaults.

    The upstream AudioSet masker uses ``context_mask_prob=0.65``,
    ``context_mask_length=10``, ``target_prob=0.25``, ``target_length=10``, and
    ``target_masks_per_context=4``. This minimal implementation samples one unioned
    target mask per item while preserving those default probabilities and lengths.
    """

    target_mask = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    masks_to_sample = max(1, config.target_masks_per_context)
    for _ in range(masks_to_sample):
        should_apply = torch.rand(batch_size, device=device) < config.target_prob
        sampled = _contiguous_mask(batch_size, length, config.target_length, device)
        target_mask |= sampled & should_apply.unsqueeze(1)

    empty_targets = ~target_mask.any(dim=1)
    if empty_targets.any():
        fallback = _contiguous_mask(batch_size, length, config.target_length, device)
        target_mask[empty_targets] = fallback[empty_targets]

    blocked_context = _contiguous_mask(batch_size, length, config.context_mask_length, device)
    context_mask = ~(target_mask | blocked_context)

    keep_count = max(1, int(length * config.context_mask_prob))
    available_count = context_mask.sum(dim=1)
    for row in torch.nonzero(available_count < keep_count, as_tuple=False).flatten().tolist():
        candidates = torch.nonzero(~target_mask[row], as_tuple=False).flatten()
        if candidates.numel() == 0:
            candidates = torch.arange(length, device=device)
        chosen = candidates[torch.randperm(candidates.numel(), device=device)[:keep_count]]
        context_mask[row] = False
        context_mask[row, chosen] = True

    if keep_count < context_mask.sum(dim=1).min().item():
        scores = torch.rand(batch_size, length, device=device).masked_fill(~context_mask, -1.0)
        keep_indices = scores.topk(keep_count, dim=1).indices
        context_mask = torch.zeros_like(context_mask)
        context_mask.scatter_(1, keep_indices, True)

    return context_mask, target_mask
