"""Default values mirrored from the upstream WavJEPA AudioSet configs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSetDefaults:
    """Minimal view of the upstream ``configs`` defaults used by this repo."""

    dataset_name: str = "AudioSet"
    sample_rate: int = 16_000
    process_seconds: float = 2.01
    in_channels: int = 1
    samples_per_audio: int = 8
    batch_size: int = 32
    learning_rate: float = 0.0004
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    weight_decay: float = 0.04
    max_steps: int = 375_000
    seed: int = 42
    feature_dim: int = 512
    encoder_dim: int = 768
    decoder_dim: int = 384
    transformer_layers: int = 12
    attention_heads: int = 12
    decoder_attention_heads: int = 12
    context_mask_prob: float = 0.65
    context_mask_length: int = 10
    target_prob: float = 0.25
    target_length: int = 10
    target_masks_per_context: int = 4
    ratio_cutoff: float = 0.1
    ema_decay: float = 0.999
    ema_end_decay: float = 0.99999
    ema_anneal_end_step: int = 100_000


AUDIOSET_DEFAULTS = AudioSetDefaults()

WAVJEPA_CONV_LAYERS_SPEC: tuple[tuple[int, int, int], ...] = (
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
)
