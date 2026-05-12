"""Configuration dataclasses and constants for the minimal WavJEPA package."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AudioSetTrainingConfig:
    """AudioSet-oriented training defaults mirrored from upstream WavJEPA configs."""

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
    checkpoint_interval_steps: int = 0
    seed: int = 42
    feature_dim: int = 512
    encoder_dim: int = 768
    predictor_dim: int = 384
    transformer_layers: int = 12
    attention_heads: int = 12
    predictor_attention_heads: int = 12
    context_mask_prob: float = 0.65
    context_mask_length: int = 10
    target_prob: float = 0.25
    target_length: int = 10
    target_masks_per_context: int = 4
    ratio_cutoff: float = 0.1
    ema_decay: float = 0.999
    ema_end_decay: float = 0.99999
    ema_anneal_end_step: int = 100_000


AUDIOSET_DEFAULTS = AudioSetTrainingConfig()

# Backward-compatible class name for users importing from older releases.
AudioSetDefaults = AudioSetTrainingConfig

WAVJEPA_CONV_LAYERS_SPEC: tuple[tuple[int, int, int], ...] = (
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
)


@dataclass(frozen=True)
class MaskConfig:
    """Configuration for AudioSet-style contiguous time-block masks."""

    context_mask_prob: float = AUDIOSET_DEFAULTS.context_mask_prob
    context_mask_length: int = AUDIOSET_DEFAULTS.context_mask_length
    target_prob: float = AUDIOSET_DEFAULTS.target_prob
    target_length: int = AUDIOSET_DEFAULTS.target_length
    target_masks_per_context: int = AUDIOSET_DEFAULTS.target_masks_per_context
    ratio_cutoff: float = AUDIOSET_DEFAULTS.ratio_cutoff


@dataclass(frozen=True)
class WavJepaConfig:
    """Model and AudioSet hyperparameters used to build a ``WavJepaModel``."""

    dataset_name: str = AUDIOSET_DEFAULTS.dataset_name
    sample_rate: int = AUDIOSET_DEFAULTS.sample_rate
    process_seconds: float = AUDIOSET_DEFAULTS.process_seconds
    samples_per_audio: int = AUDIOSET_DEFAULTS.samples_per_audio
    in_channels: int = AUDIOSET_DEFAULTS.in_channels
    feature_dim: int = AUDIOSET_DEFAULTS.feature_dim
    encoder_dim: int = AUDIOSET_DEFAULTS.encoder_dim
    predictor_dim: int = AUDIOSET_DEFAULTS.predictor_dim
    transformer_layers: int = AUDIOSET_DEFAULTS.transformer_layers
    attention_heads: int = AUDIOSET_DEFAULTS.attention_heads
    mlp_ratio: int = 4
    ema_decay: float = AUDIOSET_DEFAULTS.ema_decay
    ema_end_decay: float = AUDIOSET_DEFAULTS.ema_end_decay
    ema_anneal_end_step: int = AUDIOSET_DEFAULTS.ema_anneal_end_step
    context_mask_prob: float = AUDIOSET_DEFAULTS.context_mask_prob
    context_mask_length: int = AUDIOSET_DEFAULTS.context_mask_length
    target_prob: float = AUDIOSET_DEFAULTS.target_prob
    target_length: int = AUDIOSET_DEFAULTS.target_length
    target_masks_per_context: int = AUDIOSET_DEFAULTS.target_masks_per_context
    ratio_cutoff: float = AUDIOSET_DEFAULTS.ratio_cutoff

    @property
    def seconds(self) -> float:
        """Backward-compatible alias for the upstream ``process_seconds`` name."""

        return self.process_seconds

    @property
    def embed_dim(self) -> int:
        """Backward-compatible alias for the clearer ``encoder_dim`` field name."""

        return self.encoder_dim

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int | float | str]) -> WavJepaConfig:
        values = dict(values)
        legacy_key_map = {
            "seconds": "process_seconds",
            "embed_dim": "encoder_dim",
            "decoder_dim": "predictor_dim",
            "context_ratio": "context_mask_prob",
            "target_ratio": "target_prob",
            "min_target_patches": "target_length",
        }
        for old_key, new_key in legacy_key_map.items():
            if old_key in values and new_key not in values:
                values[new_key] = values.pop(old_key)
            else:
                values.pop(old_key, None)
        return cls(**values)


def parse_checkpoint_interval(value: str) -> int:
    """Parse a periodic checkpoint interval in global training steps.

    ``0``, empty strings, ``none``, and ``off`` disable periodic checkpointing.
    Otherwise the value must be a positive integer, such as ``10000`` to save one
    numbered checkpoint every 10,000 optimizer steps.
    """

    normalized = value.strip().lower()
    if normalized in {"", "none", "off"}:
        return 0

    try:
        interval = int(normalized)
    except ValueError as error:
        raise ValueError(
            f"Invalid checkpoint interval {value!r}; expected a positive integer or 0"
        ) from error
    if interval < 0:
        raise ValueError(
            f"Invalid checkpoint interval {interval}; expected a positive integer or 0"
        )
    return interval
