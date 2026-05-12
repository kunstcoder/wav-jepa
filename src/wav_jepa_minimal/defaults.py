"""Backward-compatible re-exports for configuration defaults.

New code should import from :mod:`wav_jepa_minimal.config` so model, masking, and
training configuration dataclasses live in one discoverable module.
"""

from __future__ import annotations

from wav_jepa_minimal.config import (
    AUDIOSET_DEFAULTS,
    WAVJEPA_CONV_LAYERS_SPEC,
    AudioSetDefaults,
    AudioSetTrainingConfig,
)

__all__ = [
    "AUDIOSET_DEFAULTS",
    "AudioSetDefaults",
    "AudioSetTrainingConfig",
    "WAVJEPA_CONV_LAYERS_SPEC",
]
