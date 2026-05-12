"""Minimal WavJEPA-style training package."""

from __future__ import annotations

from wav_jepa_minimal.config import AUDIOSET_DEFAULTS, MaskConfig, WavJepaConfig

__all__ = ["AUDIOSET_DEFAULTS", "MaskConfig", "WavJepaConfig", "WavJepaModel"]


def __getattr__(name: str) -> object:
    if name == "WavJepaModel":
        from wav_jepa_minimal.model import WavJepaModel

        return WavJepaModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
