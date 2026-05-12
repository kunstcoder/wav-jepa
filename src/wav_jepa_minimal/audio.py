"""Audio datasets and WAV loading utilities."""

from __future__ import annotations

import random
import wave
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AudioItem:
    """A single audio file and optional label."""

    path: Path
    label: str | None = None


def discover_wavs(root: Path, with_labels: bool = False) -> list[AudioItem]:
    """Return WAV files below ``root`` in a deterministic order."""

    paths = sorted(path for path in root.rglob("*.wav") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .wav files found under {root}")

    items: list[AudioItem] = []
    for path in paths:
        label = path.parent.name if with_labels else None
        items.append(AudioItem(path=path, label=label))
    return items


def load_wav_mono(path: Path) -> tuple[Tensor, int]:
    """Load a PCM WAV file as mono float32 in the range roughly [-1, 1]."""

    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if sample_width == 1:
        dtype = torch.uint8
        scale = 127.5
        offset = 128.0
    elif sample_width == 2:
        dtype = torch.int16
        scale = 32768.0
        offset = 0.0
    elif sample_width == 4:
        dtype = torch.int32
        scale = 2147483648.0
        offset = 0.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}")

    audio = torch.frombuffer(bytearray(frames), dtype=dtype).float()
    audio = (audio - offset) / scale
    if channels > 1:
        audio = audio.view(-1, channels).mean(dim=1)
    return audio, sample_rate


def fit_length(audio: Tensor, length: int, random_crop: bool = True) -> Tensor:
    """Crop or right-pad audio to ``length`` samples."""

    if audio.numel() == length:
        return audio
    if audio.numel() < length:
        return torch.nn.functional.pad(audio, (0, length - audio.numel()))

    max_start = audio.numel() - length
    start = random.randint(0, max_start) if random_crop else max_start // 2
    return audio[start : start + length]


def normalize_clip(audio: Tensor, eps: float = 1e-5) -> Tensor:
    """Mean-center and variance-normalize one waveform."""

    centered = audio - audio.mean()
    return centered / centered.std().clamp_min(eps)


class WaveDirectoryDataset(Dataset[Tensor]):
    """Dataset that returns fixed-length normalized mono waveforms."""

    def __init__(
        self,
        root: Path,
        sample_rate: int,
        seconds: float,
        with_labels: bool = False,
        random_crop: bool = True,
    ) -> None:
        self.items = discover_wavs(root, with_labels=with_labels)
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * seconds)
        self.random_crop = random_crop
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, str]:
        item = self.items[index]
        audio, sample_rate = load_wav_mono(item.path)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{item.path} has sample rate {sample_rate}; expected {self.sample_rate}. "
                "Resample files before using this minimal loader."
            )
        audio = normalize_clip(fit_length(audio, self.num_samples, self.random_crop))
        if self.with_labels:
            if item.label is None:
                raise RuntimeError("Label discovery was disabled for a labeled dataset")
            return audio, item.label
        return audio


class SyntheticWaveDataset(Dataset[Tensor]):
    """Small deterministic waveform dataset for smoke tests."""

    def __init__(self, length: int, sample_rate: int, seconds: float) -> None:
        self.length = length
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * seconds)
        self.time = torch.linspace(0, seconds, self.num_samples)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tensor:
        frequency = 110.0 + 17.0 * (index % 12)
        phase = (index % 7) / 7.0
        wave_a = torch.sin(2.0 * torch.pi * frequency * self.time + phase)
        wave_b = 0.25 * torch.sin(2.0 * torch.pi * frequency * 2.0 * self.time)
        noise = 0.01 * torch.randn_like(wave_a)
        return normalize_clip(wave_a + wave_b + noise)
