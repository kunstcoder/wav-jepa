# Minimal WavJEPA implementation plan

## Goal

Implement a minimal, readable training path inspired by the original
`labhamlet/wavjepa` codebase while excluding its full benchmark evaluation suite.
The original code uses a raw waveform CNN extractor, context/target transformer
encoders, a predictor, masking, and PyTorch Lightning/Hydra data plumbing. This repo
keeps those modeling ideas but replaces the larger framework stack with small,
standard Python modules and argparse CLIs.

## Scope

- Build a raw waveform self-supervised training loop.
- Use AudioSet as the default dataset target, matching upstream `configs/base.yaml` and
  `configs/data/audioset.yaml` defaults where practical.
- Keep the implementation minimal enough to run on local WAV directories.
- Exclude HEAR/ARCH-style evaluation code.
- Add KNN evaluation over frozen learned embeddings.
- Follow Python coding standards with type hints, clear module boundaries, and
  formatter/linter configuration.

## Upstream config values reflected

- Dataset defaults: `name=AudioSet`, `sr=16000`, `in_channels=1`,
  `samples_per_audio=8`, and `process_seconds=2.01`.
- Extractor defaults: WavJEPA conv spec `[(512, 10, 5)] + [(512, 3, 2)] * 4 +
  [(512, 2, 2)]`, `dropout=0.0`, `conv_bias=False`, and non-depthwise convolution.
- Masker defaults: `context_mask_prob=0.65`, `context_mask_length=10`,
  `target_prob=0.25`, `target_length=10`, `target_masks_per_context=4`, and
  `ratio_cutoff=0.1`.
- Optimizer/trainer defaults: AdamW `lr=0.0004`, betas `(0.9, 0.98)`,
  `weight_decay=0.04`, batch size `32`, seed `42`, and max steps `375000`.
- Transformer defaults: base size encoder dimension `768`, decoder/predictor dimension
  `384`, 12 layers, and 12 attention heads.

## Implementation updates

- Added a package under `src/wav_jepa_minimal`.
- Added `WaveDirectoryDataset` and `SyntheticWaveDataset` for real AudioSet-style WAV and
  smoke-test training inputs.
- Added upstream-aligned default constants in `defaults.py`.
- Added a compact CNN patch encoder, transformer blocks, JEPA masking, predictor, and
  EMA target encoder.
- Updated the training CLI to expose `--process-seconds` with an AudioSet default of
  `2.01` instead of the earlier generic `seconds=2.0` default.
- Added `wavjepa-train` CLI for self-supervised training and checkpoint saving.
- Added `wavjepa-knn` CLI for parent-directory-label KNN evaluation.
- Added project metadata and console scripts in `pyproject.toml`.
