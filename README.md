# Minimal WavJEPA

This repository contains a compact WavJEPA-style training implementation. It keeps the
core pre-training ingredients from the original project—raw waveform patches, a CNN
front-end, context/target transformer encoders, masked latent prediction, and an EMA
target encoder—while intentionally excluding the original benchmark evaluation stack.

The default training configuration is aligned with the upstream WavJEPA AudioSet
configs where practical for this minimal codebase: dataset name `AudioSet`, sample rate
`16000`, `process_seconds=2.01`, `samples_per_audio=8`, batch size `32`, AdamW
`lr=0.0004`, betas `(0.9, 0.98)`, weight decay `0.04`, WavJEPA convolution spec
`[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)]`, encoder dimension `768`,
decoder/predictor dimension `384`, 12 transformer layers, and AudioSet masker defaults.

Instead of HEAR/ARCH evaluation scripts, the repo provides a small KNN classifier over
frozen embeddings for quick representation checks.

## Implemented scope

- Raw PCM WAV loading with mono conversion, fixed-length crop/pad, and per-clip
  normalization.
- AudioSet-shaped synthetic waveform data for smoke tests without a local dataset.
- WavJEPA-style pretraining with a convolutional patch encoder, context transformer,
  EMA target transformer, predictor head, contiguous context/target masks, Smooth L1
  latent prediction loss, gradient clipping, and last-checkpoint saving.
- TensorBoard training monitoring for loss, mask fractions, learning rate, gradient
  norm, EMA decay, epoch loss, dataset metadata, and hyperparameters.
- Lightweight KNN evaluation over frozen utterance-level embeddings.

## Install

Install the package in editable mode:

```bash
python -m pip install -e .
```

Alternatively, install runtime dependencies from `requirements.txt` first:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Data layout

Training targets AudioSet-style WAV directories and accepts `.wav` files recursively:

```text
data/audioset/unbalanced/class_a/example.wav
```

KNN evaluation expects labels to be encoded as the parent directory name:

```text
data/knn_train/dog/a.wav
data/knn_train/rain/b.wav
data/knn_test/dog/c.wav
```

Audio is loaded with the Python standard library `wave` module, converted to mono,
center-cropped or randomly cropped/padded to `process_seconds` seconds, and normalized
per clip. Files must already match the configured sample rate because this minimal
loader does not resample audio.

## Train

```bash
wavjepa-train --data-dir data/audioset --output-dir runs/audioset-minimal --epochs 10
```

For a dependency-light smoke run without audio files:

```bash
wavjepa-train --data-dir data/audioset --synthetic --epochs 1 --steps-per-epoch 2
```

The upstream AudioSet clip length default is exposed as `--process-seconds` and defaults
to `2.01`. The latest checkpoint is written to `checkpoint_last.pt` and the model
configuration is written to `config.json` under `--output-dir`.

## TensorBoard monitoring

Training writes TensorBoard event files by default to `<output-dir>/tensorboard` and
prints the resolved log directory at startup:

```bash
wavjepa-train \
  --data-dir data/audioset \
  --output-dir runs/audioset-minimal \
  --epochs 10

tensorboard --logdir runs/audioset-minimal/tensorboard
```

Use `--tensorboard-log-dir` to choose a different event directory, or
`--no-tensorboard` to disable event writing:

```bash
wavjepa-train \
  --data-dir data/audioset \
  --output-dir runs/audioset-minimal \
  --tensorboard-log-dir runs/tb/audioset-minimal
```

Logged scalars include:

- `train/loss`
- `train/context_fraction`
- `train/target_fraction`
- `train/learning_rate`
- `train/grad_norm`
- `train/ema_decay`
- `epoch/loss`

## KNN evaluation

```bash
wavjepa-knn \
  --checkpoint runs/audioset-minimal/checkpoint_last.pt \
  --train-dir data/knn_train \
  --test-dir data/knn_test \
  --k 5
```
