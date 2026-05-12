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

## Install

```bash
python -m pip install -e .
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
center-cropped or padded to `process_seconds` seconds, and normalized per clip.

## Train

```bash
wavjepa-train --data-dir data/audioset --output-dir runs/audioset-minimal --epochs 10
```

For a dependency-light smoke run without audio files:

```bash
wavjepa-train --data-dir data/audioset --synthetic --epochs 1 --steps-per-epoch 2
```

The upstream AudioSet clip length default is exposed as `--process-seconds` and defaults
to `2.01`.

## KNN evaluation

```bash
wavjepa-knn \
  --checkpoint runs/audioset-minimal/checkpoint_last.pt \
  --train-dir data/knn_train \
  --test-dir data/knn_test \
  --k 5
```
