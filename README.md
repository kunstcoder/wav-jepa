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

To automatically save extra checkpoints at a fixed global-step period, pass
`--checkpoint-interval-steps`. For example, `--checkpoint-interval-steps 10000` saves
numbered snapshots at steps 10,000, 20,000, 30,000, and so on. Each periodic snapshot
uses the name `checkpoint_step_<step>.pt`, while normal epoch-end saving still updates
`checkpoint_last.pt`:

```bash
wavjepa-train \
  --data-dir data/audioset \
  --output-dir runs/audioset-minimal \
  --epochs 10 \
  --checkpoint-interval-steps 10000
```

Use `--checkpoint-interval-steps 0`, `--checkpoint-interval-steps none`, or omit the
flag to disable periodic step checkpoints.


## Dataclass-based configuration

The configuration source of truth is `src/wav_jepa_minimal/config.py`. The module uses
frozen Python `dataclass` objects so defaults, command-line arguments, model
construction, mask sampling, checkpoint serialization, and documentation can share the
same typed field names.

> Note: older code may still import `wav_jepa_minimal.defaults`. That file is kept as a
> compatibility re-export, but new code should prefer `wav_jepa_minimal.config`.

### Dataclasses and naming

- `AudioSetTrainingConfig` contains training/runtime defaults that mirror the upstream
  AudioSet-oriented WavJEPA configuration where this minimal implementation supports
  them. `AUDIOSET_DEFAULTS = AudioSetTrainingConfig()` is the singleton used by the CLI
  parser, including the default disabled `checkpoint_interval_steps` setting.
- `WavJepaConfig` contains the model-building subset saved to `config.json` and restored
  during KNN evaluation. It is frozen to make accidental runtime mutation less likely.
- `MaskConfig` contains only contiguous context/target mask settings used by
  `sample_context_target_masks`.
- Naming follows the role of each value: `encoder_dim` is the context/target transformer
  width, `predictor_dim` is the predictor MLP hidden width, and `process_seconds` is the
  upstream clip-duration name. The legacy CLI flag `--embed-dim` still works as an alias
  for `--encoder-dim`, and old checkpoints with `embed_dim`, `decoder_dim`, or `seconds`
  keys are normalized by `WavJepaConfig.from_dict`.

### How CLI arguments are wired

The training entry point reads defaults from the dataclass when it defines argparse
flags. For example, `--sample-rate` defaults to `AUDIOSET_DEFAULTS.sample_rate`,
`--process-seconds` defaults to `AUDIOSET_DEFAULTS.process_seconds`, `--encoder-dim`
defaults to `AUDIOSET_DEFAULTS.encoder_dim`, and `--checkpoint-interval-steps`
defaults to `AUDIOSET_DEFAULTS.checkpoint_interval_steps`. After parsing, `train.py`
passes the selected model values into `WavJepaConfig`, which is then attached to the
model:

```python
config = WavJepaConfig(
    dataset_name=args.dataset_name,
    sample_rate=args.sample_rate,
    process_seconds=args.process_seconds,
    samples_per_audio=args.samples_per_audio,
    encoder_dim=args.encoder_dim,
    predictor_dim=args.predictor_dim,
    transformer_layers=args.transformer_layers,
    attention_heads=args.attention_heads,
)
```

### Programmatic overrides

For Python usage, instantiate the dataclass directly and override only the fields that
change. Unspecified fields retain AudioSet defaults:

```python
from wav_jepa_minimal import WavJepaConfig, WavJepaModel

config = WavJepaConfig(
    sample_rate=16_000,
    process_seconds=2.01,
    encoder_dim=512,
    predictor_dim=256,
    transformer_layers=6,
)
model = WavJepaModel(config)
```

Because `WavJepaConfig` is frozen, use `dataclasses.replace` when deriving a new
configuration from an existing one:

```python
from dataclasses import replace

small_config = replace(config, encoder_dim=384, predictor_dim=192)
```

### Checkpoint serialization

`WavJepaConfig.to_dict()` converts the dataclass into a JSON-serializable dictionary
that is written to `<output-dir>/config.json` and embedded in `checkpoint_last.pt`.
`WavJepaConfig.from_dict()` performs the inverse operation and includes a small legacy
key map so checkpoints saved before the naming cleanup can still be loaded.

## Package structure

```text
src/wav_jepa_minimal/
  audio.py      # WAV discovery/loading, fixed-length crop/pad, synthetic smoke dataset
  config.py     # dataclass defaults, model config, mask config, conv spec
  defaults.py   # backward-compatible re-exports for older imports
  masking.py    # contiguous context/target mask sampling
  model.py      # convolutional patch encoder, transformers, predictor, WavJepaModel
  train.py      # wavjepa-train CLI and checkpoint writing
  knn.py        # wavjepa-knn frozen-embedding evaluation
```

This keeps configuration independent from model code, avoids circular imports, and makes
file names match their responsibilities: data utilities in `audio.py`, config objects in
`config.py`, mask algorithms in `masking.py`, model modules in `model.py`, and CLI entry
points in `train.py`/`knn.py`.

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
