"""Command-line training entry point for the minimal WavJEPA model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from wav_jepa_minimal.audio import SyntheticWaveDataset, WaveDirectoryDataset
from wav_jepa_minimal.defaults import AUDIOSET_DEFAULTS
from wav_jepa_minimal.model import WavJepaConfig, WavJepaModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, required=True, help="Directory of AudioSet WAV files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/audioset-minimal"))
    parser.add_argument("--dataset-name", default=AUDIOSET_DEFAULTS.dataset_name)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=AUDIOSET_DEFAULTS.max_steps)
    parser.add_argument("--batch-size", type=int, default=AUDIOSET_DEFAULTS.batch_size)
    parser.add_argument("--learning-rate", type=float, default=AUDIOSET_DEFAULTS.learning_rate)
    parser.add_argument("--adam-beta1", type=float, default=AUDIOSET_DEFAULTS.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=AUDIOSET_DEFAULTS.adam_beta2)
    parser.add_argument("--weight-decay", type=float, default=AUDIOSET_DEFAULTS.weight_decay)
    parser.add_argument("--sample-rate", type=int, default=AUDIOSET_DEFAULTS.sample_rate)
    parser.add_argument("--process-seconds", type=float, default=AUDIOSET_DEFAULTS.process_seconds)
    parser.add_argument(
        "--samples-per-audio", type=int, default=AUDIOSET_DEFAULTS.samples_per_audio
    )
    parser.add_argument("--embed-dim", type=int, default=AUDIOSET_DEFAULTS.encoder_dim)
    parser.add_argument("--predictor-dim", type=int, default=AUDIOSET_DEFAULTS.decoder_dim)
    parser.add_argument(
        "--transformer-layers", type=int, default=AUDIOSET_DEFAULTS.transformer_layers
    )
    parser.add_argument("--attention-heads", type=int, default=AUDIOSET_DEFAULTS.attention_heads)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=AUDIOSET_DEFAULTS.seed)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic AudioSet-shaped waves for smoke tests.",
    )
    return parser.parse_args()


def build_loader(args: argparse.Namespace) -> DataLoader[torch.Tensor]:
    if args.synthetic:
        dataset = SyntheticWaveDataset(
            length=max(args.batch_size * max(args.steps_per_epoch or 1, 1), args.batch_size),
            sample_rate=args.sample_rate,
            seconds=args.process_seconds,
        )
    else:
        dataset = WaveDirectoryDataset(
            root=args.data_dir,
            sample_rate=args.sample_rate,
            seconds=args.process_seconds,
            random_crop=True,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )


def save_checkpoint(
    output_dir: Path,
    model: WavJepaModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model.config.to_dict(),
        "epoch": epoch,
        "step": step,
    }
    torch.save(checkpoint, output_dir / "checkpoint_last.pt")
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(model.config.to_dict(), handle, indent=2)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_loader(args)
    config = WavJepaConfig(
        dataset_name=args.dataset_name,
        sample_rate=args.sample_rate,
        process_seconds=args.process_seconds,
        samples_per_audio=args.samples_per_audio,
        embed_dim=args.embed_dim,
        predictor_dim=args.predictor_dim,
        transformer_layers=args.transformer_layers,
        attention_heads=args.attention_heads,
    )
    model = WavJepaModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )

    global_step = 0
    for epoch in range(args.epochs):
        for batch_step, audio in enumerate(loader):
            if args.steps_per_epoch is not None and batch_step >= args.steps_per_epoch:
                break
            if global_step >= args.max_steps:
                break
            audio = audio.to(device)
            loss, metrics = model.forward_loss(audio)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            model.update_target_encoder(step=global_step)

            if global_step == 1 or global_step % 10 == 0:
                print(
                    f"dataset={config.dataset_name} epoch={epoch + 1} step={global_step} "
                    f"loss={metrics['loss']:.4f} "
                    f"context={metrics['context_fraction']:.2f} "
                    f"target={metrics['target_fraction']:.2f}"
                )
        save_checkpoint(args.output_dir, model, optimizer, epoch=epoch + 1, step=global_step)
        if global_step >= args.max_steps:
            break


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
