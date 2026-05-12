"""KNN evaluation over frozen WavJEPA embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from wav_jepa_minimal.audio import WaveDirectoryDataset
from wav_jepa_minimal.config import WavJepaConfig
from wav_jepa_minimal.model import WavJepaModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device) -> WavJepaModel:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = WavJepaConfig.from_dict(checkpoint["config"])
    model = WavJepaModel(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def embed_dataset(
    model: WavJepaModel,
    root: Path,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[Tensor, list[str]]:
    dataset = WaveDirectoryDataset(
        root=root,
        sample_rate=model.config.sample_rate,
        seconds=model.config.seconds,
        with_labels=True,
        random_crop=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    embeddings: list[Tensor] = []
    labels: list[str] = []
    with torch.no_grad():
        for audio, batch_labels in loader:
            embeddings.append(model.embed(audio.to(device)).cpu())
            labels.extend(batch_labels)
    return torch.cat(embeddings, dim=0), labels


def majority_vote(labels: list[str]) -> str:
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def predict_knn(
    train_embeddings: Tensor,
    train_labels: list[str],
    test_embeddings: Tensor,
    k: int,
) -> list[str]:
    normalized_train = torch.nn.functional.normalize(train_embeddings, dim=1)
    normalized_test = torch.nn.functional.normalize(test_embeddings, dim=1)
    similarities = normalized_test @ normalized_train.T
    topk = similarities.topk(k=min(k, len(train_labels)), dim=1).indices
    predictions: list[str] = []
    for neighbors in topk.tolist():
        predictions.append(majority_vote([train_labels[index] for index in neighbors]))
    return predictions


def evaluate(args: argparse.Namespace) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    train_embeddings, train_labels = embed_dataset(
        model, args.train_dir, args.batch_size, args.num_workers, device
    )
    test_embeddings, test_labels = embed_dataset(
        model, args.test_dir, args.batch_size, args.num_workers, device
    )
    predictions = predict_knn(train_embeddings, train_labels, test_embeddings, args.k)
    correct = sum(
        prediction == label
        for prediction, label in zip(predictions, test_labels, strict=True)
    )
    accuracy = correct / len(test_labels)
    print(f"knn_accuracy={accuracy:.4f} correct={correct} total={len(test_labels)} k={args.k}")
    return accuracy


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
