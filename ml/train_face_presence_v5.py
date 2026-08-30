#!/usr/bin/env python3
"""Train WallAlive's compact face-part presence gate.

The v3+v4 ensemble is good at drawing masks when a part exists, but rare
cheeks/accessories and ears can still activate on handwriting or paper folds.
This model deliberately predicts only whether each face part exists. The
segmentation ensemble remains responsible for position, contour, count, and
color. Thresholds and checkpoints are selected on a deterministic validation
split; the official ChildlikeSHAPES test split is evaluated exactly once.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from train_childlike_detector import (
    CHEEK_IDS,
    EAR_IDS,
    EYE_IDS,
    FACE_PARTS,
    MOUTH_IDS,
    ChildlikeDataset,
    SyntheticDataset,
)
from train_face_detector_v4 import FaceOnlyDataset


PART_IDS = (EYE_IDS, CHEEK_IDS, MOUTH_IDS, EAR_IDS)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        squeezed = max(6, channels // 6)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, squeezed, 1),
            nn.SiLU(),
            nn.Conv2d(squeezed, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.net(values)


class MobileResidualSE(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int):
        super().__init__()
        expanded = input_channels * 2
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, expanded, 1, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(),
            nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False),
            nn.BatchNorm2d(expanded),
            nn.SiLU(),
            SqueezeExcite(expanded),
            nn.Conv2d(expanded, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = stride == 1 and input_channels == output_channels
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        features = self.body(values)
        return self.activation(features + values) if self.skip else self.activation(features)


class FacePresenceNet(nn.Module):
    """A browser-sized multi-label classifier with global face context."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(20),
            nn.SiLU(),
            MobileResidualSE(20, 28, 2),
            MobileResidualSE(28, 40, 2),
            MobileResidualSE(40, 56, 2),
            MobileResidualSE(56, 72, 2),
            MobileResidualSE(72, 72, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.output = nn.Sequential(nn.Flatten(), nn.Dropout(0.10), nn.Linear(72, len(FACE_PARTS)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(self.features(values))


class PresenceDataset(Dataset):
    def __init__(self, source: Dataset):
        self.source = source

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, masks = self.source[index]
        presence = masks.flatten(1).any(dim=1).float()
        return image, presence


def real_sample_weights(dataset: ChildlikeDataset) -> tuple[list[float], list[int]]:
    """Moderately expose rare labels without teaching an always-on cheek."""
    weights: list[float] = []
    positives = [0] * len(FACE_PARTS)
    for path in dataset.paths:
        label = np.asarray(Image.open(dataset.label_root / path.name), dtype=np.uint8)
        present_values = set(int(value) for value in np.unique(label))
        present = [any(value in present_values for value in ids) for ids in PART_IDS]
        for channel, exists in enumerate(present):
            positives[channel] += int(exists)
        weights.append(1.0 + 1.15 * present[1] + 0.75 * present[3])
    return weights, positives


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weights: torch.Tensor,
    gamma_negative: float = 2.0,
    gamma_positive: float = 0.0,
    negative_clip: float = 0.02,
) -> torch.Tensor:
    """Asymmetric multi-label loss with hard-negative emphasis.

    Easy negative labels dominate rare multi-label classification. Clipping and
    separate focal exponents reduce their gradient while preserving hard false
    positives such as handwriting that resembles an ear or cheek mark.
    """
    probabilities = logits.sigmoid()
    positive_probability = probabilities.clamp(1e-7, 1 - 1e-7)
    negative_probability = (1 - probabilities + negative_clip).clamp(1e-7, 1)
    positive_loss = targets * torch.log(positive_probability) * positive_weights
    negative_loss = (1 - targets) * torch.log(negative_probability)
    positive_focus = (1 - positive_probability).pow(gamma_positive)
    negative_focus = (1 - negative_probability).pow(gamma_negative)
    return -(positive_loss * positive_focus + negative_loss * negative_focus).mean()


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    probability_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            probabilities = model(images.to(device).float() / 255).sigmoid().cpu().numpy()
            probability_batches.append(probabilities)
            target_batches.append(targets.numpy().astype(bool, copy=False))
    return np.concatenate(probability_batches), np.concatenate(target_batches)


def binary_metrics(probabilities: np.ndarray, truth: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    true_negative = int((~predicted & ~truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "false_positive_rate": round(false_positive / max(1, false_positive + true_negative), 4),
        "prevalence": round(float(truth.mean()), 4),
    }


def binary_auroc(probabilities: np.ndarray, truth: np.ndarray) -> float:
    """Rank-based AUROC without adding a scikit-learn training dependency."""
    positive_count = int(truth.sum())
    negative_count = len(truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        return 0.0
    order = np.argsort(probabilities, kind="mergesort")
    sorted_values = probabilities[order]
    ranks = np.arange(1, len(probabilities) + 1, dtype=np.float64)
    start = 0
    while start < len(sorted_values):
        end = start + 1
        while end < len(sorted_values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = ranks[start:end].mean()
        start = end
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    positive_rank_sum = ranks[inverse][truth].sum()
    auc = (positive_rank_sum - positive_count * (positive_count + 1) / 2) / (positive_count * negative_count)
    return round(float(auc), 4)


def calibrate_thresholds(probabilities: np.ndarray, truth: np.ndarray) -> tuple[list[float], dict[str, dict[str, float]]]:
    candidates = np.arange(0.05, 0.951, 0.01)
    thresholds: list[float] = []
    metrics: dict[str, dict[str, float]] = {}
    for channel, part in enumerate(FACE_PARTS):
        scored: list[tuple[float, float, float, float]] = []
        for threshold in candidates:
            result = binary_metrics(probabilities[:, channel], truth[:, channel], float(threshold))
            # F1 selects the operating point; precision and lower FPR resolve
            # near-ties so the gate does not recreate the rare-part flood.
            scored.append((result["f1"], result["precision"], -result["false_positive_rate"], float(threshold)))
        best = max(scored)
        threshold = round(best[3], 2)
        thresholds.append(threshold)
        metrics[part] = binary_metrics(probabilities[:, channel], truth[:, channel], threshold)
        metrics[part]["auroc"] = binary_auroc(probabilities[:, channel], truth[:, channel])
    return thresholds, metrics


def evaluate_at_thresholds(probabilities: np.ndarray, truth: np.ndarray, thresholds: list[float]) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for channel, part in enumerate(FACE_PARTS):
        report[part] = binary_metrics(probabilities[:, channel], truth[:, channel], thresholds[channel])
        report[part]["auroc"] = binary_auroc(probabilities[:, channel], truth[:, channel])
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("public/models/wallalive-face-presence-v5.onnx"))
    parser.add_argument("--face-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--synthetic-samples", type=int, default=3500)
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weight-cache", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(20260902)
    random.seed(20260902)
    started = time.perf_counter()
    all_paths = sorted((args.root / "train_images").glob("*.png"))
    random.Random(20260831).shuffle(all_paths)
    validation_paths = all_paths[-args.validation_count:]
    training_paths = all_paths[:-args.validation_count]

    real = ChildlikeDataset(args.root, "train", 96, args.face_size, augment=True, paths=training_paths)
    synthetic = SyntheticDataset(args.synthetic_samples, 96, args.face_size)
    training = ConcatDataset((
        PresenceDataset(FaceOnlyDataset(real, True, 61_000_000)),
        PresenceDataset(FaceOnlyDataset(synthetic, True, 71_000_000)),
    ))
    if args.weight_cache and args.weight_cache.exists():
        cached = np.load(args.weight_cache)
        if len(cached) != len(real):
            raise ValueError(f"Sampling cache has {len(cached)} entries, expected {len(real)}")
        real_weights = cached[:, 0].tolist()
        training_positives = cached[0, 1:5].astype(int).tolist()
    else:
        real_weights, training_positives = real_sample_weights(real)
        if args.weight_cache:
            args.weight_cache.parent.mkdir(parents=True, exist_ok=True)
            cache = np.zeros((len(real), 5), dtype=np.float32)
            cache[:, 0] = real_weights
            cache[0, 1:5] = training_positives
            np.save(args.weight_cache, cache)
    weights = real_weights + [1.35] * len(synthetic)
    samples_per_epoch = min(len(training), max(args.batch_size, args.samples_per_epoch))
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True, generator=torch.Generator().manual_seed(20260902))

    validation_base = ChildlikeDataset(args.root, "train", 96, args.face_size, augment=False, paths=validation_paths)
    official_base = ChildlikeDataset(args.root, "test", 96, args.face_size, augment=False)
    validation = PresenceDataset(FaceOnlyDataset(validation_base, False, 0))
    official_test = PresenceDataset(FaceOnlyDataset(official_base, False, 0))
    training_loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FacePresenceNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.8e-4)
    positive_weights = torch.tensor([0.72, 2.25, 0.82, 1.75], device=device).view(1, -1)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, dict[str, float]] = {}
    start_epoch = 0
    checkpoint_path = args.checkpoint or args.output.with_suffix(".pt")
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = checkpoint["best_state"]
        best_validation = checkpoint["best_validation"]
    print(json.dumps({"setup": {
        "real": len(real),
        "synthetic": len(synthetic),
        "samples_per_epoch": samples_per_epoch,
        "face_size": args.face_size,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "real_training_prevalence": dict(zip(FACE_PARTS, (round(value / len(real), 4) for value in training_positives), strict=True)),
        "device": str(device),
        "start_epoch": start_epoch,
    }}), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        for batch_index, (images, targets) in enumerate(training_loader):
            images = images.to(device).float() / 255
            targets = targets.to(device)
            loss = asymmetric_loss(model(images), targets, positive_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * images.shape[0]
            if (batch_index + 1) % 40 == 0:
                seen = min((batch_index + 1) * args.batch_size, samples_per_epoch)
                print(json.dumps({"epoch": epoch + 1, "seen": seen, "loss_so_far": round(total_loss / seen, 5)}), flush=True)
        scheduler.step()
        probabilities, truth = collect_predictions(model, validation_loader, device)
        _, validation_metrics = calibrate_thresholds(probabilities, truth)
        score = (
            validation_metrics["eye"]["f1"] * 0.12
            + validation_metrics["cheek"]["f1"] * 0.42
            + validation_metrics["mouth"]["f1"] * 0.12
            + validation_metrics["ear"]["f1"] * 0.34
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_validation = validation_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "loss": round(total_loss / samples_per_epoch, 5), "validation": validation_metrics, "score": round(score, 5)}), flush=True)
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "best_validation": best_validation,
        }, checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    validation_probabilities, validation_truth = collect_predictions(model, validation_loader, device)
    thresholds, validation_metrics = calibrate_thresholds(validation_probabilities, validation_truth)

    # This is the only official-test inference in the training pipeline, after
    # all checkpoint and threshold choices have been frozen on validation.
    official_loader = DataLoader(official_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    official_probabilities, official_truth = collect_predictions(model, official_loader, device)
    official_metrics = evaluate_at_thresholds(official_probabilities, official_truth, thresholds)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, args.face_size, args.face_size),
        args.output,
        input_names=["face_values"],
        output_names=["presence_logits"],
        dynamic_axes={"face_values": {0: "batch"}, "presence_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report = {
        "architecture": "WallAlive FacePresenceNet v5",
        "face_parts": FACE_PARTS,
        "input": [1, 3, args.face_size, args.face_size],
        "official_training_drawings": len(real),
        "validation_drawings": len(validation),
        "official_test_drawings": len(official_test),
        "synthetic_training_drawings": len(synthetic),
        "balanced_samples_per_epoch": samples_per_epoch,
        "presence_thresholds": dict(zip(FACE_PARTS, thresholds, strict=True)),
        "validation_presence_metrics": validation_metrics,
        "official_test_presence_metrics": official_metrics,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "seconds": round(time.perf_counter() - started, 1),
        "dataset_license": "ChildlikeSHAPES CC-BY-4.0",
        "test_split_used_for_selection": False,
        "role": "presence gate only; segmentation ensemble retains geometry and color",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
