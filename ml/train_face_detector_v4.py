#!/usr/bin/env python3
"""Train WallAlive's high-resolution v4 face-part recognizer.

V3 proved that hierarchical body + face parsing works in the browser, but its
small 96 px face model remains weak on rare cheeks/facial accessories and ears.
This experiment keeps the untouched ChildlikeSHAPES test split sealed, raises
the face crop to 128 px, oversamples rare annotations, and adds graph-paper and
label hard negatives that occur in real wall photos.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
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


class ResidualSEBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        squeeze_channels = max(6, output_channels // 6)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(output_channels, squeeze_channels, 1),
            nn.SiLU(),
            nn.Conv2d(squeeze_channels, output_channels, 1),
            nn.Sigmoid(),
        )
        self.skip = nn.Identity() if input_channels == output_channels else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.body(value)
        return self.activation(features * self.gate(features) + self.skip(value))


class FaceUNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualSEBlock(3, 20)
        self.enc2 = ResidualSEBlock(20, 36)
        self.enc3 = ResidualSEBlock(36, 60)
        self.bridge = ResidualSEBlock(60, 96)
        self.dec3 = ResidualSEBlock(96 + 60, 60)
        self.dec2 = ResidualSEBlock(60 + 36, 38)
        self.dec1 = ResidualSEBlock(38 + 20, 26)
        self.refine = ResidualSEBlock(26, 26)
        self.output = nn.Conv2d(26, len(FACE_PARTS), 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        one = self.enc1(value)
        two = self.enc2(self.pool(one))
        three = self.enc3(self.pool(two))
        bridge = self.bridge(self.pool(three))
        decoded = self.dec3(torch.cat((nn.functional.interpolate(bridge, size=three.shape[-2:], mode="bilinear", align_corners=False), three), 1))
        decoded = self.dec2(torch.cat((nn.functional.interpolate(decoded, size=two.shape[-2:], mode="bilinear", align_corners=False), two), 1))
        decoded = self.dec1(torch.cat((nn.functional.interpolate(decoded, size=one.shape[-2:], mode="bilinear", align_corners=False), one), 1))
        return self.output(self.refine(decoded))


def add_photo_hard_negatives(image: Image.Image, targets: np.ndarray, seed: int) -> Image.Image:
    """Add camera/paper artifacts without changing the semantic ground truth."""
    rng = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.86, 1.12))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.22))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.72, 1.35))
    if rng.random() < 0.28:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.12, 0.72)))

    if rng.random() < 0.48:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        spacing = rng.randint(max(8, image.width // 15), max(12, image.width // 9))
        color = rng.choice(((76, 139, 188, rng.randint(18, 42)), (120, 125, 138, rng.randint(15, 34)), (211, 94, 112, rng.randint(12, 26))))
        for x in range(rng.randrange(spacing), image.width, spacing):
            draw.line((x, 0, x, image.height), fill=color, width=1)
        for y in range(rng.randrange(spacing), image.height, spacing):
            draw.line((0, y, image.width, y), fill=color, width=1)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    # A small label or handwriting fragment above the head is a hard negative,
    # not an ear. Draw it only where no annotated facial part is present.
    union = targets.max(axis=0)
    occupied_rows = np.nonzero(union)[0]
    clear_top = int(occupied_rows.min()) if len(occupied_rows) else image.height // 3
    if clear_top >= 8 and rng.random() < 0.28:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        top = rng.randint(1, max(1, clear_top - 6))
        left = rng.randint(0, max(0, image.width - image.width // 3))
        label_width = rng.randint(max(8, image.width // 10), max(12, image.width // 3))
        label_height = rng.randint(3, max(4, min(clear_top - top, image.height // 10)))
        draw.rounded_rectangle((left, top, min(image.width - 1, left + label_width), top + label_height), radius=2, outline=(72, 122, 174, 70), width=1)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    return image


class FaceOnlyDataset(Dataset):
    def __init__(self, source: Dataset, augment: bool, seed_offset: int):
        self.source = source
        self.augment = augment
        self.seed_offset = seed_offset

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        _, _, _, face_image, face_targets = self.source[index]
        targets = face_targets.numpy().astype(np.uint8, copy=False)
        if self.augment:
            pil = Image.fromarray(np.moveaxis(face_image.numpy(), 0, -1), "RGB")
            pil = add_photo_hard_negatives(pil, targets, self.seed_offset + index * 104729 + random.randrange(1_000_000))
            face_image = torch.from_numpy(np.moveaxis(np.asarray(pil, dtype=np.uint8), -1, 0).copy())
        return face_image, face_targets


def rare_sample_weights(dataset: ChildlikeDataset) -> list[float]:
    weights: list[float] = []
    for path in dataset.paths:
        label = np.asarray(Image.open(dataset.label_root / path.name), dtype=np.uint8)
        present = set(int(value) for value in np.unique(label))
        weight = 1.0
        if any(value in present for value in CHEEK_IDS):
            weight += 4.0
        if any(value in present for value in EAR_IDS):
            weight += 2.4
        if any(value in present for value in MOUTH_IDS):
            weight += 1.0
        if any(value in present for value in EYE_IDS):
            weight += 0.5
        weights.append(weight)
    return weights


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = (probabilities + targets).sum(dim=(0, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    true_positive = (probabilities * targets).sum(dim=(0, 2, 3))
    false_positive = (probabilities * (1 - targets)).sum(dim=(0, 2, 3))
    false_negative = ((1 - probabilities) * targets).sum(dim=(0, 2, 3))
    score = (true_positive + 1) / (true_positive + false_positive * 0.35 + false_negative * 0.65 + 1)
    return ((1 - score).clamp_min(0) ** 0.78).mean()


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    probability_edges = nn.functional.max_pool2d(probabilities, 3, stride=1, padding=1) + nn.functional.max_pool2d(-probabilities, 3, stride=1, padding=1)
    target_edges = nn.functional.max_pool2d(targets, 3, stride=1, padding=1) + nn.functional.max_pool2d(-targets, 3, stride=1, padding=1)
    return nn.functional.smooth_l1_loss(probability_edges, target_edges)


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, positive_weights: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weights)
    return bce * 0.42 + dice_loss(logits, targets) * 0.32 + tversky_loss(logits, targets) * 0.20 + boundary_loss(logits, targets) * 0.06


def face_metrics(model: nn.Module, loader: DataLoader, device: torch.device, thresholds: list[float] | None = None) -> tuple[dict[str, float], dict[str, float]]:
    intersections = torch.zeros(len(FACE_PARTS), device=device)
    unions = torch.zeros(len(FACE_PARTS), device=device)
    presence_tp = torch.zeros(len(FACE_PARTS), device=device)
    presence_fp = torch.zeros(len(FACE_PARTS), device=device)
    presence_fn = torch.zeros(len(FACE_PARTS), device=device)
    cutoffs = torch.tensor(thresholds or [0.5] * len(FACE_PARTS), device=device).view(1, -1, 1, 1)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            predictions = model(images.to(device).float() / 255).sigmoid() >= cutoffs
            truth = targets.to(device).bool()
            intersections += (predictions & truth).sum(dim=(0, 2, 3))
            unions += (predictions | truth).sum(dim=(0, 2, 3))
            predicted_presence = predictions.flatten(2).any(dim=2)
            true_presence = truth.flatten(2).any(dim=2)
            presence_tp += (predicted_presence & true_presence).sum(dim=0)
            presence_fp += (predicted_presence & ~true_presence).sum(dim=0)
            presence_fn += (~predicted_presence & true_presence).sum(dim=0)
    iou = {part: round(float((intersections[index] / unions[index].clamp_min(1)).cpu()), 4) for index, part in enumerate(FACE_PARTS)}
    f1 = {part: round(float((2 * presence_tp[index] / (2 * presence_tp[index] + presence_fp[index] + presence_fn[index]).clamp_min(1)).cpu()), 4) for index, part in enumerate(FACE_PARTS)}
    return iou, f1


def calibrate_thresholds(model: nn.Module, loader: DataLoader, device: torch.device) -> list[float]:
    candidates = torch.arange(0.08, 0.821, 0.02, device=device)
    intersections = torch.zeros((len(candidates), len(FACE_PARTS)), device=device)
    unions = torch.zeros_like(intersections)
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            probabilities = model(images.to(device).float() / 255).sigmoid()
            truth = targets.to(device).bool()
            for threshold_index, threshold in enumerate(candidates):
                predictions = probabilities >= threshold
                intersections[threshold_index] += (predictions & truth).sum(dim=(0, 2, 3))
                unions[threshold_index] += (predictions | truth).sum(dim=(0, 2, 3))
    scores = intersections / unions.clamp_min(1)
    best = scores.argmax(dim=0)
    return [round(float(candidates[index].cpu()), 2) for index in best]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("public/models/wallalive-face-v4.onnx"))
    parser.add_argument("--face-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=28)
    # Zero is slower but works in restricted CI/sandbox environments where
    # PyTorch's shared-memory manager cannot create its IPC file.
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--synthetic-samples", type=int, default=3500)
    parser.add_argument("--samples-per-epoch", type=int, default=7000, help="Balanced draws per epoch; rare classes are sampled with replacement")
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--weight-cache", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(20260901)
    random.seed(20260901)
    started = time.perf_counter()
    all_paths = sorted((args.root / "train_images").glob("*.png"))
    random.Random(20260831).shuffle(all_paths)  # Keep exactly the sealed v3 split.
    if args.limit_train:
        training_paths = all_paths[:args.limit_train]
        validation_paths = all_paths[args.limit_train:args.limit_train + max(32, min(128, args.limit_train // 2))]
    else:
        validation_count = min(args.validation_count, max(1, len(all_paths) // 10))
        training_paths = all_paths[:-validation_count]
        validation_paths = all_paths[-validation_count:]

    real = ChildlikeDataset(args.root, "train", 96, args.face_size, augment=True, paths=training_paths)
    synthetic = SyntheticDataset(args.synthetic_samples, 96, args.face_size)
    training = ConcatDataset((FaceOnlyDataset(real, True, 41_000_000), FaceOnlyDataset(synthetic, True, 51_000_000)))
    if args.weight_cache and args.weight_cache.exists():
        cached_weights = np.load(args.weight_cache)
        if len(cached_weights) != len(real):
            raise ValueError(f"Sampling cache has {len(cached_weights)} entries, expected {len(real)}")
        real_weights = cached_weights.tolist()
    else:
        real_weights = rare_sample_weights(real)
        if args.weight_cache:
            args.weight_cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.weight_cache, np.asarray(real_weights, dtype=np.float32))
    weights = real_weights + [3.0] * len(synthetic)
    samples_per_epoch = min(len(training), max(args.batch_size, args.samples_per_epoch))
    sampler = WeightedRandomSampler(weights, num_samples=samples_per_epoch, replacement=True, generator=torch.Generator().manual_seed(20260901))
    validation_base = ChildlikeDataset(args.root, "train", 96, args.face_size, augment=False, paths=validation_paths)
    official_base = ChildlikeDataset(args.root, "test", 96, args.face_size, args.limit_test, augment=False)
    validation = FaceOnlyDataset(validation_base, False, 0)
    official_test = FaceOnlyDataset(official_base, False, 0)
    train_loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FaceUNetV4().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.8e-4)
    # Three high-recall warmup epochs teach the rare accessory/ear channels to
    # activate. The moderate continuation then suppresses the "cheek on every
    # face" failure while Dice/Tversky preserve the learned rare-part recall.
    warmup_positive_weights = torch.tensor([4.5, 18.0, 6.0, 7.5], device=device).view(1, -1, 1, 1)
    positive_weights = torch.tensor([4.0, 7.0, 5.0, 5.5], device=device).view(1, -1, 1, 1)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_iou: dict[str, float] = {}
    best_presence: dict[str, float] = {}
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
        best_iou = checkpoint["best_iou"]
        best_presence = checkpoint["best_presence"]
    print(json.dumps({"setup": {"real": len(real), "synthetic": len(synthetic), "samples_per_epoch": samples_per_epoch, "face_size": args.face_size, "parameters": sum(parameter.numel() for parameter in model.parameters()), "start_epoch": start_epoch}}), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss = 0.0
        epoch_positive_weights = warmup_positive_weights if epoch < 3 else positive_weights
        for batch_index, (images, targets) in enumerate(train_loader):
            images = images.to(device).float() / 255
            targets = targets.to(device).float()
            logits = model(images)
            loss = segmentation_loss(logits, targets, epoch_positive_weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * images.shape[0]
            if (batch_index + 1) % 100 == 0:
                seen = min((batch_index + 1) * args.batch_size, samples_per_epoch)
                print(json.dumps({"epoch": epoch + 1, "batch": batch_index + 1, "seen": seen, "loss_so_far": round(total_loss / seen, 4)}), flush=True)
        scheduler.step()
        iou, presence = face_metrics(model, validation_loader, device)
        score = iou["eye"] * 0.24 + iou["cheek"] * 0.27 + iou["mouth"] * 0.20 + iou["ear"] * 0.29
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_iou = iou
            best_presence = presence
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "loss": round(total_loss / samples_per_epoch, 4), "iou": iou, "presence_f1": presence}), flush=True)
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "best_iou": best_iou,
            "best_presence": best_presence,
        }, checkpoint_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    thresholds = calibrate_thresholds(model, validation_loader, device)
    calibrated_iou, calibrated_presence = face_metrics(model, validation_loader, device, thresholds)
    official_loader = DataLoader(official_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    official_iou, official_presence = face_metrics(model, official_loader, device, thresholds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, args.face_size, args.face_size),
        args.output,
        input_names=["face_values"],
        output_names=["face_logits"],
        dynamic_axes={"face_values": {0: "batch"}, "face_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    baseline_path = Path("public/models/wallalive-parts-v3.json")
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    report = {
        "architecture": "WallAlive ChildlikeSHAPES FaceUNet v4",
        "face_parts": FACE_PARTS,
        "input": [1, 3, args.face_size, args.face_size],
        "official_training_drawings": len(real),
        "validation_drawings": len(validation),
        "official_test_drawings": len(official_test),
        "synthetic_training_drawings": args.synthetic_samples,
        "balanced_samples_per_epoch": samples_per_epoch,
        "validation_face_iou": best_iou,
        "validation_presence_f1": best_presence,
        "calibrated_validation_face_iou": calibrated_iou,
        "calibrated_validation_presence_f1": calibrated_presence,
        "face_thresholds": dict(zip(FACE_PARTS, thresholds, strict=True)),
        "official_test_face_iou": official_iou,
        "official_test_presence_f1": official_presence,
        "v3_official_test_face_iou": baseline.get("official_test_face_iou", {}),
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "resumed_from_epoch": start_epoch,
        "seconds": round(time.perf_counter() - started, 1),
        "dataset_license": "ChildlikeSHAPES CC-BY-4.0",
        "test_split_used_for_selection": False,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
