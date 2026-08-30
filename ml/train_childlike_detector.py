#!/usr/bin/env python3
"""Train WallAlive v3 on Meta's pixel-labeled ChildlikeSHAPES drawings.

The browser model is deliberately smaller than CharSegNet/SAM, but follows its
three important ideas: predict broad anatomy before small parts, keep exact
pixel supervision, and parse an enlarged face crop separately. The official
test split is never used for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from train_part_detector import make_sample


PARTS = ("body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot")
COARSE_PARTS = ("foreground", "head", "torso", "upper_appendage", "lower_appendage")
FACE_PARTS = ("eye", "cheek", "mouth", "ear")

# Palette indices in the released ChildlikeSHAPES PNG annotations. The public
# label_definition.json omits the final Neck and Tail entries, but their palette
# indices and binary masks are present in the archive.
EYE_IDS = (4, 22)
CHEEK_IDS = (5,)
MOUTH_IDS = (3, 17, 23)
EAR_IDS = (18,)
ARM_IDS = (9, 16)
HAND_IDS = (7, 10)
LEG_IDS = (1, 14)
FOOT_IDS = (24,)
HEAD_IDS = (2, 3, 4, 5, 6, 12, 13, 17, 18, 20, 22, 23)
TORSO_IDS = (8, 11, 15, 21, 25, 26)


def mask_for(label: np.ndarray, ids: tuple[int, ...]) -> np.ndarray:
    return np.isin(label, np.asarray(ids, dtype=np.uint8))


def semantic_targets(label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    foreground = (label > 0) & (label < 255)
    parts = np.stack((
        foreground,
        mask_for(label, EYE_IDS),
        mask_for(label, CHEEK_IDS),
        mask_for(label, MOUTH_IDS),
        mask_for(label, EAR_IDS),
        mask_for(label, ARM_IDS),
        mask_for(label, HAND_IDS),
        mask_for(label, LEG_IDS),
        mask_for(label, FOOT_IDS),
    )).astype(np.uint8)
    coarse = np.stack((
        foreground,
        mask_for(label, HEAD_IDS),
        mask_for(label, TORSO_IDS),
        mask_for(label, ARM_IDS + HAND_IDS),
        mask_for(label, LEG_IDS + FOOT_IDS),
    )).astype(np.uint8)
    return parts, coarse


def letterbox(image: Image.Image, label: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    width, height = image.size
    scale = min(size / max(1, width), size / max(1, height))
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(resized, Image.Resampling.BILINEAR)
    label = label.resize(resized, Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    label_canvas = Image.new("L", (size, size), 0)
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas.paste(image, offset)
    label_canvas.paste(label, offset)
    return canvas, label_canvas


def crop_face(image: Image.Image, label: Image.Image, size: int) -> tuple[Image.Image, np.ndarray]:
    label_array = np.asarray(label, dtype=np.uint8)
    head = mask_for(label_array, HEAD_IDS)
    ys, xs = np.nonzero(head)
    if not len(xs):
        width, height = image.size
        box = (0, 0, width, max(1, round(height * 0.52)))
    else:
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        span = max(max_x - min_x + 1, max_y - min_y + 1)
        margin = max(3, round(span * 0.18))
        box = (
            max(0, min_x - margin),
            max(0, min_y - margin),
            min(image.width, max_x + margin + 1),
            min(image.height, max_y + margin + 1),
        )
    face_image, face_label = letterbox(image.crop(box), label.crop(box), size)
    face_array = np.asarray(face_label, dtype=np.uint8)
    targets = np.stack((
        mask_for(face_array, EYE_IDS),
        mask_for(face_array, CHEEK_IDS),
        mask_for(face_array, MOUTH_IDS),
        mask_for(face_array, EAR_IDS),
    )).astype(np.uint8)
    return face_image, targets


def augment_pair(image: Image.Image, label: Image.Image, seed: int) -> tuple[Image.Image, Image.Image]:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    rng = random.Random(seed)
    width, height = image.size
    angle = rng.uniform(-12, 12)
    translate = (rng.uniform(-width * 0.035, width * 0.035), rng.uniform(-height * 0.035, height * 0.035))
    scale = rng.uniform(0.90, 1.08)
    shear = rng.uniform(-4, 4)
    paper = tuple(int(value) for value in np.asarray(image).reshape(-1, 3).mean(axis=0))
    image = tvf.affine(image, angle=angle, translate=translate, scale=scale, shear=[shear, 0], interpolation=InterpolationMode.BILINEAR, fill=paper)
    label = tvf.affine(label, angle=angle, translate=translate, scale=scale, shear=[shear, 0], interpolation=InterpolationMode.NEAREST, fill=0)
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.65)))
    array = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(seed ^ 0xC41D).normal(0, rng.uniform(0.0, 2.5), array.shape)
    image = Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")
    return image, label


class ChildlikeDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int, face_size: int, limit: int = 0, augment: bool = False, paths: list[Path] | None = None):
        self.image_root = root / f"{split}_images"
        self.label_root = root / f"{split}_annos"
        self.paths = paths if paths is not None else sorted(self.image_root.glob("*.png"))
        if limit:
            self.paths = self.paths[:limit]
        self.size = size
        self.face_size = face_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        # Converting a palette PNG directly to L maps its colors to luminance
        # and destroys the class ids. Re-wrap the raw palette indices instead.
        palette_label = Image.open(self.label_root / path.name)
        label = Image.fromarray(np.asarray(palette_label, dtype=np.uint8), "L")
        if self.augment:
            image, label = augment_pair(image, label, 31_000_000 + index * 7919 + random.randrange(1_000_000))
        face_image, face_targets = crop_face(image, label, self.face_size)
        image, label = letterbox(image, label, self.size)
        parts, coarse = semantic_targets(np.asarray(label, dtype=np.uint8))
        return (
            torch.from_numpy(np.moveaxis(np.asarray(image, dtype=np.uint8), -1, 0).copy()),
            torch.from_numpy(parts),
            torch.from_numpy(coarse),
            torch.from_numpy(np.moveaxis(np.asarray(face_image, dtype=np.uint8), -1, 0).copy()),
            torch.from_numpy(face_targets),
        )


class SyntheticDataset(Dataset):
    def __init__(self, count: int, size: int, face_size: int):
        self.count = count
        self.size = size
        self.face_size = face_size

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        image, parts = make_sample(70_000_000 + index * 8191, self.size)
        image_hwc = np.moveaxis(image, 0, -1)
        face_union = parts[1:5].max(axis=0)
        ys, xs = np.nonzero(face_union)
        if len(xs):
            span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            margin = max(2, round(span * 0.28))
            box = (
                max(0, int(xs.min()) - margin),
                max(0, int(ys.min()) - margin),
                min(self.size, int(xs.max()) + margin + 1),
                min(self.size, int(ys.max()) + margin + 1),
            )
        else:
            box = (0, 0, self.size, max(1, round(self.size * 0.52)))
        face_image = Image.fromarray(image_hwc, "RGB").crop(box).resize((self.face_size, self.face_size), Image.Resampling.BILINEAR)
        face_targets = np.stack([
            np.asarray(Image.fromarray(parts[channel] * 255).crop(box).resize((self.face_size, self.face_size), Image.Resampling.NEAREST)) >= 128
            for channel in (1, 2, 3, 4)
        ]).astype(np.uint8)
        coarse = np.stack((
            parts[0],
            parts[1:5].max(axis=0),
            parts[0] & ~(parts[5:9].max(axis=0).astype(bool)),
            parts[5:7].max(axis=0),
            parts[7:9].max(axis=0),
        )).astype(np.uint8)
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(parts.copy()),
            torch.from_numpy(coarse),
            torch.from_numpy(np.moveaxis(np.asarray(face_image, dtype=np.uint8), -1, 0).copy()),
            torch.from_numpy(face_targets),
        )


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class HierarchicalPartUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 18)
        self.enc2 = ConvBlock(18, 30)
        self.enc3 = ConvBlock(30, 50)
        self.bridge = ConvBlock(50, 78)
        self.dec3 = ConvBlock(78 + 50, 50)
        self.dec2 = ConvBlock(50 + 30, 32)
        self.dec1 = ConvBlock(32 + 18, 24)
        self.coarse_output = nn.Conv2d(24, len(COARSE_PARTS), 1)
        self.semantic_refiner = ConvBlock(24 + len(COARSE_PARTS), 30)
        self.output = nn.Conv2d(30, len(PARTS) - 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        one = self.enc1(value)
        two = self.enc2(self.pool(one))
        three = self.enc3(self.pool(two))
        bridge = self.bridge(self.pool(three))
        decoded = self.dec3(torch.cat((nn.functional.interpolate(bridge, size=three.shape[-2:], mode="bilinear", align_corners=False), three), 1))
        decoded = self.dec2(torch.cat((nn.functional.interpolate(decoded, size=two.shape[-2:], mode="bilinear", align_corners=False), two), 1))
        features = self.dec1(torch.cat((nn.functional.interpolate(decoded, size=one.shape[-2:], mode="bilinear", align_corners=False), one), 1))
        coarse = self.coarse_output(features)
        refined = self.semantic_refiner(torch.cat((features, coarse.sigmoid()), 1))
        return torch.cat((coarse[:, :1], self.output(refined)), 1), coarse


class FaceUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 12)
        self.enc2 = ConvBlock(12, 20)
        self.enc3 = ConvBlock(20, 32)
        self.bridge = ConvBlock(32, 48)
        self.dec3 = ConvBlock(48 + 32, 32)
        self.dec2 = ConvBlock(32 + 20, 20)
        self.dec1 = ConvBlock(20 + 12, 16)
        self.output = nn.Conv2d(16, len(FACE_PARTS), 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        one = self.enc1(value)
        two = self.enc2(self.pool(one))
        three = self.enc3(self.pool(two))
        bridge = self.bridge(self.pool(three))
        decoded = self.dec3(torch.cat((nn.functional.interpolate(bridge, size=three.shape[-2:], mode="bilinear", align_corners=False), three), 1))
        decoded = self.dec2(torch.cat((nn.functional.interpolate(decoded, size=two.shape[-2:], mode="bilinear", align_corners=False), two), 1))
        decoded = self.dec1(torch.cat((nn.functional.interpolate(decoded, size=one.shape[-2:], mode="bilinear", align_corners=False), one), 1))
        return self.output(decoded)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = (probabilities + targets).sum(dim=(0, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor, positive_weights: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weights)
    return bce * 0.52 + dice_loss(logits, targets) * 0.48


def metrics(
    model: nn.Module,
    face_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    part_thresholds: list[float] | None = None,
    face_thresholds: list[float] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    intersections = torch.zeros(len(PARTS), device=device)
    unions = torch.zeros(len(PARTS), device=device)
    face_intersections = torch.zeros(len(FACE_PARTS), device=device)
    face_unions = torch.zeros(len(FACE_PARTS), device=device)
    part_cutoffs = torch.tensor(part_thresholds or [0.5] * len(PARTS), device=device).view(1, -1, 1, 1)
    face_cutoffs = torch.tensor(face_thresholds or [0.5] * len(FACE_PARTS), device=device).view(1, -1, 1, 1)
    model.eval()
    face_model.eval()
    with torch.no_grad():
        for images, targets, _, face_images, face_targets in loader:
            predictions = model(images.to(device).float() / 255)[0].sigmoid() >= part_cutoffs
            truth = targets.to(device).bool()
            intersections += (predictions & truth).sum(dim=(0, 2, 3))
            unions += (predictions | truth).sum(dim=(0, 2, 3))
            face_predictions = face_model(face_images.to(device).float() / 255).sigmoid() >= face_cutoffs
            face_truth = face_targets.to(device).bool()
            face_intersections += (face_predictions & face_truth).sum(dim=(0, 2, 3))
            face_unions += (face_predictions | face_truth).sum(dim=(0, 2, 3))
    return (
        {part: round(float((intersections[index] / unions[index].clamp_min(1)).cpu()), 4) for index, part in enumerate(PARTS)},
        {part: round(float((face_intersections[index] / face_unions[index].clamp_min(1)).cpu()), 4) for index, part in enumerate(FACE_PARTS)},
    )


def calibrate_thresholds(model: nn.Module, face_model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], list[float]]:
    """Choose per-class cutoffs on validation only, never on the official test."""
    candidates = torch.arange(0.24, 0.721, 0.02, device=device)
    part_intersections = torch.zeros((len(candidates), len(PARTS)), device=device)
    part_unions = torch.zeros_like(part_intersections)
    face_intersections = torch.zeros((len(candidates), len(FACE_PARTS)), device=device)
    face_unions = torch.zeros_like(face_intersections)
    model.eval()
    face_model.eval()
    with torch.no_grad():
        for images, targets, _, face_images, face_targets in loader:
            probabilities = model(images.to(device).float() / 255)[0].sigmoid()
            truth = targets.to(device).bool()
            face_probabilities = face_model(face_images.to(device).float() / 255).sigmoid()
            face_truth = face_targets.to(device).bool()
            for threshold_index, threshold in enumerate(candidates):
                predictions = probabilities >= threshold
                part_intersections[threshold_index] += (predictions & truth).sum(dim=(0, 2, 3))
                part_unions[threshold_index] += (predictions | truth).sum(dim=(0, 2, 3))
                face_predictions = face_probabilities >= threshold
                face_intersections[threshold_index] += (face_predictions & face_truth).sum(dim=(0, 2, 3))
                face_unions[threshold_index] += (face_predictions | face_truth).sum(dim=(0, 2, 3))
    part_scores = part_intersections / part_unions.clamp_min(1)
    face_scores = face_intersections / face_unions.clamp_min(1)
    part_best = part_scores.argmax(dim=0)
    face_best = face_scores.argmax(dim=0)
    return (
        [round(float(candidates[index].cpu()), 2) for index in part_best],
        [round(float(candidates[index].cpu()), 2) for index in face_best],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="Extracted ChildlikeSHAPES dataset directory")
    parser.add_argument("--output", type=Path, default=Path("public/models/wallalive-parts-v3.onnx"))
    parser.add_argument("--face-output", type=Path, default=Path("public/models/wallalive-face-v3.onnx"))
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--face-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--synthetic-samples", type=int, default=3000)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(20260831)
    random.seed(20260831)
    started = time.perf_counter()
    all_training_paths = sorted((args.root / "train_images").glob("*.png"))
    # The released UUIDs are grouped by collection order, so taking the final
    # filenames as validation can accidentally select a narrow acquisition
    # slice. Shuffle once with the run seed before creating the disjoint split.
    random.Random(20260831).shuffle(all_training_paths)
    if args.limit_train:
        training_paths = all_training_paths[:args.limit_train]
        validation_paths = all_training_paths[args.limit_train:args.limit_train + max(32, min(128, args.limit_train // 2))]
    else:
        validation_count = min(args.validation_count, max(1, len(all_training_paths) // 10))
        training_paths = all_training_paths[:-validation_count]
        validation_paths = all_training_paths[-validation_count:]
    train_real = ChildlikeDataset(args.root, "train", args.size, args.face_size, augment=True, paths=training_paths)
    datasets: list[Dataset] = [train_real]
    if args.synthetic_samples:
        datasets.append(SyntheticDataset(args.synthetic_samples, args.size, args.face_size))
    training = ConcatDataset(datasets)
    validation = ChildlikeDataset(args.root, "train", args.size, args.face_size, augment=False, paths=validation_paths)
    official_test = ChildlikeDataset(args.root, "test", args.size, args.face_size, args.limit_test, augment=False)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HierarchicalPartUNet().to(device)
    face_model = FaceUNet().to(device)
    optimizer = torch.optim.AdamW((*model.parameters(), *face_model.parameters()), lr=2.2e-3, weight_decay=1e-4)
    part_weights = torch.tensor([1.2, 13.0, 24.0, 12.0, 10.0, 7.0, 9.0, 7.0, 9.0], device=device).view(1, -1, 1, 1)
    coarse_weights = torch.tensor([1.2, 3.5, 3.0, 6.0, 6.0], device=device).view(1, -1, 1, 1)
    face_weights = torch.tensor([4.0, 14.0, 5.0, 6.0], device=device).view(1, -1, 1, 1)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_face_state: dict[str, torch.Tensor] | None = None
    best_parts: dict[str, float] = {}
    best_face: dict[str, float] = {}

    for epoch in range(args.epochs):
        model.train()
        face_model.train()
        total_loss = 0.0
        for images, targets, coarse_targets, face_images, face_targets in train_loader:
            images = images.to(device).float() / 255
            targets = targets.to(device).float()
            coarse_targets = coarse_targets.to(device).float()
            face_images = face_images.to(device).float() / 255
            face_targets = face_targets.to(device).float()
            part_logits, coarse_logits = model(images)
            face_logits = face_model(face_images)
            loss = (
                segmentation_loss(part_logits, targets, part_weights) * 0.55
                + segmentation_loss(coarse_logits, coarse_targets, coarse_weights) * 0.17
                + segmentation_loss(face_logits, face_targets, face_weights) * 0.28
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * images.shape[0]
        part_iou, face_iou = metrics(model, face_model, validation_loader, device)
        score = sum(part_iou.values()) / len(part_iou) * 0.65 + sum(face_iou.values()) / len(face_iou) * 0.35
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_parts = part_iou
            best_face = face_iou
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_face_state = {name: value.detach().cpu().clone() for name, value in face_model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "loss": round(total_loss / len(training), 4), "part_iou": part_iou, "face_iou": face_iou}), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if best_face_state is not None:
        face_model.load_state_dict(best_face_state)
    part_thresholds, face_thresholds = calibrate_thresholds(model, face_model, validation_loader, device)
    calibrated_validation_parts, calibrated_validation_face = metrics(
        model, face_model, validation_loader, device, part_thresholds, face_thresholds,
    )
    official_test_loader = DataLoader(official_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    official_part_iou, official_face_iou = metrics(
        model, face_model, official_test_loader, device, part_thresholds, face_thresholds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    face_model = face_model.to("cpu").eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, args.size, args.size),
        args.output,
        input_names=["pixel_values"],
        output_names=["part_logits", "coarse_logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "part_logits": {0: "batch"}, "coarse_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        face_model,
        torch.zeros(1, 3, args.face_size, args.face_size),
        args.face_output,
        input_names=["face_values"],
        output_names=["face_logits"],
        dynamic_axes={"face_values": {0: "batch"}, "face_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report = {
        "architecture": "WallAlive ChildlikeSHAPES PartUNet v3",
        "parts": PARTS,
        "coarse_channels": COARSE_PARTS,
        "face_parts": FACE_PARTS,
        "input": [1, 3, args.size, args.size],
        "face_input": [1, 3, args.face_size, args.face_size],
        "official_training_drawings": len(train_real),
        "validation_drawings": len(validation),
        "official_test_drawings": len(official_test),
        "synthetic_training_drawings": args.synthetic_samples,
        "validation_part_iou": best_parts,
        "validation_face_iou": best_face,
        "calibrated_validation_part_iou": calibrated_validation_parts,
        "calibrated_validation_face_iou": calibrated_validation_face,
        "part_thresholds": dict(zip(PARTS, part_thresholds, strict=True)),
        "face_thresholds": dict(zip(FACE_PARTS, face_thresholds, strict=True)),
        "official_test_part_iou": official_part_iou,
        "official_test_face_iou": official_face_iou,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "face_parameters": sum(parameter.numel() for parameter in face_model.parameters()),
        "seconds": round(time.perf_counter() - started, 1),
        "dataset_license": "ChildlikeSHAPES CC-BY-4.0",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
