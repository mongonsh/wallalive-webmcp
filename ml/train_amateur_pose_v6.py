#!/usr/bin/env python3
"""Train a compact browser pose network on real Amateur Drawings.

This adds a named 17-joint skeleton to WallAlive's existing silhouette and
part segmenters. The deterministic test split is loaded only after checkpoint
and operating-point selection on validation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from evaluate_amateur_benchmark import isolated_browser_input, local_path, polygon_mask


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
LEFT_RIGHT_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))


def gaussian_heatmaps(points: dict[str, tuple[float, float, int]], input_size: int, output_size: int, sigma: float) -> torch.Tensor:
    y_grid, x_grid = torch.meshgrid(torch.arange(output_size), torch.arange(output_size), indexing="ij")
    heatmaps = torch.zeros((len(KEYPOINT_NAMES), output_size, output_size), dtype=torch.float32)
    scale = output_size / input_size
    for index, name in enumerate(KEYPOINT_NAMES):
        x, y, visibility = points[name]
        if visibility <= 0:
            continue
        x *= scale
        y *= scale
        heatmaps[index] = torch.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma**2))
    return heatmaps


class AmateurPoseDataset(Dataset):
    def __init__(
        self,
        manifest: dict[str, Any],
        images_root: Path,
        split: str,
        input_size: int,
        output_size: int,
        augment: bool,
    ):
        self.records = [record for record in manifest["records"] if record["split"] == split]
        self.images_root = images_root
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.cached: list[tuple[torch.Tensor, torch.Tensor]] = []
        for record in self.records:
            image = Image.open(local_path(self.images_root, record["file_name"])).convert("RGB")
            mask = polygon_mask(record, image.size)
            values, _, points, _ = isolated_browser_input(image, mask, record["keypoints"], self.input_size)
            image_tensor = torch.from_numpy(np.clip(values * 255, 0, 255).astype(np.uint8))
            heatmaps = gaussian_heatmaps(points, self.input_size, self.output_size, sigma=1.35).to(torch.float16)
            self.cached.append((image_tensor, heatmaps))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        image_tensor, heatmaps = self.cached[index]
        image_tensor = image_tensor.float() / 255
        heatmaps = heatmaps.float()
        if self.augment:
            rng = random.Random(random.randrange(1 << 30) ^ (index * 104729))
            angle = rng.uniform(-13, 13)
            translate = [round(rng.uniform(-self.input_size * 0.045, self.input_size * 0.045)) for _ in range(2)]
            scale = rng.uniform(0.88, 1.10)
            shear = [rng.uniform(-5, 5), rng.uniform(-2, 2)]
            image_tensor = tvf.affine(
                image_tensor,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=1.0,
            )
            heatmaps = tvf.affine(
                heatmaps,
                angle=angle,
                translate=[round(value * self.output_size / self.input_size) for value in translate],
                scale=scale,
                shear=shear,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            if rng.random() < 0.5:
                image_tensor = image_tensor.flip(-1)
                heatmaps = heatmaps.flip(-1)
                for left, right in LEFT_RIGHT_PAIRS:
                    heatmaps[[left, right]] = heatmaps[[right, left]]
            if rng.random() < 0.45:
                image_tensor = tvf.gaussian_blur(image_tensor, kernel_size=3, sigma=[rng.uniform(0.15, 0.8)] * 2)
            contrast = rng.uniform(0.78, 1.24)
            brightness = rng.uniform(0.83, 1.15)
            image_tensor = tvf.adjust_contrast(image_tensor, contrast)
            image_tensor = tvf.adjust_brightness(image_tensor, brightness)
            if rng.random() < 0.6:
                image_tensor = image_tensor + torch.randn_like(image_tensor) * rng.uniform(0.002, 0.018)
            image_tensor = image_tensor.clamp(0, 1)
        return image_tensor, heatmaps


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class AmateurPoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(),
        )
        self.encoder2 = ConvBlock(16, 24, stride=2)
        self.encoder3 = ConvBlock(24, 40, stride=2)
        self.bridge = ConvBlock(40, 56)
        self.decoder3 = ConvBlock(56 + 24, 40)
        self.decoder2 = ConvBlock(40 + 16, 28)
        self.refine = ConvBlock(28, 28)
        self.output = nn.Conv2d(28, len(KEYPOINT_NAMES), 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        one = self.stem(value)  # half resolution
        two = self.encoder2(one)  # quarter resolution
        three = self.encoder3(two)  # eighth resolution
        bridge = self.bridge(three)
        decoded = self.decoder3(torch.cat((nn.functional.interpolate(bridge, size=two.shape[-2:], mode="bilinear", align_corners=False), two), 1))
        decoded = self.decoder2(torch.cat((nn.functional.interpolate(decoded, size=one.shape[-2:], mode="bilinear", align_corners=False), one), 1))
        return self.output(self.refine(decoded))


def heatmap_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    weighted_mse = ((probabilities - targets) ** 2 * (1 + targets * 22)).mean()
    batch, channels, height, width = logits.shape
    spatial = torch.softmax(logits.reshape(batch, channels, -1) * 2.2, dim=-1)
    x_grid = torch.arange(width, device=logits.device, dtype=logits.dtype).repeat(height)
    y_grid = torch.arange(height, device=logits.device, dtype=logits.dtype).repeat_interleave(width)
    predicted_x = (spatial * x_grid).sum(-1) / width
    predicted_y = (spatial * y_grid).sum(-1) / height
    target_flat = targets.reshape(batch, channels, -1)
    target_index = target_flat.argmax(-1)
    target_x = (target_index % width) / width
    target_y = torch.div(target_index, width, rounding_mode="floor") / height
    coordinate_loss = nn.functional.smooth_l1_loss(predicted_x, target_x) + nn.functional.smooth_l1_loss(predicted_y, target_y)
    return weighted_mse * 8 + coordinate_loss * 1.8


def coordinates_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    batch, channels, _, width = heatmaps.shape
    indices = heatmaps.reshape(batch, channels, -1).argmax(-1)
    return torch.stack((indices % width, torch.div(indices, width, rounding_mode="floor")), dim=-1).float()


def pose_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    distances: list[torch.Tensor] = []
    per_joint: list[list[float]] = [[] for _ in KEYPOINT_NAMES]
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            predicted = coordinates_from_heatmaps(model(images))
            expected = coordinates_from_heatmaps(targets)
            error = torch.linalg.vector_norm(predicted - expected, dim=-1)
            distances.append(error.cpu())
            for index in range(len(KEYPOINT_NAMES)):
                per_joint[index].extend(error[:, index].cpu().tolist())
    all_distances = torch.cat(distances).numpy()
    output_size = next(iter(loader))[1].shape[-1]
    return {
        "pck_0_05": round(float((all_distances <= output_size * 0.05).mean()), 4),
        "pck_0_10": round(float((all_distances <= output_size * 0.10).mean()), 4),
        "mean_error_input_pixels": round(float(all_distances.mean() * 2), 3),
        "per_joint_pck_0_05": {
            name: round(float((np.asarray(per_joint[index]) <= output_size * 0.05).mean()), 4)
            for index, name in enumerate(KEYPOINT_NAMES)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--output-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--samples-per-epoch", type=int, default=1408)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    if args.output_size != args.input_size // 2:
        raise ValueError("AmateurPoseNet outputs half-resolution heatmaps")
    torch.manual_seed(20260904)
    random.seed(20260904)
    np.random.seed(20260904)
    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    training = AmateurPoseDataset(manifest, args.images_root, "train", args.input_size, args.output_size, augment=True)
    validation = AmateurPoseDataset(manifest, args.images_root, "validation", args.input_size, args.output_size, augment=False)
    sampler = RandomSampler(training, replacement=True, num_samples=args.samples_per_epoch, generator=torch.Generator().manual_seed(20260904))
    train_loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AmateurPoseNet().to(device)
    print(json.dumps({
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_drawings": len(training),
        "validation_drawings": len(validation),
        "samples_per_epoch": args.samples_per_epoch,
    }), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1.2e-4)
    best_score = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] = {}

    for epoch in range(args.epochs):
        model.train()
        loss_total = 0.0
        samples = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            loss = heatmap_loss(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(images)
            samples += len(images)
        scheduler.step()
        validation_metrics = pose_metrics(model, validation_loader, device)
        score = validation_metrics["pck_0_05"] - validation_metrics["mean_error_input_pixels"] * 0.002
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_validation = validation_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({
            "epoch": epoch + 1,
            "loss": round(loss_total / max(1, samples), 5),
            "validation": validation_metrics,
        }), flush=True)

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model = model.to("cpu").eval()
    # Instantiate the sealed test data only after validation has selected the
    # checkpoint. This keeps even test preprocessing out of the training loop.
    test = AmateurPoseDataset(manifest, args.images_root, "test", args.input_size, args.output_size, augment=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    official_test = pose_metrics(model, test_loader, torch.device("cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, args.input_size, args.input_size),
        args.output,
        input_names=["pose_values"],
        output_names=["joint_heatmaps"],
        dynamic_axes={"pose_values": {0: "batch"}, "joint_heatmaps": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report = {
        "architecture": "WallAlive AmateurPoseNet v6",
        "input": [1, 3, args.input_size, args.input_size],
        "output": [1, len(KEYPOINT_NAMES), args.output_size, args.output_size],
        "keypoints": KEYPOINT_NAMES,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_drawings": len(training),
        "validation_drawings": len(validation),
        "test_drawings": len(test),
        "samples_per_epoch": args.samples_per_epoch,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "validation": best_validation,
        "official_test": official_test,
        "test_split_used_for_selection": False,
        "dataset": manifest["dataset"],
        "dataset_license": manifest["license"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
