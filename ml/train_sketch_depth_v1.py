#!/usr/bin/env python3
"""Train WallAlive's compact sketch-to-front/back-depth prior.

The browser already preserves the exact authored silhouette.  This model learns
the missing third dimension from synthetic articulated families instead of
assigning one symmetric distance-to-edge thickness to every drawing.  Training
examples are rendered analytically from unions of rotated ellipsoids, so every
input has exact front and rear surface supervision without downloaded artwork.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SIZE = 64
DEPTH_SCALE = 0.75
FAMILIES = ("biped", "quadruped", "winged", "aquatic", "radial", "branched", "machine", "chain")


@dataclass(frozen=True)
class Ellipsoid:
    x: float
    y: float
    rx: float
    ry: float
    rz: float
    angle: float = 0.0
    z: float = 0.0


def ellipsoid(x: float, y: float, rx: float, ry: float, rz: float, angle: float = 0.0, z: float = 0.0) -> Ellipsoid:
    return Ellipsoid(x, y, max(0.025, rx), max(0.025, ry), max(0.025, rz), angle, z)


def chain(parts: list[Ellipsoid], start: tuple[float, float], end: tuple[float, float], radius: float, depth: float, z0: float, z1: float, count: int | None = None) -> None:
    length = math.dist(start, end)
    samples = count or max(3, int(length / max(0.025, radius * 0.72)))
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    for index in range(samples + 1):
        amount = index / samples
        taper = 1.0 - amount * 0.24
        parts.append(ellipsoid(
            start[0] + (end[0] - start[0]) * amount,
            start[1] + (end[1] - start[1]) * amount,
            radius * 1.18 * taper,
            radius * 0.92 * taper,
            depth * taper,
            angle,
            z0 + (z1 - z0) * amount,
        ))


def make_parts(family: str, rng: random.Random) -> tuple[list[Ellipsoid], list[tuple[float, float, float, float]]]:
    """Return connected volume primitives and optional feature ellipses."""
    parts: list[Ellipsoid] = []
    features: list[tuple[float, float, float, float]] = []
    mirror = -1 if rng.random() < 0.5 else 1
    lean = rng.uniform(-0.10, 0.10)

    if family == "biped":
        body_rx, body_ry = rng.uniform(0.22, 0.33), rng.uniform(0.29, 0.42)
        parts.append(ellipsoid(lean, -0.04, body_rx, body_ry, rng.uniform(0.20, 0.32), rng.uniform(-0.12, 0.12)))
        head = (lean + rng.uniform(-0.04, 0.04), rng.uniform(0.30, 0.40))
        head_rx, head_ry = rng.uniform(0.20, 0.30), rng.uniform(0.18, 0.27)
        parts.append(ellipsoid(*head, head_rx, head_ry, rng.uniform(0.19, 0.29), rng.uniform(-0.08, 0.08), rng.uniform(-0.04, 0.07)))
        for side in (-1, 1):
            shoulder = (lean + side * body_rx * 0.76, rng.uniform(0.04, 0.15))
            elbow = (lean + side * rng.uniform(0.33, 0.48), rng.uniform(-0.05, 0.13))
            hand = (lean + side * rng.uniform(0.42, 0.60), rng.uniform(-0.26, 0.05))
            z_end = side * rng.uniform(-0.12, 0.16)
            chain(parts, shoulder, elbow, rng.uniform(0.055, 0.085), rng.uniform(0.075, 0.12), 0.0, z_end * 0.55)
            chain(parts, elbow, hand, rng.uniform(0.045, 0.073), rng.uniform(0.065, 0.11), z_end * 0.55, z_end)
            parts.append(ellipsoid(*hand, rng.uniform(0.06, 0.09), rng.uniform(0.055, 0.085), rng.uniform(0.07, 0.11), z=z_end))
            hip = (lean + side * body_rx * 0.48, -0.27)
            knee = (lean + side * rng.uniform(0.10, 0.21), rng.uniform(-0.45, -0.38))
            foot = (lean + side * rng.uniform(0.12, 0.25), rng.uniform(-0.68, -0.58))
            leg_z = -side * rng.uniform(-0.06, 0.11)
            chain(parts, hip, knee, rng.uniform(0.065, 0.095), rng.uniform(0.09, 0.14), 0.0, leg_z * 0.5)
            chain(parts, knee, foot, rng.uniform(0.055, 0.085), rng.uniform(0.08, 0.13), leg_z * 0.5, leg_z)
            parts.append(ellipsoid(foot[0] + side * 0.035, foot[1], rng.uniform(0.09, 0.14), rng.uniform(0.05, 0.075), rng.uniform(0.075, 0.12), z=leg_z))
        features.extend([(head[0] - head_rx * 0.34, head[1] + 0.025, 0.034, 0.048), (head[0] + head_rx * 0.34, head[1] + 0.025, 0.034, 0.048)])

    elif family == "quadruped":
        body = (rng.uniform(-0.05, 0.05), rng.uniform(-0.02, 0.09))
        body_rx, body_ry = rng.uniform(0.38, 0.52), rng.uniform(0.20, 0.29)
        parts.append(ellipsoid(*body, body_rx, body_ry, rng.uniform(0.24, 0.36), rng.uniform(-0.06, 0.06)))
        head = (mirror * rng.uniform(0.42, 0.56), rng.uniform(0.15, 0.27))
        parts.append(ellipsoid(*head, rng.uniform(0.18, 0.25), rng.uniform(0.18, 0.25), rng.uniform(0.19, 0.28), z=rng.uniform(-0.03, 0.10)))
        muzzle = (head[0] + mirror * 0.14, head[1] - 0.07)
        parts.append(ellipsoid(*muzzle, 0.13, 0.10, 0.14, z=0.08))
        features.append((head[0] + mirror * 0.035, head[1] + 0.045, 0.035, 0.045))
        for x in (-0.28, -0.12, 0.16, 0.31):
            hip = (x, body[1] - body_ry * 0.55)
            hoof = (x + rng.uniform(-0.07, 0.07), rng.uniform(-0.61, -0.50))
            leg_z = rng.uniform(-0.14, 0.14)
            chain(parts, hip, hoof, rng.uniform(0.05, 0.075), rng.uniform(0.075, 0.12), 0.0, leg_z)
            parts.append(ellipsoid(hoof[0] + mirror * 0.035, hoof[1], 0.075, 0.045, 0.07, z=leg_z))
        tail_start = (-mirror * body_rx * 0.8, body[1] + 0.02)
        tail_end = (-mirror * rng.uniform(0.60, 0.72), rng.uniform(0.20, 0.46))
        chain(parts, tail_start, tail_end, 0.045, 0.065, 0.0, rng.uniform(-0.16, 0.16))

    elif family == "winged":
        parts.append(ellipsoid(0, -0.02, rng.uniform(0.20, 0.28), rng.uniform(0.35, 0.46), rng.uniform(0.21, 0.31), rng.uniform(-0.10, 0.10)))
        head = (rng.uniform(-0.04, 0.04), rng.uniform(0.35, 0.45))
        parts.append(ellipsoid(*head, 0.19, 0.18, 0.20, z=0.05))
        for side in (-1, 1):
            wing_z = side * rng.uniform(-0.16, 0.18)
            parts.append(ellipsoid(side * rng.uniform(0.32, 0.42), rng.uniform(0.02, 0.14), rng.uniform(0.33, 0.46), rng.uniform(0.10, 0.17), rng.uniform(0.07, 0.13), side * rng.uniform(0.18, 0.42), wing_z))
            tip = (side * rng.uniform(0.57, 0.68), rng.uniform(-0.04, 0.20))
            chain(parts, (side * 0.18, 0.11), tip, 0.055, 0.08, 0.0, wing_z)
        beak_x = mirror * 0.24
        chain(parts, (mirror * 0.12, head[1]), (beak_x, head[1] - 0.025), 0.045, 0.055, 0.04, 0.10, 3)
        features.append((mirror * 0.055, head[1] + 0.025, 0.034, 0.045))

    elif family == "aquatic":
        body_angle = rng.uniform(-0.12, 0.12)
        parts.append(ellipsoid(0, 0.03, rng.uniform(0.42, 0.55), rng.uniform(0.22, 0.31), rng.uniform(0.22, 0.34), body_angle))
        tail_x = -mirror * rng.uniform(0.49, 0.58)
        parts.append(ellipsoid(tail_x, 0.17, 0.23, 0.095, 0.10, -mirror * 0.65, rng.uniform(-0.08, 0.08)))
        parts.append(ellipsoid(tail_x, -0.13, 0.23, 0.095, 0.10, mirror * 0.65, rng.uniform(-0.08, 0.08)))
        fin_z = rng.uniform(0.05, 0.18)
        parts.append(ellipsoid(mirror * 0.02, -0.10, 0.23, 0.085, 0.065, mirror * 0.42, fin_z))
        eye_x = mirror * rng.uniform(0.26, 0.36)
        features.append((eye_x, 0.10, 0.035, 0.045))

    elif family == "radial":
        center = (rng.uniform(-0.03, 0.03), rng.uniform(0.03, 0.13))
        parts.append(ellipsoid(*center, rng.uniform(0.24, 0.34), rng.uniform(0.23, 0.34), rng.uniform(0.24, 0.36)))
        count = rng.randint(5, 10)
        for index in range(count):
            angle = math.pi * (1.05 + 0.90 * index / max(1, count - 1)) + rng.uniform(-0.16, 0.16)
            length = rng.uniform(0.38, 0.64)
            start = (center[0] + math.cos(angle) * 0.16, center[1] + math.sin(angle) * 0.16)
            end = (center[0] + math.cos(angle) * length, center[1] + math.sin(angle) * length)
            chain(parts, start, end, rng.uniform(0.04, 0.075), rng.uniform(0.055, 0.105), 0.0, rng.uniform(-0.18, 0.18))
        features.extend([(center[0] - 0.09, center[1] + 0.04, 0.032, 0.045), (center[0] + 0.09, center[1] + 0.04, 0.032, 0.045)])

    elif family == "branched":
        trunk_bottom = (rng.uniform(-0.06, 0.06), -0.66)
        trunk_top = (rng.uniform(-0.08, 0.08), 0.15)
        chain(parts, trunk_bottom, trunk_top, rng.uniform(0.07, 0.11), rng.uniform(0.10, 0.17), 0.0, 0.0, 10)
        for index in range(rng.randint(4, 7)):
            side = -1 if index % 2 else 1
            start = (trunk_top[0] + rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.18))
            end = (side * rng.uniform(0.27, 0.58), rng.uniform(0.26, 0.57))
            z_end = rng.uniform(-0.14, 0.14)
            chain(parts, start, end, rng.uniform(0.045, 0.075), rng.uniform(0.065, 0.11), 0.0, z_end)
            parts.append(ellipsoid(*end, rng.uniform(0.15, 0.26), rng.uniform(0.13, 0.23), rng.uniform(0.13, 0.22), rng.uniform(-0.5, 0.5), z_end))
        parts.append(ellipsoid(0, 0.37, rng.uniform(0.42, 0.57), rng.uniform(0.28, 0.40), rng.uniform(0.25, 0.38), z=rng.uniform(-0.04, 0.05)))

    elif family == "machine":
        parts.append(ellipsoid(0, 0.02, rng.uniform(0.34, 0.48), rng.uniform(0.23, 0.34), rng.uniform(0.27, 0.42), rng.uniform(-0.04, 0.04)))
        for side in (-1, 1):
            wheel_z = side * rng.uniform(-0.09, 0.12)
            parts.append(ellipsoid(side * rng.uniform(0.25, 0.34), -0.29, rng.uniform(0.12, 0.17), rng.uniform(0.12, 0.17), rng.uniform(0.08, 0.13), z=wheel_z))
        if rng.random() < 0.65:
            pivot = (mirror * 0.24, 0.20)
            elbow = (mirror * rng.uniform(0.34, 0.48), rng.uniform(0.36, 0.48))
            tip = (mirror * rng.uniform(0.51, 0.66), rng.uniform(0.15, 0.35))
            chain(parts, pivot, elbow, 0.06, 0.09, 0.0, 0.12)
            chain(parts, elbow, tip, 0.05, 0.08, 0.12, -0.08)

    else:  # chain
        count = rng.randint(5, 10)
        phase = rng.uniform(-math.pi, math.pi)
        previous: tuple[float, float] | None = None
        for index in range(count):
            amount = index / max(1, count - 1)
            x = -0.58 + amount * 1.16
            y = math.sin(amount * math.pi * rng.uniform(1.0, 2.2) + phase) * rng.uniform(0.16, 0.31)
            current = (x, y)
            z = math.sin(amount * math.pi * 1.7 + phase) * rng.uniform(0.08, 0.20)
            if previous is not None:
                chain(parts, previous, current, 0.055, 0.085, z * 0.7, z, 3)
            parts.append(ellipsoid(x, y, rng.uniform(0.10, 0.16), rng.uniform(0.10, 0.16), rng.uniform(0.10, 0.17), z=z))
            previous = current

    return parts, features


def render_sample(family: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    parts, features = make_parts(family, rng)
    axis = np.linspace(-1.0, 1.0, SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis[::-1])
    front = np.zeros((SIZE, SIZE), dtype=np.float32)
    back = np.zeros_like(front)
    owner = np.full((SIZE, SIZE), -1, dtype=np.int16)
    owner_front = np.full((SIZE, SIZE), -10.0, dtype=np.float32)
    for part_index, part in enumerate(parts):
        cosine, sine = math.cos(part.angle), math.sin(part.angle)
        local_x = (xx - part.x) * cosine + (yy - part.y) * sine
        local_y = -(xx - part.x) * sine + (yy - part.y) * cosine
        radius = 1.0 - (local_x / part.rx) ** 2 - (local_y / part.ry) ** 2
        valid = radius > 0
        half_depth = part.rz * np.sqrt(np.maximum(0.0, radius))
        candidate_front = part.z + half_depth
        candidate_back = -part.z + half_depth
        front = np.where(valid, np.maximum(front, candidate_front), front)
        back = np.where(valid, np.maximum(back, candidate_back), back)
        update_owner = valid & (candidate_front > owner_front)
        owner[update_owner] = part_index
        owner_front[update_owner] = candidate_front[update_owner]

    mask = (front > 0.004) | (back > 0.004)
    mask = ndimage.binary_closing(mask, iterations=1)
    distance = ndimage.distance_transform_edt(mask).astype(np.float32)
    distance /= max(1.0, float(distance.max()))
    eroded = ndimage.binary_erosion(mask)
    contour = mask & ~eroded
    owner_edge = mask & (
        (owner != np.roll(owner, 1, 0)) | (owner != np.roll(owner, -1, 0)) |
        (owner != np.roll(owner, 1, 1)) | (owner != np.roll(owner, -1, 1))
    )
    ink = contour | (owner_edge & (distance > 0.08))
    for fx, fy, frx, fry in features:
        ring = np.abs(((xx - fx) / frx) ** 2 + ((yy - fy) / fry) ** 2 - 1.0) < 0.38
        ink |= ring & mask
    if rng.random() < 0.75:
        ink = ndimage.binary_dilation(ink, iterations=rng.randint(1, 2))
    if rng.random() < 0.20:
        dropout = np.random.default_rng(seed ^ 0xC0FFEE).random((SIZE, SIZE)) > rng.uniform(0.02, 0.10)
        ink &= dropout

    inputs = np.stack((mask.astype(np.float32), distance, ink.astype(np.float32)))
    targets = np.stack((np.clip(front / DEPTH_SCALE, 0, 1), np.clip(back / DEPTH_SCALE, 0, 1)))
    targets *= mask[None]
    return inputs, targets.astype(np.float32)


def make_dataset(count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    inputs = np.empty((count, 3, SIZE, SIZE), dtype=np.uint8)
    targets = np.empty((count, 2, SIZE, SIZE), dtype=np.float16)
    labels = np.empty(count, dtype=np.uint8)
    for index in range(count):
        family_index = index % len(FAMILIES)
        sample, target = render_sample(FAMILIES[family_index], seed + index * 1009)
        inputs[index] = np.rint(sample * 255).astype(np.uint8)
        targets[index] = target.astype(np.float16)
        labels[index] = family_index
    return torch.from_numpy(inputs), torch.from_numpy(targets), labels


class ConvBlock(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        groups = 4 if outgoing % 4 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(groups, outgoing),
            nn.SiLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(groups, outgoing),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class SketchDepthNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 28)
        self.enc3 = ConvBlock(28, 44)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(44, 28, 2, stride=2)
        self.dec2 = ConvBlock(56, 28)
        self.up1 = nn.ConvTranspose2d(28, 16, 2, stride=2)
        self.dec1 = ConvBlock(32, 20)
        self.output = nn.Conv2d(20, 2, 1)

    def forward(self, sketch_values: torch.Tensor) -> torch.Tensor:
        first = self.enc1(sketch_values)
        second = self.enc2(self.pool(first))
        latent = self.enc3(self.pool(second))
        decoded = self.dec2(torch.cat((self.up2(latent), second), dim=1))
        decoded = self.dec1(torch.cat((self.up1(decoded), first), dim=1))
        return torch.sigmoid(self.output(decoded)) * sketch_values[:, :1]


def depth_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    normalizer = mask.sum().clamp_min(1.0) * 2
    surface = ((prediction - target).abs() * mask).sum() / normalizer
    asymmetry = (((prediction[:, :1] - prediction[:, 1:]) - (target[:, :1] - target[:, 1:])).abs() * mask).sum() / (mask.sum().clamp_min(1.0))
    grad_x = ((prediction[..., 1:] - prediction[..., :-1]) - (target[..., 1:] - target[..., :-1])).abs()
    grad_y = ((prediction[..., 1:, :] - prediction[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])).abs()
    gradient = (grad_x * mask[..., 1:]).sum() / (mask[..., 1:].sum().clamp_min(1.0) * 2)
    gradient += (grad_y * mask[..., 1:, :]).sum() / (mask[..., 1:, :].sum().clamp_min(1.0) * 2)
    return surface + 0.24 * asymmetry + 0.16 * gradient


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    absolute = 0.0
    asymmetry = 0.0
    pixels = 0.0
    prediction_values: list[np.ndarray] = []
    target_values: list[np.ndarray] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, dtype=torch.float32) / 255.0
        targets = targets.to(device, dtype=torch.float32)
        prediction = model(inputs)
        mask = inputs[:, :1]
        count = float(mask.sum().item())
        absolute += float(((prediction - targets).abs() * mask).sum().item())
        asymmetry += float((((prediction[:, :1] - prediction[:, 1:]) - (targets[:, :1] - targets[:, 1:])).abs() * mask).sum().item())
        pixels += count
        selected = mask.expand_as(prediction) > 0.5
        prediction_values.append(prediction[selected].detach().cpu().numpy())
        target_values.append(targets[selected].detach().cpu().numpy())
    predicted = np.concatenate(prediction_values)
    expected = np.concatenate(target_values)
    correlation = float(np.corrcoef(predicted, expected)[0, 1])
    return {
        "surface_mae_normalized": round(absolute / max(1.0, pixels * 2), 6),
        "surface_mae_scene_units": round(absolute / max(1.0, pixels * 2) * DEPTH_SCALE, 6),
        "front_back_asymmetry_mae": round(asymmetry / max(1.0, pixels), 6),
        "surface_correlation": round(correlation, 6),
    }


def loaders(train_count: int, validation_count: int, test_count: int, batch_size: int):
    train_x, train_y, train_labels = make_dataset(train_count, 1_000_003)
    validation_x, validation_y, validation_labels = make_dataset(validation_count, 2_000_003)
    test_x, test_y, test_labels = make_dataset(test_count, 3_000_017)
    return (
        DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(TensorDataset(validation_x, validation_y), batch_size=batch_size, shuffle=False, num_workers=0),
        DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size, shuffle=False, num_workers=0),
        train_labels,
        validation_labels,
        test_labels,
    )


def export_onnx(model: nn.Module, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        (torch.zeros(1, 3, SIZE, SIZE),),
        output,
        input_names=["sketch_values"],
        output_names=["front_back_depth"],
        dynamic_axes={"sketch_values": {0: "batch"}, "front_back_depth": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


def verify_onnx(model: nn.Module, model_path: Path, test_loader: DataLoader) -> float:
    import onnxruntime as ort

    inputs, _ = next(iter(test_loader))
    values = inputs[:8].numpy().astype(np.float32) / 255.0
    with torch.inference_mode():
        expected = model.cpu()(torch.from_numpy(values)).numpy()
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"sketch_values": values})[0]
    return float(np.max(np.abs(expected - actual)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("public/models/wallalive-sketch-depth-v1.onnx"))
    parser.add_argument("--report", type=Path, default=Path("public/models/wallalive-sketch-depth-v1.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--train-count", type=int, default=6144)
    parser.add_argument("--validation-count", type=int, default=768)
    parser.add_argument("--test-count", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, validation_loader, test_loader, train_labels, validation_labels, test_labels = loaders(
        args.train_count, args.validation_count, args.test_count, args.batch_size,
    )
    model = SketchDepthNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device, dtype=torch.float32) / 255.0
            targets = targets.to(device, dtype=torch.float32)
            prediction = model(inputs)
            loss = depth_loss(prediction, targets, inputs[:, :1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            batches += 1
        scheduler.step()
        validation = evaluate(model, validation_loader, device)
        entry = {"epoch": epoch, "train_loss": round(total / max(1, batches), 6), **validation}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["surface_mae_normalized"] < best_validation:
            best_validation = validation["surface_mae_normalized"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    sealed_test = evaluate(model, test_loader, device)
    export_onnx(model, args.output)
    maximum_onnx_error = verify_onnx(model, args.output, test_loader)
    report = {
        "model": "wallalive-sketch-depth-v1",
        "architecture": "compact U-Net front/back depth prior",
        "purpose": "predict distinct visible and hidden surfaces from a segmented drawing",
        "input": [1, 3, SIZE, SIZE],
        "input_channels": ["foreground mask", "interior distance", "authored/internal contour ink"],
        "output": [1, 2, SIZE, SIZE],
        "output_channels": ["front depth", "back depth"],
        "depth_scale_scene_units": DEPTH_SCALE,
        "families": list(FAMILIES),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_examples": args.train_count,
        "validation_examples": args.validation_count,
        "sealed_test_examples": args.test_count,
        "family_counts": {
            split: {family: int(np.sum(labels == index)) for index, family in enumerate(FAMILIES)}
            for split, labels in (("train", train_labels), ("validation", validation_labels), ("test", test_labels))
        },
        "selection": "lowest validation surface MAE; sealed test opened once after selection",
        "test_split_used_for_selection": False,
        "best_epoch": min(history, key=lambda entry: float(entry["surface_mae_normalized"]))["epoch"],
        "validation": min(history, key=lambda entry: float(entry["surface_mae_normalized"])),
        "sealed_test": sealed_test,
        "onnx_max_absolute_error": maximum_onnx_error,
        "onnx_export_verified": maximum_onnx_error < 1e-4,
        "training_data": "procedural analytic unions of articulated ellipsoids; no child drawings or user pixels",
        "limitations": "single-view hidden depth is a learned plausible prior, not measurable ground truth for an unseen back",
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "sealed_test": sealed_test, "onnx_max_absolute_error": maximum_onnx_error}, indent=2))


if __name__ == "__main__":
    main()
