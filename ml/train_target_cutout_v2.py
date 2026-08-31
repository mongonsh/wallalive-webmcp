#!/usr/bin/env python3
"""Train WallAlive's point-prompted paper-scene character segmenter v2.

The browser supplies RGB plus a Gaussian heatmap centered on the child's tap.
Training uses Meta's accepted Amateur Drawings masks and adds deterministic
paper grids, handwriting-like distractors, shadows, blur, perspective and JPEG
damage.  Test records are sealed until validation chooses the checkpoint and
mask threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, RandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


def split_for(name: str) -> str:
    bucket = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def image_path(root: Path, name: str) -> Path:
    direct = root / name
    return direct if direct.exists() else root / name.removeprefix("amateur_drawings/")


def polygon_mask(annotation: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in annotation.get("segmentation", []):
        if len(polygon) >= 6:
            draw.polygon(list(zip(polygon[::2], polygon[1::2], strict=True)), fill=255)
    return mask


def letterbox(image: Image.Image, size: int, interpolation: Image.Resampling, fill: int | tuple[int, int, int]):
    scale = min(size / image.width, size / image.height)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new(image.mode, (size, size), fill)
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas.paste(image.resize(resized, interpolation), offset)
    return canvas


def compose_paper_scene(image: Image.Image, mask: Image.Image, seed: int) -> tuple[Image.Image, Image.Image]:
    """Place one annotated character on a photographed-paper hard negative.

    The paper is deliberately a stronger closed contour than the drawing. Its
    pixels remain negative, so the network must follow the prompted character
    rather than returning the sheet, label, grid, or wall.
    """
    rng = random.Random(seed ^ 0x5A17)
    size = image.width
    wall = Image.new("RGB", image.size, tuple(rng.randint(174, 235) for _ in range(3)))
    wall_draw = ImageDraw.Draw(wall)
    # A torn, perspective-skewed paper silhouette with a dark drop shadow.
    inset_x = rng.randint(2, 14)
    inset_y = rng.randint(3, 17)
    paper = [(inset_x + rng.randint(0, 7), inset_y + rng.randint(0, 6)),
             (size - inset_x - rng.randint(0, 9), inset_y + rng.randint(0, 9)),
             (size - inset_x - rng.randint(0, 8), size - inset_y - rng.randint(0, 10)),
             (inset_x + rng.randint(0, 9), size - inset_y - rng.randint(0, 8))]
    shadow = [(x + rng.randint(2, 5), y + rng.randint(3, 7)) for x, y in paper]
    wall_draw.polygon(shadow, fill=tuple(rng.randint(105, 155) for _ in range(3)))
    paper_color = tuple(rng.randint(231, 255) for _ in range(3))
    wall_draw.polygon(paper, fill=paper_color)
    paper_mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(paper_mask).polygon(paper, fill=255)
    if rng.random() < 0.82:
        grid = wall.copy()
        grid_draw = ImageDraw.Draw(grid)
        spacing = rng.randint(8, 16)
        grid_color = tuple(max(0, channel - rng.randint(7, 22)) for channel in paper_color)
        for coordinate in range(rng.randrange(spacing), size, spacing):
            grid_draw.line((coordinate, 0, coordinate, size), fill=grid_color, width=1)
            grid_draw.line((0, coordinate, size, coordinate), fill=grid_color, width=1)
        wall.paste(grid, mask=paper_mask)
    values = np.asarray(mask) >= 128
    ys, xs = np.nonzero(values)
    padding = 3
    box = (max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding),
           min(size, int(xs.max()) + padding + 1), min(size, int(ys.max()) + padding + 1))
    character = image.crop(box)
    character_mask = mask.crop(box)
    target_extent = rng.randint(round(size * 0.38), round(size * 0.68))
    scale = target_extent / max(character.size)
    resized = (max(1, round(character.width * scale)), max(1, round(character.height * scale)))
    character = character.resize(resized, Image.Resampling.BILINEAR)
    character_mask = character_mask.resize(resized, Image.Resampling.NEAREST)
    angle = rng.uniform(-11, 11)
    character = character.rotate(angle, Image.Resampling.BILINEAR, expand=True, fillcolor=paper_color)
    character_mask = character_mask.rotate(angle, Image.Resampling.NEAREST, expand=True, fillcolor=0)
    x = rng.randint(max(inset_x + 4, 0), max(inset_x + 4, size - inset_x - character.width - 4))
    y = rng.randint(max(inset_y + 4, 0), max(inset_y + 4, size - inset_y - character.height - 4))
    wall.paste(character, (x, y), character_mask)
    target = Image.new("L", image.size, 0)
    target.paste(character_mask, (x, y))
    # Neighboring handwriting and a label are negatives even when they share
    # the same ink color as the prompted character.
    draw = ImageDraw.Draw(wall)
    ink = tuple(rng.randint(65, 175) for _ in range(3))
    for _ in range(rng.randint(2, 7)):
        start_x = rng.randint(inset_x, max(inset_x, size - inset_x - 12))
        start_y = rng.randint(inset_y, max(inset_y, size - inset_y - 6))
        draw.line([(start_x, start_y), (start_x + rng.randint(5, 18), start_y + rng.randint(-5, 5)),
                   (start_x + rng.randint(12, 25), start_y + rng.randint(-4, 6))], fill=ink, width=rng.randint(1, 2))
    return wall, target


def add_scene_damage(image: Image.Image, mask: Image.Image, seed: int) -> Image.Image:
    rng = random.Random(seed)
    paper = tuple(rng.randint(226, 255) for _ in range(3))
    base = image.copy()
    draw = ImageDraw.Draw(base)
    width, height = base.size
    ink = tuple(rng.randint(45, 170) for _ in range(3))
    if rng.random() < 0.75:
        spacing = rng.randint(8, 18)
        grid = tuple(min(255, channel + rng.randint(2, 14)) for channel in paper)
        for x in range(rng.randrange(spacing), width, spacing):
            draw.line((x, 0, x, height), fill=grid, width=1)
        for y in range(rng.randrange(spacing), height, spacing):
            draw.line((0, y, width, y), fill=grid, width=1)
    mask_values = np.asarray(mask) > 0
    for _ in range(rng.randint(3, 16)):
        points = []
        start_x = rng.randrange(width)
        start_y = rng.randrange(height)
        for step in range(rng.randint(2, 6)):
            points.append((start_x + step * rng.randint(2, 8), start_y + rng.randint(-8, 8)))
        # Do not teach the network to erase strokes that cross the character.
        if any(0 <= x < width and 0 <= y < height and mask_values[y, x] for x, y in points):
            continue
        draw.line(points, fill=ink, width=rng.randint(1, 3), joint="curve")
    yy, xx = np.mgrid[:height, :width]
    shade = ((xx / max(1, width - 1) - 0.5) * rng.uniform(-38, 38)
             + (yy / max(1, height - 1) - 0.5) * rng.uniform(-30, 30))[..., None]
    array = np.clip(np.asarray(base, dtype=np.float32) + shade, 0, 255).astype(np.uint8)
    result = Image.fromarray(array)
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(0.76, 1.28))
    result = ImageEnhance.Color(result).enhance(rng.uniform(0.72, 1.38))
    if rng.random() < 0.55:
        result = result.filter(ImageFilter.GaussianBlur(rng.uniform(0.12, 1.0)))
    if rng.random() < 0.42:
        buffer = io.BytesIO()
        result.save(buffer, format="JPEG", quality=rng.randint(42, 84))
        result = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return result


def prompted_tensor(image: Image.Image, mask: Image.Image, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = random.Random(seed)
    mask_values = np.asarray(mask) >= 128
    ys, xs = np.nonzero(mask_values)
    if not len(xs):
        raise ValueError("empty character mask")
    pick = rng.randrange(len(xs))
    point_x, point_y = float(xs[pick]), float(ys[pick])
    y_grid, x_grid = np.mgrid[:mask.height, :mask.width]
    sigma = rng.uniform(mask.width * 0.035, mask.width * 0.075)
    prompt = np.exp(-((x_grid - point_x) ** 2 + (y_grid - point_y) ** 2) / (2 * sigma**2)).astype(np.float32)
    rgb = np.moveaxis(np.asarray(image, dtype=np.float32) / 255.0, -1, 0)
    values = np.concatenate((rgb, prompt[None]), axis=0)
    return torch.from_numpy(values), torch.from_numpy(mask_values.astype(np.float32)[None])


class AmateurTargetDataset(Dataset):
    def __init__(self, payload: dict[str, Any], root: Path, split: str, size: int, augment: bool, paper_scene: bool = False):
        images = {int(item["id"]): item for item in payload["images"]}
        annotations = {int(item["image_id"]): item for item in payload["annotations"] if item.get("segmentation")}
        self.records = [(images[image_id], annotation) for image_id, annotation in annotations.items()
                        if image_id in images and split_for(images[image_id]["file_name"]) == split]
        self.root = root
        self.size = size
        self.augment = augment
        self.paper_scene = paper_scene
        self.cached: list[tuple[Image.Image, Image.Image]] = []
        for metadata, annotation in self.records:
            try:
                image = Image.open(image_path(root, metadata["file_name"])).convert("RGB")
                image.load()
            except (OSError, SyntaxError):
                continue
            mask = polygon_mask(annotation, image.size)
            if not np.asarray(mask).any():
                continue
            prepared_image = letterbox(image, size, Image.Resampling.BILINEAR, (255, 255, 255))
            prepared_mask = letterbox(mask, size, Image.Resampling.NEAREST, 0)
            if not np.asarray(prepared_mask).any():
                continue
            self.cached.append((prepared_image, prepared_mask))

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, index: int):
        source, source_mask = self.cached[index]
        seed = (index + 1) * 104729 + (random.randrange(1 << 30) if self.augment else 20260831)
        rng = random.Random(seed)
        image, mask = source.copy(), source_mask.copy()
        if self.augment:
            angle = rng.uniform(-17, 17)
            translate = [round(rng.uniform(-self.size * 0.08, self.size * 0.08)) for _ in range(2)]
            scale = rng.uniform(0.76, 1.14)
            shear = [rng.uniform(-8, 8), rng.uniform(-4, 4)]
            image = tvf.affine(image, angle, translate, scale, shear, InterpolationMode.BILINEAR, fill=tuple(rng.randint(225, 255) for _ in range(3)))
            mask = tvf.affine(mask, angle, translate, scale, shear, InterpolationMode.NEAREST, fill=0)
            if rng.random() < 0.42:
                distortion = rng.uniform(0.025, 0.09) * self.size
                starts = [[0, 0], [self.size - 1, 0], [self.size - 1, self.size - 1], [0, self.size - 1]]
                ends = [[rng.uniform(0, distortion), rng.uniform(0, distortion)],
                        [self.size - 1 - rng.uniform(0, distortion), rng.uniform(0, distortion)],
                        [self.size - 1 - rng.uniform(0, distortion), self.size - 1 - rng.uniform(0, distortion)],
                        [rng.uniform(0, distortion), self.size - 1 - rng.uniform(0, distortion)]]
                image = tvf.perspective(image, starts, ends, InterpolationMode.BILINEAR, fill=255)
                mask = tvf.perspective(mask, starts, ends, InterpolationMode.NEAREST, fill=0)
        if not np.asarray(mask).any():
            image, mask = source.copy(), source_mask.copy()
        if self.paper_scene or (self.augment and rng.random() < 0.72):
            image, mask = compose_paper_scene(image, mask, seed)
        image = add_scene_damage(image, mask, seed)
        return prompted_tensor(image, mask, seed)


class Block(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU(),
                                 nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU())

    def forward(self, value: torch.Tensor):
        return self.net(value)


class TargetCutoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = Block(4, 18)
        self.two = Block(18, 32)
        self.three = Block(32, 52)
        self.bridge = Block(52, 76)
        self.dec3 = Block(76 + 52, 52)
        self.dec2 = Block(52 + 32, 34)
        self.dec1 = Block(34 + 18, 24)
        self.output = nn.Conv2d(24, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, value: torch.Tensor):
        one = self.one(value)
        two = self.two(self.pool(one))
        three = self.three(self.pool(two))
        bridge = self.bridge(self.pool(three))
        decoded = self.dec3(torch.cat((nn.functional.interpolate(bridge, size=three.shape[-2:], mode="bilinear", align_corners=False), three), 1))
        decoded = self.dec2(torch.cat((nn.functional.interpolate(decoded, size=two.shape[-2:], mode="bilinear", align_corners=False), two), 1))
        decoded = self.dec1(torch.cat((nn.functional.interpolate(decoded, size=one.shape[-2:], mode="bilinear", align_corners=False), one), 1))
        return self.output(decoded)


def loss_for(logits: torch.Tensor, target: torch.Tensor):
    probability = logits.sigmoid()
    intersection = (probability * target).sum((0, 2, 3))
    dice = 1 - ((2 * intersection + 1) / ((probability + target).sum((0, 2, 3)) + 1)).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(2.2, device=logits.device))
    return bce * 0.52 + dice * 0.48


def metrics(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float):
    intersection = union = prompt_hits = total = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).bool()
            predicted = model(inputs).sigmoid() >= threshold
            intersection += int((predicted & targets).sum())
            union += int((predicted | targets).sum())
            prompt_hits += int((predicted[:, 0] & (inputs[:, 3] >= 0.98)).flatten(1).any(1).sum())
            total += len(inputs)
    return {"iou": round(intersection / max(1, union), 4), "prompt_hit_rate": round(prompt_hits / max(1, total), 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--samples-per-epoch", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model-name", default="wallalive-target-cutout-v2")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.manual_seed(20260831)
    random.seed(20260831)
    np.random.seed(20260831)
    payload = json.loads(args.annotations.read_text())
    train = AmateurTargetDataset(payload, args.images_root, "train", args.size, True)
    validation_clean = AmateurTargetDataset(payload, args.images_root, "validation", args.size, False)
    validation_scene = AmateurTargetDataset(payload, args.images_root, "validation", args.size, False, True)
    validation = ConcatDataset([validation_clean, validation_scene])
    sampler = RandomSampler(train, replacement=True, num_samples=args.samples_per_epoch, generator=torch.Generator().manual_seed(20260831))
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TargetCutoutNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_validation = {}
    print(json.dumps({"device": str(device), "train": len(train), "validation": len(validation), "parameters": sum(p.numel() for p in model.parameters())}), flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = count = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            loss = loss_for(model(inputs), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(inputs)
            count += len(inputs)
        scheduler.step()
        candidates = [(threshold, metrics(model, validation_loader, device, threshold)) for threshold in (0.36, 0.42, 0.48, 0.54, 0.60)]
        threshold, result = max(candidates, key=lambda item: item[1]["iou"] + item[1]["prompt_hit_rate"] * 0.08)
        score = result["iou"] + result["prompt_hit_rate"] * 0.08
        if score > best_score:
            best_score, best_epoch, best_threshold, best_validation = score, epoch + 1, threshold, result
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "loss": round(total_loss / max(1, count), 5), "threshold": threshold, "validation": result}), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    sealed_test_clean = AmateurTargetDataset(payload, args.images_root, "test", args.size, False)
    sealed_test_scene = AmateurTargetDataset(payload, args.images_root, "test", args.size, False, True)
    sealed_test = ConcatDataset([sealed_test_clean, sealed_test_scene])
    test_loader = DataLoader(sealed_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    official_test = metrics(model, test_loader, torch.device("cpu"), best_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 4, args.size, args.size), args.output,
                      input_names=["prompted_image"], output_names=["target_mask"],
                      dynamic_axes={"prompted_image": {0: "batch"}, "target_mask": {0: "batch"}}, opset_version=17, dynamo=False)
    report = {"model": args.model_name, "architecture": "point-prompted compact U-Net", "input": [1, 4, args.size, args.size],
              "output": [1, 1, args.size, args.size], "parameters": sum(p.numel() for p in model.parameters()),
              "training_drawings": len(train), "validation_drawings": len(validation), "sealed_test_drawings": len(sealed_test),
              "samples_per_epoch": args.samples_per_epoch, "epochs": args.epochs, "best_epoch": best_epoch,
              "threshold": best_threshold, "validation": best_validation, "official_test": official_test,
              "test_split_used_for_selection": False, "dataset": "Meta Amateur Drawings Dataset v1.0", "dataset_license": "MIT",
              "augmentations": ["point prompt", "negative paper boundary", "paper grid", "neighbor strokes", "perspective", "shadow", "blur", "JPEG"],
              "seconds": round(time.perf_counter() - started, 1)}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
