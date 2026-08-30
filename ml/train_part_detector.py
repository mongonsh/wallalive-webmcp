#!/usr/bin/env python3
"""Train WallAlive's compact drawing-part segmentation model.

The training data is generated locally: every sample has an exact silhouette and
semantic masks for the visible eyes, cheeks, mouth, ears, arms, hands, legs and
feet.  Heavy paper, camera, color and geometry augmentation prevents the model
from memorising one character.  The exported ONNX model contains no user image.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PARTS = ("body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot")


def _palette(rng: random.Random) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    paper = tuple(rng.randint(218, 255) for _ in range(3))
    if rng.random() < 0.72:
        dominant = rng.randrange(3)
        ink = [rng.randint(35, 135) for _ in range(3)]
        ink[dominant] = rng.randint(115, 225)
        ink = tuple(ink)
    else:
        value = rng.randint(20, 120)
        ink = (value, value + rng.randint(-8, 8), value + rng.randint(-8, 8))
    fill = tuple(round(paper[index] * 0.72 + ink[index] * 0.28) for index in range(3))
    return paper, ink, fill


def _ellipse(draw: ImageDraw.ImageDraw, box: tuple[float, float, float, float], *, fill=None, outline=None, width=1) -> None:
    draw.ellipse(tuple(round(value) for value in box), fill=fill, outline=outline, width=width)


def _line(draw: ImageDraw.ImageDraw, points, *, fill, width) -> None:
    draw.line([(round(x), round(y)) for x, y in points], fill=fill, width=width, joint="curve")


def make_sample(seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    paper, ink, body_fill = _palette(rng)
    image = Image.new("RGB", (size, size), paper)
    draw = ImageDraw.Draw(image)
    masks = [Image.new("L", (size, size), 0) for _ in PARTS]
    mask_draws = [ImageDraw.Draw(mask) for mask in masks]

    # Real capture nuisances: paper grid, fold/shadow gradients and unrelated strokes.
    if rng.random() < 0.55:
        grid = tuple(min(255, channel + rng.randint(2, 18)) for channel in paper)
        spacing = rng.randint(7, 15)
        for x in range(rng.randint(0, spacing), size, spacing):
            draw.line((x, 0, x, size), fill=grid, width=1)
        for y in range(rng.randint(0, spacing), size, spacing):
            draw.line((0, y, size, y), fill=grid, width=1)

    cx = rng.uniform(size * 0.43, size * 0.57)
    cy = rng.uniform(size * 0.47, size * 0.58)
    body_w = rng.uniform(size * 0.31, size * 0.56)
    body_h = rng.uniform(size * 0.38, size * 0.67)
    left, top, right, bottom = cx - body_w / 2, cy - body_h / 2, cx + body_w / 2, cy + body_h / 2
    stroke = rng.randint(max(1, size // 48), max(2, size // 20))
    filled = rng.random() < 0.34
    shape_kind = rng.choice(("ellipse", "rounded", "blob", "spiky"))
    if shape_kind == "ellipse":
        _ellipse(draw, (left, top, right, bottom), fill=body_fill if filled else paper, outline=ink, width=stroke)
        _ellipse(mask_draws[0], (left, top, right, bottom), fill=255)
    elif shape_kind == "rounded":
        radius = round(min(body_w, body_h) * rng.uniform(0.18, 0.38))
        box = tuple(round(value) for value in (left, top, right, bottom))
        draw.rounded_rectangle(box, radius=radius, fill=body_fill if filled else paper, outline=ink, width=stroke)
        mask_draws[0].rounded_rectangle(box, radius=radius, fill=255)
    else:
        count = rng.randint(10, 18)
        points = []
        for index in range(count):
            angle = index / count * math.tau - math.pi / 2
            wave = 1 + rng.uniform(-0.12, 0.12)
            if shape_kind == "spiky" and index % 2:
                wave *= rng.uniform(0.78, 0.92)
            points.append((cx + math.cos(angle) * body_w * 0.5 * wave, cy + math.sin(angle) * body_h * 0.5 * wave))
        polygon = [(round(x), round(y)) for x, y in points]
        draw.polygon(polygon, fill=body_fill if filled else paper, outline=ink, width=stroke)
        mask_draws[0].polygon(polygon, fill=255)

    # Ears are explicit semantic instances and are also unioned into the silhouette.
    ear_count = rng.choices((0, 1, 2), weights=(0.18, 0.12, 0.70))[0]
    ear_sides = rng.sample((-1, 1), ear_count) if ear_count < 2 else [-1, 1]
    for side in ear_sides:
        ex = cx + side * body_w * rng.uniform(0.25, 0.37)
        ey = top + body_h * rng.uniform(-0.02, 0.13)
        ew = body_w * rng.uniform(0.12, 0.25)
        eh = body_h * rng.uniform(0.14, 0.30)
        points = [(ex - ew * 0.5, ey + eh * 0.5), (ex, ey - eh * 0.5), (ex + ew * 0.5, ey + eh * 0.5)]
        polygon = [(round(x), round(y)) for x, y in points]
        draw.polygon(polygon, fill=body_fill if filled else paper, outline=ink, width=stroke)
        mask_draws[4].polygon(polygon, fill=255)
        mask_draws[0].polygon(polygon, fill=255)

    # Arms, hands, legs and feet vary independently; asymmetric creatures remain valid.
    limb_width = max(stroke + 1, round(size * rng.uniform(0.035, 0.075)))
    for side in (-1, 1):
        if rng.random() < 0.78:
            shoulder = (cx + side * body_w * 0.42, cy + body_h * rng.uniform(-0.12, 0.12))
            hand = (shoulder[0] + side * body_w * rng.uniform(0.24, 0.52), shoulder[1] + body_h * rng.uniform(-0.06, 0.28))
            _line(draw, (shoulder, hand), fill=ink, width=stroke)
            _line(mask_draws[5], (shoulder, hand), fill=255, width=limb_width)
            _line(mask_draws[0], (shoulder, hand), fill=255, width=limb_width)
            radius = size * rng.uniform(0.035, 0.065)
            hand_box = (hand[0] - radius, hand[1] - radius, hand[0] + radius, hand[1] + radius)
            _ellipse(draw, hand_box, fill=body_fill if filled else paper, outline=ink, width=max(1, stroke - 1))
            _ellipse(mask_draws[6], hand_box, fill=255)
            _ellipse(mask_draws[0], hand_box, fill=255)
        if rng.random() < 0.92:
            hip = (cx + side * body_w * rng.uniform(0.12, 0.27), bottom - body_h * 0.03)
            foot = (hip[0] + side * body_w * rng.uniform(-0.04, 0.18), hip[1] + body_h * rng.uniform(0.16, 0.31))
            _line(draw, (hip, foot), fill=ink, width=stroke)
            _line(mask_draws[7], (hip, foot), fill=255, width=limb_width)
            _line(mask_draws[0], (hip, foot), fill=255, width=limb_width)
            fw = size * rng.uniform(0.075, 0.13)
            fh = size * rng.uniform(0.035, 0.07)
            foot_box = (foot[0] - fw / 2, foot[1] - fh / 2, foot[0] + fw / 2, foot[1] + fh / 2)
            _ellipse(draw, foot_box, fill=body_fill if filled else paper, outline=ink, width=max(1, stroke - 1))
            _ellipse(mask_draws[8], foot_box, fill=255)
            _ellipse(mask_draws[0], foot_box, fill=255)

    face_y = top + body_h * rng.uniform(0.30, 0.43)
    eye_count = rng.choices((1, 2, 3), weights=(0.10, 0.84, 0.06))[0]
    if eye_count == 1:
        eye_xs = [cx + rng.uniform(-0.05, 0.05) * body_w]
    elif eye_count == 2:
        separation = body_w * rng.uniform(0.14, 0.25)
        eye_xs = [cx - separation, cx + separation]
    else:
        separation = body_w * rng.uniform(0.12, 0.19)
        eye_xs = [cx - separation, cx, cx + separation]
    for ex in eye_xs:
        ew = body_w * rng.uniform(0.075, 0.16)
        eh = body_h * rng.uniform(0.055, 0.13)
        box = (ex - ew / 2, face_y - eh / 2, ex + ew / 2, face_y + eh / 2)
        if rng.random() < 0.14:
            _line(draw, ((box[0], face_y), (box[2], face_y)), fill=ink, width=stroke)
            _line(mask_draws[1], ((box[0], face_y), (box[2], face_y)), fill=255, width=max(3, stroke + 2))
        else:
            _ellipse(draw, box, fill=ink if rng.random() < 0.24 else None, outline=ink, width=stroke)
            _ellipse(mask_draws[1], box, fill=255)
            if rng.random() < 0.55 and ew > size * 0.04:
                pr = min(ew, eh) * rng.uniform(0.12, 0.25)
                _ellipse(draw, (ex - pr, face_y - pr, ex + pr, face_y + pr), fill=ink)

    if eye_count >= 2 and rng.random() < 0.70:
        cheek_y = face_y + body_h * rng.uniform(0.13, 0.21)
        for side in (-1, 1):
            x = cx + side * body_w * rng.uniform(0.24, 0.32)
            cw = body_w * rng.uniform(0.07, 0.13)
            ch = body_h * rng.uniform(0.035, 0.07)
            box = (x - cw / 2, cheek_y - ch / 2, x + cw / 2, cheek_y + ch / 2)
            _ellipse(draw, box, fill=None, outline=ink, width=max(1, stroke - 1))
            _ellipse(mask_draws[2], box, fill=255)

    mouth_y = face_y + body_h * rng.uniform(0.15, 0.28)
    mouth_w = body_w * rng.uniform(0.10, 0.28)
    mouth_h = body_h * rng.uniform(0.035, 0.10)
    mouth_box = tuple(round(value) for value in (cx - mouth_w / 2, mouth_y - mouth_h / 2, cx + mouth_w / 2, mouth_y + mouth_h / 2))
    mouth_kind = rng.choice(("smile", "frown", "line", "open"))
    if mouth_kind == "open":
        draw.ellipse(mouth_box, outline=ink, width=stroke)
        mask_draws[3].ellipse(mouth_box, fill=255)
    elif mouth_kind == "line":
        _line(draw, ((mouth_box[0], mouth_y), (mouth_box[2], mouth_y)), fill=ink, width=stroke)
        _line(mask_draws[3], ((mouth_box[0], mouth_y), (mouth_box[2], mouth_y)), fill=255, width=max(3, stroke + 2))
    else:
        start, end = (5, 175) if mouth_kind == "frown" else (185, 355)
        draw.arc(mouth_box, start=start, end=end, fill=ink, width=stroke)
        mask_draws[3].arc(mouth_box, start=start, end=end, fill=255, width=max(3, stroke + 2))

    # Add distractors outside the body to force target-vs-background discrimination.
    for _ in range(rng.randint(0, 8)):
        x1, y1 = rng.uniform(0, size), rng.uniform(0, size)
        if left - 4 < x1 < right + 4 and top - 4 < y1 < bottom + 4:
            continue
        x2, y2 = x1 + rng.uniform(-10, 10), y1 + rng.uniform(-10, 10)
        _line(draw, ((x1, y1), (x2, y2)), fill=ink, width=max(1, stroke - 1))

    # Shared affine transform means every semantic label stays exactly registered.
    angle = rng.uniform(-17, 17)
    translate = (rng.uniform(-size * 0.055, size * 0.055), rng.uniform(-size * 0.045, size * 0.045))
    scale = rng.uniform(0.88, 1.08)
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as tvf

    image = tvf.affine(image, angle=angle, translate=translate, scale=scale, shear=[rng.uniform(-5, 5), 0], interpolation=InterpolationMode.BILINEAR, fill=paper)
    masks = [tvf.affine(mask, angle=angle, translate=translate, scale=scale, shear=[0, 0], interpolation=InterpolationMode.NEAREST, fill=0) for mask in masks]

    if rng.random() < 0.66:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.15, 1.05)))
    array = np.asarray(image, dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    shadow = ((xx / max(1, size - 1) - 0.5) * rng.uniform(-22, 22) + (yy / max(1, size - 1) - 0.5) * rng.uniform(-18, 18))[..., None]
    noise = np.random.default_rng(seed ^ 0xA11CE).normal(0, rng.uniform(0.8, 5.0), array.shape)
    array = np.clip(array + shadow + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.28:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="JPEG", quality=rng.randint(45, 86))
        array = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), dtype=np.uint8)

    labels = np.stack([(np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8) for mask in masks], axis=0)
    return np.moveaxis(array, -1, 0), labels


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PartUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(3, 16)
        self.enc2 = ConvBlock(16, 28)
        self.enc3 = ConvBlock(28, 48)
        self.bridge = ConvBlock(48, 72)
        self.dec3 = ConvBlock(72 + 48, 48)
        self.dec2 = ConvBlock(48 + 28, 28)
        self.dec1 = ConvBlock(28 + 16, 20)
        self.output = nn.Conv2d(20, len(PARTS), 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        one = self.enc1(x)
        two = self.enc2(self.pool(one))
        three = self.enc3(self.pool(two))
        bridge = self.bridge(self.pool(three))
        up_three = torch.nn.functional.interpolate(bridge, size=three.shape[-2:], mode="bilinear", align_corners=False)
        up_three = self.dec3(torch.cat((up_three, three), dim=1))
        up_two = torch.nn.functional.interpolate(up_three, size=two.shape[-2:], mode="bilinear", align_corners=False)
        up_two = self.dec2(torch.cat((up_two, two), dim=1))
        up_one = torch.nn.functional.interpolate(up_two, size=one.shape[-2:], mode="bilinear", align_corners=False)
        return self.output(self.dec1(torch.cat((up_one, one), dim=1)))


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + targets.sum(dim=(0, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()


def build_dataset(count: int, size: int, seed_offset: int) -> TensorDataset:
    images = np.empty((count, 3, size, size), dtype=np.uint8)
    labels = np.empty((count, len(PARTS), size, size), dtype=np.uint8)
    for index in range(count):
        images[index], labels[index] = make_sample(seed_offset + index * 7919, size)
        if (index + 1) % 500 == 0:
            print(f"generated {index + 1}/{count}", flush=True)
    return TensorDataset(torch.from_numpy(images), torch.from_numpy(labels))


def channel_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    intersections = torch.zeros(len(PARTS), device=device)
    unions = torch.zeros(len(PARTS), device=device)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images.to(device).float() / 255).sigmoid() > 0.45
            truth = labels.to(device).bool()
            intersections += (predictions & truth).sum(dim=(0, 2, 3))
            unions += (predictions | truth).sum(dim=(0, 2, 3))
    return {part: round(float((intersections[index] / unions[index].clamp_min(1)).cpu()), 4) for index, part in enumerate(PARTS)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--validation", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("public/models/wallalive-parts-v1.onnx"))
    args = parser.parse_args()
    torch.manual_seed(20260831)
    started = time.perf_counter()
    training = build_dataset(args.samples, args.size, 10_000)
    validation = build_dataset(args.validation, args.size, 900_000)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PartUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
    positive_weights = torch.tensor([1.5, 7.0, 10.0, 12.0, 7.0, 8.0, 10.0, 8.0, 10.0], device=device).view(1, -1, 1, 1)
    best_score = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images = images.to(device).float() / 255
            labels = labels.to(device).float()
            logits = model(images)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=positive_weights)
            loss = bce * 0.52 + dice_loss(logits, labels) * 0.48
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * images.shape[0]
        metrics = channel_metrics(model, validation_loader, device)
        score = sum(metrics.values()) / len(metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_metrics = metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({"epoch": epoch + 1, "loss": round(total / args.samples, 4), "iou": metrics}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cpu_model = model.to("cpu").eval()
    if best_state is not None:
        cpu_model.load_state_dict(best_state)
    dummy = torch.zeros(1, 3, args.size, args.size)
    torch.onnx.export(
        cpu_model,
        dummy,
        args.output,
        input_names=["pixel_values"],
        output_names=["part_logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "part_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report = {
        "architecture": "WallAlive PartUNet v1",
        "parts": PARTS,
        "input": [1, 3, args.size, args.size],
        "validation_iou": best_metrics,
        "best_epoch": best_epoch,
        "training_samples": args.samples,
        "validation_samples": args.validation,
        "epochs": args.epochs,
        "parameters": sum(parameter.numel() for parameter in cpu_model.parameters()),
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
