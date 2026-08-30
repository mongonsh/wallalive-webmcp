#!/usr/bin/env python3
"""Train WallAlive's category-independent 2D topology recognizer.

Unlike the fixed 17-joint human pose model, this network predicts fields that
are meaningful for any connected drawing: foreground, medial centerline,
endpoints, and junctions.  A small structural-class head is an audit signal;
the variable graph decoded from the fields is the actual output contract.

The generator covers bipeds, quadrupeds, winged/aquatic forms, radial forms,
plants, machines and unconstrained chains.  Every training target is created
from a graph before the pixels are rendered, so labels remain exact under
camera, paper, color, blur and affine augmentation.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf


TOPOLOGY_CLASSES = ("biped", "quadruped", "winged", "aquatic", "radial", "branched", "machine", "chain")
FIELD_NAMES = ("foreground", "centerline", "endpoint", "junction")


@dataclass(frozen=True)
class Graph:
    nodes: tuple[tuple[float, float], ...]
    edges: tuple[tuple[int, int], ...]
    center: int


def _graph(kind: str, rng: random.Random) -> Graph:
    flip = -1 if rng.random() < 0.5 else 1
    jitter = lambda amount=0.025: rng.uniform(-amount, amount)
    if kind == "biped":
        nodes = ((.50, .49), (.50, .24), (.34, .40), (.18, .58), (.66, .40), (.82, .58),
                 (.40, .67), (.34, .90), (.60, .67), (.66, .90))
        edges = ((0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (0, 6), (6, 7), (0, 8), (8, 9))
    elif kind == "quadruped":
        nodes = ((.50, .49), (.24, .43), (.12, .34), (.76, .43), (.91, .33),
                 (.33, .60), (.29, .88), (.45, .60), (.43, .89), (.63, .60), (.61, .89), (.76, .59), (.79, .88))
        edges = ((0, 1), (1, 2), (0, 3), (3, 4), (0, 5), (5, 6), (0, 7), (7, 8), (0, 9), (9, 10), (0, 11), (11, 12))
    elif kind == "winged":
        nodes = ((.50, .50), (.50, .24), (.30, .47), (.08, .36), (.70, .47), (.92, .36), (.43, .76), (.37, .90), (.57, .76), (.63, .90))
        edges = ((0, 1), (0, 2), (2, 3), (0, 4), (4, 5), (0, 6), (6, 7), (0, 8), (8, 9))
    elif kind == "aquatic":
        nodes = ((.48, .50), (.23, .48), (.10, .40), (.10, .60), (.70, .49), (.91, .32), (.91, .67), (.48, .29), (.48, .71))
        edges = ((0, 1), (1, 2), (1, 3), (0, 4), (4, 5), (4, 6), (0, 7), (0, 8))
    elif kind == "radial":
        count = rng.randint(4, 8)
        nodes = [(0.5, 0.5)]
        edges = []
        for index in range(count):
            angle = math.tau * index / count + rng.uniform(-.15, .15)
            radius = rng.uniform(.30, .43)
            nodes.append((.5 + math.cos(angle) * radius, .5 + math.sin(angle) * radius))
            edges.append((0, index + 1))
        return Graph(tuple(nodes), tuple(edges), 0)
    elif kind == "branched":
        nodes = ((.50, .88), (.50, .64), (.50, .42), (.38, .30), (.28, .18), (.44, .17),
                 (.62, .31), (.55, .17), (.73, .18), (.35, .48), (.22, .38), (.66, .49), (.80, .39))
        edges = ((0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (2, 6), (6, 7), (6, 8), (1, 9), (9, 10), (1, 11), (11, 12))
    elif kind == "machine":
        nodes = ((.42, .64), (.42, .46), (.57, .39), (.69, .27), (.82, .35), (.88, .25),
                 (.82, .45), (.29, .72), (.25, .88), (.55, .72), (.58, .88))
        edges = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (4, 6), (0, 7), (7, 8), (0, 9), (9, 10))
    else:
        count = rng.randint(4, 9)
        nodes = []
        for index in range(count):
            amount = index / max(1, count - 1)
            nodes.append((.13 + amount * .74, .50 + math.sin(amount * math.tau * rng.uniform(.65, 1.35)) * rng.uniform(.10, .25)))
        return Graph(tuple(nodes), tuple((index, index + 1) for index in range(count - 1)), count // 2)
    adjusted = tuple((.5 + (x - .5) * flip + jitter(), y + jitter()) for x, y in nodes)
    return Graph(adjusted, edges, 0)


def _degrees(graph: Graph) -> list[int]:
    result = [0] * len(graph.nodes)
    for start, end in graph.edges:
        result[start] += 1
        result[end] += 1
    return result


def _palette(rng: random.Random):
    paper = tuple(rng.randint(226, 255) for _ in range(3))
    ink = tuple(rng.randint(20, 155) for _ in range(3))
    channel = rng.randrange(3)
    ink = tuple(rng.randint(125, 230) if index == channel else value for index, value in enumerate(ink))
    fill = tuple(min(255, round(paper[index] * rng.uniform(.55, .82) + ink[index] * rng.uniform(.18, .45))) for index in range(3))
    return paper, ink, fill


def _gaussian(draw: ImageDraw.ImageDraw, point: tuple[float, float], size: int, radius: float):
    x, y = point
    box = ((x * size - radius), (y * size - radius), (x * size + radius), (y * size + radius))
    draw.ellipse(tuple(round(value) for value in box), fill=255)


def render_sample(seed: int, size: int, field_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    rng = random.Random(seed)
    kind_index = seed % len(TOPOLOGY_CLASSES)
    kind = TOPOLOGY_CLASSES[kind_index]
    graph = _graph(kind, rng)
    paper, ink, fill = _palette(rng)
    image = Image.new("RGB", (size, size), paper)
    foreground = Image.new("L", (size, size), 0)
    line_width = rng.randint(max(6, size // 16), max(10, size // 9))
    stroke = rng.randint(max(1, size // 64), max(2, size // 28))
    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(foreground)

    if rng.random() < .55:
        grid = tuple(min(255, channel + rng.randint(3, 15)) for channel in paper)
        spacing = rng.randint(7, 15)
        for value in range(rng.randrange(spacing), size, spacing):
            image_draw.line((value, 0, value, size), fill=grid, width=1)
            image_draw.line((0, value, size, value), fill=grid, width=1)

    pixels = [(round(x * size), round(y * size)) for x, y in graph.nodes]
    for start, end in graph.edges:
        width = round(line_width * rng.uniform(.56, 1.03))
        image_draw.line((pixels[start], pixels[end]), fill=fill, width=width, joint="curve")
        image_draw.line((pixels[start], pixels[end]), fill=ink, width=max(1, stroke), joint="curve")
        mask_draw.line((pixels[start], pixels[end]), fill=255, width=width, joint="curve")
    degrees = _degrees(graph)
    for index, (x, y) in enumerate(pixels):
        radius = line_width * rng.uniform(.34, .65) * (1.35 if degrees[index] >= 3 else 1)
        box = (x - radius, y - radius, x + radius, y + radius)
        image_draw.ellipse(tuple(round(value) for value in box), fill=fill, outline=ink, width=stroke)
        mask_draw.ellipse(tuple(round(value) for value in box), fill=255)

    # Draw face marks on a plausible terminal without changing topology labels.
    face_index = min(range(len(graph.nodes)), key=lambda index: graph.nodes[index][1])
    fx, fy = pixels[face_index]
    eye_radius = max(1.2, line_width * .09)
    for side in (-1, 1):
        image_draw.ellipse((fx + side * line_width * .19 - eye_radius, fy - eye_radius, fx + side * line_width * .19 + eye_radius, fy + eye_radius), fill=ink)
    image_draw.arc((fx - line_width * .23, fy + line_width * .07, fx + line_width * .23, fy + line_width * .31), 5, 175, fill=ink, width=max(1, stroke))

    centerline = Image.new("L", (field_size, field_size), 0)
    endpoint = Image.new("L", (field_size, field_size), 0)
    junction = Image.new("L", (field_size, field_size), 0)
    center_draw = ImageDraw.Draw(centerline)
    endpoint_draw = ImageDraw.Draw(endpoint)
    junction_draw = ImageDraw.Draw(junction)
    field_pixels = [(round(x * field_size), round(y * field_size)) for x, y in graph.nodes]
    for start, end in graph.edges:
        center_draw.line((field_pixels[start], field_pixels[end]), fill=255, width=2, joint="curve")
    for index, point in enumerate(graph.nodes):
        if degrees[index] == 1:
            _gaussian(endpoint_draw, point, field_size, 2.0)
        elif degrees[index] >= 3:
            _gaussian(junction_draw, point, field_size, 2.2)

    foreground = foreground.resize((field_size, field_size), Image.Resampling.BILINEAR)
    target_images = [foreground, centerline, endpoint.filter(ImageFilter.GaussianBlur(.7)), junction.filter(ImageFilter.GaussianBlur(.7))]

    # Shared affine keeps pixels and graph fields registered.
    angle = rng.uniform(-14, 14)
    translate = [round(rng.uniform(-size * .035, size * .035)) for _ in range(2)]
    scale = rng.uniform(.88, 1.08)
    shear = [rng.uniform(-4, 4), rng.uniform(-2, 2)]
    image = tvf.affine(image, angle=angle, translate=translate, scale=scale, shear=shear, interpolation=InterpolationMode.BILINEAR, fill=paper)
    field_translate = [round(value * field_size / size) for value in translate]
    target_images = [tvf.affine(target, angle=angle, translate=field_translate, scale=scale, shear=shear, interpolation=InterpolationMode.BILINEAR, fill=0) for target in target_images]

    if rng.random() < .7:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(.1, .9)))
    values = np.asarray(image, dtype=np.float32)
    yy, xx = np.mgrid[:size, :size]
    shadow = ((xx / (size - 1) - .5) * rng.uniform(-24, 24) + (yy / (size - 1) - .5) * rng.uniform(-20, 20))[..., None]
    values = np.clip(values + shadow + np.random.default_rng(seed ^ 0x7A110).normal(0, rng.uniform(.8, 4.5), values.shape), 0, 255).astype(np.uint8)
    if rng.random() < .3:
        buffer = io.BytesIO()
        Image.fromarray(values).save(buffer, "JPEG", quality=rng.randint(46, 88))
        values = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"), dtype=np.uint8)
    fields = np.stack([np.asarray(target, dtype=np.float32) / 255 for target in target_images], axis=0).copy()
    return np.moveaxis(values, -1, 0).copy(), fields, kind_index


class TopologyDataset(Dataset):
    def __init__(self, count: int, seed: int, size: int, field_size: int):
        self.count, self.seed, self.size, self.field_size = count, seed, size, field_size

    def __len__(self):
        return self.count

    def __getitem__(self, index: int):
        image, fields, topology = render_sample(self.seed + index * 104729, self.size, self.field_size)
        return torch.from_numpy(image).float() / 255, torch.from_numpy(fields), topology, torch.ones(len(FIELD_NAMES), 1, 1)


QUICKDRAW_CLASSES = {
    "snowman": "biped",
    "cat": "quadruped",
    "bird": "winged",
    "fish": "aquatic",
    "octopus": "radial",
    "tree": "branched",
    "car": "machine",
    "snake": "chain",
}


class QuickDrawTopologyDataset(Dataset):
    """Real Google Quick, Draw! vectors used only for structural classification.

    Stroke order is not a body skeleton, so endpoint/junction field loss is
    deliberately disabled.  This prevents pseudo-labels from contaminating the
    graph decoder while still adapting the shared encoder to real drawings.
    """

    def __init__(self, root: Path, split: str, per_class: int, size: int, field_size: int):
        offsets = {"train": 0, "validation": per_class, "test": per_class + max(160, per_class // 5)}
        counts = {"train": per_class, "validation": max(160, per_class // 5), "test": max(160, per_class // 5)}
        self.records: list[tuple[str, list[list[list[int]]]]] = []
        for category in QUICKDRAW_CLASSES:
            path = root / f"{category}.ndjson"
            if not path.exists():
                raise FileNotFoundError(f"Missing Quick, Draw! sample: {path}")
            start = offsets[split]
            stop = start + counts[split]
            with path.open() as source:
                for index, line in enumerate(source):
                    if index >= stop:
                        break
                    if index >= start:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            # HTTP range samples end in one intentionally partial record.
                            break
                        if record.get("recognized", True) and record.get("drawing"):
                            self.records.append((category, record["drawing"]))
        self.size = size
        self.field_size = field_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        category, strokes = self.records[index]
        rng = random.Random(index * 104729 + sum(map(ord, category)))
        paper, ink, fill = _palette(rng)
        canvas = Image.new("RGB", (self.size, self.size), paper)
        draw = ImageDraw.Draw(canvas)
        ink_mask_image = Image.new("L", (self.size, self.size), 0)
        ink_mask_draw = ImageDraw.Draw(ink_mask_image)
        all_x = [x for stroke in strokes for x in stroke[0]]
        all_y = [y for stroke in strokes for y in stroke[1]]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)
        scale = min((self.size * .78) / max(1, max_x - min_x), (self.size * .78) / max(1, max_y - min_y))
        offset_x = (self.size - (max_x - min_x) * scale) / 2 - min_x * scale
        offset_y = (self.size - (max_y - min_y) * scale) / 2 - min_y * scale
        width = rng.randint(max(1, self.size // 48), max(2, self.size // 24))
        for x_values, y_values in strokes:
            points = [(round(x * scale + offset_x), round(y * scale + offset_y)) for x, y in zip(x_values, y_values)]
            if len(points) >= 2:
                draw.line(points, fill=ink, width=width, joint="curve")
                ink_mask_draw.line(points, fill=255, width=width, joint="curve")
            elif points:
                x, y = points[0]
                draw.ellipse((x - width, y - width, x + width, y + width), fill=ink)
                ink_mask_draw.ellipse((x - width, y - width, x + width, y + width), fill=255)
        # The deployed extractor sends a filled transparent cutout, not only a
        # one-pixel vector trace.  Train the real-data branch on both domains so
        # a correctly filled fish does not suddenly resemble a machine.
        if rng.random() < .62:
            ink_mask = np.asarray(ink_mask_image) > 0
            close_radius = rng.randint(max(2, self.size // 48), max(3, self.size // 18))
            grid_y, grid_x = np.ogrid[-close_radius : close_radius + 1, -close_radius : close_radius + 1]
            disk = grid_x * grid_x + grid_y * grid_y <= close_radius * close_radius
            barrier = ndimage.binary_closing(ink_mask, structure=disk)
            enclosed = ndimage.binary_fill_holes(barrier)
            if enclosed.mean() < .82:
                values = np.asarray(canvas).copy()
                values[enclosed] = fill
                values[ink_mask] = ink
                canvas = Image.fromarray(values, mode="RGB")
        if rng.random() < .55:
            canvas = canvas.filter(ImageFilter.GaussianBlur(rng.uniform(.15, .65)))
        values = np.moveaxis(np.asarray(canvas, dtype=np.uint8), -1, 0).copy()
        fields = np.zeros((len(FIELD_NAMES), self.field_size, self.field_size), dtype=np.float32)
        topology = TOPOLOGY_CLASSES.index(QUICKDRAW_CLASSES[category])
        return torch.from_numpy(values).float() / 255, torch.from_numpy(fields), topology, torch.zeros(len(FIELD_NAMES), 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU(),
                                 nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU())

    def forward(self, value):
        return self.net(value)


class TopologyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 24, 5, stride=2, padding=2, bias=False), nn.BatchNorm2d(24), nn.SiLU())
        self.encoder2 = ConvBlock(24, 36, 2)
        self.encoder3 = ConvBlock(36, 56, 2)
        self.bridge = ConvBlock(56, 72)
        self.decoder3 = ConvBlock(72 + 36, 56)
        self.decoder2 = ConvBlock(56 + 24, 36)
        self.fields = nn.Conv2d(36, len(FIELD_NAMES), 1)
        # Family labels depend on arrangement, not only feature presence.  A
        # 4x4 pooled map retains "legs below body" and "tail beside body"
        # evidence that the former global-average head destroyed.
        self.classes = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(72 * 4 * 4, 112),
            nn.SiLU(),
            nn.Dropout(.08),
            nn.Linear(112, len(TOPOLOGY_CLASSES)),
        )

    def forward(self, value):
        one = self.stem(value)
        two = self.encoder2(one)
        three = self.encoder3(two)
        bridge = self.bridge(three)
        decoded = self.decoder3(torch.cat((nn.functional.interpolate(bridge, size=two.shape[-2:], mode="bilinear", align_corners=False), two), 1))
        decoded = self.decoder2(torch.cat((nn.functional.interpolate(decoded, size=one.shape[-2:], mode="bilinear", align_corners=False), one), 1))
        return self.fields(decoded), self.classes(bridge)


def loss_fn(field_logits, class_logits, fields, classes, field_validity):
    weights = torch.tensor((1.0, 3.2, 4.5, 4.5), device=field_logits.device).view(1, -1, 1, 1)
    positive = torch.tensor((2.0, 7.0, 10.0, 10.0), device=field_logits.device).view(1, -1, 1, 1)
    bce = nn.functional.binary_cross_entropy_with_logits(field_logits, fields, reduction="none", pos_weight=positive)
    probabilities = field_logits.sigmoid()
    intersection = (probabilities * fields).sum((2, 3))
    dice = 1 - ((2 * intersection + 1) / (probabilities.sum((2, 3)) + fields.sum((2, 3)) + 1))
    valid = field_validity.to(field_logits.device)
    field_bce = (bce * weights * valid).sum() / (valid.sum() * field_logits.shape[-1] * field_logits.shape[-2]).clamp_min(1)
    field_dice = (dice * weights.flatten() * valid.flatten(1)).sum() / valid.sum().clamp_min(1)
    return field_bce + field_dice * .7 + nn.functional.cross_entropy(class_logits, classes) * .7


def dilate(values: torch.Tensor, radius: int = 1):
    return nn.functional.max_pool2d(values.float(), radius * 2 + 1, stride=1, padding=radius) > .5


def metrics(model, loader, device):
    field_intersections = torch.zeros(len(FIELD_NAMES))
    field_unions = torch.zeros(len(FIELD_NAMES))
    center_precision_tp = center_recall_tp = center_pred = center_true = 0
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, fields, classes, _ in loader:
            logits, class_logits = model(images.to(device))
            predicted = logits.sigmoid().cpu() > .5
            actual = fields > .5
            field_intersections += (predicted & actual).sum((0, 2, 3))
            field_unions += (predicted | actual).sum((0, 2, 3))
            center_prediction = predicted[:, 1:2]
            center_actual = actual[:, 1:2]
            center_precision_tp += int((center_prediction & dilate(center_actual)).sum())
            center_recall_tp += int((center_actual & dilate(center_prediction)).sum())
            center_pred += int(center_prediction.sum())
            center_true += int(center_actual.sum())
            correct += int((class_logits.cpu().argmax(1) == classes).sum())
            total += len(classes)
    precision = center_precision_tp / max(1, center_pred)
    recall = center_recall_tp / max(1, center_true)
    return {
        "field_iou": {name: round(float(field_intersections[index] / field_unions[index].clamp_min(1)), 4) for index, name in enumerate(FIELD_NAMES)},
        "centerline_f1_tolerance_1px": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "topology_accuracy": round(correct / max(1, total), 4),
    }


def classification_accuracy(model, loader, device):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for images, _, classes, _ in loader:
            _, logits = model(images.to(device))
            correct += int((logits.cpu().argmax(1) == classes).sum())
            total += len(classes)
    return round(correct / max(1, total), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("public/models/wallalive-topology-v10.onnx"))
    parser.add_argument("--input-size", type=int, default=96)
    parser.add_argument("--field-size", type=int, default=48)
    parser.add_argument("--training", type=int, default=4500)
    parser.add_argument("--validation", type=int, default=800)
    parser.add_argument("--test", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--quickdraw-dir", type=Path)
    parser.add_argument("--quickdraw-per-class", type=int, default=1400)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    torch.manual_seed(7317)
    started = time.perf_counter()
    synthetic_training = TopologyDataset(args.training, 731700, args.input_size, args.field_size)
    validation = TopologyDataset(args.validation, 1731700, args.input_size, args.field_size)
    real_training = QuickDrawTopologyDataset(args.quickdraw_dir, "train", args.quickdraw_per_class, args.input_size, args.field_size) if args.quickdraw_dir else None
    real_validation = QuickDrawTopologyDataset(args.quickdraw_dir, "validation", args.quickdraw_per_class, args.input_size, args.field_size) if args.quickdraw_dir else None
    training = ConcatDataset([synthetic_training, real_training]) if real_training else synthetic_training
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    real_validation_loader = DataLoader(real_validation, batch_size=args.batch_size, shuffle=False) if real_validation else None
    model = TopologyNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.4e-3, weight_decay=8e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1.5e-4)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    best_validation = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, fields, classes, field_validity in train_loader:
            images, fields, classes, field_validity = images.to(device), fields.to(device), classes.to(device), field_validity.to(device)
            optimizer.zero_grad(set_to_none=True)
            field_logits, class_logits = model(images)
            loss = loss_fn(field_logits, class_logits, fields, classes, field_validity)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(images)
        scheduler.step()
        current = metrics(model, validation_loader, device)
        real_validation_accuracy = classification_accuracy(model, real_validation_loader, device) if real_validation_loader else None
        # Graph fields are the product contract; the friendly family label is
        # useful but receives less selection weight than endpoints/junctions.
        score = (current["centerline_f1_tolerance_1px"] + current["field_iou"]["foreground"]
                 + current["field_iou"]["endpoint"] + current["field_iou"]["junction"]
                 + current["topology_accuracy"] + (real_validation_accuracy or 0) * .7)
        print(json.dumps({"epoch": epoch, "loss": round(running / len(training), 4), **current, "quickdraw_validation_accuracy": real_validation_accuracy}), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = {**current, "quickdraw_accuracy": real_validation_accuracy}
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    # The test generator is instantiated only after validation chose the checkpoint.
    test = TopologyDataset(args.test, 2717300, args.input_size, args.field_size)
    official_test = metrics(model, DataLoader(test, batch_size=args.batch_size, shuffle=False), torch.device("cpu"))
    real_test = QuickDrawTopologyDataset(args.quickdraw_dir, "test", args.quickdraw_per_class, args.input_size, args.field_size) if args.quickdraw_dir else None
    real_test_accuracy = classification_accuracy(model, DataLoader(real_test, batch_size=args.batch_size, shuffle=False), torch.device("cpu")) if real_test else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, args.input_size, args.input_size), args.output,
                      input_names=["topology_values"], output_names=["topology_fields", "topology_logits"],
                      dynamic_axes={"topology_values": {0: "batch"}, "topology_fields": {0: "batch"}, "topology_logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
    report = {
        "architecture": "WallAlive TopologyNet v10",
        "contract": "variable foreground-centerline-endpoint-junction graph; fixed human pose is optional only",
        "input": [1, 3, args.input_size, args.input_size],
        "outputs": {"topology_fields": [1, len(FIELD_NAMES), args.field_size, args.field_size], "topology_logits": [1, len(TOPOLOGY_CLASSES)]},
        "field_names": FIELD_NAMES,
        "topology_classes": TOPOLOGY_CLASSES,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_samples": len(training), "synthetic_training_samples": len(synthetic_training), "real_training_samples": len(real_training) if real_training else 0,
        "validation_samples": len(validation), "test_samples": len(test),
        "epochs": args.epochs, "best_epoch": best_epoch, "validation": best_validation, "official_test": official_test,
        "quickdraw_test_accuracy": real_test_accuracy,
        "test_split_used_for_selection": False,
        "training_domains": ["procedural graph-supervised drawings", "Google Quick, Draw! real vectors (classification only)", "filled isolated drawing cutouts", "paper-grid", "camera blur", "compression", "color and affine perturbations"],
        "filled_quickdraw_augmentation_probability": 0.62,
        "quickdraw_categories": QUICKDRAW_CLASSES,
        "real_data_claim": "Quick, Draw! training/validation/test records are disjoint. Real strokes supervise only topology class; they never provide pseudo endpoint or junction labels.",
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
