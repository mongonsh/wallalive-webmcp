#!/usr/bin/env python3
"""Train the mixed-domain, point-prompted WallAlive cutout model v3.

V2 learned from 353 Meta Amateur Drawings.  V3 keeps that real-camera domain
but adds the official ChildlikeSHAPES training split, where every body region is
pixel-labelled.  It also trains against a deliberately duplicated character so
the point prompt—not visual identity—decides which of several figures survives.

The ChildlikeSHAPES official test directory and the filename-hashed Meta test
split stay sealed until checkpoint and threshold selection are complete.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

from train_target_cutout_v2 import (
    AmateurTargetDataset,
    add_scene_damage,
    compose_paper_scene,
    prompted_tensor,
)


def letterbox_pair(image: Image.Image, mask: Image.Image, size: int) -> tuple[Image.Image, Image.Image]:
    scale = min(size / image.width, size / image.height)
    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    target = Image.new("L", (size, size), 0)
    offset = ((size - resized[0]) // 2, (size - resized[1]) // 2)
    canvas.paste(image.resize(resized, Image.Resampling.BILINEAR), offset)
    target.paste(mask.resize(resized, Image.Resampling.NEAREST), offset)
    return canvas, target


def foreground_from_label(path: Path) -> Image.Image:
    # Palette indices are the semantic IDs.  0 is background; 255 is conflict
    # or unlabeled and must never be taught as character foreground.
    values = np.asarray(Image.open(path), dtype=np.uint8)
    return Image.fromarray((((values > 0) & (values < 255)) * 255).astype(np.uint8), "L")


def add_duplicate_hard_negative(image: Image.Image, target: Image.Image, seed: int) -> Image.Image:
    """Paste a second, visually identical figure while keeping it negative.

    This directly trains the failure case where two children draw the same role
    on one wall.  The selected figure remains positive; only the prompt can
    disambiguate the otherwise identical neighbor.
    """
    values = np.asarray(target) >= 128
    ys, xs = np.nonzero(values)
    if not len(xs):
        return image
    rng = random.Random(seed ^ 0xD091CA7E)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    character = image.crop(box)
    character_mask = target.crop(box)
    occupied = box
    for _ in range(18):
        scale = rng.uniform(0.48, 0.82)
        width = max(5, round(character.width * scale))
        height = max(5, round(character.height * scale))
        if width >= image.width - 4 or height >= image.height - 4:
            continue
        x = rng.randint(2, image.width - width - 2)
        y = rng.randint(2, image.height - height - 2)
        overlap_width = max(0, min(x + width, occupied[2]) - max(x, occupied[0]))
        overlap_height = max(0, min(y + height, occupied[3]) - max(y, occupied[1]))
        if overlap_width * overlap_height > width * height * 0.03:
            continue
        resized = character.resize((width, height), Image.Resampling.BILINEAR)
        resized_mask = character_mask.resize((width, height), Image.Resampling.NEAREST)
        result = image.copy()
        result.paste(resized, (x, y), resized_mask)
        return result
    return image


def augment_pair(image: Image.Image, mask: Image.Image, seed: int) -> tuple[Image.Image, Image.Image]:
    rng = random.Random(seed)
    size = image.width
    angle = rng.uniform(-18, 18)
    translate = [round(rng.uniform(-size * 0.09, size * 0.09)) for _ in range(2)]
    scale = rng.uniform(0.72, 1.16)
    shear = [rng.uniform(-9, 9), rng.uniform(-5, 5)]
    paper = tuple(rng.randint(222, 255) for _ in range(3))
    image = tvf.affine(image, angle, translate, scale, shear, InterpolationMode.BILINEAR, fill=paper)
    mask = tvf.affine(mask, angle, translate, scale, shear, InterpolationMode.NEAREST, fill=0)
    if rng.random() < 0.48:
        distortion = rng.uniform(0.02, 0.095) * size
        starts = [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]]
        ends = [
            [rng.uniform(0, distortion), rng.uniform(0, distortion)],
            [size - 1 - rng.uniform(0, distortion), rng.uniform(0, distortion)],
            [size - 1 - rng.uniform(0, distortion), size - 1 - rng.uniform(0, distortion)],
            [rng.uniform(0, distortion), size - 1 - rng.uniform(0, distortion)],
        ]
        image = tvf.perspective(image, starts, ends, InterpolationMode.BILINEAR, fill=paper)
        mask = tvf.perspective(mask, starts, ends, InterpolationMode.NEAREST, fill=0)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.72, 1.28))
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 0.8)))
    return image, mask


class ChildlikeTargetDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        size: int,
        augment: bool,
        paths: list[Path] | None = None,
        paper_scene: bool = False,
        duplicate: bool = False,
    ):
        self.image_root = root / f"{split}_images"
        self.label_root = root / f"{split}_annos"
        self.paths = paths if paths is not None else sorted(self.image_root.glob("*.png"))
        self.size = size
        self.augment = augment
        self.paper_scene = paper_scene
        self.duplicate = duplicate

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        # The official release contains a small number of annotation PNGs with
        # only background/conflict IDs. Skip them deterministically instead of
        # letting an empty polygon poison a training batch.
        source_image = source_mask = None
        path = self.paths[index]
        for offset in range(min(32, len(self.paths))):
            path = self.paths[(index + offset) % len(self.paths)]
            candidate_mask = foreground_from_label(self.label_root / path.name)
            if np.asarray(candidate_mask).any():
                source_image = Image.open(path).convert("RGB")
                source_mask = candidate_mask
                break
        if source_image is None or source_mask is None:
            raise ValueError("ChildlikeSHAPES sample window contains no labelled character pixels")
        seed = (index + 1) * 130363 + (random.randrange(1 << 30) if self.augment else 20260902)
        image, mask = letterbox_pair(source_image, source_mask, self.size)
        if self.augment:
            image, mask = augment_pair(image, mask, seed)
        if not np.asarray(mask).any():
            image, mask = letterbox_pair(source_image, source_mask, self.size)
        rng = random.Random(seed)
        if self.paper_scene or (self.augment and rng.random() < 0.78):
            image, mask = compose_paper_scene(image, mask, seed)
        if self.duplicate or (self.augment and rng.random() < 0.58):
            image = add_duplicate_hard_negative(image, mask, seed)
        image = add_scene_damage(image, mask, seed)
        return prompted_tensor(image, mask, seed)


class ResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        groups = 8 if outputs % 8 == 0 else 4 if outputs % 4 == 0 else 2
        self.project = nn.Conv2d(inputs, outputs, 1, bias=False) if inputs != outputs else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(groups, outputs),
        )
        self.activation = nn.SiLU()

    def forward(self, value: torch.Tensor):
        return self.activation(self.net(value) + self.project(value))


class TargetCutoutNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = ResidualBlock(4, 20)
        self.two = ResidualBlock(20, 36)
        self.three = ResidualBlock(36, 60)
        self.bridge = ResidualBlock(60, 88)
        self.dec3 = ResidualBlock(88 + 60, 60)
        self.dec2 = ResidualBlock(60 + 36, 38)
        self.dec1 = ResidualBlock(38 + 20, 26)
        self.output = nn.Conv2d(26, 1, 1)
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
    false_positive = (probability * (1 - target)).sum((0, 2, 3))
    false_negative = ((1 - probability) * target).sum((0, 2, 3))
    tversky = 1 - ((intersection + 1) / (intersection + 0.42 * false_positive + 0.58 * false_negative + 1)).mean()
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(2.0, device=logits.device))
    probability_dx = probability[..., :, 1:] - probability[..., :, :-1]
    probability_dy = probability[..., 1:, :] - probability[..., :-1, :]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    boundary = nn.functional.l1_loss(probability_dx, target_dx) + nn.functional.l1_loss(probability_dy, target_dy)
    return bce * 0.43 + tversky * 0.49 + boundary * 0.08


def empty_metric_state():
    return {
        "intersection": 0,
        "union": 0,
        "prompt_hits": 0,
        "total": 0,
        "boundary_hits": 0,
        "boundary_predicted": 0,
        "boundary_target": 0,
        "per_image": [],
    }


def finalize_metric_state(state: dict[str, int | list[float]]):
    per_image = state["per_image"]
    assert isinstance(per_image, list)
    return {
        "global_iou": round(int(state["intersection"]) / max(1, int(state["union"])), 4),
        "mean_iou": round(float(np.mean(per_image)), 4),
        "boundary_f1_1px": round(int(state["boundary_hits"]) / max(1, int(state["boundary_predicted"]) + int(state["boundary_target"])), 4),
        "prompt_hit_rate": round(int(state["prompt_hits"]) / max(1, int(state["total"])), 4),
        "drawings": int(state["total"]),
    }


def segmentation_metrics_for_thresholds(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: tuple[float, ...],
):
    states = {threshold: empty_metric_state() for threshold in thresholds}
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device).bool()
            target_eroded = ~nn.functional.max_pool2d((~targets).float(), 3, stride=1, padding=1).bool()
            target_edge = targets & ~target_eroded
            target_near = nn.functional.max_pool2d(target_edge.float(), 3, stride=1, padding=1).bool()
            probability = model(inputs).sigmoid()
            for threshold, state in states.items():
                predicted = probability >= threshold
                both = (predicted & targets).flatten(1).sum(1)
                either = (predicted | targets).flatten(1).sum(1)
                state["intersection"] = int(state["intersection"]) + int(both.sum())
                state["union"] = int(state["union"]) + int(either.sum())
                per_image = state["per_image"]
                assert isinstance(per_image, list)
                per_image.extend((both / either.clamp_min(1)).cpu().tolist())
                state["prompt_hits"] = int(state["prompt_hits"]) + int((predicted[:, 0] & (inputs[:, 3] >= 0.98)).flatten(1).any(1).sum())
                state["total"] = int(state["total"]) + len(inputs)
                pred_eroded = ~nn.functional.max_pool2d((~predicted).float(), 3, stride=1, padding=1).bool()
                pred_edge = predicted & ~pred_eroded
                pred_near = nn.functional.max_pool2d(pred_edge.float(), 3, stride=1, padding=1).bool()
                state["boundary_hits"] = int(state["boundary_hits"]) + int((pred_edge & target_near).sum()) + int((target_edge & pred_near).sum())
                state["boundary_predicted"] = int(state["boundary_predicted"]) + int(pred_edge.sum())
                state["boundary_target"] = int(state["boundary_target"]) + int(target_edge.sum())
    return {threshold: finalize_metric_state(state) for threshold, state in states.items()}


def evaluate_domains_for_thresholds(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    thresholds: tuple[float, ...],
):
    by_domain = {
        name: segmentation_metrics_for_thresholds(model, loader, device, thresholds)
        for name, loader in loaders.items()
    }
    return {
        threshold: {name: results[threshold] for name, results in by_domain.items()}
        for threshold in thresholds
    }


def selection_score(results: dict[str, dict[str, float]]) -> float:
    # Each domain matters equally even though ChildlikeSHAPES is much larger.
    domain_scores = [
        values["mean_iou"] * 0.72 + values["boundary_f1_1px"] * 0.20 + values["prompt_hit_rate"] * 0.08
        for values in results.values()
    ]
    return float(np.mean(domain_scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amateur-annotations", type=Path, required=True)
    parser.add_argument("--amateur-images-root", type=Path, required=True)
    parser.add_argument("--childlike-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--model-name", default="wallalive-target-cutout-v3")
    args = parser.parse_args()
    started = time.perf_counter()
    torch.manual_seed(20260902)
    random.seed(20260902)
    np.random.seed(20260902)

    amateur_payload = json.loads(args.amateur_annotations.read_text())
    childlike_paths = sorted((args.childlike_root / "train_images").glob("*.png"))
    # Match the established ChildlikeSHAPES split contract: UUIDs retain some
    # collection ordering, so a seeded shuffle happens before reserving
    # validation rather than taking a lexicographic acquisition slice.
    random.Random(20260831).shuffle(childlike_paths)
    validation_count = min(args.validation_count, max(1, len(childlike_paths) // 10))
    childlike_train_paths = childlike_paths[:-validation_count]
    childlike_validation_paths = childlike_paths[-validation_count:]
    childlike_train = ChildlikeTargetDataset(args.childlike_root, "train", args.size, True, childlike_train_paths)
    amateur_train = AmateurTargetDataset(amateur_payload, args.amateur_images_root, "train", args.size, True)
    training = ConcatDataset([childlike_train, amateur_train])
    # A small but critical real-camera domain must not disappear beneath the
    # larger ChildlikeSHAPES corpus. Give it 42% of sampled training examples.
    weights = [0.58 / max(1, len(childlike_train))] * len(childlike_train)
    weights += [0.42 / max(1, len(amateur_train))] * len(amateur_train)
    sampler = WeightedRandomSampler(weights, args.samples_per_epoch, replacement=True, generator=torch.Generator().manual_seed(20260902))

    validation_sets = {
        "childlike_clean": ChildlikeTargetDataset(args.childlike_root, "train", args.size, False, childlike_validation_paths),
        "childlike_wall_multi": ChildlikeTargetDataset(args.childlike_root, "train", args.size, False, childlike_validation_paths, True, True),
        "amateur_clean": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "validation", args.size, False),
        "amateur_wall": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "validation", args.size, False, True),
    }
    train_loader = DataLoader(training, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) for name, dataset in validation_sets.items()}
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TargetCutoutNetV3().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.1e-3, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=8e-5)
    best_score = -1.0
    best_state = None
    best_epoch = 0
    best_threshold = 0.5
    best_validation: dict[str, dict[str, float]] = {}
    print(json.dumps({
        "device": str(device),
        "childlike_train": len(childlike_train),
        "amateur_train": len(amateur_train),
        "validation": {name: len(dataset) for name, dataset in validation_sets.items()},
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }), flush=True)

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
        thresholds = (0.36, 0.42, 0.48, 0.54, 0.60, 0.66)
        threshold_results = evaluate_domains_for_thresholds(model, validation_loaders, device, thresholds)
        candidates = [(threshold, results, selection_score(results)) for threshold, results in threshold_results.items()]
        threshold, results, score = max(candidates, key=lambda item: item[2])
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_threshold = threshold
            best_validation = results
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        print(json.dumps({
            "epoch": epoch + 1,
            "loss": round(total_loss / max(1, count), 5),
            "threshold": threshold,
            "score": round(score, 5),
            "validation": results,
        }), flush=True)

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    sealed_sets = {
        "childlike_official": ChildlikeTargetDataset(args.childlike_root, "test", args.size, False),
        "childlike_official_wall_multi": ChildlikeTargetDataset(args.childlike_root, "test", args.size, False, paper_scene=True, duplicate=True),
        "amateur_official": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "test", args.size, False),
        "amateur_official_wall": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "test", args.size, False, True),
    }
    sealed_loaders = {name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) for name, dataset in sealed_sets.items()}
    official_test = evaluate_domains_for_thresholds(model, sealed_loaders, torch.device("cpu"), (best_threshold,))[best_threshold]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 4, args.size, args.size),
        args.output,
        input_names=["prompted_image"],
        output_names=["target_mask"],
        dynamic_axes={"prompted_image": {0: "batch"}, "target_mask": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    report = {
        "model": args.model_name,
        "architecture": "mixed-domain residual point-prompted U-Net",
        "input": [1, 4, args.size, args.size],
        "output": [1, 1, args.size, args.size],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_drawings": {"childlike": len(childlike_train), "amateur": len(amateur_train)},
        "validation_drawings": {name: len(dataset) for name, dataset in validation_sets.items()},
        "sealed_test_drawings": {name: len(dataset) for name, dataset in sealed_sets.items()},
        "samples_per_epoch": args.samples_per_epoch,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "threshold": best_threshold,
        "validation": best_validation,
        "official_test": official_test,
        "test_split_used_for_selection": False,
        "datasets": [
            {"name": "ChildlikeSHAPES", "license": "CC-BY-4.0"},
            {"name": "Meta Amateur Drawings Dataset v1.0", "license": "MIT"},
        ],
        "augmentations": [
            "point prompt", "negative paper boundary", "paper grid", "neighbor strokes",
            "identical-character hard negative", "perspective", "shadow", "blur", "JPEG",
        ],
        "seconds": round(time.perf_counter() - started, 1),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
