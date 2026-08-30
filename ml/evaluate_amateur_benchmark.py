#!/usr/bin/env python3
"""Evaluate WallAlive's browser body/part model on real amateur drawings.

The input is the compact manifest produced by ``prepare_amateur_benchmark.py``.
Each corrected silhouette is converted to the same 512px transparent cutout
layout emitted by ``app/lib/drawing.ts`` and then downsampled to the model's
96px browser input. Keypoints provide an independent semantic localization
check even though Amateur Drawings does not contain dense per-part masks.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from scipy import ndimage


PARTS = ("body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot")
KEYPOINTS_BY_PART = {
    "eye": ("left_eye", "right_eye"),
    "ear": ("left_ear", "right_ear"),
    "arm": ("left_elbow", "right_elbow", "left_wrist", "right_wrist"),
    "hand": ("left_wrist", "right_wrist"),
    "leg": ("left_knee", "right_knee", "left_ankle", "right_ankle"),
    "foot": ("left_ankle", "right_ankle"),
}


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def local_path(images_root: Path, file_name: str) -> Path:
    path = images_root / file_name
    if path.exists():
        return path
    return images_root / file_name.removeprefix("amateur_drawings/")


def polygon_mask(record: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in record["segmentation"]:
        if len(polygon) >= 6:
            draw.polygon(list(zip(polygon[::2], polygon[1::2], strict=True)), fill=255)
    return mask


def isolated_browser_input(image: Image.Image, mask: Image.Image, keypoints: dict[str, Any], model_size: int):
    binary = np.asarray(mask) > 0
    ys, xs = np.nonzero(binary)
    if not len(xs):
        raise ValueError("Empty corrected silhouette")
    span = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    padding = max(2, round(span * 0.12))
    box = (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(image.width, int(xs.max()) + padding + 1),
        min(image.height, int(ys.max()) + padding + 1),
    )
    crop_image = image.crop(box).convert("RGB")
    crop_mask = mask.crop(box)
    crop_rgb = np.asarray(crop_image).copy()
    crop_alpha = np.asarray(crop_mask) > 0
    crop_rgb[~crop_alpha] = 255
    crop_image = Image.fromarray(crop_rgb, "RGB")

    scale = min(448 / crop_image.width, 448 / crop_image.height)
    resized = (max(1, round(crop_image.width * scale)), max(1, round(crop_image.height * scale)))
    offset = ((512 - resized[0]) // 2, (512 - resized[1]) // 2)
    texture = Image.new("RGB", (512, 512), (255, 255, 255))
    texture_mask = Image.new("L", (512, 512), 0)
    texture.paste(crop_image.resize(resized, Image.Resampling.BILINEAR), offset)
    texture_mask.paste(crop_mask.resize(resized, Image.Resampling.NEAREST), offset)
    input_image = texture.resize((model_size, model_size), Image.Resampling.BILINEAR)
    target = np.asarray(texture_mask.resize((model_size, model_size), Image.Resampling.NEAREST)) > 0

    point_scale = model_size / 512
    transformed: dict[str, tuple[float, float, int]] = {}
    for name, point in keypoints.items():
        transformed[name] = (
            ((float(point["x"]) - box[0]) * scale + offset[0]) * point_scale,
            ((float(point["y"]) - box[1]) * scale + offset[1]) * point_scale,
            int(point["visibility"]),
        )
    values = np.moveaxis(np.asarray(input_image, dtype=np.float32) / 255, -1, 0)
    return values, target, transformed, texture


def component_count(mask: np.ndarray, part: str) -> int:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.reshape(-1))[1:] if count else np.asarray([], dtype=np.int64)
    minimum = 3 if part in ("eye", "mouth") else 4
    valid = int(((areas >= minimum) & (areas <= mask.size * (0.18 if part in ("arm", "leg") else 0.14))).sum())
    return min(3 if part == "mouth" else 10 if part in ("arm", "hand", "leg", "foot") else 6, valid)


def point_hit(mask: np.ndarray, x: float, y: float, radius: int = 4) -> bool:
    x0 = max(0, math.floor(x - radius))
    y0 = max(0, math.floor(y - radius))
    x1 = min(mask.shape[1], math.ceil(x + radius + 1))
    y1 = min(mask.shape[0], math.ceil(y + radius + 1))
    if x0 >= x1 or y0 >= y1:
        return False
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    return bool((mask[y0:y1, x0:x1] & disk).any())


def rounded(value: float) -> float:
    return round(float(value), 4)


def evaluate_split(
    records: list[dict[str, Any]],
    images_root: Path,
    session: ort.InferenceSession,
    thresholds: dict[str, float],
    model_size: int,
    batch_size: int,
) -> dict[str, Any]:
    body_intersection = 0
    body_union = 0
    body_iou_per_drawing: list[tuple[float, str]] = []
    body_keypoint_hits = 0
    body_keypoint_total = 0
    semantic_hits = {part: 0 for part in KEYPOINTS_BY_PART}
    semantic_total = {part: 0 for part in KEYPOINTS_BY_PART}
    component_errors = {part: [] for part in KEYPOINTS_BY_PART}
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    points: list[dict[str, tuple[float, float, int]]] = []
    names: list[str] = []

    def consume_batch() -> None:
        nonlocal body_intersection, body_union, body_keypoint_hits, body_keypoint_total
        if not inputs:
            return
        batch = np.stack(inputs).astype(np.float32, copy=False)
        outputs = session.run(None, {session.get_inputs()[0].name: batch})
        probabilities = sigmoid(outputs[0])
        for index, target in enumerate(targets):
            predictions = probabilities[index] >= np.asarray([thresholds[part] for part in PARTS])[:, None, None]
            intersection = int((predictions[0] & target).sum())
            union = int((predictions[0] | target).sum())
            body_intersection += intersection
            body_union += union
            body_iou_per_drawing.append((intersection / max(1, union), names[index]))
            for x, y, visibility in points[index].values():
                if visibility <= 0:
                    continue
                body_keypoint_total += 1
                body_keypoint_hits += int(point_hit(predictions[0], x, y, 3))
            for part, keypoint_names in KEYPOINTS_BY_PART.items():
                channel = PARTS.index(part)
                visible_points = [points[index][name] for name in keypoint_names if name in points[index] and points[index][name][2] > 0]
                semantic_total[part] += len(visible_points)
                semantic_hits[part] += sum(point_hit(predictions[channel], x, y) for x, y, _ in visible_points)
                expected_instances = sum(1 for name in keypoint_names[-2:] if name in points[index] and points[index][name][2] > 0)
                if part in ("arm", "leg"):
                    expected_instances = min(2, expected_instances)
                component_errors[part].append(abs(component_count(predictions[channel], part) - expected_instances))
        inputs.clear()
        targets.clear()
        points.clear()
        names.clear()

    for record in records:
        image = Image.open(local_path(images_root, record["file_name"])).convert("RGB")
        mask = polygon_mask(record, image.size)
        values, target, transformed, _ = isolated_browser_input(image, mask, record["keypoints"], model_size)
        inputs.append(values)
        targets.append(target)
        points.append(transformed)
        names.append(record["file_name"])
        if len(inputs) >= batch_size:
            consume_batch()
    consume_batch()

    body_iou_per_drawing.sort()
    return {
        "drawings": len(records),
        "body_iou": rounded(body_intersection / max(1, body_union)),
        "body_visible_keypoint_coverage": rounded(body_keypoint_hits / max(1, body_keypoint_total)),
        "semantic_keypoint_hit_rate": {
            part: rounded(semantic_hits[part] / max(1, semantic_total[part])) for part in KEYPOINTS_BY_PART
        },
        "component_count_mae": {
            part: rounded(np.mean(component_errors[part])) for part in KEYPOINTS_BY_PART
        },
        "worst_body_iou": [
            {"file_name": name, "iou": rounded(iou)} for iou, name in body_iou_per_drawing[:12]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("public/models/wallalive-parts-v3.onnx"))
    parser.add_argument("--model-report", type=Path, default=Path("public/models/wallalive-parts-v3.json"))
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    manifest = json.loads(args.manifest.read_text())
    model_report = json.loads(args.model_report.read_text())
    thresholds = model_report["part_thresholds"]
    model_size = int(model_report["input"][-1])
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    splits = ("train", "validation", "test") if args.split == "all" else (args.split,)
    report = {
        "dataset": manifest["dataset"],
        "dataset_license": manifest["license"],
        "model": args.model.name,
        "model_thresholds": thresholds,
        "evaluation_contract": "corrected silhouette -> WallAlive 512px cutout layout -> exact 96px ONNX input",
        "splits": {
            split: evaluate_split(
                [record for record in manifest["records"] if record["split"] == split],
                args.images_root,
                session,
                thresholds,
                model_size,
                args.batch_size,
            )
            for split in splits
        },
        "seconds": rounded(time.perf_counter() - started),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
