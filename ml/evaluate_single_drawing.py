#!/usr/bin/env python3
"""Run WallAlive's browser face stack on one drawing and render evidence.

This is a diagnostic evaluator, not a training shortcut.  It reproduces the
browser's square-fit first pass, learned head crop, v3/v4 logit blend, and
validation-selected thresholds.  The resulting overlays make failures on a
real user drawing visible before changing thresholds or retraining.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
from scipy import ndimage


PARTS = ("eye", "cheek", "mouth", "ear")
BODY_PARTS = ("body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot")
BODY_THRESHOLDS = (0.54, 0.72, 0.24, 0.70, 0.72, 0.72, 0.60, 0.72, 0.72)
BODY_SIZE = 96
FACE_V3_SIZE = 96
FACE_SIZE = 128
POSE_SIZE = 48
POSE_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
POSE_EDGES = (
    ("left_eye", "nose"), ("nose", "right_eye"),
    ("left_ear", "left_eye"), ("right_eye", "right_ear"),
    ("left_ear", "left_shoulder"), ("right_ear", "right_shoulder"),
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def square_fit(image: Image.Image, size: int) -> tuple[Image.Image, tuple[float, float, float, float]]:
    # Match the browser canvas exactly: transparent cutout pixels reveal the
    # white canvas rather than becoming black during PIL's RGB conversion.
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    image = background.convert("RGB")
    scale = min(size / image.width, size / image.height)
    width = image.width * scale
    height = image.height * scale
    offset_x = (size - width) / 2
    offset_y = (size - height) / 2
    canvas = Image.new("RGB", (size, size), "white")
    resized = image.resize((max(1, round(width)), max(1, round(height))), Image.Resampling.BILINEAR)
    canvas.paste(resized, (round(offset_x), round(offset_y)))
    return canvas, (offset_x, offset_y, width, height)


def tensor(image: Image.Image) -> np.ndarray:
    return np.moveaxis(np.asarray(image, dtype=np.float32) / 255.0, -1, 0)[None]


def run(model: Path, values: np.ndarray) -> list[np.ndarray]:
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: values.astype(np.float32, copy=False)})


def components(mask: np.ndarray, minimum: int = 3, maximum_fraction: float = 0.34) -> list[dict[str, float | int]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    found: list[dict[str, float | int]] = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < minimum or len(xs) > mask.size * maximum_fraction:
            continue
        found.append({
            "pixels": int(len(xs)),
            "min_x": int(xs.min()),
            "min_y": int(ys.min()),
            "max_x": int(xs.max()),
            "max_y": int(ys.max()),
            "center_x": round(float(xs.mean()), 2),
            "center_y": round(float(ys.mean()), 2),
        })
    return sorted(found, key=lambda item: int(item["pixels"]), reverse=True)


def locate_head(coarse_logits: np.ndarray, content_rect: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    mask = sigmoid(coarse_logits[0, 1]) >= 0.42
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    candidates = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if len(xs) >= BODY_SIZE * BODY_SIZE * 0.004:
            candidates.append((xs, ys))
    content_x, content_y, content_width, content_height = content_rect
    if not candidates:
        return content_x, content_y, content_width, content_height * 0.52
    min_x = min(int(xs.min()) for xs, _ in candidates)
    min_y = min(int(ys.min()) for _, ys in candidates)
    max_x = max(int(xs.max()) for xs, _ in candidates)
    max_y = max(int(ys.max()) for _, ys in candidates)
    span = max(max_x - min_x + 1, max_y - min_y + 1)
    margin = max(3.0, span * 0.18)
    x = max(content_x, min_x - margin)
    y = max(content_y, min_y - margin)
    right = min(content_x + content_width, max_x + margin + 1)
    bottom = min(content_y + content_height, max_y + margin + 1)
    return x, y, max(1.0, right - x), max(1.0, bottom - y)


def model_rect_to_source(rect: tuple[float, float, float, float], content_rect: tuple[float, float, float, float], image: Image.Image) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    offset_x, offset_y, draw_width, draw_height = content_rect
    left = (x - offset_x) / draw_width * image.width
    top = (y - offset_y) / draw_height * image.height
    right = (x + width - offset_x) / draw_width * image.width
    bottom = (y + height - offset_y) / draw_height * image.height
    return (
        max(0, round(left)),
        max(0, round(top)),
        min(image.width, round(right)),
        min(image.height, round(bottom)),
    )


def blend_face_logits(v3: np.ndarray, v4: np.ndarray, weights: list[float]) -> np.ndarray:
    resized_v3 = np.stack([
        np.asarray(
            Image.fromarray(channel.astype(np.float32), mode="F").resize((FACE_SIZE, FACE_SIZE), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        for channel in v3[0]
    ])[None]
    weight_array = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1, 1)
    return resized_v3 * (1 - weight_array) + v4 * weight_array


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    tint = Image.new("RGBA", base.size, (*color, 0))
    alpha = Image.fromarray((mask.astype(np.uint8) * 118), mode="L")
    tint.putalpha(alpha)
    return Image.alpha_composite(base, tint).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--models", type=Path, default=Path("public/models"))
    parser.add_argument("--ensemble", type=Path, default=Path("public/models/wallalive-face-ensemble-v4.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    source = Image.open(args.image)
    if args.crop:
        source = source.crop(tuple(args.crop))
    prepared, content_rect = square_fit(source, BODY_SIZE)
    part_logits, coarse_logits = run(args.models / "wallalive-parts-v3.onnx", tensor(prepared))
    pose_logits = run(args.models / "wallalive-amateur-pose-v6.onnx", tensor(prepared))[0][0]
    head_rect = locate_head(coarse_logits, content_rect)
    source_head_box = model_rect_to_source(head_rect, content_rect, source)
    head = source.crop(source_head_box)
    face_v3, _ = square_fit(head, FACE_V3_SIZE)
    face_v4, _ = square_fit(head, FACE_SIZE)
    v3 = run(args.models / "wallalive-face-v3.onnx", tensor(face_v3))[0]
    v4 = run(args.models / "wallalive-face-v4.onnx", tensor(face_v4))[0]
    config = json.loads(args.ensemble.read_text())
    weights = [float(config["blend_weight_v4"][part]) for part in PARTS]
    thresholds = [float(config["thresholds"][part]) for part in PARTS]
    logits = blend_face_logits(v3, v4, weights)
    probabilities = sigmoid(logits)[0]
    masks = probabilities >= np.asarray(thresholds)[:, None, None]

    prepared.save(args.output / "whole-input.png")
    face_v4.save(args.output / "face-input.png")
    palette = ((42, 220, 105), (255, 79, 117), (73, 155, 255), (255, 177, 55))
    report = {
        "image": str(args.image),
        "requested_crop": args.crop,
        "source_size": list(source.size),
        "head_crop": list(source_head_box),
        "parts": {},
        "whole_character_parts": {},
        "pose": {},
    }
    contact = Image.new("RGB", (FACE_SIZE * 2, FACE_SIZE * 2), "white")
    draw = ImageDraw.Draw(contact)
    for index, part in enumerate(PARTS):
        part_overlay = overlay(face_v4, masks[index], palette[index])
        x = (index % 2) * FACE_SIZE
        y = (index // 2) * FACE_SIZE
        contact.paste(part_overlay, (x, y))
        draw.text((x + 4, y + 4), part, fill="black", stroke_width=2, stroke_fill="white")
        part_overlay.save(args.output / f"{part}.png")
        report["parts"][part] = {
            "threshold": thresholds[index],
            "maximum_probability": round(float(probabilities[index].max()), 4),
            "components": components(masks[index], minimum=3 if part in ("eye", "mouth") else 4),
        }
    contact.save(args.output / "contact-sheet.png")
    whole_probabilities = sigmoid(part_logits)[0]
    whole_masks = whole_probabilities >= np.asarray(BODY_THRESHOLDS)[:, None, None]
    whole_palette = (
        (38, 208, 137), (95, 102, 255), (255, 93, 143),
        (64, 154, 255), (255, 179, 64), (161, 92, 255),
        (235, 78, 68), (45, 200, 211), (142, 107, 72),
    )
    whole_contact = Image.new("RGB", (BODY_SIZE * 3, BODY_SIZE * 3), "white")
    whole_draw = ImageDraw.Draw(whole_contact)
    for index, part in enumerate(BODY_PARTS):
        maximum_fraction = 0.18 if part in ("arm", "leg") else 0.14 if part in ("hand", "foot") else 0.34 if part == "body" else 0.13 if part == "ear" else 0.09
        minimum = 3 if part in ("eye", "mouth") else 4
        part_overlay = overlay(prepared, whole_masks[index], whole_palette[index])
        x = (index % 3) * BODY_SIZE
        y = (index // 3) * BODY_SIZE
        whole_contact.paste(part_overlay, (x, y))
        whole_draw.text((x + 3, y + 3), part, fill="black", stroke_width=2, stroke_fill="white")
        part_overlay.save(args.output / f"whole-{part}.png")
        report["whole_character_parts"][part] = {
            "threshold": BODY_THRESHOLDS[index],
            "maximum_probability": round(float(whole_probabilities[index].max()), 4),
            "components": components(whole_masks[index], minimum=minimum, maximum_fraction=maximum_fraction),
        }
    whole_contact.save(args.output / "whole-contact-sheet.png")
    pose_points: dict[str, tuple[float, float, float]] = {}
    for index, name in enumerate(POSE_JOINTS):
        flat = int(np.argmax(pose_logits[index]))
        x = (flat % POSE_SIZE + 0.5) * BODY_SIZE / POSE_SIZE
        y = (flat // POSE_SIZE + 0.5) * BODY_SIZE / POSE_SIZE
        confidence = float(sigmoid(pose_logits[index]).max())
        pose_points[name] = (x, y, confidence)
        report["pose"][name] = {"x": round(x, 2), "y": round(y, 2), "confidence": round(confidence, 4)}
    pose_overlay = prepared.convert("RGB")
    pose_draw = ImageDraw.Draw(pose_overlay)
    for left, right in POSE_EDGES:
        x1, y1, c1 = pose_points[left]
        x2, y2, c2 = pose_points[right]
        color = (35, 211, 143) if min(c1, c2) >= 0.48 else (160, 160, 160)
        pose_draw.line((x1, y1, x2, y2), fill=color, width=2)
    for name, (x, y, confidence) in pose_points.items():
        color = (255, 72, 129) if confidence >= 0.48 else (150, 150, 150)
        pose_draw.ellipse((x - 2.5, y - 2.5, x + 2.5, y + 2.5), fill=color, outline="white", width=1)
    pose_overlay.resize((BODY_SIZE * 3, BODY_SIZE * 3), Image.Resampling.NEAREST).save(args.output / "pose-skeleton.png")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
