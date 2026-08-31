#!/usr/bin/env python3
"""Visual parity check for the browser target-cutout v2 candidate search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter


SIZE = 128
SCALES = (0.46, 0.62, 0.8, 1.0)


def candidate(image: Image.Image, target: tuple[float, float], scale: float):
    base = min(image.size)
    side = min(max(96, base * scale), max(image.size))
    target_x, target_y = target[0] * image.width, target[1] * image.height
    x = min(max(0, target_x - side / 2), max(0, image.width - side))
    y = min(max(0, target_y - side / 2), max(0, image.height - side))
    crop = image.crop((x, y, x + side, y + side)).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    prompt = (min(1, max(0, (target_x - x) / side)), min(1, max(0, (target_y - y) / side)))
    rgb = np.moveaxis(np.asarray(crop, dtype=np.float32) / 255.0, -1, 0)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    prompt_x, prompt_y = prompt[0] * (SIZE - 1), prompt[1] * (SIZE - 1)
    heatmap = np.exp(-((xx - prompt_x) ** 2 + (yy - prompt_y) ** 2) / (2 * (SIZE * 0.055) ** 2)).astype(np.float32)
    return crop, np.concatenate((rgb, heatmap[None]), axis=0), prompt


def prompted_component(probability: np.ndarray, prompt: tuple[float, float], threshold: float):
    active = probability >= threshold
    px, py = round(prompt[0] * (SIZE - 1)), round(prompt[1] * (SIZE - 1))
    if not active[py, px]:
        choices = []
        for y in range(max(0, py - 15), min(SIZE, py + 16)):
            for x in range(max(0, px - 15), min(SIZE, px + 16)):
                if active[y, x]:
                    choices.append(((x - px) ** 2 + (y - py) ** 2, -probability[y, x], x, y))
        if not choices:
            return np.zeros_like(active)
        _, _, px, py = min(choices)
    mask = np.zeros_like(active)
    stack = [(px, py)]
    mask[py, px] = True
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < SIZE and 0 <= ny < SIZE and active[ny, nx] and not mask[ny, nx]:
                mask[ny, nx] = True
                stack.append((nx, ny))
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("public/models/wallalive-target-cutout-v2.onnx"))
    parser.add_argument("--report", type=Path, default=Path("public/models/wallalive-target-cutout-v2.json"))
    parser.add_argument("--target-x", type=float, default=0.42)
    parser.add_argument("--target-y", type=float, default=0.42)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--opening", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("/private/tmp/wallalive-target-cutout-eval.png"))
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    threshold = args.threshold if args.threshold is not None else json.loads(args.report.read_text())["threshold"]
    candidates = [candidate(image, (args.target_x, args.target_y), scale) for scale in SCALES]
    batch = np.stack([item[1] for item in candidates])
    logits = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"]).run(None, {"prompted_image": batch})[0][:, 0]
    sheet = Image.new("RGB", (SIZE * 3, SIZE * len(SCALES)), "white")
    summaries = []
    for row, (scale, (crop, _, prompt), output) in enumerate(zip(SCALES, candidates, logits, strict=True)):
        probability = 1 / (1 + np.exp(-output))
        original = probability >= threshold
        if args.opening:
            radius = args.opening
            opened = Image.fromarray((original * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(radius * 2 + 1)).filter(ImageFilter.MaxFilter(radius * 2 + 1))
            mask = prompted_component(np.asarray(opened, dtype=np.float32) / 255, prompt, 0.5) & original
        else:
            mask = prompted_component(probability, prompt, threshold)
        overlay = np.asarray(crop).copy()
        overlay[mask] = overlay[mask] * 0.45 + np.array([255, 103, 77]) * 0.55
        cutout = np.full_like(overlay, 255)
        cutout[mask] = np.asarray(crop)[mask]
        for column, panel in enumerate((crop, Image.fromarray(overlay.astype(np.uint8)), Image.fromarray(cutout))):
            sheet.paste(panel, (column * SIZE, row * SIZE))
        draw = ImageDraw.Draw(sheet)
        draw.text((4, row * SIZE + 4), f"scale {scale}", fill=(24, 49, 46))
        summaries.append({"scale": scale, "area": round(float(mask.mean()), 4), "mean_probability": round(float(probability[mask].mean()) if mask.any() else 0, 4), "prompt_probability": round(float(probability[round(prompt[1] * 127), round(prompt[0] * 127)]), 4)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(json.dumps({"threshold": threshold, "target": [args.target_x, args.target_y], "candidates": summaries, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
