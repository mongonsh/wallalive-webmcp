#!/usr/bin/env python3
"""Render sealed TargetCutout-v3 predictions for human visual inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from evaluate_target_cutout_onnx import OnnxCutoutModel
from train_target_cutout_v2 import AmateurTargetDataset
from train_target_cutout_v3 import ChildlikeTargetDataset


def mask_edge(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    eroded = mask.copy()
    for y in range(3):
        for x in range(3):
            eroded &= padded[y:y + mask.shape[0], x:x + mask.shape[1]]
    return mask & ~eroded


def render_sample(inputs: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor, size: int) -> Image.Image:
    rgb = np.clip(inputs[:3].permute(1, 2, 0).numpy() * 255, 0, 255).astype(np.uint8)
    truth = target[0].numpy() >= 0.5
    predicted = prediction[0].numpy() >= 0.5
    truth_edge = mask_edge(truth)
    predicted_edge = mask_edge(predicted)
    overlay = rgb.copy()
    overlay[truth_edge] = (53, 220, 132)
    overlay[predicted_edge] = (255, 64, 134)
    overlay[truth_edge & predicted_edge] = (255, 224, 72)
    prompt = np.unravel_index(int(torch.argmax(inputs[3])), inputs[3].shape)
    panel = Image.fromarray(overlay).resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    scale = size / inputs.shape[-1]
    x, y = int((prompt[1] + 0.5) * scale), int((prompt[0] + 0.5) * scale)
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255), outline=(20, 29, 33), width=2)
    intersection = np.logical_and(truth, predicted).sum()
    union = np.logical_or(truth, predicted).sum()
    draw.rectangle((0, size - 19, size, size), fill=(10, 20, 24))
    draw.text((5, size - 16), f"IoU {intersection / max(1, union):.2f}", fill="white", font=ImageFont.load_default())
    return panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--amateur-annotations", type=Path, required=True)
    parser.add_argument("--amateur-images-root", type=Path, required=True)
    parser.add_argument("--childlike-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--panel-size", type=int, default=220)
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    size = int(report["input"][-1])
    threshold = float(report["threshold"])
    amateur_payload = json.loads(args.amateur_annotations.read_text())
    domains = {
        "Childlike · clean": ChildlikeTargetDataset(args.childlike_root, "test", size, False),
        "Childlike · wall + twin": ChildlikeTargetDataset(args.childlike_root, "test", size, False, paper_scene=True, duplicate=True),
        "Amateur · clean": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "test", size, False),
        "Amateur · wall": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "test", size, False, True),
    }
    model = OnnxCutoutModel(args.model)
    margin = 16
    label_height = 28
    sheet = Image.new(
        "RGB",
        (margin * 2 + args.samples * args.panel_size, margin * 2 + len(domains) * (label_height + args.panel_size)),
        (244, 240, 232),
    )
    draw = ImageDraw.Draw(sheet)
    for row, (label, dataset) in enumerate(domains.items()):
        y = margin + row * (label_height + args.panel_size)
        draw.text((margin, y + 5), f"{label}  ·  green=truth  pink=prediction  yellow=match", fill=(16, 42, 39), font=ImageFont.load_default())
        indices = np.linspace(0, len(dataset) - 1, args.samples, dtype=int)
        batch_inputs = torch.stack([dataset[int(index)][0] for index in indices])
        batch_targets = torch.stack([dataset[int(index)][1] for index in indices])
        with torch.no_grad():
            batch_predictions = model(batch_inputs).sigmoid() >= threshold
        for column in range(args.samples):
            panel = render_sample(batch_inputs[column], batch_targets[column], batch_predictions[column], args.panel_size)
            sheet.paste(panel, (margin + column * args.panel_size, y + label_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
