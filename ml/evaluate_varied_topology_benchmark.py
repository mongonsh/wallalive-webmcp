#!/usr/bin/env python3
"""Evaluate topology-v10 on the attributed post-split drawing benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


CLASSES = ("biped", "quadruped", "winged", "aquatic", "radial", "branched", "machine", "chain")
FIELDS = ("foreground", "centerline", "endpoint", "junction")


def image_tensor(path: Path) -> np.ndarray:
    source = Image.open(path).convert("RGBA")
    white = Image.new("RGBA", source.size, "white")
    white.alpha_composite(source)
    resized = white.convert("RGB").resize((96, 96), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)[None] / 255


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("eval/varied-drawings/manifest.json"))
    parser.add_argument("--model", type=Path, default=Path("public/models/wallalive-topology-v10.onnx"))
    parser.add_argument("--output", type=Path, default=Path("eval/varied-drawings/topology-v10-results.json"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    cases = []
    for benchmark_case in manifest["cases"]:
        fields, logits = session.run(None, {"topology_values": image_tensor(args.manifest.parent / benchmark_case["input"])})
        shifted = logits[0] - logits[0].max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        order = probabilities.argsort()[::-1]
        predicted = CLASSES[int(order[0])]
        cases.append({
            "id": benchmark_case["id"],
            "source_line_index": benchmark_case["source_line_index"],
            "expected": benchmark_case["expected_topology"],
            "predicted": predicted,
            "confidence": round(float(probabilities[order[0]]), 4),
            "passed": predicted == benchmark_case["expected_topology"],
            "top3": [[CLASSES[int(index)], round(float(probabilities[index]), 4)] for index in order[:3]],
            "positive_field_pixels": {
                name: int((fields[0, index] >= 0).sum()) for index, name in enumerate(FIELDS)
            },
        })
    report = {
        "model": "wallalive-topology-v10",
        "training_split_ends_before_line": 1960,
        "passed": all(item["passed"] for item in cases),
        "cases": cases,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
