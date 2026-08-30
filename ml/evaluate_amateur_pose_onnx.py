#!/usr/bin/env python3
"""Verify exported AmateurPoseNet coordinates with ONNX Runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from train_amateur_pose_v6 import AmateurPoseDataset, KEYPOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    training_report = json.loads(args.report.read_text())
    input_size = int(training_report["input"][-1])
    output_size = int(training_report["output"][-1])
    dataset = AmateurPoseDataset(manifest, args.images_root, "test", input_size, output_size, augment=False)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    distances: list[np.ndarray] = []
    per_joint: list[list[float]] = [[] for _ in KEYPOINT_NAMES]
    for start in range(0, len(dataset), args.batch_size):
        samples = [dataset[index] for index in range(start, min(len(dataset), start + args.batch_size))]
        images = np.stack([sample[0].numpy() for sample in samples]).astype(np.float32, copy=False)
        targets = np.stack([sample[1].numpy() for sample in samples])
        logits = session.run(None, {session.get_inputs()[0].name: images})[0]
        predicted_indices = logits.reshape(len(samples), len(KEYPOINT_NAMES), -1).argmax(-1)
        expected_indices = targets.reshape(len(samples), len(KEYPOINT_NAMES), -1).argmax(-1)
        predicted = np.stack((predicted_indices % output_size, predicted_indices // output_size), axis=-1)
        expected = np.stack((expected_indices % output_size, expected_indices // output_size), axis=-1)
        error = np.linalg.norm(predicted - expected, axis=-1)
        distances.append(error)
        for index in range(len(KEYPOINT_NAMES)):
            per_joint[index].extend(error[:, index].tolist())
    all_distances = np.concatenate(distances)
    metrics = {
        "pck_0_05": round(float((all_distances <= output_size * 0.05).mean()), 4),
        "pck_0_10": round(float((all_distances <= output_size * 0.10).mean()), 4),
        "mean_error_input_pixels": round(float(all_distances.mean() * input_size / output_size), 3),
        "per_joint_pck_0_05": {
            name: round(float((np.asarray(per_joint[index]) <= output_size * 0.05).mean()), 4)
            for index, name in enumerate(KEYPOINT_NAMES)
        },
    }
    if metrics != training_report["official_test"]:
        raise RuntimeError(f"ONNX metrics diverged from the selected PyTorch checkpoint: {metrics}")
    training_report["onnx_official_test"] = metrics
    training_report["onnx_export_verified"] = True
    args.report.write_text(json.dumps(training_report, indent=2) + "\n")
    print(json.dumps({"onnx_export_verified": True, "official_test": metrics}, indent=2))


if __name__ == "__main__":
    main()
