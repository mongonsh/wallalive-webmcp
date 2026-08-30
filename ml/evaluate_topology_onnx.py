#!/usr/bin/env python3
"""Independently verify the exported WallAlive topology ONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from torch.utils.data import DataLoader

from train_topology_v10 import FIELD_NAMES, QuickDrawTopologyDataset, TopologyDataset


def _dilate(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, ((0, 0), (0, 0), (1, 1), (1, 1)))
    output = np.zeros_like(values, dtype=bool)
    for y in range(3):
        for x in range(3):
            output |= padded[:, :, y:y + values.shape[2], x:x + values.shape[3]]
    return output


def evaluate_fields(session: ort.InferenceSession, loader: DataLoader):
    intersections = np.zeros(len(FIELD_NAMES), dtype=np.int64)
    unions = np.zeros(len(FIELD_NAMES), dtype=np.int64)
    precision_tp = recall_tp = predicted_total = actual_total = 0
    correct = total = 0
    for images, fields, classes, _ in loader:
        field_logits, class_logits = session.run(None, {"topology_values": images.numpy()})
        predicted = field_logits >= 0
        actual = fields.numpy() > .5
        intersections += (predicted & actual).sum(axis=(0, 2, 3))
        unions += (predicted | actual).sum(axis=(0, 2, 3))
        center_prediction = predicted[:, 1:2]
        center_actual = actual[:, 1:2]
        precision_tp += int((center_prediction & _dilate(center_actual)).sum())
        recall_tp += int((center_actual & _dilate(center_prediction)).sum())
        predicted_total += int(center_prediction.sum())
        actual_total += int(center_actual.sum())
        correct += int((class_logits.argmax(1) == classes.numpy()).sum())
        total += len(classes)
    precision = precision_tp / max(1, predicted_total)
    recall = recall_tp / max(1, actual_total)
    return {
        "field_iou": {name: round(float(intersections[index] / max(1, unions[index])), 4) for index, name in enumerate(FIELD_NAMES)},
        "centerline_f1_tolerance_1px": round(2 * precision * recall / max(1e-9, precision + recall), 4),
        "topology_accuracy": round(correct / max(1, total), 4),
    }


def evaluate_classes(session: ort.InferenceSession, loader: DataLoader):
    correct = total = 0
    for images, _, classes, _ in loader:
        _, logits = session.run(None, {"topology_values": images.numpy()})
        correct += int((logits.argmax(1) == classes.numpy()).sum())
        total += len(classes)
    return round(correct / max(1, total), 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("public/models/wallalive-topology-v10.onnx"))
    parser.add_argument("--quickdraw-dir", type=Path, required=True)
    parser.add_argument("--quickdraw-per-class", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-report", action="store_true")
    args = parser.parse_args()
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    synthetic = TopologyDataset(800, 2717300, 96, 48)
    quickdraw = QuickDrawTopologyDataset(args.quickdraw_dir, "test", args.quickdraw_per_class, 96, 48)
    synthetic_metrics = evaluate_fields(session, DataLoader(synthetic, batch_size=args.batch_size, shuffle=False))
    quickdraw_accuracy = evaluate_classes(session, DataLoader(quickdraw, batch_size=args.batch_size, shuffle=False))
    result = {
        "onnx_checked": True,
        "providers": session.get_providers(),
        "inputs": {item.name: item.shape for item in session.get_inputs()},
        "outputs": {item.name: item.shape for item in session.get_outputs()},
        "synthetic_test": synthetic_metrics,
        "quickdraw_test_accuracy": quickdraw_accuracy,
    }
    report_path = args.model.with_suffix(".json")
    if args.update_report:
        report = json.loads(report_path.read_text())
        if synthetic_metrics != report["official_test"] or quickdraw_accuracy != report["quickdraw_test_accuracy"]:
            raise RuntimeError("ONNX results do not exactly reproduce the sealed PyTorch test report")
        report["onnx_official_test"] = synthetic_metrics
        report["onnx_quickdraw_test_accuracy"] = quickdraw_accuracy
        report["onnx_export_verified"] = True
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
