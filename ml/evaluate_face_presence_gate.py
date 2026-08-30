#!/usr/bin/env python3
"""Calibrate a FacePresenceNet gate against browser-decoded face masks.

The segmentation ensemble configuration is already frozen. This evaluator
uses only the held-out validation split to decide whether a classifier gate is
worth applying per part, then makes one report on the official test split.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluate_face_ensemble import OnnxSession, aligned_logits, component_count
from train_childlike_detector import FACE_PARTS, ChildlikeDataset


class GatedFaceDataset(Dataset):
    def __init__(self, source96: ChildlikeDataset, source128: ChildlikeDataset, source64: ChildlikeDataset):
        if source96.paths != source128.paths or source96.paths != source64.paths:
            raise ValueError("All face datasets must contain identical drawings")
        self.source96 = source96
        self.source128 = source128
        self.source64 = source64

    def __len__(self) -> int:
        return len(self.source96)

    def __getitem__(self, index: int):
        *_, image96, _ = self.source96[index]
        *_, image128, targets128 = self.source128[index]
        *_, image64, _ = self.source64[index]
        return image96, image128, image64, targets128


def collect_gate_evidence(
    v3: OnnxSession,
    v4: OnnxSession,
    presence: OnnxSession,
    loader: DataLoader,
    weights: list[float],
    thresholds: list[float],
) -> dict[str, np.ndarray]:
    weight_array = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1, 1)
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1, 1, 1)
    probability_rows: list[np.ndarray] = []
    predicted_count_rows: list[np.ndarray] = []
    true_count_rows: list[np.ndarray] = []
    prediction_masks: list[np.ndarray] = []
    truth_masks: list[np.ndarray] = []
    for batch_index, (images96, images128, images64, targets) in enumerate(loader):
        logits3, logits4 = aligned_logits(v3, v4, images96, images128)
        combined = (1.0 - weight_array) * logits3 + weight_array * logits4
        predictions = 1.0 / (1.0 + np.exp(-combined)) >= threshold_array
        truth = targets.numpy().astype(bool, copy=False)
        presence_values = images64.numpy().astype(np.float32, copy=False) / 255.0
        probabilities = 1.0 / (1.0 + np.exp(-presence.run(presence_values)))
        predicted_counts = np.zeros((len(predictions), len(FACE_PARTS)), dtype=np.int16)
        true_counts = np.zeros_like(predicted_counts)
        for drawing_index in range(len(predictions)):
            for channel, part in enumerate(FACE_PARTS):
                predicted_counts[drawing_index, channel] = component_count(predictions[drawing_index, channel], part)
                true_counts[drawing_index, channel] = component_count(truth[drawing_index, channel], part, cap=False)
        probability_rows.append(probabilities)
        predicted_count_rows.append(predicted_counts)
        true_count_rows.append(true_counts)
        prediction_masks.append(predictions)
        truth_masks.append(truth)
        if (batch_index + 1) % 20 == 0:
            print(json.dumps({"evidence_batches": batch_index + 1}), flush=True)
    return {
        "probabilities": np.concatenate(probability_rows),
        "predicted_counts": np.concatenate(predicted_count_rows),
        "true_counts": np.concatenate(true_count_rows),
        "prediction_masks": np.concatenate(prediction_masks),
        "truth_masks": np.concatenate(truth_masks),
    }


def component_scores(predicted: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    true_positive = int((predicted & truth).sum())
    false_positive = int((predicted & ~truth).sum())
    false_negative = int((~predicted & truth).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    false_positive_rate = false_positive / max(1, int((~truth).sum()))
    return precision, recall, f1, false_positive_rate


def calibrate_gates(evidence: dict[str, np.ndarray]) -> tuple[list[float], dict[str, dict[str, float]]]:
    probabilities = evidence["probabilities"]
    segment_presence = evidence["predicted_counts"] > 0
    truth_presence = evidence["true_counts"] > 0
    candidates = np.concatenate(([0.0], np.arange(0.05, 0.951, 0.01)))
    selected: list[float] = []
    report: dict[str, dict[str, float]] = {}
    for channel, part in enumerate(FACE_PARTS):
        scored: list[tuple[float, float, float, float]] = []
        for threshold in candidates:
            predicted = segment_presence[:, channel] & (probabilities[:, channel] >= threshold)
            precision, _, f1, false_positive_rate = component_scores(predicted, truth_presence[:, channel])
            scored.append((f1, precision, -false_positive_rate, float(threshold)))
        best = max(scored)
        threshold = round(best[3], 2)
        selected.append(threshold)
        baseline = component_scores(segment_presence[:, channel], truth_presence[:, channel])
        gated = component_scores(
            segment_presence[:, channel] & (probabilities[:, channel] >= threshold),
            truth_presence[:, channel],
        )
        report[part] = {
            "gate_threshold": threshold,
            "baseline_precision": round(baseline[0], 4),
            "baseline_recall": round(baseline[1], 4),
            "baseline_f1": round(baseline[2], 4),
            "gated_precision": round(gated[0], 4),
            "gated_recall": round(gated[1], 4),
            "gated_f1": round(gated[2], 4),
            "gated_false_positive_rate": round(gated[3], 4),
        }
    return selected, report


def evaluate_gates(evidence: dict[str, np.ndarray], thresholds: list[float]) -> dict[str, dict[str, float]]:
    probabilities = evidence["probabilities"]
    original_counts = evidence["predicted_counts"]
    true_counts = evidence["true_counts"]
    predictions = evidence["prediction_masks"]
    truth_masks = evidence["truth_masks"]
    drawing_count = len(original_counts)
    report: dict[str, dict[str, float]] = {}
    for channel, part in enumerate(FACE_PARTS):
        segment_presence = original_counts[:, channel] > 0
        true_presence = true_counts[:, channel] > 0
        gate_open = probabilities[:, channel] >= thresholds[channel]
        gated_presence = segment_presence & gate_open
        gated_counts = original_counts[:, channel] * gate_open.astype(np.int16)
        precision, recall, f1, false_positive_rate = component_scores(gated_presence, true_presence)
        baseline_precision, baseline_recall, baseline_f1, baseline_fpr = component_scores(segment_presence, true_presence)
        gated_mask = predictions[:, channel] & gate_open[:, None, None]
        intersection = int((gated_mask & truth_masks[:, channel]).sum())
        union = int((gated_mask | truth_masks[:, channel]).sum())
        report[part] = {
            "gate_threshold": thresholds[channel],
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "false_positive_rate_on_absent": round(false_positive_rate, 4),
            "detection_rate_on_present": round(float((gated_presence & true_presence).sum() / max(1, true_presence.sum())), 4),
            "count_mae": round(float(np.abs(gated_counts - true_counts[:, channel]).mean()), 4),
            "count_exact_rate": round(float((gated_counts == true_counts[:, channel]).mean()), 4),
            "gated_pixel_iou": round(intersection / max(1, union), 4),
            "baseline_precision": round(baseline_precision, 4),
            "baseline_recall": round(baseline_recall, 4),
            "baseline_f1": round(baseline_f1, 4),
            "baseline_false_positive_rate": round(baseline_fpr, 4),
            "prevalence": round(float(true_presence.mean()), 4),
            "drawings": drawing_count,
        }
    return report


def make_dataset(root: Path, split: str, paths: list[Path] | None = None) -> GatedFaceDataset:
    return GatedFaceDataset(
        ChildlikeDataset(root, split, 96, 96, augment=False, paths=paths),
        ChildlikeDataset(root, split, 96, 128, augment=False, paths=paths),
        ChildlikeDataset(root, split, 96, 64, augment=False, paths=paths),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v3", type=Path, default=Path("public/models/wallalive-face-v3.onnx"))
    parser.add_argument("--v4", type=Path, default=Path("public/models/wallalive-face-v4.onnx"))
    parser.add_argument("--presence", type=Path, required=True)
    parser.add_argument("--ensemble-config", type=Path, default=Path("public/models/wallalive-face-ensemble-v4.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation-count", type=int, default=1000)
    args = parser.parse_args()

    started = time.perf_counter()
    ensemble = json.loads(args.ensemble_config.read_text())
    weights = [ensemble["blend_weight_v4"][part] for part in FACE_PARTS]
    segmentation_thresholds = [ensemble["thresholds"][part] for part in FACE_PARTS]
    all_paths = sorted((args.root / "train_images").glob("*.png"))
    random.Random(20260831).shuffle(all_paths)
    validation_paths = all_paths[-args.validation_count:]
    validation = make_dataset(args.root, "train", validation_paths)
    official_test = make_dataset(args.root, "test")
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(official_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    v3 = OnnxSession(args.v3)
    v4 = OnnxSession(args.v4)
    presence = OnnxSession(args.presence)
    validation_evidence = collect_gate_evidence(v3, v4, presence, validation_loader, weights, segmentation_thresholds)
    gate_thresholds, validation_report = calibrate_gates(validation_evidence)
    print(json.dumps({"selected_gate_thresholds": dict(zip(FACE_PARTS, gate_thresholds, strict=True)), "validation": validation_report}, indent=2), flush=True)

    test_evidence = collect_gate_evidence(v3, v4, presence, test_loader, weights, segmentation_thresholds)
    test_report = evaluate_gates(test_evidence, gate_thresholds)
    report = {
        "architecture": "WallAlive frozen face ensemble + validation-calibrated FacePresenceNet v5 gate",
        "face_parts": FACE_PARTS,
        "gate_thresholds": dict(zip(FACE_PARTS, gate_thresholds, strict=True)),
        "validation_browser_component_metrics": validation_report,
        "official_test_browser_component_metrics": test_report,
        "validation_drawings": len(validation),
        "official_test_drawings": len(official_test),
        "seconds": round(time.perf_counter() - started, 1),
        "segmentation_configuration_frozen": True,
        "test_split_used_for_selection": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
