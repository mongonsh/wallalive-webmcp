#!/usr/bin/env python3
"""Calibrate a v3/v4 face ensemble without touching the official test split.

The two models make complementary mistakes: v3 is compact and stable, while
v4 resolves small facial accessories at 128 px. This evaluator searches blend
weights and thresholds on the held-out validation drawings, freezes them, and
then reports one official-test result. It deliberately never searches on test.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    import onnxruntime as ort
except ModuleNotFoundError:  # The training environment only needs ONNX's reference backend.
    ort = None
    import onnx
    from onnx.reference import ReferenceEvaluator

from train_childlike_detector import FACE_PARTS, ChildlikeDataset


HISTOGRAM_BINS = 256
BLEND_WEIGHTS = np.linspace(0.0, 1.0, 11, dtype=np.float32)


class PairedFaceDataset(Dataset):
    """Return aligned 96 px and 128 px crops for the same drawing."""

    def __init__(self, source96: ChildlikeDataset, source128: ChildlikeDataset):
        if source96.paths != source128.paths:
            raise ValueError("The paired face datasets must contain identical drawings")
        self.source96 = source96
        self.source128 = source128

    def __len__(self) -> int:
        return len(self.source96)

    def __getitem__(self, index: int):
        *_, image96, _ = self.source96[index]
        *_, image128, targets128 = self.source128[index]
        return image96, image128, targets128


class OnnxSession:
    def __init__(self, path: Path):
        if ort is not None:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = max(1, min(4, torch.get_num_threads()))
            self.session: Any = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        else:
            model = onnx.load(path)
            self.session = ReferenceEvaluator(model)
            self.input_name = model.graph.input[0].name

    def run(self, values: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: values})[0]


def make_session(path: Path) -> OnnxSession:
    return OnnxSession(path)


def logits(session: OnnxSession, images: torch.Tensor) -> np.ndarray:
    values = images.numpy().astype(np.float32, copy=False) / 255.0
    return session.run(values)


def aligned_logits(
    v3: OnnxSession,
    v4: OnnxSession,
    images96: torch.Tensor,
    images128: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    v3_logits = torch.from_numpy(logits(v3, images96))
    resized_v3 = torch.nn.functional.interpolate(v3_logits, size=(128, 128), mode="bilinear", align_corners=False).numpy()
    return resized_v3, logits(v4, images128)


def validation_histograms(
    v3: OnnxSession,
    v4: OnnxSession,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(BLEND_WEIGHTS), len(FACE_PARTS), HISTOGRAM_BINS)
    true_histograms = np.zeros(shape, dtype=np.int64)
    all_histograms = np.zeros(shape, dtype=np.int64)
    total_truth = np.zeros(len(FACE_PARTS), dtype=np.int64)
    for batch_index, (images96, images128, targets) in enumerate(loader):
        logits3, logits4 = aligned_logits(v3, v4, images96, images128)
        truth = targets.numpy().astype(bool, copy=False)
        total_truth += truth.sum(axis=(0, 2, 3))
        for weight_index, weight in enumerate(BLEND_WEIGHTS):
            probabilities = 1.0 / (1.0 + np.exp(-((1.0 - weight) * logits3 + weight * logits4)))
            bins = np.minimum(HISTOGRAM_BINS - 1, (probabilities * HISTOGRAM_BINS).astype(np.int16))
            for channel in range(len(FACE_PARTS)):
                channel_bins = bins[:, channel].reshape(-1)
                channel_truth = truth[:, channel].reshape(-1)
                all_histograms[weight_index, channel] += np.bincount(channel_bins, minlength=HISTOGRAM_BINS)
                true_histograms[weight_index, channel] += np.bincount(channel_bins[channel_truth], minlength=HISTOGRAM_BINS)
        if (batch_index + 1) % 10 == 0:
            print(json.dumps({"validation_batches": batch_index + 1}), flush=True)
    return true_histograms, all_histograms, total_truth


def select_configuration(
    true_histograms: np.ndarray,
    all_histograms: np.ndarray,
    total_truth: np.ndarray,
) -> tuple[list[float], list[float], dict[str, float]]:
    selected_weights: list[float] = []
    selected_thresholds: list[float] = []
    selected_ious: dict[str, float] = {}
    for channel, part in enumerate(FACE_PARTS):
        best = (-1.0, 0, 0)
        for weight_index in range(len(BLEND_WEIGHTS)):
            predicted = np.cumsum(all_histograms[weight_index, channel, ::-1])[::-1]
            intersection = np.cumsum(true_histograms[weight_index, channel, ::-1])[::-1]
            union = total_truth[channel] + predicted - intersection
            scores = intersection / np.maximum(1, union)
            # Exclude effectively empty and always-on masks from calibration.
            valid_scores = scores[10:246]
            threshold_bin = int(valid_scores.argmax()) + 10
            score = float(scores[threshold_bin])
            if score > best[0]:
                best = (score, weight_index, threshold_bin)
        score, weight_index, threshold_bin = best
        selected_weights.append(round(float(BLEND_WEIGHTS[weight_index]), 2))
        selected_thresholds.append(round(threshold_bin / HISTOGRAM_BINS, 4))
        selected_ious[part] = round(score, 4)
    return selected_weights, selected_thresholds, selected_ious


def evaluate(
    v3: OnnxSession,
    v4: OnnxSession,
    loader: DataLoader,
    weights: list[float],
    thresholds: list[float],
) -> tuple[dict[str, float], dict[str, float]]:
    intersections = np.zeros(len(FACE_PARTS), dtype=np.int64)
    unions = np.zeros(len(FACE_PARTS), dtype=np.int64)
    presence_tp = np.zeros(len(FACE_PARTS), dtype=np.int64)
    presence_fp = np.zeros(len(FACE_PARTS), dtype=np.int64)
    presence_fn = np.zeros(len(FACE_PARTS), dtype=np.int64)
    weight_array = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1, 1)
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1, 1, 1)
    for batch_index, (images96, images128, targets) in enumerate(loader):
        logits3, logits4 = aligned_logits(v3, v4, images96, images128)
        combined = (1.0 - weight_array) * logits3 + weight_array * logits4
        predictions = 1.0 / (1.0 + np.exp(-combined)) >= threshold_array
        truth = targets.numpy().astype(bool, copy=False)
        intersections += (predictions & truth).sum(axis=(0, 2, 3))
        unions += (predictions | truth).sum(axis=(0, 2, 3))
        predicted_presence = predictions.any(axis=(2, 3))
        true_presence = truth.any(axis=(2, 3))
        presence_tp += (predicted_presence & true_presence).sum(axis=0)
        presence_fp += (predicted_presence & ~true_presence).sum(axis=0)
        presence_fn += (~predicted_presence & true_presence).sum(axis=0)
        if (batch_index + 1) % 10 == 0:
            print(json.dumps({"test_batches": batch_index + 1}), flush=True)
    iou = {
        part: round(float(intersections[channel] / max(1, unions[channel])), 4)
        for channel, part in enumerate(FACE_PARTS)
    }
    presence_f1 = {
        part: round(float(2 * presence_tp[channel] / max(1, 2 * presence_tp[channel] + presence_fp[channel] + presence_fn[channel])), 4)
        for channel, part in enumerate(FACE_PARTS)
    }
    return iou, presence_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v3", type=Path, default=Path("public/models/wallalive-face-v3.onnx"))
    parser.add_argument("--v4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--validation-count", type=int, default=1000)
    args = parser.parse_args()

    started = time.perf_counter()
    all_paths = sorted((args.root / "train_images").glob("*.png"))
    random.Random(20260831).shuffle(all_paths)
    validation_paths = all_paths[-args.validation_count:]
    validation = PairedFaceDataset(
        ChildlikeDataset(args.root, "train", 96, 96, augment=False, paths=validation_paths),
        ChildlikeDataset(args.root, "train", 96, 128, augment=False, paths=validation_paths),
    )
    test = PairedFaceDataset(
        ChildlikeDataset(args.root, "test", 96, 96, augment=False),
        ChildlikeDataset(args.root, "test", 96, 128, augment=False),
    )
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    v3 = make_session(args.v3)
    v4 = make_session(args.v4)
    true_histograms, all_histograms, total_truth = validation_histograms(v3, v4, validation_loader)
    weights, thresholds, validation_iou = select_configuration(true_histograms, all_histograms, total_truth)
    print(json.dumps({"selected": {"weights": weights, "thresholds": thresholds, "validation_iou": validation_iou}}), flush=True)
    test_iou, test_presence = evaluate(v3, v4, test_loader, weights, thresholds)
    report = {
        "architecture": "WallAlive validation-calibrated v3+v4 face ensemble",
        "face_parts": FACE_PARTS,
        "blend_weight_v4": dict(zip(FACE_PARTS, weights, strict=True)),
        "thresholds": dict(zip(FACE_PARTS, thresholds, strict=True)),
        "validation_face_iou": validation_iou,
        "official_test_face_iou": test_iou,
        "official_test_presence_f1": test_presence,
        "validation_drawings": len(validation),
        "official_test_drawings": len(test),
        "seconds": round(time.perf_counter() - started, 1),
        "test_split_used_for_selection": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
