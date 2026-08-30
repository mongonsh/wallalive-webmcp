#!/usr/bin/env python3
"""Learn a tiny component-quality gate for WallAlive face segmentation.

Unlike an image-level presence classifier, this model judges each decoded mask
component independently. Features come entirely from local v3/v4 logits and
geometry, so the browser can reproduce the linear decision with a few dozen
arithmetic operations and no additional neural-network download.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from evaluate_face_ensemble import OnnxSession, PairedFaceDataset, aligned_logits, component_count
from train_childlike_detector import FACE_PARTS, ChildlikeDataset


FEATURE_NAMES = (
    "log_area",
    "area_fraction",
    "width_fraction",
    "height_fraction",
    "log_aspect",
    "fill_fraction",
    "center_x",
    "center_y",
    "center_distance",
    "border_touch",
    "mean_probability",
    "minimum_probability",
    "maximum_probability",
    "probability_std",
    "probability_q25",
    "probability_q75",
    "strong_pixel_fraction",
    "v3_mean_probability",
    "v4_mean_probability",
    "model_disagreement",
    "component_rank",
    "component_count",
)
TARGET_GATED_PARTS = frozenset(("cheek", "ear"))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def valid_components(mask: np.ndarray, part: str) -> list[np.ndarray]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    candidates: list[np.ndarray] = []
    for label_index in range(1, count + 1):
        indices = np.flatnonzero(labels.reshape(-1) == label_index)
        if 6 <= len(indices) <= mask.size * 0.34:
            candidates.append(indices)
    candidates.sort(key=len, reverse=True)
    maximum = 3 if part == "mouth" else 6
    return candidates[:maximum]


def component_features(
    indices: np.ndarray,
    combined_logits: np.ndarray,
    v3_logits: np.ndarray,
    v4_logits: np.ndarray,
    threshold: float,
    rank: int,
    count: int,
    size: int = 128,
) -> np.ndarray:
    ys, xs = np.divmod(indices, size)
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    area = len(indices)
    probabilities = sigmoid(combined_logits.reshape(-1)[indices])
    v3_probabilities = sigmoid(v3_logits.reshape(-1)[indices])
    v4_probabilities = sigmoid(v4_logits.reshape(-1)[indices])
    center_x = float(xs.mean() / (size - 1))
    center_y = float(ys.mean() / (size - 1))
    return np.asarray((
        math.log1p(area),
        area / (size * size),
        width / size,
        height / size,
        math.log(max(1e-6, width / height)),
        area / (width * height),
        center_x,
        center_y,
        math.hypot(center_x - 0.5, center_y - 0.5),
        float(xs.min() == 0 or ys.min() == 0 or xs.max() == size - 1 or ys.max() == size - 1),
        float(probabilities.mean()),
        float(probabilities.min()),
        float(probabilities.max()),
        float(probabilities.std()),
        float(np.quantile(probabilities, 0.25)),
        float(np.quantile(probabilities, 0.75)),
        float((probabilities >= min(0.99, threshold + 0.12)).mean()),
        float(v3_probabilities.mean()),
        float(v4_probabilities.mean()),
        float(np.abs(v3_probabilities - v4_probabilities).mean()),
        rank / max(1, count - 1),
        count / 6,
    ), dtype=np.float32)


def collect_evidence(
    v3: OnnxSession,
    v4: OnnxSession,
    loader: DataLoader,
    weights: list[float],
    thresholds: list[float],
) -> dict[str, Any]:
    records: dict[str, list[dict[str, Any]]] = {part: [] for part in FACE_PARTS}
    truth_counts: list[np.ndarray] = []
    truth_pixels: list[np.ndarray] = []
    drawing_offset = 0
    weight_array = np.asarray(weights, dtype=np.float32).reshape(1, -1, 1, 1)
    threshold_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1, 1, 1)
    for batch_index, (images96, images128, targets) in enumerate(loader):
        logits3, logits4 = aligned_logits(v3, v4, images96, images128)
        combined = (1.0 - weight_array) * logits3 + weight_array * logits4
        probabilities = sigmoid(combined)
        predictions = probabilities >= threshold_array
        truth = targets.numpy().astype(bool, copy=False)
        for drawing_index in range(len(predictions)):
            drawing_truth_counts = np.zeros(len(FACE_PARTS), dtype=np.int16)
            drawing_truth_pixels = np.zeros(len(FACE_PARTS), dtype=np.int32)
            for channel, part in enumerate(FACE_PARTS):
                truth_mask = truth[drawing_index, channel]
                drawing_truth_counts[channel] = component_count(truth_mask, part, cap=False)
                drawing_truth_pixels[channel] = int(truth_mask.sum())
                candidates = valid_components(predictions[drawing_index, channel], part)
                for rank, indices in enumerate(candidates):
                    truth_flat = truth_mask.reshape(-1)
                    intersection = int(truth_flat[indices].sum())
                    truth_fraction = intersection / max(1, int(truth_mask.sum()))
                    predicted_fraction = intersection / len(indices)
                    # A correct instance must overlap the annotated part, not
                    # merely coexist somewhere else on a positive drawing.
                    matched = intersection >= 3 and (predicted_fraction >= 0.08 or truth_fraction >= 0.08)
                    records[part].append({
                        "drawing": drawing_offset + drawing_index,
                        "features": component_features(
                            indices,
                            combined[drawing_index, channel],
                            logits3[drawing_index, channel],
                            logits4[drawing_index, channel],
                            thresholds[channel],
                            rank,
                            len(candidates),
                        ),
                        "matched": matched,
                        "area": len(indices),
                        "intersection": intersection,
                    })
            truth_counts.append(drawing_truth_counts)
            truth_pixels.append(drawing_truth_pixels)
        drawing_offset += len(predictions)
        if (batch_index + 1) % 20 == 0:
            print(json.dumps({"component_batches": batch_index + 1}), flush=True)
    return {
        "records": records,
        "truth_counts": np.stack(truth_counts),
        "truth_pixels": np.stack(truth_pixels),
        "drawings": drawing_offset,
    }


def score_records(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
    truth_counts: np.ndarray,
    truth_pixels: np.ndarray,
) -> dict[str, float]:
    drawing_count = len(truth_counts)
    predicted_counts = np.zeros(drawing_count, dtype=np.int16)
    predicted_pixels = 0
    intersections = 0
    for record, score in zip(records, scores, strict=True):
        if score < threshold:
            continue
        predicted_counts[record["drawing"]] += 1
        predicted_pixels += record["area"]
        intersections += record["intersection"]
    predicted_presence = predicted_counts > 0
    true_presence = truth_counts > 0
    true_positive = int((predicted_presence & true_presence).sum())
    false_positive = int((predicted_presence & ~true_presence).sum())
    false_negative = int((~predicted_presence & true_presence).sum())
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    union = int(truth_pixels.sum()) + predicted_pixels - intersections
    pixel_iou = intersections / max(1, union)
    false_positive_rate = false_positive / max(1, int((~true_presence).sum()))
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate_on_absent": false_positive_rate,
        "count_mae": float(np.abs(predicted_counts - truth_counts).mean()),
        "count_exact_rate": float((predicted_counts == truth_counts).mean()),
        "pixel_iou": pixel_iou,
        "score": pixel_iou * 0.55 + f1 * 0.35 + float((predicted_counts == truth_counts).mean()) * 0.10,
    }


def rounded_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        name: round(float(value), 4) if isinstance(value, (int, float, np.integer, np.floating)) else value
        for name, value in metrics.items()
    }


def fit_gates(evidence: dict[str, Any], meta_train_count: int) -> tuple[dict[str, Any], dict[str, Any]]:
    configuration: dict[str, Any] = {}
    calibration_report: dict[str, Any] = {}
    candidates = np.concatenate(([0.0], np.arange(0.05, 0.951, 0.01)))
    for channel, part in enumerate(FACE_PARTS):
        all_records = evidence["records"][part]
        training_records = [record for record in all_records if record["drawing"] < meta_train_count]
        calibration_records = [record for record in all_records if record["drawing"] >= meta_train_count]
        x_train = np.stack([record["features"] for record in training_records])
        y_train = np.asarray([record["matched"] for record in training_records], dtype=np.uint8)
        x_calibration = np.stack([record["features"] for record in calibration_records])
        scaler = StandardScaler().fit(x_train)
        local_truth_counts = evidence["truth_counts"][meta_train_count:, channel]
        local_truth_pixels = evidence["truth_pixels"][meta_train_count:, channel]
        # Reindex calibration drawing ids for metric aggregation.
        local_records = [{**record, "drawing": record["drawing"] - meta_train_count} for record in calibration_records]
        baseline_scores = np.ones(len(local_records), dtype=np.float32)
        baseline = score_records(local_records, baseline_scores, 0.0, local_truth_counts, local_truth_pixels)
        if part not in TARGET_GATED_PARTS:
            # The acceptance contract targets rare cheek/accessory and ear
            # failures. Eyes and mouths already exceed 0.92 browser F1, so a
            # learned filter is intentionally a no-op for those classes.
            configuration[part] = {
                "threshold": 0.0,
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficient": [0.0] * len(FEATURE_NAMES),
                "intercept": 0.0,
            }
            calibration_report[part] = {
                "training_components": len(training_records),
                "training_positive_components": int(y_train.sum()),
                "calibration_components": len(calibration_records),
                "baseline": rounded_metrics(baseline),
                "selected": rounded_metrics({**baseline, "threshold": 0.0}),
                "decision": "no-op; high-performing common part outside rare-part gate scope",
            }
            continue
        best: tuple[float, float, float, Any, np.ndarray, dict[str, float]] | None = None
        minimum_recall = baseline["recall"] - (0.03 if part == "cheek" else 0.05)
        positive_weight = len(y_train) / max(1, 2 * int(y_train.sum()))
        negative_weight = len(y_train) / max(1, 2 * int((1 - y_train).sum()))
        sample_weights = np.where(y_train, positive_weight, negative_weight)
        model_candidates: list[tuple[str, float, Any]] = []
        for regularization in (0.05, 0.2, 1.0, 5.0):
            model_candidates.append(("linear", regularization, LogisticRegression(
                C=regularization,
                class_weight="balanced",
                max_iter=2000,
                random_state=20260903,
            )))
        for regularization in (0.003, 0.03, 0.3, 1.0):
            model_candidates.append(("mlp-tanh-10", regularization, MLPClassifier(
                hidden_layer_sizes=(10,),
                activation="tanh",
                solver="lbfgs",
                alpha=regularization,
                max_iter=2500,
                random_state=20260903,
            )))
        for model_kind, regularization, model in model_candidates:
            model.fit(scaler.transform(x_train), y_train, sample_weight=sample_weights)
            probabilities = model.predict_proba(scaler.transform(x_calibration))[:, 1]
            for threshold in candidates:
                metrics = score_records(local_records, probabilities, float(threshold), local_truth_counts, local_truth_pixels)
                if metrics["recall"] < minimum_recall:
                    continue
                selection = (metrics["score"], metrics["pixel_iou"], metrics["f1"])
                if best is None or selection > best[:3]:
                    best = (*selection, model, probabilities, {**metrics, "threshold": float(threshold), "regularization": regularization, "model": model_kind})
        if best is None:
            raise RuntimeError(f"No component gate candidates for {part}")
        model = best[3]
        selected_metrics = best[5]
        # A zero threshold is an explicit no-op. Keep it whenever the learned
        # gate cannot beat the browser baseline on held-out calibration.
        if selected_metrics["score"] <= baseline["score"]:
            selected_metrics = {
                **baseline,
                "threshold": 0.0,
                "regularization": selected_metrics["regularization"],
                "model": selected_metrics["model"],
            }
        serialized: dict[str, Any] = {
            "model": selected_metrics["model"],
            "threshold": round(selected_metrics["threshold"], 4),
            "mean": scaler.mean_.round(8).tolist(),
            "scale": scaler.scale_.round(8).tolist(),
        }
        if isinstance(model, LogisticRegression):
            serialized.update({
                "coefficient": model.coef_[0].round(8).tolist(),
                "intercept": round(float(model.intercept_[0]), 8),
            })
        else:
            serialized.update({
                "hidden_weight": model.coefs_[0].round(8).tolist(),
                "hidden_bias": model.intercepts_[0].round(8).tolist(),
                "output_weight": model.coefs_[1][:, 0].round(8).tolist(),
                "output_bias": round(float(model.intercepts_[1][0]), 8),
            })
        configuration[part] = serialized
        calibration_report[part] = {
            "training_components": len(training_records),
            "training_positive_components": int(y_train.sum()),
            "calibration_components": len(calibration_records),
            "baseline": rounded_metrics(baseline),
            "selected": rounded_metrics(selected_metrics),
            "minimum_calibration_recall": round(minimum_recall, 4),
        }
    return configuration, calibration_report


def evaluate_configuration(evidence: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for channel, part in enumerate(FACE_PARTS):
        records = evidence["records"][part]
        features = np.stack([record["features"] for record in records])
        config = configuration[part]
        normalized = (features - np.asarray(config["mean"])) / np.asarray(config["scale"])
        if config.get("model") == "mlp-tanh-10":
            hidden = np.tanh(normalized @ np.asarray(config["hidden_weight"]) + np.asarray(config["hidden_bias"]))
            logits = hidden @ np.asarray(config["output_weight"]) + config["output_bias"]
        else:
            logits = normalized @ np.asarray(config["coefficient"]) + config["intercept"]
        probabilities = sigmoid(logits)
        selected = score_records(
            records,
            probabilities,
            config["threshold"],
            evidence["truth_counts"][:, channel],
            evidence["truth_pixels"][:, channel],
        )
        baseline = score_records(
            records,
            np.ones(len(records), dtype=np.float32),
            0.0,
            evidence["truth_counts"][:, channel],
            evidence["truth_pixels"][:, channel],
        )
        report[part] = {
            "threshold": config["threshold"],
            "baseline": rounded_metrics(baseline),
            "selected": rounded_metrics(selected),
        }
    return report


def make_dataset(root: Path, split: str, paths: list[Path] | None = None) -> PairedFaceDataset:
    return PairedFaceDataset(
        ChildlikeDataset(root, split, 96, 96, augment=False, paths=paths),
        ChildlikeDataset(root, split, 96, 128, augment=False, paths=paths),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v3", type=Path, default=Path("public/models/wallalive-face-v3.onnx"))
    parser.add_argument("--v4", type=Path, default=Path("public/models/wallalive-face-v4.onnx"))
    parser.add_argument("--ensemble-config", type=Path, default=Path("public/models/wallalive-face-ensemble-v4.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-count", type=int, default=1000)
    parser.add_argument("--meta-train-count", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    started = time.perf_counter()
    ensemble = json.loads(args.ensemble_config.read_text())
    weights = [ensemble["blend_weight_v4"][part] for part in FACE_PARTS]
    thresholds = [ensemble["thresholds"][part] for part in FACE_PARTS]
    all_paths = sorted((args.root / "train_images").glob("*.png"))
    random.Random(20260831).shuffle(all_paths)
    validation_paths = all_paths[-args.validation_count:]
    validation = make_dataset(args.root, "train", validation_paths)
    official_test = make_dataset(args.root, "test")
    v3 = OnnxSession(args.v3)
    v4 = OnnxSession(args.v4)

    validation_evidence = collect_evidence(
        v3,
        v4,
        DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
        weights,
        thresholds,
    )
    configuration, calibration_report = fit_gates(validation_evidence, args.meta_train_count)
    print(json.dumps({"configuration": {part: {"threshold": value["threshold"]} for part, value in configuration.items()}, "calibration": calibration_report}, indent=2), flush=True)

    official_evidence = collect_evidence(
        v3,
        v4,
        DataLoader(official_test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
        weights,
        thresholds,
    )
    official_report = evaluate_configuration(official_evidence, configuration)
    report = {
        "architecture": "WallAlive per-component standardized logistic confidence gate v5",
        "features": FEATURE_NAMES,
        "face_parts": FACE_PARTS,
        "configuration": configuration,
        "validation_meta_training_drawings": args.meta_train_count,
        "validation_calibration_drawings": len(validation) - args.meta_train_count,
        "calibration_browser_metrics": calibration_report,
        "official_test_browser_metrics": official_report,
        "official_test_drawings": len(official_test),
        "seconds": round(time.perf_counter() - started, 1),
        "segmentation_configuration_frozen": True,
        "test_split_used_for_selection": False,
        "browser_runtime": "closed-form linear score; no additional model download",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
