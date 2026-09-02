#!/usr/bin/env python3
"""Re-run the sealed target-cutout evaluation through the exported ONNX graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_target_cutout_v2 import AmateurTargetDataset
from train_target_cutout_v3 import ChildlikeTargetDataset, evaluate_domains_for_thresholds


class OnnxCutoutModel(nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def forward(self, value: torch.Tensor):
        result = self.session.run(None, {"prompted_image": value.detach().cpu().numpy()})[0]
        return torch.from_numpy(result)


def assert_metric_parity(expected: dict, actual: dict, tolerance: float = 0.0001) -> None:
    for domain, expected_metrics in expected.items():
        if domain not in actual:
            raise AssertionError(f"missing ONNX evaluation domain: {domain}")
        for name, expected_value in expected_metrics.items():
            actual_value = actual[domain].get(name)
            if isinstance(expected_value, float) and abs(expected_value - float(actual_value)) > tolerance:
                raise AssertionError(f"{domain}.{name}: PyTorch {expected_value} != ONNX {actual_value}")
            if isinstance(expected_value, int) and expected_value != actual_value:
                raise AssertionError(f"{domain}.{name}: PyTorch {expected_value} != ONNX {actual_value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--amateur-annotations", type=Path, required=True)
    parser.add_argument("--amateur-images-root", type=Path, required=True)
    parser.add_argument("--childlike-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update-report", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.report.read_text())
    size = int(report["input"][-1])
    threshold = float(report["threshold"])
    amateur_payload = json.loads(args.amateur_annotations.read_text())
    sealed_sets = {
        "childlike_official": ChildlikeTargetDataset(args.childlike_root, "test", size, False),
        "childlike_official_wall_multi": ChildlikeTargetDataset(args.childlike_root, "test", size, False, paper_scene=True, duplicate=True),
        "amateur_official": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "test", size, False),
        "amateur_official_wall": AmateurTargetDataset(amateur_payload, args.amateur_images_root, "test", size, False, True),
    }
    loaders = {
        name: DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for name, dataset in sealed_sets.items()
    }
    model = OnnxCutoutModel(args.model)
    results = evaluate_domains_for_thresholds(model, loaders, torch.device("cpu"), (threshold,))[threshold]
    expected = report.get("official_test")
    if isinstance(expected, dict) and "childlike_official" in expected:
        assert_metric_parity(expected, results)
    payload = {
        "model": args.model.name,
        "threshold": threshold,
        "onnx_official_test": results,
        "onnx_export_verified": True,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.update_report:
        report["onnx_official_test"] = results
        report["onnx_export_verified"] = True
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
