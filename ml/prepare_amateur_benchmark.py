#!/usr/bin/env python3
"""Create a compact, deterministic benchmark from a sampled Amateur Drawings tar.

The official COCO-style annotation file is roughly 275 MiB. This script streams
it twice, retaining only records whose PNG is present in ``--images-root``.
No drawing pixels are copied into the repository; the output is a compact local
manifest that can drive evaluation or fine-tuning from a byte-range tar sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import ijson
from PIL import Image


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def normalized_relative_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    marker = "amateur_drawings/"
    index = relative.find(marker)
    return relative[index:] if index >= 0 else relative


def split_for(file_name: str) -> str:
    # The tar is grouped by UUID prefix, so use the full filename hash rather
    # than archive order. 70/15/15 yields useful training data and two disjoint
    # checks even for a small byte-range sample.
    bucket = int(hashlib.sha256(file_name.encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def valid_pngs(root: Path) -> set[str]:
    found: set[str] = set()
    for path in sorted(root.rglob("*.png")):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception:
            continue
        found.add(normalized_relative_path(path, root))
    return found


def stream_items(path: Path, prefix: str):
    with path.open("rb") as source:
        yield from ijson.items(source, prefix, use_float=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sampled_names = valid_pngs(args.images_root)
    if not sampled_names:
        raise RuntimeError(f"No valid PNG files found below {args.images_root}")

    images_by_id: dict[int, dict[str, Any]] = {}
    for image in stream_items(args.annotations, "images.item"):
        if image["file_name"] in sampled_names:
            images_by_id[int(image["id"])] = image

    annotations_by_image: dict[int, dict[str, Any]] = {}
    for annotation in stream_items(args.annotations, "annotations.item"):
        image_id = int(annotation["image_id"])
        if image_id in images_by_id:
            annotations_by_image[image_id] = annotation

    records: list[dict[str, Any]] = []
    for image_id, image in sorted(images_by_id.items()):
        annotation = annotations_by_image.get(image_id)
        if not annotation or not annotation.get("segmentation"):
            continue
        file_name = image["file_name"]
        local_path = args.images_root / file_name
        if not local_path.exists():
            # Allow --images-root to point directly at the extracted
            # ``amateur_drawings`` directory as well as its parent.
            local_path = args.images_root / file_name.removeprefix("amateur_drawings/")
        with Image.open(local_path) as local_image:
            local_width, local_height = local_image.size
        keypoint_values = annotation.get("keypoints", [])
        keypoints = {
            name: {
                "x": keypoint_values[index * 3],
                "y": keypoint_values[index * 3 + 1],
                "visibility": int(keypoint_values[index * 3 + 2]),
            }
            for index, name in enumerate(KEYPOINT_NAMES)
            if len(keypoint_values) >= (index + 1) * 3
        }
        records.append({
            "id": image_id,
            "file_name": file_name,
            "split": split_for(file_name),
            "local_width": local_width,
            "local_height": local_height,
            "source_width": int(image["width"]),
            "source_height": int(image["height"]),
            "bbox": annotation["bbox"],
            "segmentation": annotation["segmentation"],
            "area": annotation["area"],
            "keypoints": keypoints,
        })

    counts = {split: sum(record["split"] == split for record in records) for split in ("train", "validation", "test")}
    output = {
        "dataset": "Meta Amateur Drawings Dataset v1.0",
        "source": "https://github.com/facebookresearch/AnimatedDrawings",
        "license": "MIT",
        "sampling": "complete PNG entries in a 128 MiB byte-range prefix of the official uncompressed tar",
        "keypoint_names": KEYPOINT_NAMES,
        "split_policy": "sha256(file_name) modulo 100: train 0-69, validation 70-84, test 85-99",
        "counts": counts,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, separators=(",", ":")) + "\n")
    print(json.dumps({"sampled_pngs": len(sampled_names), "matched_records": len(records), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
