#!/usr/bin/env python3
"""Build a small, attributable real-drawing benchmark for WallAlive.

The geometry comes from human drawings in Google's Quick, Draw! dataset.  The
script only rasterizes the source strokes and recovers enclosed regions so the
same transparent RGBA image that WallAlive would upload can be evaluated by an
image-to-rig model.  It does not invent or relabel semantic parts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage


DATASET_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CASES = {
    "cat": {
        "key_id": "6311191834001408",
        "source_line_index": 2464,
        "topology": "quadruped",
        "expected_visible_parts": ["head", "body", "eye", "ear", "leg", "tail"],
        "body_color": "#e8a1aa",
        "line_color": "#6d2530",
    },
    "bird": {
        "key_id": "6031164919775232",
        "source_line_index": 2446,
        "topology": "winged",
        "expected_visible_parts": ["head", "body", "eye", "beak", "wing", "leg"],
        "body_color": "#8fc8ef",
        "line_color": "#173d65",
    },
    "fish": {
        "key_id": "4816885604417536",
        "source_line_index": 2495,
        "topology": "aquatic",
        "expected_visible_parts": ["body", "eye", "mouth", "fin", "tail"],
        "body_color": "#78d7cf",
        "line_color": "#174f55",
    },
    "octopus": {
        "key_id": "5973746168889344",
        "source_line_index": 2481,
        "topology": "radial",
        "expected_visible_parts": ["head", "eye", "tentacle"],
        "body_color": "#c5a2e8",
        "line_color": "#4b2867",
    },
    "tree": {
        "key_id": "5835180709249024",
        "source_line_index": 2422,
        "topology": "branched",
        "expected_visible_parts": ["trunk", "branch", "canopy"],
        "body_color": "#9dce82",
        "line_color": "#31572c",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing wallalive-qdraw-<category>.ndjson slices.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eval/varied-drawings"))
    return parser.parse_args()


def read_record(path: Path, key_id: str) -> tuple[int, dict[str, Any]]:
    with path.open(encoding="utf-8", errors="ignore") as source:
        for line_index, line in enumerate(source):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # HTTP range slices end in one deliberately incomplete line.
                continue
            if str(record.get("key_id")) == key_id:
                return line_index, record
    raise RuntimeError(f"Quick, Draw! key {key_id} was not found in {path}")


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def render_record(record: dict[str, Any], body_color: str, line_color: str) -> Image.Image:
    size = 512
    margin = 42
    strokes = record["drawing"]
    points = [(float(x), float(y)) for xs, ys in strokes for x, y in zip(xs, ys)]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)
    scale = min((size - margin * 2) / max(1.0, max_x - min_x), (size - margin * 2) / max(1.0, max_y - min_y))
    offset_x = (size - (max_x - min_x) * scale) / 2 - min_x * scale
    offset_y = (size - (max_y - min_y) * scale) / 2 - min_y * scale

    line_image = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(line_image)
    transformed_strokes: list[list[tuple[float, float]]] = []
    for xs, ys in strokes:
        transformed = [(x * scale + offset_x, y * scale + offset_y) for x, y in zip(xs, ys)]
        transformed_strokes.append(transformed)
        if len(transformed) == 1:
            x, y = transformed[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=255)
        else:
            draw.line(transformed, fill=255, width=10, joint="curve")

    line_mask = np.asarray(line_image) > 0
    object_span = max((max_x - min_x) * scale, (max_y - min_y) * scale)
    # Children's outlines are commonly split across strokes (for example, the
    # two sides of a trunk).  Add invisible caps only between nearby endpoints
    # of long, axis-aligned strokes.  The caps participate in region filling but
    # never alter the visible source strokes.
    endpoints: list[tuple[int, tuple[float, float]]] = []
    for stroke_index, transformed in enumerate(transformed_strokes):
        path_length = sum(
            float(np.hypot(b[0] - a[0], b[1] - a[1]))
            for a, b in zip(transformed, transformed[1:])
        )
        if len(transformed) > 1 and path_length >= object_span * 0.18:
            endpoints.extend([(stroke_index, transformed[0]), (stroke_index, transformed[-1])])
    candidates = []
    for left in range(len(endpoints)):
        for right in range(left + 1, len(endpoints)):
            left_stroke, a = endpoints[left]
            right_stroke, b = endpoints[right]
            if left_stroke == right_stroke:
                continue
            dx, dy = abs(b[0] - a[0]), abs(b[1] - a[1])
            distance = float(np.hypot(dx, dy))
            aligned = dx <= object_span * 0.055 or dy <= object_span * 0.055
            if aligned and distance <= object_span * 0.22:
                candidates.append((distance, left, right))
    closure_image = Image.new("L", (size, size), 0)
    closure_draw = ImageDraw.Draw(closure_image)
    consumed: set[int] = set()
    for _, left, right in sorted(candidates):
        if left in consumed or right in consumed:
            continue
        closure_draw.line([endpoints[left][1], endpoints[right][1]], fill=255, width=10)
        consumed.update((left, right))

    closure_mask = np.asarray(closure_image) > 0
    close_radius = int(np.clip(round(object_span * 0.058), 12, 26))
    grid_y, grid_x = np.ogrid[-close_radius : close_radius + 1, -close_radius : close_radius + 1]
    disk = grid_x * grid_x + grid_y * grid_y <= close_radius * close_radius
    barrier = ndimage.binary_closing(line_mask | closure_mask, structure=disk, iterations=1)
    enclosed = ndimage.binary_fill_holes(barrier)
    alpha = ndimage.binary_dilation(line_mask | enclosed, iterations=2)

    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[alpha, :3] = hex_rgb(body_color)
    rgba[line_mask, :3] = hex_rgb(line_color)
    rgba[alpha, 3] = 255
    return Image.fromarray(rgba, mode="RGBA")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_records: dict[str, Any] = {}
    manifest_cases = []

    for category, spec in CASES.items():
        source_path = args.source_dir / f"{category}.ndjson"
        if not source_path.exists():
            source_path = args.source_dir / f"wallalive-qdraw-{category}.ndjson"
        line_index, record = read_record(source_path, spec["key_id"])
        if line_index != spec["source_line_index"]:
            raise RuntimeError(f"Selected {category} drawing moved from source line {spec['source_line_index']} to {line_index}")
        if not record.get("recognized"):
            raise RuntimeError(f"Selected {category} drawing is not marked recognized")
        image_path = args.output_dir / f"{category}.png"
        render_record(record, spec["body_color"], spec["line_color"]).save(image_path)
        source_records[category] = {
            "key_id": str(record["key_id"]),
            "source_line_index": line_index,
            "recognized": bool(record["recognized"]),
            "drawing": record["drawing"],
        }
        manifest_cases.append(
            {
                "id": category,
                "input": image_path.name,
                "source_category": category,
                "source_key_id": str(record["key_id"]),
                "source_line_index": line_index,
                "source_url": DATASET_URL.format(category=category),
                "expected_topology": spec["topology"],
                "expected_visible_parts": spec["expected_visible_parts"],
                "render_palette": {
                    "body": spec["body_color"],
                    "line": spec["line_color"],
                },
            }
        )

    (args.output_dir / "source-strokes.json").write_text(
        json.dumps(source_records, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "name": "WallAlive varied human-drawing benchmark v1",
        "purpose": "End-to-end drawing-to-rigged-3D evaluation, not training data",
        "source": "Google Quick, Draw! simplified drawings",
        "source_repository": "https://github.com/googlecreativelab/quickdraw-dataset",
        "license": "CC BY 4.0",
        "license_url": LICENSE_URL,
        "transformation": "Geometry-preserving stroke rasterization, bounded invisible endpoint caps, small-gap closure, enclosed-region fill, and a fixed per-category palette",
        "selection_policy": "Every source line is >= 2400, beyond topology-v10's Quick, Draw! training/validation/test window ending before line 1960",
        "cases": manifest_cases,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "cases": len(manifest_cases)}, indent=2))


if __name__ == "__main__":
    main()
