#!/usr/bin/env python3
"""Prepare a real single drawing for watertight semantic 3D QA.

The browser remains the product path. This offline evaluator mirrors its math
with a higher-resolution signed-distance volume, runs the checked-in face ONNX
ensemble, and emits geometry/semantic evidence that Blender can export and
turntable-render. It is intended for regression evidence on difficult photos.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import measure, morphology

from evaluate_single_drawing import (
    FACE_SIZE,
    PARTS,
    blend_face_logits,
    locate_head,
    model_rect_to_source,
    run,
    sigmoid,
    square_fit,
    tensor,
)


TOPOLOGY_CLASSES = ("biped", "quadruped", "winged", "aquatic", "radial", "branched", "machine", "chain")
TOPOLOGY_SEMANTICS = {
    "biped": ("body", "head", "ear", "arm", "hand", "leg", "foot"),
    "quadruped": ("body", "head", "ear", "leg", "foot", "tail"),
    "winged": ("body", "head", "beak", "wing", "leg", "foot", "tail"),
    "aquatic": ("body", "head", "fin", "tail"),
    "radial": ("body", "head", "tentacle"),
    "branched": ("body", "trunk", "branch", "canopy"),
    "machine": ("body", "linkage"),
    "chain": ("body", "segment"),
}
FACE_KINDS_BY_TOPOLOGY = {
    "biped": frozenset(("eye", "pupil", "cheek", "mouth", "ear", "marking")),
    "quadruped": frozenset(("eye", "pupil", "cheek", "mouth", "ear", "marking")),
    "winged": frozenset(("eye", "pupil", "cheek", "mouth", "marking")),
    "aquatic": frozenset(("eye", "pupil", "cheek", "mouth", "marking")),
    "radial": frozenset(("eye", "pupil", "cheek", "mouth", "marking")),
    "branched": frozenset(),
    "machine": frozenset(),
    "chain": frozenset(),
}


def hex_color(values: np.ndarray) -> str:
    channels = np.clip(np.round(values), 0, 255).astype(np.uint8)
    return "#" + "".join(f"{channel:02x}" for channel in channels[:3])


def drawing_palette(texture_rgb: np.ndarray, mask: np.ndarray) -> tuple[str, str]:
    samples = texture_rgb[mask].astype(np.float32)
    maximum = samples.max(axis=1)
    minimum = samples.min(axis=1)
    chroma = maximum - minimum
    brightness = samples.mean(axis=1)
    if float(np.percentile(chroma, 90)) >= 8:
        ink_score = chroma * 2 + (255 - brightness) * 0.2
        ink_samples = samples[ink_score >= np.percentile(ink_score, 95)]
        ink = np.median(ink_samples, axis=0)
        average = float(ink.mean())
        ink = np.clip(average + (ink - average) * 2.25, 0, 255)
    else:
        ink = np.median(samples[brightness <= np.percentile(brightness, 8)], axis=0)
    if float(np.median(chroma)) >= 15:
        body = np.median(samples, axis=0)
    else:
        quiet = samples[(brightness >= np.percentile(brightness, 62)) & (chroma <= np.percentile(chroma, 72))]
        body = np.median(quiet if len(quiet) else samples, axis=0)
        body = body * 0.42 + 255 * 0.58
    return hex_color(body), hex_color(ink)


def estimate_ink(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3].astype(np.float32)
    height, width = rgb.shape[:2]
    border_width = max(3, round(min(width, height) * 0.045))
    border = np.concatenate((
        rgb[:border_width].reshape(-1, 3),
        rgb[-border_width:].reshape(-1, 3),
        rgb[:, :border_width].reshape(-1, 3),
        rgb[:, -border_width:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=-1)
    maximum = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    chroma = maximum - minimum
    background_chroma = float(background.max() - background.min())
    darkness = background.mean() - rgb.mean(axis=-1)
    colored = (chroma >= max(5.0, background_chroma + 1.5)) & (distance >= 6.0)
    dark = (darkness >= 24.0) & (distance >= 22.0)
    # Prefer hue/chroma isolation when the photo contains enough colored ink;
    # mixing graph-paper darkness into that mask can cut the character apart.
    if int(colored.sum()) >= image.shape[0] * image.shape[1] * 0.002:
        return colored
    return colored | dark


def choose_character_mask(ink: np.ndarray) -> np.ndarray:
    # Phone compression and faint colored pencil routinely break a closed
    # outline by several pixels. Bridge those short gaps before the exterior
    # flood fill, while the component scorer still rejects paper/text clutter.
    radius = max(3, round(min(ink.shape) * 0.018))
    connected = morphology.closing(ink, morphology.disk(radius))
    connected = morphology.dilation(connected, morphology.disk(1))
    filled = ndimage.binary_fill_holes(connected)
    # Score enclosed regions rather than the connected ink itself. A character
    # outline may touch nearby writing through graph-paper compression, while
    # its interior is still a distinct bounded region.
    enclosed = filled & ~connected
    labels = measure.label(enclosed, connectivity=2)
    height, width = ink.shape
    center = np.asarray([height * 0.52, width * 0.5])
    best_label = 0
    best_score = -np.inf
    for region in measure.regionprops(labels):
        min_y, min_x, max_y, max_x = region.bbox
        if region.area < height * width * 0.012:
            continue
        span = max(max_x - min_x, max_y - min_y)
        border_touch = min_x <= 1 or min_y <= 1 or max_x >= width - 1 or max_y >= height - 1
        distance = np.linalg.norm(np.asarray(region.centroid) - center) / np.hypot(height, width)
        density = region.area / max(1, (max_x - min_x) * (max_y - min_y))
        score = np.log1p(region.area) + density * 2.2 - distance * 5.5 - border_touch * 2.0 - (span > max(height, width) * 0.88) * 2.5
        if score > best_score:
            best_label = region.label
            best_score = score
    if not best_label:
        raise ValueError("No enclosed central character could be isolated")
    mask = morphology.dilation(labels == best_label, morphology.disk(max(1, radius // 2)))
    mask = morphology.closing(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, max_size=max(63, round(mask.size * 0.003) - 1))
    return mask


def normalized_contour(mask: np.ndarray) -> np.ndarray:
    contours = measure.find_contours(mask.astype(np.float32), 0.5, fully_connected="high")
    if not contours:
        raise ValueError("The isolated character has no closed contour")
    contour = max(contours, key=len)
    stride = max(1, len(contour) // 180)
    contour = contour[::stride]
    height, width = mask.shape
    return np.column_stack(((contour[:, 1] / (width - 1) - 0.5) * 2, (0.5 - contour[:, 0] / (height - 1)) * 2))


def semantic_features(
    source: Image.Image,
    head_box: tuple[int, int, int, int],
    probabilities: np.ndarray,
    thresholds: list[float],
    source_box: tuple[int, int, int, int],
    canvas_offset: tuple[float, float],
    canvas_scale: float,
    canvas_size: int,
    local_depth: np.ndarray,
) -> list[dict[str, object]]:
    head_width = max(1, head_box[2] - head_box[0])
    head_height = max(1, head_box[3] - head_box[1])
    face_scale = min(FACE_SIZE / head_width, FACE_SIZE / head_height)
    face_offset_x = (FACE_SIZE - head_width * face_scale) / 2
    face_offset_y = (FACE_SIZE - head_height * face_scale) / 2
    source_left, source_top, _, _ = source_box
    canvas_offset_x, canvas_offset_y = canvas_offset
    features: list[dict[str, object]] = []
    for channel, kind in enumerate(PARTS):
        mask = probabilities[channel] >= thresholds[channel]
        labels = measure.label(mask, connectivity=2)
        minimum = 3 if kind in ("eye", "mouth") else 4
        candidates = sorted(measure.regionprops(labels), key=lambda region: region.area, reverse=True)
        limit = 3 if kind == "mouth" else 6
        accepted = 0
        for region in candidates:
            if region.area < minimum or region.area > mask.size * 0.34:
                continue
            component = labels == region.label
            contours = measure.find_contours(component.astype(np.float32), 0.5)
            if not contours:
                continue
            face_contour = max(contours, key=len)
            source_x = head_box[0] + (face_contour[:, 1] - face_offset_x) / face_scale
            source_y = head_box[1] + (face_contour[:, 0] - face_offset_y) / face_scale
            mesh_x = canvas_offset_x + (source_x - source_left) * canvas_scale
            mesh_y = canvas_offset_y + (source_y - source_top) * canvas_scale
            normalized = np.column_stack(((mesh_x / (canvas_size - 1) - 0.5) * 2, (0.5 - mesh_y / (canvas_size - 1)) * 2))
            center_x = float(normalized[:, 0].mean())
            center_y = float(normalized[:, 1].mean())
            feature_width = float(normalized[:, 0].max() - normalized[:, 0].min())
            feature_height = float(normalized[:, 1].max() - normalized[:, 1].min())
            if any(existing["kind"] == kind and math.hypot(
                float(existing["center"][0]) - center_x,
                float(existing["center"][1]) - center_y,
            ) <= max(feature_width, feature_height, float(existing["size"][0]), float(existing["size"][1])) * 0.72 for existing in features):
                continue
            pixel_x = int(np.clip(round((center_x / 2 + 0.5) * (canvas_size - 1)), 0, canvas_size - 1))
            pixel_y = int(np.clip(round((0.5 - center_y / 2) * (canvas_size - 1)), 0, canvas_size - 1))
            features.append({
                "kind": kind,
                "confidence": round(float(probabilities[channel][component].mean()), 4),
                "center": [center_x, center_y, float(local_depth[pixel_y, pixel_x])],
                "size": [feature_width, feature_height],
                "outline": normalized[::max(1, len(normalized) // 64)].round(6).tolist(),
            })
            accepted += 1
            if accepted >= limit:
                break
    return features


def topology_endpoints(
    fields: np.ndarray,
    content_rect: tuple[float, float, float, float],
    source: Image.Image,
    source_box: tuple[int, int, int, int],
    canvas_offset: tuple[float, float],
    canvas_scale: float,
    canvas_size: int,
) -> list[list[float]]:
    probabilities = sigmoid(fields[0, 2])
    labels = measure.label(probabilities >= 0.42, connectivity=2)
    offset_x, offset_y, draw_width, draw_height = content_rect
    source_left, source_top, _, _ = source_box
    canvas_offset_x, canvas_offset_y = canvas_offset
    endpoints: list[list[float]] = []
    for region in sorted(measure.regionprops(labels), key=lambda item: float(probabilities[labels == item.label].max()), reverse=True):
        if region.area > probabilities.size * 0.06:
            continue
        component = labels == region.label
        ys, xs = np.nonzero(component)
        weights = probabilities[component]
        model_x = (float(np.average(xs, weights=weights)) + 0.5) * 2
        model_y = (float(np.average(ys, weights=weights)) + 0.5) * 2
        source_x = (model_x - offset_x) / draw_width * source.width
        source_y = (model_y - offset_y) / draw_height * source.height
        mesh_x = canvas_offset_x + (source_x - source_left) * canvas_scale
        mesh_y = canvas_offset_y + (source_y - source_top) * canvas_scale
        point = [(mesh_x / (canvas_size - 1) - 0.5) * 2, (0.5 - mesh_y / (canvas_size - 1)) * 2, 0.0]
        if abs(point[0]) > 1.08 or abs(point[1]) > 1.08:
            continue
        if any(math.hypot(point[0] - existing[0], point[1] - existing[1]) < 0.09 for existing in endpoints):
            continue
        endpoints.append([round(float(point[0]), 6), round(float(point[1]), 6), 0.0])
        if len(endpoints) >= 12:
            break
    return endpoints


def skeleton_endpoints(mask: np.ndarray, root_pixel: tuple[int, int]) -> list[list[float]]:
    skeleton = morphology.skeletonize(mask)
    neighbors = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant") - skeleton
    ys, xs = np.nonzero(skeleton & (neighbors == 1))
    root_y, root_x = root_pixel
    ranked = sorted(zip(xs, ys, strict=True), key=lambda point: np.hypot(point[0] - root_x, point[1] - root_y), reverse=True)
    selected: list[tuple[int, int]] = []
    for point in ranked:
        if np.hypot(point[0] - root_x, point[1] - root_y) < min(mask.shape) * 0.18:
            continue
        if any(np.hypot(point[0] - other[0], point[1] - other[1]) < min(mask.shape) * 0.1 for other in selected):
            continue
        selected.append(point)
        if len(selected) >= 8:
            break
    height, width = mask.shape
    return [[round((x / (width - 1) - 0.5) * 2, 6), round((0.5 - y / (height - 1)) * 2, 6), 0.0] for x, y in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    parser.add_argument("--models", type=Path, default=Path("public/models"))
    parser.add_argument("--ensemble", type=Path, default=Path("public/models/wallalive-face-ensemble-v4.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=128)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    source_rgba = Image.open(args.image).convert("RGBA")
    if args.crop:
        source_rgba = source_rgba.crop(tuple(args.crop))
    alpha = np.asarray(source_rgba)[..., 3]
    source = Image.alpha_composite(Image.new("RGBA", source_rgba.size, "white"), source_rgba).convert("RGB")
    rgb = np.asarray(source)
    ink = estimate_ink(rgb)
    alpha_fraction = float((alpha > 32).mean())
    source_mask = alpha > 32 if 0.01 < alpha_fraction < 0.96 else choose_character_mask(ink)
    regions = measure.regionprops(source_mask.astype(np.uint8))
    min_y, min_x, max_y, max_x = regions[0].bbox
    padding = max(3, round(max(max_x - min_x, max_y - min_y) * 0.08))
    source_box = (
        max(0, min_x - padding), max(0, min_y - padding),
        min(source.width, max_x + padding), min(source.height, max_y + padding),
    )
    cropped_image = source.crop(source_box)
    cropped_mask = Image.fromarray((source_mask[source_box[1]:source_box[3], source_box[0]:source_box[2]] * 255).astype(np.uint8))
    scale = min((args.resolution - 12) / cropped_image.width, (args.resolution - 12) / cropped_image.height)
    draw_width = max(1, round(cropped_image.width * scale))
    draw_height = max(1, round(cropped_image.height * scale))
    offset_x = (args.resolution - draw_width) / 2
    offset_y = (args.resolution - draw_height) / 2
    texture = Image.new("RGBA", (args.resolution, args.resolution), (0, 0, 0, 0))
    resized_image = cropped_image.resize((draw_width, draw_height), Image.Resampling.LANCZOS).convert("RGBA")
    resized_mask = cropped_mask.resize((draw_width, draw_height), Image.Resampling.NEAREST)
    resized_image.putalpha(resized_mask)
    texture.paste(resized_image, (round(offset_x), round(offset_y)), resized_image)
    mask_canvas = Image.new("L", (args.resolution, args.resolution), 0)
    mask_canvas.paste(resized_mask, (round(offset_x), round(offset_y)))
    mask = np.asarray(mask_canvas) > 0
    mask = morphology.closing(mask, morphology.disk(1))

    inside_distance = ndimage.distance_transform_edt(mask)
    outside_distance = ndimage.distance_transform_edt(~mask)
    signed_distance = inside_distance - outside_distance
    maximum_distance = max(1.0, float(inside_distance.max()))
    maximum_depth = args.resolution * 0.17
    normalized_distance = np.clip(inside_distance / maximum_distance, 0, 1)
    # A raw distance transform has sharp medial-axis ridges that render like a
    # cut gemstone. Blur only the depth prior (never the source mask), then use
    # a sinusoidal lens profile for a rounded clay/plush surface.
    smooth_distance = ndimage.gaussian_filter(normalized_distance, sigma=max(1.0, args.resolution * 0.012))
    smooth_distance /= max(1e-6, float(smooth_distance.max()))
    local_depth_pixels = maximum_depth * np.sin(np.clip(smooth_distance, 0, 1) * math.pi / 2) ** 0.72
    z_resolution = 64
    z_coordinates = np.linspace(-maximum_depth, maximum_depth, z_resolution, dtype=np.float32)
    field = np.minimum(signed_distance[None], local_depth_pixels[None] - np.abs(z_coordinates[:, None, None])).astype(np.float32)
    vertices_zyx, faces, _, _ = measure.marching_cubes(
        field,
        level=0,
        spacing=(2 * maximum_depth / (z_resolution - 1), 1.0, 1.0),
        allow_degenerate=False,
    )
    z = vertices_zyx[:, 0] - maximum_depth
    y = vertices_zyx[:, 1]
    x = vertices_zyx[:, 2]
    vertices = np.column_stack(((x / (args.resolution - 1) - 0.5) * 2, (0.5 - y / (args.resolution - 1)) * 2, z / ((args.resolution - 1) / 2))).astype(np.float32)
    uv = np.column_stack((x / (args.resolution - 1), 1 - y / (args.resolution - 1))).astype(np.float32)
    local_depth = (local_depth_pixels / ((args.resolution - 1) / 2)).astype(np.float32)

    prepared, content_rect = square_fit(source, 96)
    _, coarse_logits = run(args.models / "wallalive-parts-v3.onnx", tensor(prepared))
    topology_fields, topology_logits = run(args.models / "wallalive-topology-v10.onnx", tensor(prepared))
    shifted_topology_logits = topology_logits[0] - topology_logits[0].max()
    topology_probabilities = np.exp(shifted_topology_logits) / np.exp(shifted_topology_logits).sum()
    topology_index = int(np.argmax(topology_probabilities))
    topology_kind = TOPOLOGY_CLASSES[topology_index]
    topology_confidence = float(topology_probabilities[topology_index])
    head_rect = locate_head(coarse_logits, content_rect)
    head_box = model_rect_to_source(head_rect, content_rect, source)
    head = source.crop(head_box)
    face_v3, _ = square_fit(head, 96)
    face_v4, _ = square_fit(head, FACE_SIZE)
    v3 = run(args.models / "wallalive-face-v3.onnx", tensor(face_v3))[0]
    v4 = run(args.models / "wallalive-face-v4.onnx", tensor(face_v4))[0]
    ensemble = json.loads(args.ensemble.read_text())
    weights = [float(ensemble["blend_weight_v4"][part]) for part in PARTS]
    thresholds = [float(ensemble["thresholds"][part]) for part in PARTS]
    probabilities = sigmoid(blend_face_logits(v3, v4, weights))[0]
    features = semantic_features(
        source, head_box, probabilities, thresholds, source_box,
        (offset_x, offset_y), scale, args.resolution, local_depth,
    )
    allowed_face_kinds = FACE_KINDS_BY_TOPOLOGY[topology_kind]
    features = [feature for feature in features if str(feature["kind"]) in allowed_face_kinds]

    root_y, root_x = np.unravel_index(int(inside_distance.argmax()), inside_distance.shape)
    endpoints = topology_endpoints(
        topology_fields, content_rect, source, source_box,
        (offset_x, offset_y), scale, args.resolution,
    )
    for endpoint in skeleton_endpoints(mask, (root_y, root_x)):
        if not any(math.hypot(endpoint[0] - existing[0], endpoint[1] - existing[1]) < 0.1 for existing in endpoints):
            endpoints.append(endpoint)
    texture_rgb = np.asarray(texture.convert("RGB"))
    body_color, line_color = drawing_palette(texture_rgb, mask)

    np.savez_compressed(args.output / "mesh.npz", vertices=vertices, faces=faces.astype(np.int32), uv=uv)
    background_rgb = tuple(int(body_color[index:index + 2], 16) for index in (1, 3, 5))
    final_texture = Image.new("RGB", texture.size, background_rgb)
    final_texture.paste(texture.convert("RGB"), (0, 0), texture.getchannel("A"))
    final_texture.save(args.output / "texture.png")
    contour = normalized_contour(mask)
    report = {
        "source_image": str(args.image),
        "requested_crop": args.crop,
        "source_size": list(source.size),
        "source_character_box": list(source_box),
        "mesh": {"vertices": len(vertices), "triangles": len(faces), "closed_volume_method": "signed-distance lens marching cubes", "depth_ratio": round(float(np.ptp(vertices[:, 2]) / max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))), 4)},
        "body_color": body_color,
        "line_color": line_color,
        "contour": contour.round(6).tolist(),
        "root": [round((root_x / (args.resolution - 1) - 0.5) * 2, 6), round((0.5 - root_y / (args.resolution - 1)) * 2, 6), 0.0],
        "skeleton_endpoints": endpoints,
        "topology_kind": topology_kind,
        "topology_confidence": round(topology_confidence, 5),
        "topology_class_probabilities": [round(float(value), 5) for value in topology_probabilities],
        "semantic_features": features,
        "semantic_kinds": sorted(set([str(feature["kind"]) for feature in features] + list(TOPOLOGY_SEMANTICS[topology_kind]))),
        "back_prior": "symmetric filled volume; no copied face texture",
    }
    (args.output / "rig.json").write_text(json.dumps(report, indent=2) + "\n")

    diagnostic = texture.convert("RGB").resize((512, 512), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(diagnostic)
    for feature in features:
        center_x, center_y, _ = feature["center"]
        px = (center_x / 2 + 0.5) * 512
        py = (0.5 - center_y / 2) * 512
        draw.ellipse((px - 6, py - 6, px + 6, py + 6), outline="#b8ff39", width=3)
        draw.text((px + 8, py - 7), str(feature["kind"]), fill="#17332f", stroke_width=2, stroke_fill="white")
    diagnostic.save(args.output / "diagnostic.png")
    print(json.dumps({key: report[key] for key in ("mesh", "topology_kind", "topology_confidence", "body_color", "line_color", "semantic_kinds", "back_prior")}, indent=2))


if __name__ == "__main__":
    main()
