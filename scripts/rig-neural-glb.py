#!/usr/bin/env python3
"""Attach WallAlive's variable drawing graph to a colored static neural GLB."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.smoothing import filter_taubin


def srgb_to_linear_u8(colors: np.ndarray) -> np.ndarray:
    """Encode authored sRGB colors as the linear vertex colors glTF expects."""
    values = np.clip(colors.astype(np.float32) / 255.0, 0, 1)
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return np.rint(linear * 255).astype(np.uint8)


def project_front_colors(vertices: np.ndarray, texture_path: Path) -> np.ndarray:
    """Project approved front artwork while keeping sides/back feature-free."""
    image = np.asarray(Image.open(texture_path).convert("RGBA"))
    foreground = image[..., 3] > 32
    rows, columns = np.where(foreground)
    if not len(rows):
        raise ValueError(f"Front texture has no visible foreground: {texture_path}")
    left, right = int(columns.min()), int(columns.max())
    top, bottom = int(rows.min()), int(rows.max())
    horizontal = np.clip((vertices[:, 1] - vertices[:, 1].min()) / max(1e-8, float(np.ptp(vertices[:, 1]))), 0, 1)
    vertical = np.clip((vertices[:, 0].max() - vertices[:, 0]) / max(1e-8, float(np.ptp(vertices[:, 0]))), 0, 1)
    sample_x = np.rint(left + horizontal * (right - left)).astype(np.int32)
    sample_y = np.rint(top + vertical * (bottom - top)).astype(np.int32)
    sampled = image[sample_y, sample_x, :3].astype(np.float32)

    visible_pixels = image[..., :3][foreground].astype(np.float32)
    chroma = np.ptp(visible_pixels, axis=1)
    neutral = visible_pixels[(chroma < 22) & (visible_pixels.mean(axis=1) > 120)]
    base = np.median(neutral if len(neutral) else visible_pixels, axis=0)
    center_depth = float(np.median(vertices[:, 2]))
    front_range = max(1e-8, float(vertices[:, 2].max()) - center_depth)
    front_weight = np.clip((vertices[:, 2] - center_depth) / front_range, 0, 1) ** 0.55
    colors = base[None] * (1 - front_weight[:, None]) + sampled * front_weight[:, None]
    return np.column_stack((np.rint(colors).astype(np.uint8), np.full(len(colors), 255, dtype=np.uint8)))


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
    return normals


def endpoint_roles(root: np.ndarray, endpoints: np.ndarray, topology: str) -> list[str]:
    if topology == "radial":
        return ["tentacle"] * len(endpoints)
    if topology == "branched":
        trunk = int(np.argmin(endpoints[:, 1])) if len(endpoints) else -1
        return ["trunk" if index == trunk else "branch" for index in range(len(endpoints))]
    if topology == "machine":
        return ["linkage"] * len(endpoints)
    if topology == "chain":
        return ["segment"] * len(endpoints)
    if topology == "aquatic":
        tail = int(np.argmax(np.abs(endpoints[:, 0] - root[0]))) if len(endpoints) else -1
        return ["tail" if index == tail else "fin" for index in range(len(endpoints))]
    if topology == "winged":
        return ["leg" if endpoint[1] < root[1] - 0.24 else "wing" for endpoint in endpoints]
    if topology == "quadruped":
        farthest = int(np.argmax(np.abs(endpoints[:, 0] - root[0]))) if len(endpoints) else -1
        return [
            "ear" if endpoint[1] > root[1] + 0.22
            else "tail" if index == farthest and endpoint[1] > root[1] - 0.34
            else "leg"
            for index, endpoint in enumerate(endpoints)
        ]
    return [
        "ear" if endpoint[1] > root[1] + 0.26
        else "leg" if endpoint[1] < root[1] - 0.3
        else "arm"
        for endpoint in endpoints
    ]


def select_biped_endpoints(root: np.ndarray, endpoints: np.ndarray) -> tuple[np.ndarray, list[str]]:
    selected: list[np.ndarray] = []
    roles: list[str] = []
    used: set[int] = set()

    def take_sides(candidates: list[int], role: str) -> None:
        if not candidates:
            return
        ordered = sorted(candidates, key=lambda index: float(endpoints[index, 0]))
        choices = [ordered[0]]
        if ordered[-1] != ordered[0] and endpoints[ordered[-1], 0] - endpoints[ordered[0], 0] >= 0.16:
            choices.append(ordered[-1])
        for index in choices:
            if index in used:
                continue
            used.add(index)
            selected.append(endpoints[index])
            roles.append(role)

    take_sides([index for index, point in enumerate(endpoints) if point[1] > root[1] + 0.23 and abs(point[0] - root[0]) > 0.12], "ear")
    take_sides([index for index, point in enumerate(endpoints) if point[1] < root[1] - 0.62], "leg")
    take_sides([
        index for index, point in enumerate(endpoints)
        if root[1] - 0.58 <= point[1] <= root[1] + 0.12 and abs(point[0] - root[0]) > 0.24
    ], "arm")
    if not selected:
        return endpoints, endpoint_roles(root, endpoints, "biped")
    return np.asarray(selected, dtype=np.float32), roles


class BinaryBuilder:
    def __init__(self):
        self.data = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add_view(self, payload: bytes, target: int | None = None) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(payload)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def add_accessor(self, values: np.ndarray, component_type: int, kind: str, target: int | None = None, bounds: bool = False) -> int:
        values = np.ascontiguousarray(values)
        view = self.add_view(values.tobytes(), target)
        accessor = {"bufferView": view, "componentType": component_type, "count": len(values), "type": kind}
        if bounds:
            accessor["min"] = values.min(axis=0).astype(float).tolist()
            accessor["max"] = values.max(axis=0).astype(float).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def project_vertices(vertices: np.ndarray, front_axis: int, horizontal_axis: int, up_axis: int) -> np.ndarray:
    projected = vertices[:, [horizontal_axis, up_axis]].copy()
    minimum = projected.min(axis=0)
    span = np.maximum(projected.max(axis=0) - minimum, 1e-6)
    return (projected - minimum) / span * 2 - 1


def map_drawing_point(point: np.ndarray, bounds: np.ndarray, front_axis: int, horizontal_axis: int, up_axis: int) -> np.ndarray:
    mapped = bounds.mean(axis=0)
    mapped[horizontal_axis] = bounds[0, horizontal_axis] + (point[0] * 0.5 + 0.5) * (bounds[1, horizontal_axis] - bounds[0, horizontal_axis])
    mapped[up_axis] = bounds[0, up_axis] + (point[1] * 0.5 + 0.5) * (bounds[1, up_axis] - bounds[0, up_axis])
    mapped[front_axis] = bounds[:, front_axis].mean()
    return mapped.astype(np.float32)


def skin_graph(projected: np.ndarray, root: np.ndarray, endpoints: np.ndarray):
    valid = np.linalg.norm(endpoints - root[None], axis=1) >= 0.16
    endpoints = endpoints[valid]
    if not len(endpoints):
        endpoints = np.asarray([[root[0], -0.82]], dtype=np.float32)
    evidence = []
    for endpoint in endpoints:
        delta = endpoint - root
        length_squared = max(1e-8, float(delta @ delta))
        amount = np.clip(((projected - root[None]) @ delta) / length_squared, 0, 1)
        nearest = root[None] + amount[:, None] * delta[None]
        distance = np.linalg.norm(projected - nearest, axis=1)
        radius = max(0.07, math.sqrt(length_squared) * 0.12)
        radial = np.clip(1 - distance / (radius * 2.3), 0, 1)
        along = np.clip((amount - 0.24) / 0.28, 0, 1)
        evidence.append(radial * radial * along)
    scores = np.stack(evidence, axis=1)
    best = np.argmax(scores, axis=1)
    branch_weight = np.clip(scores[np.arange(len(projected)), best], 0, 0.96).astype(np.float32)
    joints = np.zeros((len(projected), 4), dtype=np.uint16)
    weights = np.zeros((len(projected), 4), dtype=np.float32)
    joints[:, 1] = best.astype(np.uint16) + 1
    weights[:, 0] = 1 - branch_weight
    weights[:, 1] = branch_weight
    return endpoints, joints, weights


def export(input_glb: Path, rig_path: Path, output: Path, front_texture: Path | None = None, smooth_iterations: int = 0) -> dict:
    source = trimesh.load(input_glb, force="mesh")
    if not isinstance(source, trimesh.Trimesh):
        raise TypeError(f"Expected a mesh, got {type(source).__name__}")
    if smooth_iterations > 0:
        filter_taubin(source, lamb=0.43, nu=0.45, iterations=smooth_iterations)
    source_vertices = np.asarray(source.vertices, dtype=np.float32)
    # TripoSR's MarchingCubeHelper emits [grid-Z, grid-Y, grid-X], so its GLB
    # accessor is [up, right, camera-depth]. Convert to Three.js [right, up,
    # depth] and put the authored front on -Z. prepareNeuralCharacter applies
    # the same 180-degree display turn used by live AniGen assets, bringing this
    # front back toward the camera. The axis swap and Z reflection cancel, so
    # the source winding remains outward.
    vertices = np.column_stack((source_vertices[:, 1], source_vertices[:, 0], -source_vertices[:, 2])).astype(np.float32)
    faces = np.asarray(source.faces, dtype=np.uint32).copy()
    colors = project_front_colors(source_vertices, front_texture) if front_texture else np.asarray(source.visual.vertex_colors, dtype=np.uint8)
    if colors.shape[1] == 3:
        colors = np.column_stack((colors, np.full(len(colors), 255, dtype=np.uint8)))
    colors[:, :3] = srgb_to_linear_u8(colors[:, :3])
    normals = vertex_normals(vertices, faces)
    rig = json.loads(rig_path.read_text())
    root_2d = np.asarray(rig["root"][:2], dtype=np.float32)
    endpoints_2d = np.asarray(rig["skeleton_endpoints"], dtype=np.float32)[:, :2]
    if rig["topology_kind"] == "biped":
        endpoints_2d, roles = select_biped_endpoints(root_2d, endpoints_2d)
    else:
        roles = endpoint_roles(root_2d, endpoints_2d, rig["topology_kind"])

    front_axis = 2
    horizontal_axis = 0
    up_axis = 1
    projected = project_vertices(vertices, front_axis, horizontal_axis, up_axis)
    endpoints_2d, joints, weights = skin_graph(projected, root_2d, endpoints_2d)
    if len(roles) != len(endpoints_2d):
        roles = endpoint_roles(root_2d, endpoints_2d, rig["topology_kind"])
    anchors_2d = root_2d[None] + (endpoints_2d - root_2d[None]) * 0.48
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0))).astype(np.float32)
    root_3d = map_drawing_point(root_2d, bounds, front_axis, horizontal_axis, up_axis)
    anchors_3d = np.stack([map_drawing_point(point, bounds, front_axis, horizontal_axis, up_axis) for point in anchors_2d])
    inverse_binds = np.repeat(np.eye(4, dtype=np.float32)[None], len(endpoints_2d) + 1, axis=0)
    inverse_binds[0, 3, :3] = -root_3d
    for index, anchor in enumerate(anchors_3d, start=1):
        inverse_binds[index, 3, :3] = -anchor

    binary = BinaryBuilder()
    position = binary.add_accessor(vertices, 5126, "VEC3", 34962, bounds=True)
    normal = binary.add_accessor(normals, 5126, "VEC3", 34962)
    color = binary.add_accessor(colors, 5121, "VEC4", 34962)
    binary.accessors[color]["normalized"] = True
    joint = binary.add_accessor(joints, 5123, "VEC4", 34962)
    weight = binary.add_accessor(weights, 5126, "VEC4", 34962)
    indices = binary.add_accessor(faces.reshape(-1), 5125, "SCALAR", 34963)
    inverse_bind = binary.add_accessor(inverse_binds, 5126, "MAT4")
    nodes = [
        {"name": "WallAlive neural character", "children": [1, 2]},
        {"name": "Neural watertight surface", "mesh": 0, "skin": 0},
        {"name": "Root", "translation": root_3d.astype(float).tolist(), "children": list(range(3, 3 + len(endpoints_2d)))},
    ]
    for index, (endpoint, anchor, role) in enumerate(zip(endpoints_2d, anchors_3d, roles, strict=True), start=1):
        nodes.append({
            "name": f"{role}.{index:02d}",
            "translation": (anchor - root_3d).astype(float).tolist(),
            "extras": {"semantic_kind": role, "drawing_endpoint": endpoint.astype(float).tolist()},
        })
    document = {
        "asset": {"version": "2.0", "generator": "WallAlive neural mesh variable-graph rigger"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{
            "name": "Colored neural reconstruction",
            "primitives": [{
                "attributes": {"POSITION": position, "NORMAL": normal, "COLOR_0": color, "JOINTS_0": joint, "WEIGHTS_0": weight},
                "indices": indices,
                "material": 0,
            }],
            "extras": {"topology_kind": rig["topology_kind"], "depth_source": "neural full-volume reconstruction"},
        }],
        "skins": [{"name": "Variable drawing graph skin", "inverseBindMatrices": inverse_bind, "skeleton": 2, "joints": list(range(2, 3 + len(endpoints_2d)))}],
        "materials": [{"name": "Neural vertex colors", "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.72}}],
        "accessors": binary.accessors,
        "bufferViews": binary.views,
        "buffers": [{"byteLength": len(binary.data)}],
        "extras": {
            "viewable_degrees": 360,
            "rig_source": "WallAlive learned variable topology graph",
            "static_mesh_source": input_glb.name,
            "front_color_source": front_texture.name if front_texture else "source neural vertex colors",
            "smooth_iterations": smooth_iterations,
            "hidden_surface_marks": "none projected; sides and back use the inferred neutral body palette",
        },
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    while len(binary.data) % 4:
        binary.data.append(0)
    total = 12 + 8 + len(json_bytes) + 8 + len(binary.data)
    payload = bytearray(struct.pack("<III", 0x46546C67, 2, total))
    payload.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    payload.extend(json_bytes)
    payload.extend(struct.pack("<II", len(binary.data), 0x004E4942))
    payload.extend(binary.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "output": str(output),
        "vertices": len(vertices),
        "triangles": len(faces),
        "bones": len(endpoints_2d) + 1,
        "topology_kind": rig["topology_kind"],
        "semantic_bones": [node["name"] for node in nodes[3:]],
        "source_watertight": bool(source.is_watertight),
        "front_texture": str(front_texture) if front_texture else None,
        "smooth_iterations": smooth_iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--front-texture", type=Path)
    parser.add_argument("--smooth-iterations", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(export(args.input, args.rig, args.output, args.front_texture, args.smooth_iterations), indent=2))


if __name__ == "__main__":
    main()
