#!/usr/bin/env python3
"""Export WallAlive's prepared closed surface as a rigged GLB without Blender.

This is the deterministic CI/evaluation exporter. The product renderer builds
the same continuous signed-distance surface and skin weights directly in Three.js.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def rgba(hex_value: str) -> list[float]:
    value = hex_value.lstrip("#")
    return [int(value[index:index + 2], 16) / 255 for index in (0, 2, 4)] + [1.0]


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


def graph_skin(vertices: np.ndarray, root: np.ndarray, raw_endpoints: list[list[float]]):
    endpoints = np.asarray(raw_endpoints, dtype=np.float32)
    if endpoints.size:
        endpoints = endpoints.reshape(-1, 3)
        distances = np.linalg.norm(endpoints[:, :2] - root[None, :2], axis=1)
        endpoints = endpoints[distances >= 0.18]
    else:
        endpoints = np.empty((0, 3), dtype=np.float32)
    if not len(endpoints):
        farthest = vertices[int(np.argmax(np.linalg.norm(vertices[:, :2] - root[None, :2], axis=1)))].copy()
        farthest[2] = 0
        endpoints = farthest[None]

    anchors = root[None] + (endpoints - root[None]) * 0.48
    xy = vertices[:, :2]
    influences = []
    for endpoint in endpoints:
        delta = endpoint[:2] - root[:2]
        length_squared = max(1e-8, float(delta @ delta))
        amount = np.clip(((xy - root[None, :2]) @ delta) / length_squared, 0, 1)
        projected = root[None, :2] + amount[:, None] * delta[None]
        distance = np.linalg.norm(xy - projected, axis=1)
        radius = max(0.055, math.sqrt(length_squared) * 0.115)
        radial = np.clip(1 - distance / (radius * 1.9), 0, 1)
        along = np.clip((amount - 0.34) / 0.28, 0, 1)
        influences.append(radial * radial * along)
    evidence = np.stack(influences, axis=1)
    best = np.argmax(evidence, axis=1)
    branch_weight = np.clip(evidence[np.arange(len(vertices)), best], 0, 0.96).astype(np.float32)
    joints = np.zeros((len(vertices), 4), dtype=np.uint16)
    weights = np.zeros((len(vertices), 4), dtype=np.float32)
    joints[:, 1] = best.astype(np.uint16) + 1
    weights[:, 0] = 1 - branch_weight
    weights[:, 1] = branch_weight
    return endpoints, anchors, joints, weights


def vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    triangles = vertices[faces]
    face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-8)
    return normals


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

    def add_accessor(self, values: np.ndarray, component_type: int, value_type: str, target: int | None = None, bounds: bool = False) -> int:
        values = np.ascontiguousarray(values)
        view = self.add_view(values.tobytes(), target)
        accessor = {
            "bufferView": view,
            "componentType": component_type,
            "count": len(values),
            "type": value_type,
        }
        if bounds:
            accessor["min"] = values.min(axis=0).astype(float).tolist()
            accessor["max"] = values.max(axis=0).astype(float).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def export(input_dir: Path, output: Path) -> dict:
    payload = np.load(input_dir / "mesh.npz")
    vertices = payload["vertices"].astype(np.float32)
    faces = payload["faces"].astype(np.uint32)
    uv = payload["uv"].astype(np.float32)
    rig = json.loads((input_dir / "rig.json").read_text())
    root = np.asarray(rig["root"], dtype=np.float32)
    endpoints, anchors, joints, weights = graph_skin(vertices, root, rig["skeleton_endpoints"])
    roles = endpoint_roles(root, endpoints, rig["topology_kind"])
    normals = vertex_normals(vertices, faces)

    front_mask = vertices[faces][:, :, 2].mean(axis=1) >= 0
    front_faces = faces[front_mask].reshape(-1)
    back_faces = faces[~front_mask].reshape(-1)
    inverse_binds = np.repeat(np.eye(4, dtype=np.float32)[None], len(endpoints) + 1, axis=0)
    for index, anchor in enumerate(anchors, start=1):
        inverse_binds[index, 3, :3] = -anchor

    binary = BinaryBuilder()
    position_accessor = binary.add_accessor(vertices, 5126, "VEC3", 34962, bounds=True)
    normal_accessor = binary.add_accessor(normals, 5126, "VEC3", 34962)
    uv_accessor = binary.add_accessor(uv, 5126, "VEC2", 34962)
    joint_accessor = binary.add_accessor(joints, 5123, "VEC4", 34962)
    weight_accessor = binary.add_accessor(weights, 5126, "VEC4", 34962)
    front_index_accessor = binary.add_accessor(front_faces, 5125, "SCALAR", 34963)
    back_index_accessor = binary.add_accessor(back_faces, 5125, "SCALAR", 34963)
    inverse_bind_accessor = binary.add_accessor(inverse_binds, 5126, "MAT4")
    image_view = binary.add_view((input_dir / "texture.png").read_bytes())

    common_attributes = {
        "POSITION": position_accessor,
        "NORMAL": normal_accessor,
        "TEXCOORD_0": uv_accessor,
        "JOINTS_0": joint_accessor,
        "WEIGHTS_0": weight_accessor,
    }
    nodes = [
        {"name": "WallAlive character", "children": [1, 2]},
        {"name": "Continuous semantic surface", "mesh": 0, "skin": 0},
        {"name": "Root", "children": list(range(3, 3 + len(endpoints)))},
    ]
    for index, (endpoint, anchor, role) in enumerate(zip(endpoints, anchors, roles, strict=True), start=1):
        nodes.append({
            "name": f"{role}.{index:02d}",
            "translation": anchor.astype(float).tolist(),
            "extras": {"semantic_kind": role, "endpoint": endpoint.astype(float).tolist()},
        })
    document = {
        "asset": {"version": "2.0", "generator": "WallAlive continuous semantic GLB exporter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{
            "name": "Watertight signed-distance character",
            "primitives": [
                {"attributes": common_attributes, "indices": front_index_accessor, "material": 0},
                {"attributes": common_attributes, "indices": back_index_accessor, "material": 1},
            ],
            "extras": {
                "topology_kind": rig["topology_kind"],
                "topology_confidence": rig["topology_confidence"],
                "semantic_kinds": rig["semantic_kinds"],
                "back_prior": rig["back_prior"],
            },
        }],
        "skins": [{
            "name": "Variable topology graph skin",
            "inverseBindMatrices": inverse_bind_accessor,
            "skeleton": 2,
            "joints": list(range(2, 3 + len(endpoints))),
        }],
        "materials": [
            {"name": "Authored front artwork", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 0.65}},
            {"name": "Symmetric inferred back", "pbrMetallicRoughness": {"baseColorFactor": rgba(rig["body_color"]), "metallicFactor": 0, "roughnessFactor": 0.72}},
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
        "textures": [{"sampler": 0, "source": 0}],
        "images": [{"name": "Exact isolated drawing", "bufferView": image_view, "mimeType": "image/png"}],
        "accessors": binary.accessors,
        "bufferViews": binary.views,
        "buffers": [{"byteLength": len(binary.data)}],
        "extras": {"viewable_degrees": 360, "reconstruction": "signed-distance lens marching cubes + learned variable graph skinning"},
    }
    json_bytes = json.dumps(document, separators=(",", ":")).encode()
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    while len(binary.data) % 4:
        binary.data.append(0)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(binary.data)
    glb = bytearray(struct.pack("<III", 0x46546C67, 2, total_length))
    glb.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    glb.extend(json_bytes)
    glb.extend(struct.pack("<II", len(binary.data), 0x004E4942))
    glb.extend(binary.data)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(glb)
    return {
        "output": str(output),
        "bytes": len(glb),
        "vertices": len(vertices),
        "triangles": len(faces),
        "bones": len(endpoints) + 1,
        "topology_kind": rig["topology_kind"],
        "graph_bones": [node["name"] for node in nodes[3:]],
        "back_prior": rig["back_prior"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.input_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
