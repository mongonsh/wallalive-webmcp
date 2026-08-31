#!/usr/bin/env python3
"""Build a colored, watertight, skinned GLB from prepared WallAlive evidence.

Run with Blender:
  blender --background --python scripts/build-semantic-glb.py -- \
    --input-dir /tmp/wallalive-drawing --output /tmp/wallalive.glb
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend", type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])


def srgb(hex_value: str) -> tuple[float, float, float, float]:
    value = hex_value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4)) + (1.0,)


def material(name: str, color: str, roughness: float = 0.55, metallic: float = 0.0):
    result = bpy.data.materials.new(name)
    result.diffuse_color = srgb(color)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = srgb(color)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return result


def texture_material(texture_path: Path, fallback: str):
    result = material("Authored front artwork", fallback, 0.62)
    nodes = result.node_tree.nodes
    links = result.node_tree.links
    shader = nodes.get("Principled BSDF")
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.name = "Exact isolated drawing"
    image_node.image = bpy.data.images.load(str(texture_path.resolve()))
    image_node.interpolation = "Linear"
    links.new(image_node.outputs["Color"], shader.inputs["Base Color"])
    return result


def curve_object(name: str, points: list[list[float]], z: float, bevel: float, curve_material, cyclic: bool = True):
    if len(points) < 2:
        return None
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 2
    data.bevel_depth = bevel
    data.bevel_resolution = 3
    spline = data.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for target, point in zip(spline.points, points, strict=True):
        target.co = (float(point[0]), float(point[1]), z, 1.0)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    data.materials.append(curve_material)
    bpy.context.collection.objects.link(obj)
    return obj


def create_feature(feature: dict, line_material, eye_material):
    kind = feature["kind"]
    center_x, center_y, center_z = feature["center"]
    size_x, size_y = feature["size"]
    if kind == "eye":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=20, location=(center_x, center_y, center_z + 0.018))
        lens = bpy.context.object
        lens.name = f"semantic-eye-{center_x:+.3f}"
        lens.scale = (max(0.018, size_x * 0.45), max(0.018, size_y * 0.45), max(0.014, min(size_x, size_y) * 0.15))
        lens.data.materials.append(eye_material)
        lens["semantic_kind"] = "eye"
        lens["confidence"] = feature["confidence"]
        bpy.context.view_layer.objects.active = lens
        lens.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        lens.select_set(False)
    outline = feature.get("outline", [])
    if outline:
        result = curve_object(f"semantic-{kind}-outline", outline, center_z + 0.036, max(0.005, min(size_x, size_y) * 0.045), line_material)
        if result:
            result["semantic_kind"] = kind
            result["confidence"] = feature["confidence"]


def distance_to_segment(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    delta = end - start
    denominator = max(1e-8, float(delta @ delta))
    amount = np.clip(((points - start) @ delta) / denominator, 0, 1)
    projected = start[None] + amount[:, None] * delta[None]
    return np.linalg.norm(points - projected, axis=1)


def semantic_endpoint_records(root: list[float], endpoints: list[list[float]]):
    """Select one evidence-backed endpoint per anatomical role and side."""
    candidates: dict[tuple[str, str], list[tuple[float, list[float]]]] = {}
    for endpoint in endpoints:
        dx = endpoint[0] - root[0]
        dy = endpoint[1] - root[1]
        if math.hypot(dx, dy) < 0.18:
            continue
        side = "L" if dx < 0 else "R"
        role = None
        score = 0.0
        if dy > 0.26 and abs(dx) > 0.08:
            role, score = "Ear", dy + abs(dx) * 0.15
        elif abs(dx) > 0.22 and -0.5 < dy < 0.28:
            role, score = "Arm", abs(dx) + max(0.0, -dy) * 0.1
        elif dy < -0.3:
            role, score = "Leg", -dy + abs(dx) * 0.08
        if role:
            candidates.setdefault((role, side), []).append((score, endpoint))
    records = []
    for (role, side), values in sorted(candidates.items()):
        endpoint = max(values, key=lambda item: item[0])[1]
        records.append({"role": role, "side": side, "name": f"{role}.{side}", "endpoint": endpoint})
    paired_ears = [record for record in records if record["role"] == "Ear"]
    if len(paired_ears) == 2:
        paired_height = max(float(record["endpoint"][1]) for record in paired_ears)
        for record in paired_ears:
            # A broken attachment often lowers one endpoint. The independently
            # detected opposite ear supplies a conservative bilateral prior.
            if float(record["endpoint"][1]) < paired_height - 0.08:
                record["endpoint"] = [
                    float(record["endpoint"][0]), paired_height,
                    float(record["endpoint"][2]),
                ]
    return records


def create_semantic_appendages(records, root: list[float], body_material):
    objects = []
    for record in records:
        endpoint = np.asarray(record["endpoint"][:2], dtype=np.float32)
        root_xy = np.asarray(root[:2], dtype=np.float32)
        delta = endpoint - root_xy
        distance = max(0.001, float(np.linalg.norm(delta)))
        direction = delta / distance
        role = record["role"]
        anchor_fraction = 0.62 if role == "Ear" else 0.7
        anchor = root_xy + direction * distance * anchor_fraction
        # Endpoint heatmaps land on the last confident stroke, commonly just
        # inside a faint outer tip. Extend only along that measured branch.
        outer = root_xy + delta * {"Ear": 1.2, "Arm": 1.14, "Leg": 1.12}[role]
        center = (anchor + outer) * 0.5
        length = max(0.13, float(np.linalg.norm(outer - anchor)) * (1.12 if role == "Ear" else 1.04))
        thickness = {"Ear": 0.085, "Arm": 0.062, "Leg": 0.072}[role]
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=28,
            ring_count=18,
            location=(float(center[0]), float(center[1]), 0.0),
            rotation=(0.0, 0.0, math.atan2(-float(direction[0]), float(direction[1]))),
        )
        obj = bpy.context.object
        obj.name = f"semantic-{role.lower()}-{record['side'].lower()}"
        obj.scale = (thickness, length * 0.55, thickness * (1.05 if role == "Ear" else 0.9))
        obj.data.materials.append(body_material)
        obj["semantic_kind"] = role.lower()
        obj["evidence"] = "learned topology endpoint + medial skeleton"
        obj["endpoint"] = [float(value) for value in record["endpoint"]]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.select_set(False)
        objects.append((obj, record["name"]))
    return objects


def build_armature(mesh_object, vertices: np.ndarray, root: list[float], endpoint_records):
    armature = bpy.data.armatures.new("WallAlive semantic armature")
    rig = bpy.data.objects.new("WallAlive semantic rig", armature)
    bpy.context.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    root_bone = armature.edit_bones.new("Root")
    root_bone.head = (root[0], root[1] - 0.12, 0)
    root_bone.tail = (root[0], root[1] + 0.16, 0)
    records: list[tuple[str, np.ndarray]] = []
    for record in endpoint_records:
        endpoint = record["endpoint"]
        name = record["name"]
        bone = armature.edit_bones.new(name)
        bone.head = tuple(root)
        bone.tail = tuple(endpoint)
        bone.parent = root_bone
        records.append((name, np.asarray(endpoint[:2], dtype=np.float32)))
    bpy.ops.object.mode_set(mode="OBJECT")

    modifier = mesh_object.modifiers.new("Semantic skinning", "ARMATURE")
    modifier.object = rig
    mesh_object.parent = rig
    root_group = mesh_object.vertex_groups.new(name="Root")
    branch_groups = [(mesh_object.vertex_groups.new(name=name), endpoint) for name, endpoint in records]
    xy = vertices[:, :2]
    root_xy = np.asarray(root[:2], dtype=np.float32)
    if not branch_groups:
        root_group.add(list(range(len(vertices))), 1.0, "REPLACE")
    else:
        evidence = np.stack([
            np.exp(-((distance_to_segment(xy, root_xy, endpoint) / 0.2) ** 2)) * 0.58
            for _, endpoint in branch_groups
        ], axis=1)
        root_weights = np.full(len(vertices), 0.42, dtype=np.float32)
        totals = root_weights + evidence.sum(axis=1)
        root_weights /= totals
        evidence /= totals[:, None]
        for index, weight in enumerate(root_weights):
            root_group.add([index], float(weight), "REPLACE")
        for branch_index, (group, _) in enumerate(branch_groups):
            for index, weight in enumerate(evidence[:, branch_index]):
                if weight > 0.002:
                    group.add([index], float(weight), "REPLACE")
    return rig, len(records) + 1


def main() -> None:
    args = arguments()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    payload = np.load(args.input_dir / "mesh.npz")
    vertices = payload["vertices"].astype(np.float32)
    faces = payload["faces"].astype(np.int32)
    uv = payload["uv"].astype(np.float32)
    rig_data = json.loads((args.input_dir / "rig.json").read_text())

    mesh = bpy.data.meshes.new("Watertight signed-distance character")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    mesh.update(calc_edges=True)
    body = bpy.data.objects.new("WallAlive volumetric body", mesh)
    bpy.context.collection.objects.link(body)
    body["reconstruction_method"] = "silhouette signed-distance lens marching cubes"
    body["back_prior"] = rig_data["back_prior"]
    body["source_semantic_kinds"] = ",".join(rig_data["semantic_kinds"])

    front_material = texture_material(args.input_dir / "texture.png", rig_data["body_color"])
    back_material = material("Inferred symmetric back fill", rig_data["body_color"], 0.7)
    line_material = material("Authored line color", rig_data["line_color"], 0.42)
    eye_material = material("Eye paper fill", "#fffdf7", 0.34)
    mesh.materials.append(front_material)
    mesh.materials.append(back_material)
    uv_layer = mesh.uv_layers.new(name="Authored front UV")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        # Marching-cubes winding can point inward depending on the scalar-field
        # convention, so surface position is the reliable front/back test.
        is_authored_front = polygon.center.z > 0
        polygon.material_index = 0 if is_authored_front else 1
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = tuple(uv[vertex_index]) if is_authored_front else (0.001, 0.001)

    contour = rig_data["contour"]
    curve_object("authored-front-silhouette", contour, 0.016, 0.008, line_material)
    curve_object("inferred-back-silhouette", contour, -0.016, 0.007, line_material)
    for feature in rig_data["semantic_features"]:
        create_feature(feature, line_material, eye_material)

    endpoint_records = semantic_endpoint_records(rig_data["root"], rig_data["skeleton_endpoints"])
    appendages = create_semantic_appendages(endpoint_records, rig_data["root"], back_material)
    rig, bone_count = build_armature(body, vertices, rig_data["root"], endpoint_records)
    for appendage, bone_name in appendages:
        world_transform = appendage.matrix_world.copy()
        appendage.parent = rig
        appendage.parent_type = "BONE"
        appendage.parent_bone = bone_name
        appendage.matrix_world = world_transform
    rig["bone_count"] = bone_count
    rig["weight_contract"] = "normalized multi-bone semantic skinning"
    rig["viewable_degrees"] = 360

    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(
        filepath=str(args.output.resolve()),
        export_format="GLB",
        use_selection=False,
        # The reconstruction already uses X horizontal, Y vertical, Z depth.
        # Keep that frame so depth validation and downstream AR placement agree.
        export_yup=False,
        export_skins=True,
        export_animations=True,
        export_apply=False,
    )
    if args.blend:
        args.blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()))
    print(json.dumps({
        "output": str(args.output),
        "vertices": len(vertices),
        "triangles": len(faces),
        "bones": bone_count,
        "semantic_features": len(rig_data["semantic_features"]),
        "semantic_appendages": [record["name"] for record in endpoint_records],
        "body_color": rig_data["body_color"],
        "line_color": rig_data["line_color"],
        "back_prior": rig_data["back_prior"],
    }))


if __name__ == "__main__":
    main()
