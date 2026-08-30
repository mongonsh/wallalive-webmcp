#!/usr/bin/env python3
"""Render four orthogonal views of a GLB for WallAlive 360-degree QA."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=512)
    return parser.parse_args(arguments)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    all_meshes = [item for item in bpy.context.scene.objects if item.type == "MESH" and not item.hide_render]
    # Blender may create an unmaterialed Icosphere custom shape to display the
    # imported armature.  It is not part of the GLB surface and must not enlarge
    # the camera bounds.
    meshes = [item for item in all_meshes if len(item.data.materials) > 0] or all_meshes
    if not meshes:
        raise RuntimeError("GLB contains no mesh objects")

    points = [item.matrix_world @ Vector(corner) for item in meshes for corner in item.bound_box]
    minimum = Vector(tuple(min(point[axis] for point in points) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in points) for axis in range(3)))
    center = (minimum + maximum) * 0.5
    span = maximum - minimum
    longest = max(span)
    print(json.dumps({
        "mesh_bounds": [
            {"name": item.name, "dimensions": list(item.dimensions), "location": list(item.matrix_world.translation)}
            for item in meshes
        ],
        "minimum": list(minimum),
        "maximum": list(maximum),
    }))

    camera_data = bpy.data.cameras.new("WallAlive QA Camera")
    camera = bpy.data.objects.new("WallAlive QA Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera
    camera_data.lens = 55
    camera_data.sensor_width = 36

    world = bpy.context.scene.world or bpy.data.worlds.new("WallAlive QA World")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.035, 0.048, 0.047, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

    for name, location, energy, color, size in (
        ("Key", center + Vector((-longest * 2.2, -longest * 2.4, longest * 2.7)), 1100, (1.0, 0.83, 0.67), longest * 2.0),
        ("Fill", center + Vector((longest * 2.4, -longest * 0.8, longest * 1.2)), 850, (0.55, 0.82, 1.0), longest * 1.8),
        ("Rim", center + Vector((longest * 0.2, longest * 2.4, longest * 2.0)), 950, (0.75, 1.0, 0.55), longest * 1.6),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.color = color
        data.shape = "DISK"
        data.size = size
        light = bpy.data.objects.new(name, data)
        light.location = location
        bpy.context.scene.collection.objects.link(light)
        look_at(light, center)

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.look = "AgX - Medium High Contrast"
    distance = longest * 0.72 / math.tan(camera_data.angle * 0.5)
    target = center + Vector((0, 0, span.z * 0.03))
    outputs = []
    for degrees in (0, 90, 180, 270):
        angle = math.radians(degrees)
        camera.location = center + Vector((math.sin(angle) * distance, -math.cos(angle) * distance, longest * 0.12))
        look_at(camera, target)
        output = args.output_dir / f"view-{degrees:03d}.png"
        scene.render.filepath = str(output.resolve())
        bpy.ops.render.render(write_still=True)
        outputs.append(str(output))
    print(json.dumps({"input": str(args.input), "mesh_objects": len(meshes), "span": list(span), "views": outputs}))


if __name__ == "__main__":
    main()
