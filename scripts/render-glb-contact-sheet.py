#!/usr/bin/env python3
"""Render a colored GLB from eight orbital angles without Blender."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="NEURAL 3D")
    args = parser.parse_args()

    loaded = trimesh.load(args.input, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected one mesh, got {type(loaded).__name__}")
    vertices = np.asarray(loaded.vertices)
    faces = np.asarray(loaded.faces)
    visual = loaded.visual if hasattr(loaded.visual, "vertex_colors") else loaded.visual.to_color()
    raw_colors = np.asarray(visual.vertex_colors)
    colors = raw_colors[:, :3].astype(np.float32) / 255 if raw_colors.ndim == 2 else np.full((len(vertices), 3), (0.94, 0.88, 0.84), dtype=np.float32)
    if len(faces) > 24_000:
        selected = np.linspace(0, len(faces) - 1, 24_000, dtype=int)
        faces = faces[selected]
    face_colors = colors[faces].mean(axis=1)

    spans = np.ptp(vertices, axis=0)
    up = int(np.argmax(spans))
    horizontal = [axis for axis in range(3) if axis != up]
    # Use the wider remaining axis as image horizontal and the narrower as
    # camera depth.  This keeps tall character assets upright across exporters.
    horizontal.sort(key=lambda axis: spans[axis], reverse=True)
    plotted = vertices[:, [horizontal[0], horizontal[1], up]]
    center = (plotted.min(axis=0) + plotted.max(axis=0)) / 2
    plotted -= center
    extent = max(np.ptp(plotted[:, 0]), np.ptp(plotted[:, 2])) * 0.58
    depth_extent = max(0.1, np.ptp(plotted[:, 1]) * 0.62)

    figure = plt.figure(figsize=(12, 6), facecolor="#f5efdc")
    for index, azimuth in enumerate(np.linspace(-90, 225, 8)):
        axis = figure.add_subplot(2, 4, index + 1, projection="3d")
        surface = axis.plot_trisurf(
            plotted[:, 0], plotted[:, 1], plotted[:, 2],
            triangles=faces, linewidth=0, antialiased=False,
            color="#f0e7df", shade=True,
        )
        surface.set_facecolors(face_colors)
        axis.set_proj_type("ortho")
        axis.view_init(elev=7, azim=float(azimuth))
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-depth_extent, depth_extent)
        axis.set_zlim(-extent, extent)
        axis.set_box_aspect((extent * 2, depth_extent * 2, extent * 2))
        axis.set_axis_off()
        axis.set_facecolor("#f5efdc")
        axis.set_title(f"{args.title} · {index * 45}°", fontsize=9, color="#17332f", fontweight="bold", pad=0)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, wspace=0.01, hspace=0.02)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=figure.get_facecolor())
    print(args.output)


if __name__ == "__main__":
    main()
