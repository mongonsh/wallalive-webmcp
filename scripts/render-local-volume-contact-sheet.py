#!/usr/bin/env python3
"""Render prepared WallAlive volume meshes from four angles for visual QA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np


def rgb(hex_value: str) -> tuple[float, float, float]:
    value = hex_value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255 for index in (0, 2, 4))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    views = (("FRONT", -90), ("3/4", -45), ("SIDE", 0), ("BACK", 90))
    figure = plt.figure(figsize=(12, 3 * len(args.cases)), facecolor="#f5efdc")
    light = LightSource(azdeg=315, altdeg=48)
    for row, case_name in enumerate(args.cases):
        case_dir = args.input_root / case_name
        mesh = np.load(case_dir / "mesh.npz")
        vertices = mesh["vertices"]
        faces = mesh["faces"]
        if len(faces) > 16000:
            faces = faces[np.linspace(0, len(faces) - 1, 16000, dtype=int)]
        rig = json.loads((case_dir / "rig.json").read_text())
        color = rgb(rig["body_color"])
        spans = np.ptp(vertices, axis=0)
        for column, (label, azimuth) in enumerate(views):
            axis = figure.add_subplot(len(args.cases), len(views), row * len(views) + column + 1, projection="3d")
            axis.plot_trisurf(
                # Matplotlib treats its third coordinate as vertical. WallAlive
                # uses Y-up and Z-depth, so plot X/Z/Y to keep camera labels true.
                vertices[:, 0], vertices[:, 2], vertices[:, 1],
                triangles=faces, color=color, linewidth=0, antialiased=False,
                shade=True, lightsource=light,
            )
            axis.set_proj_type("ortho")
            axis.view_init(elev=9, azim=azimuth)
            axis.set_xlim(-0.85, 0.85)
            axis.set_ylim(-0.42, 0.42)
            axis.set_zlim(-0.85, 0.85)
            axis.set_box_aspect((max(spans[0], 0.1), max(spans[2], 0.1), max(spans[1], 0.1)))
            axis.set_axis_off()
            axis.set_facecolor("#f5efdc")
            axis.set_title(
                f"{case_name.upper()} · {rig['topology_kind'].upper()}\n{label}",
                fontsize=10, color="#17332f", fontweight="bold", pad=0,
            )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.01, wspace=0.02, hspace=0.04)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150, facecolor=figure.get_facecolor())
    print(args.output)


if __name__ == "__main__":
    main()
