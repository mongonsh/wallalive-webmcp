#!/usr/bin/env python3
"""Turn a sparse background-removal mask into one solid character cutout.

Bright character interiors can be mistaken for a white background.  This keeps
the remover's outer silhouette, fills enclosed holes, and restores the original
RGB artwork inside that silhouette.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = Image.open(args.image).convert("RGBA")
    removed = Image.open(args.mask_source).convert("RGBA").resize(source.size, Image.Resampling.LANCZOS)
    mask = np.asarray(removed)[..., 3] > 12
    labels, count = ndimage.label(mask)
    if count == 0:
        raise ValueError("The background remover did not find a foreground silhouette")
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    mask = labels == int(np.argmax(sizes) + 1)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)

    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    alpha = np.clip((inside - outside + 1.5) / 3.0, 0, 1)
    rgba = np.asarray(source).copy()
    rgba[..., 3] = np.rint(alpha * 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(args.output)


if __name__ == "__main__":
    main()
