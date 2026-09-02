import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";
import { buildArtworkShellGeometry } from "../app/lib/artwork-shell.ts";

const source = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");

const characterContour = [
  { x: -0.24, y: 0.62 }, { x: 0.24, y: 0.62 }, { x: 0.31, y: 0.38 },
  { x: 0.62, y: 0.24 }, { x: 0.58, y: 0.08 }, { x: 0.31, y: 0.12 },
  { x: 0.27, y: -0.22 }, { x: 0.38, y: -0.61 }, { x: 0.10, y: -0.64 },
  { x: 0.02, y: -0.26 }, { x: -0.02, y: -0.26 }, { x: -0.10, y: -0.64 },
  { x: -0.38, y: -0.61 }, { x: -0.27, y: -0.22 }, { x: -0.31, y: 0.12 },
  { x: -0.58, y: 0.08 }, { x: -0.62, y: 0.24 }, { x: -0.31, y: 0.38 },
];

test("builds a watertight rounded shell while preserving the exact drawing silhouette and UVs", () => {
  const result = buildArtworkShellGeometry(characterContour, null, 0.15, 1, 2);
  const position = result.geometry.getAttribute("position");
  const uv = result.geometry.getAttribute("uv");
  const index = result.geometry.getIndex();
  assert.ok(index, "the closed shell must share indexed boundary vertices");
  assert.ok(result.frontTriangleCount > 100);
  assert.equal(result.frontTriangleCount, result.backTriangleCount);
  assert.ok(result.sideTriangleCount >= characterContour.length * 8);

  const xs = Array.from({ length: position.count }, (_, vertex) => position.getX(vertex));
  const ys = Array.from({ length: position.count }, (_, vertex) => position.getY(vertex));
  const zs = Array.from({ length: position.count }, (_, vertex) => position.getZ(vertex));
  assert.ok(Math.abs(Math.min(...xs) + 0.62) < 1e-5);
  assert.ok(Math.abs(Math.max(...xs) - 0.62) < 1e-5);
  assert.ok(Math.abs(Math.min(...ys) + 0.64) < 1e-5);
  assert.ok(Math.abs(Math.max(...ys) - 0.62) < 1e-5);
  assert.ok(Math.max(...zs) > 0.1 && Math.min(...zs) < -0.1);

  characterContour.forEach((point) => {
    const matching = Array.from({ length: position.count }, (_, vertex) => vertex).filter((vertex) => (
      Math.abs(position.getX(vertex) - point.x) < 1e-5
      && Math.abs(position.getY(vertex) - point.y) < 1e-5
    ));
    assert.ok(matching.some((vertex) => position.getZ(vertex) > 0));
    assert.ok(matching.some((vertex) => position.getZ(vertex) < 0));
  });

  for (let vertex = 0; vertex < position.count; vertex += 1) {
    assert.ok(Math.abs(uv.getX(vertex) - (position.getX(vertex) / 1.4 + 0.5)) < 1e-5);
    assert.ok(Math.abs(uv.getY(vertex) - (position.getY(vertex) / 1.4 + 0.5)) < 1e-5);
  }

  const edgeUse = new Map();
  for (let offset = 0; offset < index.count; offset += 3) {
    const triangle = [index.getX(offset), index.getX(offset + 1), index.getX(offset + 2)];
    for (let edge = 0; edge < 3; edge += 1) {
      const a = triangle[edge];
      const b = triangle[(edge + 1) % 3];
      const key = a < b ? `${a}:${b}` : `${b}:${a}`;
      edgeUse.set(key, (edgeUse.get(key) ?? 0) + 1);
    }
  }
  assert.equal([...edgeUse.values()].filter((uses) => uses !== 2).length, 0, "every shell edge must be watertight");
});

test("skins one continuous reconstructed surface instead of attaching generic capsules", () => {
  assert.match(source, /new THREE\.SkinnedMesh\(compactGeometry, \[sideMaterial, frontMaterial, backMaterial\]\)/);
  assert.match(source, /buildArtworkShellGeometry\(contour, depth, requestedHalfDepth, inflation, 3, reliefParts\)/);
  assert.match(source, /contour-preserving rounded 3D puppet/);
  assert.match(source, /setAttribute\("skinIndex"/);
  assert.match(source, /setAttribute\("skinWeight"/);
  assert.match(source, /one safe root bone over one continuous surface/);
  assert.match(source, /projectedSemanticFeatures: false/);
  assert.match(source, /const requestedHalfDepth = Math\.min\(0\.16/);
  assert.doesNotMatch(source, /MarchingCubes|volume\.blur|volume\.setCell/);
  assert.doesNotMatch(source, /DecalGeometry|raised-lens|raised-pupil|addInkFeature/);
  assert.doesNotMatch(source, /volumetric-appendage/);
  assert.doesNotMatch(source, /new THREE\.CapsuleGeometry/);
});

test("verified facial parts shape subtle relief on the same closed surface", () => {
  const baseline = buildArtworkShellGeometry(characterContour, null, 0.15, 1, 3);
  const relieved = buildArtworkShellGeometry(characterContour, null, 0.15, 1, 3, [{
    kind: "nose",
    center: { x: 0, y: 0.12 },
    size: { x: 0.22, y: 0.18 },
    confidence: 0.95,
  }]);
  const frontAt = (result, x, y) => {
    const positions = result.geometry.getAttribute("position");
    let best = { distance: Infinity, z: -Infinity };
    for (let vertex = 0; vertex < positions.count; vertex += 1) {
      const z = positions.getZ(vertex);
      if (z <= 0) continue;
      const distance = Math.hypot(positions.getX(vertex) - x, positions.getY(vertex) - y);
      if (distance < best.distance) best = { distance, z };
    }
    return best.z;
  };

  assert.ok(frontAt(relieved, 0, 0.12) > frontAt(baseline, 0, 0.12) + 0.012);
  assert.ok(relieved.maximumFrontDepth > baseline.maximumFrontDepth);
  assert.equal(relieved.geometry.getIndex().count, baseline.geometry.getIndex().count, "relief must not attach separate primitives");
});
