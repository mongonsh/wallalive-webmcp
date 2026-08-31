import assert from "node:assert/strict";
import test from "node:test";
import { classifySurfaceMaterial, hasRecognizableArtworkSurface } from "../app/lib/mesh-materials.ts";

test("normalizes Marching Cubes gradients before assigning the drawing texture", () => {
  const marchingCubesGradient = 1 / 64;

  assert.equal(classifySurfaceMaterial(0, 0, marchingCubesGradient, 0.12), 1);
  assert.equal(classifySurfaceMaterial(0, 0, -marchingCubesGradient, -0.12), 2);
  assert.equal(classifySurfaceMaterial(marchingCubesGradient, 0, 0, 0.02), 0);
});

test("does not put the face texture on a rear-facing triangle", () => {
  assert.equal(classifySurfaceMaterial(0, 0, -1, 0.12), 0);
  assert.equal(classifySurfaceMaterial(0, 0, 1, -0.12), 0);
});

test("refuses to present a fallback with no meaningful artwork-facing surface", () => {
  assert.equal(hasRecognizableArtworkSurface(0, 5_000, true), false);
  assert.equal(hasRecognizableArtworkSurface(900, 5_000, true), false);
  assert.equal(hasRecognizableArtworkSurface(2_400, 5_000, true), true);
  assert.equal(hasRecognizableArtworkSurface(2_400, 5_000, false), false);
});
