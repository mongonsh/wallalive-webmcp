import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");

test("skins one continuous reconstructed surface instead of attaching generic capsules", () => {
  assert.match(source, /new THREE\.SkinnedMesh\(compactGeometry, \[sideMaterial, frontMaterial, backMaterial\]\)/);
  assert.match(source, /activeVertexCount = volume\.geometry\.drawRange\.count/);
  assert.match(source, /setAttribute\("skinIndex"/);
  assert.match(source, /setAttribute\("skinWeight"/);
  assert.match(source, /one safe root bone over one continuous surface/);
  assert.match(source, /projectedSemanticFeatures: false/);
  assert.match(source, /const bodyHalfDepth = Math\.min\(0\.16/);
  assert.doesNotMatch(source, /DecalGeometry|raised-lens|raised-pupil|addInkFeature/);
  assert.doesNotMatch(source, /volumetric-appendage/);
  assert.doesNotMatch(source, /new THREE\.CapsuleGeometry/);
});
