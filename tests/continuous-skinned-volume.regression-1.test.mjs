import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");

test("skins one continuous reconstructed surface instead of attaching generic capsules", () => {
  assert.match(source, /new THREE\.SkinnedMesh\(compactGeometry, volumeMaterial\)/);
  assert.match(source, /activeVertexCount = volume\.geometry\.drawRange\.count/);
  assert.match(source, /setAttribute\("skinIndex"/);
  assert.match(source, /setAttribute\("skinWeight"/);
  assert.match(source, /variable graph bones over one continuous surface/);
  assert.doesNotMatch(source, /volumetric-appendage/);
  assert.doesNotMatch(source, /new THREE\.CapsuleGeometry/);
});
