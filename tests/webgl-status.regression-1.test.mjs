import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("a missing WebGL renderer reports an accessible story mode instead of endless GLB loading", async () => {
  const [page, stage] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(stage, /onRendererCapability\(false\)/);
  assert.match(stage, /onRendererCapability\(true\)/);
  assert.match(page, /onRendererCapability=\{handleRendererCapability\}/);
  assert.match(page, /3D ASSET VERIFIED · PREVIEW NEEDS WEBGL/);
  assert.match(page, /FULL NEURAL ASSET · PREVIEW PAUSED/);
  assert.match(page, /ACCESSIBLE STORY MODE/);
});
