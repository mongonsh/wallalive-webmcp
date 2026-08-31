import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
test("gates rectangular camera patches on learned character evidence before 3D", async () => {
  const quality = await readFile(new URL("../app/lib/character-quality.ts", import.meta.url), "utf8");
  const recognition = await readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(quality, /rectangularity >= 0\.72/);
  assert.match(quality, /axisAlignedEdgeFraction >= 0\.48/);
  assert.match(quality, /segmentation confidence is deliberately not character evidence/i);
  assert.match(quality, /evidence\.length > 0/);
  assert.match(quality, /requireCharacterExtraction/);
  assert.match(recognition, /requireCharacterExtraction\(\{/);
  assert.doesNotMatch(page, /catch \(error\) \{\s*setDrawing\(next, source\)/);
});
