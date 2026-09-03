import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { accessibleWorldObjectIds, getAccessibleWorldInteraction } from "../app/lib/world-interactions.ts";

// Regression: ISSUE-001 — visible quest controls did nothing when WebGL was unavailable
// Found by /qa on 2026-09-03
// Report: .gstack/qa-reports/qa-report-wallalive-webmcp-2026-09-03.md
test("every listed quest object has a semantic interaction fallback", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.equal(accessibleWorldObjectIds.length, 15);
  for (const objectId of accessibleWorldObjectIds) {
    assert.match(page, new RegExp(`"${objectId}"`));
    const world = objectId.split("-")[0];
    const interaction = getAccessibleWorldInteraction(world, objectId);
    assert.equal(interaction?.id, objectId);
    assert.equal(interaction?.world, world);
    assert.ok(interaction?.label);
    assert.ok(interaction?.verb);
    assert.ok(interaction?.story);
  }

  assert.equal(getAccessibleWorldInteraction("museum", "studio-projector"), null);
  assert.match(page, /if \(!activated\)[\s\S]+getAccessibleWorldInteraction[\s\S]+handleWorldInteraction\(fallbackInteraction/);
  assert.match(page, /renderedIn3D: activated, accessibleFallback: !activated/);
});
