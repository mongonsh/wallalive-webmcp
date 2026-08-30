import test from "node:test";
import assert from "node:assert/strict";

import { sameSemanticInstance } from "../app/lib/model-math.ts";

// Regression: ISSUE-002 — disconnected pixels from one limb became duplicate anatomy
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("merges nearby same-axis fragments without collapsing real multi-limb branches", () => {
  const leg = { kind: "leg", center: { x: 0.56, y: 0.76 }, size: { x: 0.08, y: 0.23 }, rotation: 0.18 };
  const nearbyFragment = { kind: "leg", center: { x: 0.60, y: 0.81 }, size: { x: 0.07, y: 0.18 }, rotation: 0.24 };
  const crossingLeg = { kind: "leg", center: { x: 0.59, y: 0.78 }, size: { x: 0.07, y: 0.2 }, rotation: 1.08 };
  const separateLeg = { kind: "leg", center: { x: 0.78, y: 0.78 }, size: { x: 0.08, y: 0.22 }, rotation: 0.2 };

  assert.equal(sameSemanticInstance(leg, nearbyFragment), true, "one fragmented leg should decode once");
  assert.equal(sameSemanticInstance(leg, crossingLeg), false, "crossing multi-limb branches must remain separate");
  assert.equal(sameSemanticInstance(leg, separateLeg), false, "spatially separate legs must remain separate");
  assert.equal(sameSemanticInstance(leg, { ...nearbyFragment, kind: "arm" }), false, "different anatomy classes never merge");
});
