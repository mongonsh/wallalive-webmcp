import test from "node:test";
import assert from "node:assert/strict";

import { sameSemanticInstance } from "../app/lib/model-math.ts";

// Regression: ISSUE-002 — adjacent cheek-mask islands survived as duplicate marks
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("groups adjacent cheek-mark islands but preserves bilateral cheeks", () => {
  const leftCheek = { kind: "cheek", center: { x: 0.38, y: 0.48 }, size: { x: 0.04, y: 0.04 }, rotation: -0.7 };
  const adjacentHatch = { kind: "cheek", center: { x: 0.45, y: 0.51 }, size: { x: 0.035, y: 0.04 }, rotation: -0.8 };
  const rightCheek = { kind: "cheek", center: { x: 0.66, y: 0.49 }, size: { x: 0.04, y: 0.04 }, rotation: 0.7 };

  assert.equal(sameSemanticInstance(leftCheek, adjacentHatch), true);
  assert.equal(sameSemanticInstance(leftCheek, rightCheek), false);
});
