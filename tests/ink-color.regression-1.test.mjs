import test from "node:test";
import assert from "node:assert/strict";

import { selectDominantInkColor } from "../app/lib/model-math.ts";

// Regression: ISSUE-003 — blue graph-paper lines desaturated the red character
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("preserves the dominant red ink instead of averaging it with vivid grid lines", () => {
  const redStroke = Array.from({ length: 120 }, (_, index) => ({
    r: 192 + index % 8,
    g: 171 + index % 5,
    b: 177 + index % 6,
  }));
  const blueGrid = Array.from({ length: 16 }, (_, index) => ({
    r: 150 + index % 4,
    g: 184 + index % 4,
    b: 207 + index % 5,
  }));

  const color = selectDominantInkColor([...redStroke, ...blueGrid]);

  assert.ok(color);
  assert.ok(color.r > color.g + 25, `expected a visibly red result, got ${JSON.stringify(color)}`);
  assert.ok(color.r > color.b + 20, `blue grid must not shift the result purple, got ${JSON.stringify(color)}`);
});
