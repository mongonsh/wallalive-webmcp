import test from "node:test";
import assert from "node:assert/strict";

import { averageLogitConfidence } from "../app/lib/model-math.ts";

// Regression: ISSUE-001 — raw segmentation logits escaped as confidence values above 1
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("converts learned component logits into bounded probabilities", () => {
  const logits = new Float32Array([8, 4, 0, -4, -8]);

  const high = averageLogitConfidence(logits, [0, 1]);
  const balanced = averageLogitConfidence(logits, [1, 3]);
  const low = averageLogitConfidence(logits, [3, 4]);

  assert.ok(high > 0.98 && high <= 1, `high-confidence logits must remain a probability, got ${high}`);
  assert.ok(Math.abs(balanced - 0.5) < 1e-6, `symmetric logits should average to 0.5, got ${balanced}`);
  assert.ok(low >= 0 && low < 0.02, `negative logits must map near zero, got ${low}`);
  assert.equal(averageLogitConfidence(logits, []), 0);
});
