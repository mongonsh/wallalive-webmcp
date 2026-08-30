import test from "node:test";
import assert from "node:assert/strict";

import { mergeLearnedPartHints } from "../app/lib/drawing.ts";

// Regression: ISSUE-002 — an unrelated large face region donated shape/color to a cheek
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("rejects a weak pixel-region match instead of misplacing a learned feature", () => {
  const extraction = {
    previewUrl: "data:image/png;base64,preview",
    textureUrl: "data:image/png;base64,texture",
    contour: [],
    skeleton: [],
    analysis: { shapeHint: "round", dominantColor: "#f5eee5", secondaryColor: "#fff", coveragePercent: 10, aspectRatio: 1, edgeEnergy: "bold", sourceWidth: 100, sourceHeight: 100, skeletonPoints: 1 },
    semanticRegions: [{ id: "wrong-large-region", x: 0.2, y: 0.2, width: 0.2, height: 0.2, color: "#00ff00", pixelCount: 200, density: 0.8 }],
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f5eee5",
      lineColor: "#b4435c",
      joints: [],
      detectedKinds: ["body"],
      parts: [{ id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 1.1, y: 1.2, z: 0.5 }, rotation: 0, color: "#f5eee5", confidence: 1, source: "silhouette-branch" }],
    },
  };

  const result = mergeLearnedPartHints(extraction, [{ kind: "cheek", center: { x: 0.6, y: 0.4 }, size: { x: 0.03, y: 0.03 }, rotation: 0, confidence: 0.9 }], 10);
  const cheek = result.rig.parts.find((part) => part.kind === "cheek");

  assert.ok(cheek);
  assert.ok(Math.abs(cheek.center.x - 0.14) < 1e-9);
  assert.ok(Math.abs(cheek.center.y - 0.14) < 1e-9);
  assert.equal(cheek.center.z, 0);
  assert.equal(cheek.color, "#b4435c");
  assert.notEqual(cheek.center.x, extraction.semanticRegions[0].x);
});
