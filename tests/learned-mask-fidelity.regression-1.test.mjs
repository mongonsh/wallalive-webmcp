import test from "node:test";
import assert from "node:assert/strict";

import { mergeLearnedPartHints } from "../app/lib/drawing.ts";

// Regression: ISSUE-003 — a neighboring stroke blob stretched a learned eye
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("keeps the neural mask contour, position, and local color for a face part", () => {
  const extraction = {
    previewUrl: "data:image/png;base64,preview",
    textureUrl: "data:image/png;base64,texture",
    contour: [],
    skeleton: [],
    analysis: { shapeHint: "round", dominantColor: "#d64f63", secondaryColor: "#fff", coveragePercent: 10, aspectRatio: 1, edgeEnergy: "bold", sourceWidth: 100, sourceHeight: 100, skeletonPoints: 1 },
    semanticRegions: [{ id: "wrong-neighbor", x: -0.3, y: 0.3, width: 0.4, height: 0.35, color: "#33aaff", pixelCount: 300, density: 0.8 }],
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f5eee5",
      lineColor: "#d64f63",
      joints: [],
      detectedKinds: ["body", "pupil"],
      parts: [
        { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 1.1, y: 1.2, z: 0.5 }, rotation: 0, color: "#f5eee5", confidence: 1, source: "silhouette-branch" },
        { id: "pupil-right", kind: "pupil", side: "right", parentId: "body", center: { x: 0.1, y: 0.2, z: 0 }, size: { x: 0.08, y: 0.08, z: 0.02 }, rotation: 0, color: "#333", confidence: 0.6, source: "image-region" },
      ],
    },
  };
  const neuralOutline = [{ x: 0.58, y: 0.35 }, { x: 0.62, y: 0.35 }, { x: 0.62, y: 0.39 }, { x: 0.58, y: 0.39 }];

  const result = mergeLearnedPartHints(extraction, [{
    kind: "eye",
    center: { x: 0.6, y: 0.37 },
    size: { x: 0.04, y: 0.04 },
    outline: neuralOutline,
    color: "#d94a5f",
    rotation: 0,
    confidence: 0.91,
  }], 8);
  const eye = result.rig.parts.find((part) => part.kind === "eye");

  assert.ok(eye);
  assert.ok(Math.abs(eye.center.x - 0.14) < 1e-9);
  assert.ok(Math.abs(eye.center.y - 0.182) < 1e-9);
  assert.equal(eye.center.z, 0);
  assert.equal(eye.color, "#d94a5f");
  assert.deepEqual(eye.outline, neuralOutline.map(({ x, y }) => ({ x: (x - 0.5) * 1.4, y: (0.5 - y) * 1.4, z: 0 })));
  assert.notEqual(eye.center.x, extraction.semanticRegions[0].x);
  assert.equal(result.rig.parts.some((part) => part.kind === "pupil"), false, "classical pupils must not duplicate a learned eye mask");
});
