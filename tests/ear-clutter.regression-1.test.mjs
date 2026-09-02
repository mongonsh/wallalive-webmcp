import test from "node:test";
import assert from "node:assert/strict";

import { mergeLearnedPartHints } from "../app/lib/drawing.ts";

// Regression: ISSUE-005 — a paper label above the drawing became a giant ear
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("rejects oversized ear clutter and completes the learned bilateral pair", () => {
  const extraction = {
    previewUrl: "preview",
    textureUrl: "texture",
    contour: [],
    skeleton: [],
    semanticRegions: [],
    analysis: { shapeHint: "round", dominantColor: "#d64f63", secondaryColor: "#fff", coveragePercent: 10, aspectRatio: 1, edgeEnergy: "bold", sourceWidth: 100, sourceHeight: 100, skeletonPoints: 1 },
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f5eee5",
      lineColor: "#d64f63",
      joints: [],
      detectedKinds: ["body"],
      parts: [{ id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.7, y: 1, z: 0.5 }, rotation: 0, color: "#f5eee5", confidence: 1, source: "silhouette-branch" }],
    },
  };
  const hint = (kind, center, size, confidence = 0.9) => ({ kind, center, size, rotation: 0, confidence });
  const result = mergeLearnedPartHints(extraction, [
    hint("eye", { x: 0.43, y: 0.42 }, { x: 0.05, y: 0.05 }),
    hint("eye", { x: 0.57, y: 0.42 }, { x: 0.05, y: 0.05 }),
    hint("ear", { x: 0.47, y: 0.17 }, { x: 0.13, y: 0.07 }),
    hint("ear", { x: 0.64, y: 0.25 }, { x: 0.045, y: 0.055 }),
  ], 12);
  const ears = result.rig.parts.filter((part) => part.kind === "ear");

  assert.equal(ears.length, 2);
  assert.ok(ears.every((ear) => ear.size.x < 0.1), `oversized clutter survived: ${JSON.stringify(ears)}`);
  assert.ok(ears.some((ear) => ear.center.x < 0));
  assert.ok(ears.some((ear) => ear.center.x > 0));
});

test("does not claim silhouette branches are ears when the learned model found no ears", () => {
  const body = { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.8, y: 1.1, z: 0.46 }, rotation: 0, color: "#f5eee5", confidence: 1, source: "silhouette-branch" };
  const ear = (side, x) => ({
    id: `ear-${side}`,
    kind: "ear",
    side,
    parentId: "body",
    center: { x, y: 0.48, z: 0 },
    anchor: { x: x * 0.62, y: 0.28, z: 0 },
    size: { x: 0.11, y: 0.18, z: 0.08 },
    rotation: 0,
    color: "#f5eee5",
    confidence: 0.76,
    source: "silhouette-branch",
  });
  const extraction = {
    previewUrl: "preview",
    textureUrl: "texture",
    contour: [],
    skeleton: [],
    semanticRegions: [],
    analysis: { shapeHint: "tall", dominantColor: "#d64f63", secondaryColor: "#fff", coveragePercent: 8, aspectRatio: 0.8, edgeEnergy: "soft", sourceWidth: 100, sourceHeight: 120, skeletonPoints: 5 },
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f5eee5",
      lineColor: "#d64f63",
      joints: [],
      detectedKinds: ["body", "ear"],
      parts: [body, ear("left", -0.28), ear("right", 0.28)],
    },
  };
  const result = mergeLearnedPartHints(extraction, [
    { kind: "eye", center: { x: 0.41, y: 0.39 }, size: { x: 0.08, y: 0.08 }, rotation: 0, confidence: 0.94 },
    { kind: "eye", center: { x: 0.59, y: 0.39 }, size: { x: 0.08, y: 0.08 }, rotation: 0, confidence: 0.93 },
    { kind: "mouth", center: { x: 0.5, y: 0.52 }, size: { x: 0.08, y: 0.04 }, rotation: 0, confidence: 0.84 },
  ], 11);

  const ears = result.rig.parts.filter((part) => part.kind === "ear");
  assert.equal(ears.length, 0);
});

test("does not turn the upper corners of a wide rectangular cartoon into ears", () => {
  const body = { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 1.05, y: 0.76, z: 0.46 }, rotation: 0, color: "#f5a53b", confidence: 1, source: "silhouette-branch" };
  const corner = (side, x) => ({
    id: `ear-${side}`, kind: "ear", side, parentId: "body",
    center: { x, y: 0.34, z: 0 }, anchor: { x: x * 0.62, y: 0.2, z: 0 },
    size: { x: 0.09, y: 0.1, z: 0.07 }, rotation: 0, color: "#f5a53b",
    confidence: 0.76, source: "silhouette-branch",
  });
  const extraction = {
    previewUrl: "preview", textureUrl: "texture", contour: [], skeleton: [], semanticRegions: [],
    analysis: { shapeHint: "wide", dominantColor: "#f5a53b", secondaryColor: "#fff", coveragePercent: 24, aspectRatio: 1.43, edgeEnergy: "bold", sourceWidth: 1264, sourceHeight: 842, skeletonPoints: 12 },
    rig: { version: "wallalive-semantic-rig-v2", bodyColor: "#f5a53b", lineColor: "#173331", joints: [], detectedKinds: ["body", "ear"], parts: [body, corner("left", -0.48), corner("right", 0.48)] },
  };
  const result = mergeLearnedPartHints(extraction, [
    { kind: "eye", center: { x: 0.43, y: 0.34 }, size: { x: 0.08, y: 0.1 }, rotation: 0, confidence: 0.94 },
    { kind: "eye", center: { x: 0.57, y: 0.34 }, size: { x: 0.08, y: 0.1 }, rotation: 0, confidence: 0.93 },
    { kind: "mouth", center: { x: 0.5, y: 0.5 }, size: { x: 0.13, y: 0.06 }, rotation: 0, confidence: 0.88 },
  ], 10);

  assert.equal(result.rig.parts.filter((part) => part.kind === "ear").length, 0);
});
