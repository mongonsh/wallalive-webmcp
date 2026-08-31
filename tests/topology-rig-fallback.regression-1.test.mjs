import test from "node:test";
import assert from "node:assert/strict";

import { mergeLearnedPartHints } from "../app/lib/drawing.ts";

test("turns a high-confidence side topology endpoint into an articulated arm and hand", () => {
  const extraction = {
    previewUrl: "preview",
    textureUrl: "texture",
    contour: [],
    skeleton: [],
    semanticRegions: [],
    analysis: { shapeHint: "round", dominantColor: "#c76a7d", secondaryColor: "#f4f1eb", coveragePercent: 7, aspectRatio: 0.8, edgeEnergy: "soft", sourceWidth: 200, sourceHeight: 260, skeletonPoints: 7 },
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f4f1eb",
      lineColor: "#c76a7d",
      joints: [],
      detectedKinds: ["body"],
      parts: [{ id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.8, y: 1.1, z: 0.45 }, rotation: 0, color: "#f4f1eb", confidence: 1, source: "silhouette-branch" }],
    },
  };
  const topology = {
    model: "wallalive-topology-v10",
    latencyMs: 20,
    kind: "biped",
    kindConfidence: 0.94,
    fieldConfidence: 0.82,
    applicable: true,
    nodes: [
      { id: "root", role: "root", x: 0.5, y: 0.5, confidence: 0.91 },
      { id: "hand", role: "endpoint", x: 0.18, y: 0.52, confidence: 0.93 },
    ],
    edges: [{ id: "edge", from: "root", to: "hand", confidence: 0.9, path: [{ x: 0.5, y: 0.5 }, { x: 0.36, y: 0.51 }, { x: 0.18, y: 0.52 }] }],
  };

  const result = mergeLearnedPartHints(extraction, [
    { kind: "eye", center: { x: 0.42, y: 0.4 }, size: { x: 0.07, y: 0.08 }, rotation: 0, confidence: 0.93 },
    { kind: "eye", center: { x: 0.58, y: 0.4 }, size: { x: 0.07, y: 0.08 }, rotation: 0, confidence: 0.92 },
  ], 18, undefined, topology);

  const arm = result.rig.parts.find((part) => part.kind === "arm");
  const hand = result.rig.parts.find((part) => part.kind === "hand");
  assert.ok(arm);
  assert.ok(hand);
  assert.equal(arm.source, "learned-topology");
  assert.equal(hand.parentId, arm.id);
  assert.ok(arm.path.length >= 2);
});
