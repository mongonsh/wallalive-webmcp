import assert from "node:assert/strict";
import test from "node:test";
import { mergeLearnedPartHints } from "../app/lib/drawing.ts";

const extraction = {
  textureUrl: "data:image/png;base64,AA==",
  previewUrl: "data:image/jpeg;base64,AA==",
  contour: [{ x: -0.4, y: -0.4 }, { x: 0.4, y: -0.4 }, { x: 0.4, y: 0.4 }, { x: -0.4, y: 0.4 }],
  skeleton: [{ x: 0, y: 0, radius: 0.2 }],
  rig: {
    version: "wallalive-semantic-rig-v2",
    bodyColor: "#fffaf0",
    lineColor: "#b93f55",
    parts: [{ id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.9, y: 0.9, z: 0.3 }, rotation: 0, color: "#fffaf0", confidence: 1, source: "image-region" }],
    joints: [],
    detectedKinds: ["body"],
  },
  analysis: { dominantColor: "#b93f55", secondaryColor: "#fffaf0", coveragePercent: 8, aspectRatio: 1, shapeHint: "round", edgeEnergy: "soft", sourceWidth: 512, sourceHeight: 512, skeletonPoints: 1 },
};

test("promotes the learned pose nose only inside an evidence-backed two-eye face", () => {
  const result = mergeLearnedPartHints(extraction, [
    { kind: "eye", center: { x: 0.39, y: 0.4 }, size: { x: 0.08, y: 0.07 }, rotation: 0, confidence: 0.96 },
    { kind: "eye", center: { x: 0.61, y: 0.4 }, size: { x: 0.08, y: 0.07 }, rotation: 0, confidence: 0.96 },
    { kind: "mouth", center: { x: 0.5, y: 0.58 }, size: { x: 0.15, y: 0.05 }, rotation: 0, confidence: 0.91 },
  ], 18, {
    model: "wallalive-amateur-pose-v6",
    latencyMs: 9,
    applicable: true,
    joints: [{ name: "nose", x: 0.5, y: 0.48, confidence: 0.88 }],
  });
  const nose = result.rig.parts.find((part) => part.kind === "nose");
  assert.ok(nose, "expected the explicit pose landmark to become a semantic nose");
  assert.equal(nose.source, "learned-pose");
  assert.equal(nose.side, "center");
  assert.ok(nose.confidence > 0.7);
});
test("does not invent a nose without a two-eye face", () => {
  const result = mergeLearnedPartHints(extraction, [
    { kind: "eye", center: { x: 0.5, y: 0.4 }, size: { x: 0.08, y: 0.07 }, rotation: 0, confidence: 0.96 },
  ], 18, {
    model: "wallalive-amateur-pose-v6",
    latencyMs: 9,
    applicable: true,
    joints: [{ name: "nose", x: 0.5, y: 0.48, confidence: 0.99 }],
  });
  assert.equal(result.rig.parts.some((part) => part.kind === "nose"), false);
});
