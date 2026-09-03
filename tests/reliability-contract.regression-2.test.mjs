import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { assessReconstructionReadiness } from "../app/lib/character-quality.ts";
import { appendCaptureTarget } from "../app/lib/drawing.ts";

test("explicit cast targets preserve human order, reject double taps, clamp coordinates, and cap the cast", () => {
  let targets = appendCaptureTarget([], { x: 0.2, y: 0.25 });
  targets = appendCaptureTarget(targets, { x: 0.21, y: 0.26 });
  assert.deepEqual(targets, [{ x: 0.2, y: 0.25 }], "a nearby double tap must not create a duplicate figure");
  targets = appendCaptureTarget(targets, { x: 0.8, y: 0.3 });
  targets = appendCaptureTarget(targets, { x: -2, y: 4 });
  targets = appendCaptureTarget(targets, { x: 0.35, y: 0.7 }, 3);
  assert.deepEqual(targets, [{ x: 0.2, y: 0.25 }, { x: 0.8, y: 0.3 }, { x: 0, y: 1 }]);
});

test("a human-reviewed limb path unlocks motion without pretending an automatic pose model succeeded", () => {
  const contour = Array.from({ length: 20 }, (_, index) => ({
    x: Math.cos(index / 20 * Math.PI * 2) * 0.4,
    y: Math.sin(index / 20 * Math.PI * 2) * 0.48,
  }));
  const drawing = {
    textureUrl: "data:image/png;base64,AA==",
    previewUrl: "data:image/png;base64,AA==",
    contour,
    skeleton: [{ x: 0, y: 0, radius: 0.2 }],
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#67c7ee",
      lineColor: "#17324a",
      parts: [
        { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.8, y: 0.95, z: 0.25 }, rotation: 0, color: "#67c7ee", confidence: 1, source: "image-region", reviewed: true },
        { id: "manual-arm", kind: "arm", side: "left", parentId: "body", center: { x: -0.45, y: 0.05, z: 0 }, anchor: { x: -0.2, y: 0.1, z: 0 }, size: { x: 0.1, y: 0.4, z: 0.08 }, rotation: 0.8, color: "#67c7ee", confidence: 1, source: "structural-inference", reviewed: true, path: [{ x: -0.2, y: 0.1, z: 0 }, { x: -0.45, y: 0.05, z: 0 }] },
      ],
      joints: [{ id: "joint-manual-arm", parentId: "body", childId: "manual-arm", x: -0.2, y: 0.1 }],
      detectedKinds: ["body", "arm"],
    },
    analysis: { dominantColor: "#67c7ee", secondaryColor: "#17324a", coveragePercent: 18, aspectRatio: 0.85, shapeHint: "tall", edgeEnergy: "bold", sourceWidth: 640, sourceHeight: 640, skeletonPoints: 1 },
    cutoutRecognition: { model: "wallalive-target-cutout-v3", latencyMs: 20, confidence: 0.91, areaPercent: 18, cropScale: 1 },
    characterValidation: { accepted: true, score: 0.86, rectangularity: 0.4, axisAlignedEdgeFraction: 0.2, evidence: ["human-reviewed articulated branch"], reason: "Accepted." },
  };
  const readiness = assessReconstructionReadiness(drawing);
  assert.equal(readiness.cutoutReady, true);
  assert.equal(readiness.motionReady, true);
});

test("the product uses exact-target multi-character recognition and exposes repair as a real WebMCP handoff", async () => {
  const [page, recognition] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8"),
  ]);
  assert.match(page, /recognizeDrawingsAtImageTargets\(pending\.url, targets\)/);
  assert.match(page, /name: "inspect_reconstruction_readiness"/);
  assert.match(page, /name: "request_character_repair"/);
  assert.match(page, /agentChangedRig: false/);
  assert.match(page, /name: "stage_next_learning_challenge"/);
  assert.match(page, /activeFigureIndexRef/);
  assert.doesNotMatch(page, /Each figure has its own cutout, skeleton, and movement rig/);
  assert.match(recognition, /targets\.slice\(0, 6\)/);
  assert.match(recognition, /no connected-component scan/i);
});
