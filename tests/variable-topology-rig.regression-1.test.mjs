import test from "node:test";
import assert from "node:assert/strict";

import { mergeLearnedPartHints } from "../app/lib/drawing.ts";

function extraction() {
  return {
    previewUrl: "preview",
    textureUrl: "texture",
    contour: [],
    skeleton: [],
    semanticRegions: [],
    analysis: { shapeHint: "round", dominantColor: "#70445f", secondaryColor: "#eadfec", coveragePercent: 8, aspectRatio: 1, edgeEnergy: "soft", sourceWidth: 256, sourceHeight: 256, skeletonPoints: 5 },
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#eadfec",
      lineColor: "#70445f",
      joints: [],
      detectedKinds: ["body"],
      parts: [{ id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 1, y: 1, z: 0.5 }, rotation: 0, color: "#eadfec", confidence: 1, source: "silhouette-branch" }],
    },
  };
}

function topology(kind, endpoints) {
  const root = { id: "root", role: "root", x: 0.5, y: 0.5, confidence: 0.96 };
  const nodes = [root, ...endpoints.map((point, index) => ({ id: `endpoint-${index}`, role: "endpoint", ...point, confidence: 0.92 - index * 0.02 }))];
  return {
    model: "wallalive-topology-v10",
    latencyMs: 15,
    kind,
    kindConfidence: 0.94,
    fieldConfidence: 0.86,
    applicable: true,
    nodes,
    edges: endpoints.map((point, index) => ({
      id: `edge-${index}`,
      from: "root",
      to: `endpoint-${index}`,
      confidence: 0.9,
      path: [{ x: 0.5, y: 0.5 }, { x: (0.5 + point.x) / 2, y: (0.5 + point.y) / 2 }, point],
    })),
  };
}

test("creates family-specific articulated parts from topology even when the anatomy detector finds no face", () => {
  const cases = [
    { kind: "quadruped", endpoints: [{ x: 0.32, y: 0.88 }], expected: ["leg", "foot"] },
    { kind: "winged", endpoints: [{ x: 0.08, y: 0.42 }], expected: ["wing"] },
    { kind: "aquatic", endpoints: [{ x: 0.05, y: 0.52 }, { x: 0.64, y: 0.24 }], expected: ["tail", "fin"] },
    { kind: "radial", endpoints: [{ x: 0.18, y: 0.84 }, { x: 0.82, y: 0.82 }], expected: ["tentacle"] },
    { kind: "branched", endpoints: [{ x: 0.16, y: 0.16 }, { x: 0.84, y: 0.14 }], expected: ["branch"] },
  ];

  for (const benchmark of cases) {
    const result = mergeLearnedPartHints(extraction(), [], 21, undefined, topology(benchmark.kind, benchmark.endpoints));
    assert.equal(result.rig.topologyKind, benchmark.kind);
    assert.equal(result.rig.topologyConfidence, 0.94);
    for (const kind of benchmark.expected) assert.ok(result.rig.parts.some((part) => part.kind === kind), `${benchmark.kind} should create ${kind}`);
    assert.ok(result.rig.parts.filter((part) => part.source === "learned-topology").every((part) => part.path?.length >= 2 || part.kind === "foot"));
  }
});

test("does not relabel fish, radial creatures, or trees as humanoid limbs", () => {
  for (const kind of ["aquatic", "radial", "branched"]) {
    const base = extraction();
    base.rig.parts.push({ id: "wrong-arm", kind: "arm", side: "left", parentId: "body", center: { x: -0.4, y: 0, z: 0 }, anchor: { x: 0, y: 0, z: 0 }, size: { x: 0.1, y: 0.4, z: 0.1 }, rotation: 0, color: "#eadfec", confidence: 0.5, source: "silhouette-branch" });
    const result = mergeLearnedPartHints(base, [], 21, undefined, topology(kind, [{ x: 0.08, y: 0.5 }]));
    assert.equal(result.rig.parts.some((part) => part.kind === "arm"), false);
  }
});
