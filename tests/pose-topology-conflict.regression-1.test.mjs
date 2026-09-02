import test from "node:test";
import assert from "node:assert/strict";

import { assessHumanoidPoseEvidence } from "../app/lib/learned-parts.ts";
import { selectBipedRigParts } from "../app/lib/drawing.ts";

const hint = (kind) => ({
  kind,
  center: { x: 0.5, y: 0.5 },
  size: { x: 0.1, y: 0.1 },
  rotation: 0,
  confidence: 0.8,
});

const joint = (name, x, y, confidence = 0.7) => ({ name, x, y, confidence });
const rectangularCartoonJoints = [
  joint("left_shoulder", 0.68, 0.34), joint("right_shoulder", 0.32, 0.34),
  joint("left_elbow", 0.78, 0.44), joint("right_elbow", 0.22, 0.44),
  joint("left_wrist", 0.9, 0.47), joint("right_wrist", 0.1, 0.47),
  joint("left_hip", 0.56, 0.58), joint("right_hip", 0.44, 0.58),
  joint("left_knee", 0.57, 0.72), joint("right_knee", 0.43, 0.72),
  joint("left_ankle", 0.6, 0.88), joint("right_ankle", 0.4, 0.88),
];

test("keeps a strong humanoid pose when the coarse topology head calls a rectangular cartoon branched", () => {
  // Mirrors the observed brick-cartoon failure: two eyes, mouth, arms, hands,
  // feet, and two separated pose chains are present, but the low-resolution
  // part mask merges both legs into one component.
  const hints = [
    hint("eye"), hint("eye"), hint("mouth"),
    hint("arm"), hint("arm"), hint("hand"), hint("hand"),
    hint("leg"), hint("foot"), hint("foot"),
  ];
  const evidence = assessHumanoidPoseEvidence(
    hints,
    rectangularCartoonJoints,
    { kind: "branched", kindConfidence: 0.9141 },
  );

  assert.equal(evidence.semanticOverride, true);
  assert.equal(evidence.applicable, true);
  assert.equal(evidence.counts.legs, 1);
  assert.equal(evidence.confidentCoreJoints, 12);
});

test("does not force a face drawn on a tree into a humanoid rig", () => {
  const hints = [hint("eye"), hint("eye"), hint("mouth"), hint("arm"), hint("leg")];
  const evidence = assessHumanoidPoseEvidence(
    hints,
    rectangularCartoonJoints,
    { kind: "branched", kindConfidence: 0.93 },
  );

  assert.equal(evidence.semanticOverride, false);
  assert.equal(evidence.applicable, false);
});

test("retains the existing biped path without requiring distal-part overrides", () => {
  const hints = [hint("arm"), hint("leg")];
  const evidence = assessHumanoidPoseEvidence(
    hints,
    rectangularCartoonJoints.map((item) => ({ ...item, confidence: 0.2 })),
    { kind: "biped", kindConfidence: 0.72 },
  );

  assert.equal(evidence.semanticOverride, false);
  assert.equal(evidence.applicable, true);
});

test("two pose-backed legs replace one higher-confidence merged centre blob", () => {
  const part = (id, side, confidence) => ({
    id, kind: "leg", side, parentId: "body",
    center: { x: 0.5, y: 0.7, z: 0 }, size: { x: 0.1, y: 0.25, z: 0.1 },
    rotation: 0, color: "#f2b84b", confidence, source: "learned-pose",
  });
  const selected = selectBipedRigParts([
    part("merged-leg", "center", 0.97),
    part("left-leg", "left", 0.72),
    part("right-leg", "right", 0.71),
  ], 2);

  assert.deepEqual(selected.map(({ id }) => id).sort(), ["left-leg", "right-leg"]);
});
