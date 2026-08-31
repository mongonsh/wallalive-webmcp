import assert from "node:assert/strict";
import test from "node:test";

import { inspectCharacterCapabilities, validateCharacterMove } from "../app/lib/creative-show.ts";

const part = (id, kind, source, confidence, path = undefined) => ({
  id,
  kind,
  source,
  confidence,
  path,
  parentId: kind === "body" ? undefined : "body",
  side: id.includes("right") ? "right" : id.includes("left") ? "left" : "center",
  center: { x: 0, y: 0, z: 0 },
  size: { x: 0.12, y: 0.25, z: 0.1 },
  rotation: 0,
  color: "#ffffff",
});

const drawing = (parts, poseApplicable = true) => ({
  rig: { parts, joints: [], detectedKinds: [...new Set(parts.map((candidate) => candidate.kind))] },
  poseRecognition: { applicable: poseApplicable },
  topologyRecognition: { applicable: false },
});

test("WebMCP direction exposes and enforces per-character rig abilities", () => {
  const armPath = [{ x: 0, y: 0, z: 0 }, { x: 0.3, y: 0.1, z: 0 }];
  const legPath = [{ x: 0, y: 0, z: 0 }, { x: 0.1, y: -0.4, z: 0 }];
  const capabilities = inspectCharacterCapabilities([
    drawing([part("body", "body", "image-region", 1), part("arm-right", "arm", "learned-pose", 0.91, armPath)]),
    drawing([part("body", "body", "image-region", 1), part("leg-left", "leg", "learned-pose", 0.88, legPath)]),
  ]);

  assert.ok(capabilities[0].availableActions.includes("wave"));
  assert.ok(!capabilities[0].availableActions.includes("walk"));
  assert.ok(capabilities[1].availableActions.includes("walk"));
  assert.ok(!capabilities[1].availableActions.includes("wave"));
  assert.equal(validateCharacterMove(capabilities, 0, "wave").ok, true);
  assert.match(validateCharacterMove(capabilities, 1, "wave").error, /No verified waving branch/);
  assert.match(validateCharacterMove(capabilities, 9, "hop").error, /does not exist/);
});
