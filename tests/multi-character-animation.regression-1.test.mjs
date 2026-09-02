import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { chooseCharacterTargets, selectAnimatableRigParts } from "../app/lib/drawing.ts";
import { areDuplicateRecognizedDrawings } from "../app/lib/learned-parts.ts";

test("keeps separate drawing figures as separate recognition prompts", () => {
  const primary = { x: 0.72, y: 0.48 };
  const candidates = [
    { target: { x: 0.23, y: 0.52 }, score: 0.82, span: 0.22 },
    { target: { x: 0.72, y: 0.48 }, score: 0.79, span: 0.24 },
    { target: { x: 0.725, y: 0.485 }, score: 0.76, span: 0.2 },
    { target: { x: 0.49, y: 0.5 }, score: 0.72, span: 0.18 },
  ];

  const selected = chooseCharacterTargets(candidates, primary, 4);
  assert.deepEqual(selected[0], primary, "the child-selected figure stays first");
  assert.equal(selected.length, 3, "near-duplicate prompts merge but separate figures remain");
  assert.ok(selected.some((target) => target.x < 0.3));
  assert.ok(selected.some((target) => target.x > 0.65));
});

test("keeps identical-looking figures when their source crops do not overlap", () => {
  const drawing = (sourceTarget, sourceBounds) => ({ textureUrl: "data:image/png;base64,SAME", sourceTarget, sourceBounds });
  assert.equal(areDuplicateRecognizedDrawings(
    drawing({ x: 0.22, y: 0.5 }, { x: 0.08, y: 0.24, width: 0.28, height: 0.54 }),
    drawing({ x: 0.76, y: 0.5 }, { x: 0.62, y: 0.24, width: 0.28, height: 0.54 }),
  ), false);
});

test("merges repeated prompts whose recognized source crops overlap", () => {
  const drawing = (sourceTarget, sourceBounds) => ({ textureUrl: "data:image/png;base64,SAME", sourceTarget, sourceBounds });
  assert.equal(areDuplicateRecognizedDrawings(
    drawing({ x: 0.38, y: 0.49 }, { x: 0.2, y: 0.16, width: 0.42, height: 0.68 }),
    drawing({ x: 0.56, y: 0.5 }, { x: 0.205, y: 0.165, width: 0.415, height: 0.675 }),
  ), true);
});

test("only learned pose/topology limb paths become deformation bones", () => {
  const part = (id, kind, source, confidence, path = undefined) => ({
    id,
    kind,
    source,
    confidence,
    path,
    parentId: "body",
    side: "left",
    center: { x: 0, y: 0, z: 0 },
    size: { x: 0.1, y: 0.3, z: 0.1 },
    rotation: 0,
    color: "#fff",
  });
  const rig = {
    version: "wallalive-semantic-rig-v2",
    bodyColor: "#fff",
    lineColor: "#000",
    joints: [],
    detectedKinds: ["body", "arm", "leg"],
    parts: [
      part("body", "body", "image-region", 1),
      part("arm-left", "arm", "learned-pose", 0.81, [{ x: -0.1, y: 0.1, z: 0 }, { x: -0.4, y: -0.1, z: 0 }]),
      part("leg-left", "leg", "learned-topology", 0.78, [{ x: -0.1, y: -0.2, z: 0 }, { x: -0.2, y: -0.7, z: 0 }]),
      part("arm-false", "arm", "structural-inference", 0.99, [{ x: 0.1, y: 0.1, z: 0 }, { x: 0.4, y: -0.1, z: 0 }]),
      part("leg-weak", "leg", "learned-pose", 0.22, [{ x: 0.1, y: -0.2, z: 0 }, { x: 0.2, y: -0.7, z: 0 }]),
    ],
  };

  assert.deepEqual(
    selectAnimatableRigParts(rig, { poseApplicable: true, topologyApplicable: true }).map((candidate) => candidate.id),
    ["arm-left", "leg-left"],
  );
});

test("the stage and page carry an ensemble instead of one rigid blob", async () => {
  const [stage, page, recognition] = await Promise.all([
    readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8"),
  ]);

  assert.doesNotMatch(stage, /const structuralParts: typeof rig\.parts = \[\]/);
  assert.match(stage, /characters: DrawingExtraction\[\] \| null/);
  assert.match(stage, /articulatedCharacters\.forEach/);
  assert.match(stage, /ensembleActions\?: CharacterAction\[\] \| null/);
  assert.match(stage, /const instanceAction = directedActions\?\.\[instanceIndex\] \?\? currentAction/);
  assert.match(stage, /wallaliveBasePosition/);
  assert.match(recognition, /recognizeDrawingsFromImageUrl/);
  assert.match(recognition, /recognizeDrawingsFromVideo/);
  assert.match(page, /captureEnsembleRef/);
  assert.match(page, /figures found/i);
});
