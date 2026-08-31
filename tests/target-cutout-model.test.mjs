import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";
import { makeTransparentArtworkPixels } from "../app/lib/target-cutout.ts";

test("uses drawing-specific segmentation before general point-guided fallback", async () => {
  const report = JSON.parse(await readFile(new URL("../public/models/wallalive-target-cutout-v2.json", import.meta.url), "utf8"));
  const model = await stat(new URL("../public/models/wallalive-target-cutout-v2.onnx", import.meta.url));
  const runtime = await readFile(new URL("../app/lib/target-cutout.ts", import.meta.url), "utf8");
  const trainer = await readFile(new URL("../ml/train_target_cutout_v2.py", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");
  const wall = await readFile(new URL("../app/components/DrawingWall.tsx", import.meta.url), "utf8");
  const learnedParts = await readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8");

  assert.equal(report.model, "wallalive-target-cutout-v2");
  assert.equal(report.test_split_used_for_selection, false);
  assert.ok(report.training_drawings >= 300);
  assert.ok(report.sealed_test_drawings >= 140);
  assert.ok(report.official_test.iou >= 0.5, `expected paper-scene test IoU >= 0.5, got ${report.official_test.iou}`);
  assert.ok(report.official_test.prompt_hit_rate >= 0.9);
  assert.ok(model.size > 900_000 && model.size < 1_300_000, `expected a substantive compact target model, got ${model.size} bytes`);
  assert.match(runtime, /wallalive-target-cutout-v2\.onnx/);
  assert.match(runtime, /interactive_segmenter_v2\/magic_touch/);
  assert.match(runtime, /InteractiveSegmenter\.createFromModelPath/);
  assert.match(runtime, /brushMode: 1 as import/);
  assert.match(runtime, /brushMode: 2 as import/);
  assert.doesNotMatch(runtime, /vision\.BrushMode/);
  assert.match(runtime, /mediapipe-magic-touch-v2/);
  assert.match(runtime, /targeted-local-extraction-v3/);
  assert.ok(runtime.indexOf("isolateWithCompactDrawingModel(frame, started)") < runtime.lastIndexOf("isolateWithTargetedLocalExtraction(frame, started)"));
  assert.ok(runtime.indexOf("isolateWithMagicTouch(frame)") < runtime.lastIndexOf("isolateWithTargetedLocalExtraction(frame, started)"));
  assert.doesNotMatch(runtime, /sourceCrop\(frame, decoded, false\)/);
  assert.match(runtime, /for \(const scale of \[0\.42, 0\.56\]/);
  assert.match(runtime, /coveragePercent <= 1/);
  assert.match(runtime, /maximumValue <= 1 \? 1 : 128/);
  assert.match(runtime, /promptedComponent/);
  assert.match(runtime, /confidence < 0\.56/);
  assert.match(trainer, /negative paper boundary/);
  assert.match(trainer, /compose_paper_scene/);
  assert.match(page, /REVIEW RIG/);
  assert.match(page, /Artwork preserved/);
  assert.doesNotMatch(page, /% clean cutout/);
  assert.match(page, /Add missing/);
  assert.match(page, /TAP INSIDE THE CHARACTER/);
  assert.match(page, /enum: \["local-articulated", "neural-full"\]/);
  assert.match(page, /characters=\{character\.created && !neuralAsset \? captureEnsemble : null\}/);
  assert.match(page, /INSTANT 3D · PRIVATE/);
  assert.match(stage, /setWorld: \(world: ARWorld\) => void/);
  assert.match(stage, /handlesRef\.current\?\.setWorld\(world\)/);
  assert.doesNotMatch(stage, /textureUrl, visible, world\]\)/);
  assert.match(wall, /makeTransparentArtworkPixels\(image\.data, canvas\.width, canvas\.height\)/);
  assert.match(learnedParts, /existing\.textureUrl === drawing\.textureUrl/);
});

test("authored wall export removes only edge-connected paper and preserves enclosed white artwork", () => {
  const width = 7;
  const height = 7;
  const pixels = new Uint8ClampedArray(width * height * 4).fill(255);
  const paint = (x, y, r, g, b) => {
    const offset = (y * width + x) * 4;
    pixels[offset] = r;
    pixels[offset + 1] = g;
    pixels[offset + 2] = b;
    pixels[offset + 3] = 255;
  };
  for (let x = 2; x <= 4; x += 1) {
    paint(x, 2, 20, 30, 40);
    paint(x, 4, 20, 30, 40);
  }
  for (let y = 2; y <= 4; y += 1) {
    paint(2, y, 20, 30, 40);
    paint(4, y, 20, 30, 40);
  }

  const result = makeTransparentArtworkPixels(pixels, width, height);
  assert.equal(result[(0 * width + 0) * 4 + 3], 0, "edge-connected white wall should be transparent");
  assert.equal(result[(2 * width + 2) * 4 + 3], 255, "ink outline should remain opaque");
  assert.equal(result[(3 * width + 3) * 4 + 3], 255, "enclosed white eye/body detail should remain opaque");
});
