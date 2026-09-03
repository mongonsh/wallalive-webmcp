import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";
import { assessFlatArtworkIsolation, makeTransparentArtworkPixels } from "../app/lib/target-cutout.ts";

test("uses drawing-specific segmentation before general point-guided fallback", async () => {
  const report = JSON.parse(await readFile(new URL("../public/models/wallalive-target-cutout-v3.json", import.meta.url), "utf8"));
  const model = await stat(new URL("../public/models/wallalive-target-cutout-v3.onnx", import.meta.url));
  const runtime = await readFile(new URL("../app/lib/target-cutout.ts", import.meta.url), "utf8");
  const trainer = await readFile(new URL("../ml/train_target_cutout_v3.py", import.meta.url), "utf8");
  const evaluator = await readFile(new URL("../ml/evaluate_target_cutout_onnx.py", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");
  const wall = await readFile(new URL("../app/components/DrawingWall.tsx", import.meta.url), "utf8");
  const learnedParts = await readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8");

  assert.equal(report.model, "wallalive-target-cutout-v3");
  assert.equal(report.test_split_used_for_selection, false);
  assert.equal(report.onnx_export_verified, true);
  assert.ok(report.training_drawings.childlike >= 12_000);
  assert.ok(report.training_drawings.amateur >= 340);
  assert.ok(report.sealed_test_drawings.childlike_official >= 1_900);
  assert.ok(report.sealed_test_drawings.amateur_official >= 70);
  assert.ok(report.official_test.childlike_official.global_iou >= 0.7);
  assert.ok(report.official_test.childlike_official_wall_multi.global_iou >= 0.88);
  assert.ok(report.official_test.amateur_official_wall.global_iou >= 0.9);
  assert.ok(Object.values(report.official_test).every((result) => result.prompt_hit_rate >= 0.99));
  assert.deepEqual(report.onnx_official_test, report.official_test);
  assert.ok(model.size > 1_400_000 && model.size < 1_800_000, `expected a substantive compact target model, got ${model.size} bytes`);
  assert.match(runtime, /wallalive-target-cutout-v3\.onnx/);
  assert.match(runtime, /const MODEL_SIZE = 160/);
  assert.match(runtime, /const MASK_THRESHOLD = 0\.60/);
  assert.match(runtime, /interactive_segmenter_v2\/magic_touch/);
  assert.match(runtime, /InteractiveSegmenter\.createFromModelPath/);
  assert.match(runtime, /brushMode: 1 as import/);
  assert.match(runtime, /brushMode: 2 as import/);
  assert.doesNotMatch(runtime, /vision\.BrushMode/);
  assert.match(runtime, /mediapipe-magic-touch-v2/);
  assert.match(runtime, /targeted-local-extraction-v3/);
  assert.match(runtime, /flat-artwork-alpha-v1/);
  assert.ok(runtime.indexOf("isolateWithFlatArtworkBackground(frame, started)") < runtime.indexOf("isolateWithCompactDrawingModel(frame, started)"));
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
  assert.match(trainer, /identical-character hard negative/);
  assert.match(trainer, /ChildlikeTargetDataset/);
  assert.match(evaluator, /assert_metric_parity/);
  assert.match(page, /REVIEW RIG/);
  assert.match(page, /Artwork preserved/);
  assert.doesNotMatch(page, /% clean cutout/);
  assert.match(page, /Add missing/);
  assert.match(page, /SELECT THE CAST · UP TO 6/);
  assert.match(page, /recognizeDrawingsAtImageTargets\(pending\.url, targets\)/);
  assert.match(page, /enum: \["local-articulated", "neural-full"\]/);
  assert.match(page, /characters=\{character\.created && !neuralAsset \? captureEnsemble : null\}/);
  assert.match(page, /STATIC PUPPET · PRIVATE/);
  assert.match(stage, /setWorld: \(world: ARWorld\) => void/);
  assert.match(stage, /handlesRef\.current\?\.setWorld\(world\)/);
  assert.doesNotMatch(stage, /textureUrl, visible, world\]\)/);
  assert.match(wall, /makeTransparentArtworkPixels\(image\.data, canvas\.width, canvas\.height\)/);
  assert.match(learnedParts, /areDuplicateRecognizedDrawings/);
  assert.match(learnedParts, /left\.textureUrl !== right\.textureUrl/);
  assert.match(learnedParts, /sourceBounds/);
});

test("documents the exact seven-graph primary browser stack", async () => {
  const names = [
    "wallalive-target-cutout-v3.onnx",
    "wallalive-parts-v3.onnx",
    "wallalive-face-v3.onnx",
    "wallalive-face-v4.onnx",
    "wallalive-amateur-pose-v6.onnx",
    "wallalive-topology-v10.onnx",
    "wallalive-sketch-depth-v1.onnx",
  ];
  const files = await Promise.all(names.map((name) => stat(new URL(`../public/models/${name}`, import.meta.url))));
  const bytes = files.reduce((total, file) => total + file.size, 0);
  assert.equal(files.length, 7);
  assert.equal(bytes, 7_600_857);
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

test("accepts a non-rectangular cartoon on a flat background without downsampling its edge", () => {
  const width = 12;
  const height = 12;
  const pixels = new Uint8ClampedArray(width * height * 4).fill(255);
  const paint = (x, y, r = 246, g = 174, b = 56) => {
    const offset = (y * width + x) * 4;
    pixels[offset] = r;
    pixels[offset + 1] = g;
    pixels[offset + 2] = b;
    pixels[offset + 3] = 255;
  };
  // A filled body plus narrow arms and separated feet: deliberately not a
  // page-like solid rectangle.
  for (let y = 3; y <= 8; y += 1) for (let x = 4; x <= 7; x += 1) paint(x, y);
  paint(2, 5); paint(3, 5); paint(8, 5); paint(9, 5);
  paint(4, 9); paint(7, 9);
  const cleaned = makeTransparentArtworkPixels(pixels, width, height);
  const assessment = assessFlatArtworkIsolation(pixels, cleaned, width, height);

  assert.equal(assessment.borderShare, 1);
  assert.equal(assessment.usable, true);
  assert.ok(assessment.boxFill < 0.83);
});

test("does not bypass prompted segmentation for a flat paper rectangle", () => {
  const width = 12;
  const height = 12;
  const pixels = new Uint8ClampedArray(width * height * 4).fill(220);
  for (let pixel = 0; pixel < width * height; pixel += 1) pixels[pixel * 4 + 3] = 255;
  for (let y = 2; y <= 9; y += 1) {
    for (let x = 2; x <= 9; x += 1) {
      const offset = (y * width + x) * 4;
      pixels[offset] = 252;
      pixels[offset + 1] = 250;
      pixels[offset + 2] = 245;
    }
  }
  const cleaned = makeTransparentArtworkPixels(pixels, width, height);
  const assessment = assessFlatArtworkIsolation(pixels, cleaned, width, height);

  assert.equal(assessment.usable, false);
  assert.ok(assessment.boxFill >= 0.83);
});
