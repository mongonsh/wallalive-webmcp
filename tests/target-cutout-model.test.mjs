import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";

test("ships a point-prompted paper-scene model with sealed clean and hard-negative evaluation", async () => {
  const report = JSON.parse(await readFile(new URL("../public/models/wallalive-target-cutout-v2.json", import.meta.url), "utf8"));
  const model = await stat(new URL("../public/models/wallalive-target-cutout-v2.onnx", import.meta.url));
  const runtime = await readFile(new URL("../app/lib/target-cutout.ts", import.meta.url), "utf8");
  const trainer = await readFile(new URL("../ml/train_target_cutout_v2.py", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.equal(report.model, "wallalive-target-cutout-v2");
  assert.equal(report.test_split_used_for_selection, false);
  assert.ok(report.training_drawings >= 300);
  assert.ok(report.sealed_test_drawings >= 140);
  assert.ok(report.official_test.iou >= 0.5, `expected paper-scene test IoU >= 0.5, got ${report.official_test.iou}`);
  assert.ok(report.official_test.prompt_hit_rate >= 0.9);
  assert.ok(model.size > 900_000 && model.size < 1_300_000, `expected a substantive compact target model, got ${model.size} bytes`);
  assert.match(runtime, /wallalive-target-cutout-v2\.onnx/);
  assert.match(runtime, /promptedComponent/);
  assert.match(runtime, /confidence < 0\.56/);
  assert.match(trainer, /negative paper boundary/);
  assert.match(trainer, /compose_paper_scene/);
  assert.match(page, /CHECK PARTS/);
  assert.match(page, /Add missing/);
  assert.match(page, /TAP INSIDE THE CHARACTER/);
});
