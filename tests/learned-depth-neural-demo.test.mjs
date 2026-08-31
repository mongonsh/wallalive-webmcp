import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);

test("ships the verified eight-family learned front/back depth model", async () => {
  const report = JSON.parse(await readFile(new URL("../public/models/wallalive-sketch-depth-v1.json", import.meta.url), "utf8"));
  const model = await stat(new URL("../public/models/wallalive-sketch-depth-v1.onnx", import.meta.url));
  const inference = await readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8");
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");

  assert.equal(report.model, "wallalive-sketch-depth-v1");
  assert.equal(report.onnx_export_verified, true);
  assert.equal(report.test_split_used_for_selection, false);
  assert.equal(report.families.length, 8);
  assert.deepEqual(Object.values(report.family_counts.test), Array(8).fill(96));
  assert.equal(report.sealed_test_examples, 768);
  assert.ok(report.sealed_test.surface_mae_normalized < 0.04);
  assert.ok(report.sealed_test.surface_correlation > 0.91);
  assert.ok(report.sealed_test.front_back_asymmetry_mae > 0.04);
  assert.ok(report.onnx_max_absolute_error < 0.00001);
  assert.ok(model.size > 300_000 && model.size < 400_000);

  assert.match(inference, /DEPTH_MODEL_PATH = "\/models\/wallalive-sketch-depth-v1\.onnx"/);
  assert.match(inference, /front_back_depth/);
  assert.match(inference, /depthRecognition/);
  assert.match(stage, /localFront - fieldZ/);
  assert.match(stage, /localBack \+ fieldZ/);
  assert.doesNotMatch(stage, /Math\.min\(signedEdge, depth - Math\.abs\(fieldZ\)\)/);
});

test("the exact-drawing neural judge asset is closed, colored, full-volume, and actively skinned", async () => {
  const evaluator = fileURLToPath(new URL("../scripts/evaluate-anigen.mjs", import.meta.url));
  const asset = fileURLToPath(new URL("../public/pip-neural-demo.glb", import.meta.url));
  const { stdout } = await execFileAsync(process.execPath, [evaluator, "--inspect", asset], { maxBuffer: 1_000_000 });
  const { inspection } = JSON.parse(stdout);

  assert.equal(inspection.vertices, 68_326);
  assert.equal(inspection.triangles, 136_648);
  assert.ok(inspection.depthRatio > 0.70 && inspection.depthRatio < 0.72);
  assert.equal(inspection.bones, 7);
  assert.equal(inspection.activeJoints, 7);
  assert.ok(inspection.branchInfluenceCoverage > 0.12);
  assert.equal(inspection.weightNormalizationRatio, 1);
  assert.equal(inspection.boundaryEdges, 0);
  assert.equal(inspection.nonManifoldEdges, 0);
  assert.equal(inspection.closedSurface, true);
  assert.equal(inspection.trueVolume, true);
  assert.equal(inspection.rigged, true);
  assert.equal(inspection.colorData, true);
  assert.equal(inspection.assetExtras.smooth_iterations, 30);
  assert.match(inspection.assetExtras.front_color_source, /front-v2-solid-cutout\.png$/);
  assert.match(inspection.assetExtras.hidden_surface_marks, /none projected/);
});
