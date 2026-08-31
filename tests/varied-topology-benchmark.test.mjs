import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import test from "node:test";

const execFileAsync = promisify(execFile);

test("keeps the varied filled-drawing benchmark outside every topology-v10 split", async () => {
  const root = new URL("../eval/varied-drawings/", import.meta.url);
  const manifest = JSON.parse(await readFile(new URL("manifest.json", root), "utf8"));
  const strokes = JSON.parse(await readFile(new URL("source-strokes.json", root), "utf8"));
  const results = JSON.parse(await readFile(new URL("topology-v10-results.json", root), "utf8"));
  const expected = [
    ["cat", "quadruped"],
    ["bird", "winged"],
    ["fish", "aquatic"],
    ["octopus", "radial"],
    ["tree", "branched"],
    ["car", "machine"],
    ["snake", "chain"],
    ["snowman", "biped"],
  ];

  assert.deepEqual(manifest.cases.map((item) => [item.id, item.expected_topology]), expected);
  assert.match(manifest.selection_policy, /source line is >= 2400/);
  assert.match(manifest.selection_policy, /ending before line 1960/);
  assert.equal(results.model, "wallalive-topology-v10");
  assert.equal(results.training_split_ends_before_line, 1960);
  assert.equal(results.passed, true);
  assert.deepEqual(results.cases.map((item) => [item.id, item.predicted]), expected);

  for (const item of manifest.cases) {
    assert.ok(item.source_line_index >= 2400);
    assert.equal(strokes[item.id].source_line_index, item.source_line_index);
    assert.equal(strokes[item.id].key_id, item.source_key_id);
    const png = await readFile(new URL(item.input, root));
    assert.equal(png.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
    assert.equal(png.readUInt32BE(16), 512);
    assert.equal(png.readUInt32BE(20), 512);
  }
});

test("bundled AniGen reference asset is closed volumetric colored and skinned", async () => {
  const evaluator = fileURLToPath(new URL("../scripts/evaluate-anigen.mjs", import.meta.url));
  const asset = fileURLToPath(new URL("../public/anigen-demo.glb", import.meta.url));
  const { stdout } = await execFileAsync(process.execPath, [evaluator, "--inspect", asset], { maxBuffer: 1_000_000 });
  const { inspection } = JSON.parse(stdout);
  assert.ok(inspection.vertices > 150_000);
  assert.ok(inspection.triangles > 300_000);
  assert.ok(inspection.depthRatio > 0.2);
  assert.ok(inspection.bones >= 20);
  assert.equal(inspection.weightNormalizationRatio, 1);
  assert.equal(inspection.boundaryEdges, 0);
  assert.equal(inspection.nonManifoldEdges, 0);
  assert.equal(inspection.closedSurface, true);
  assert.equal(inspection.trueVolume, true);
  assert.equal(inspection.rigged, true);
  assert.equal(inspection.colorData, true);
});

test("all eight held-out drawing families produce asymmetric closed learned-depth rigs", async () => {
  const report = JSON.parse(await readFile(new URL("../eval/varied-drawings/local-depth-v1-results.json", import.meta.url), "utf8"));
  const topology = JSON.parse(await readFile(new URL("../eval/varied-drawings/topology-v10-results.json", import.meta.url), "utf8"));
  assert.equal(report.model, "wallalive-sketch-depth-v1");
  assert.equal(report.cases, 8);
  assert.equal(report.passed, true);
  assert.equal(topology.cases.length, 8);
  assert.equal(topology.passed, true);
  for (const item of report.results) {
    assert.ok(item.vertices > 10_000);
    assert.ok(item.triangles > 20_000);
    assert.ok(item.depth_ratio > 0.17);
    assert.ok(item.mean_front_back_asymmetry > 0.005);
    assert.equal(item.bones, item.active_joints);
    assert.ok(item.branch_influence_coverage > 0.12);
    assert.equal(item.closed, true);
    assert.equal(item.true_volume, true);
    assert.equal(item.rigged, true);
    assert.equal(item.color, true);
  }
});
