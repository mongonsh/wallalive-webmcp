import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { parseAniGenPreview } from "../app/lib/anigen.ts";
import { extractMedialSkeleton, inferSemanticRig, inkAroundEnclosedRegion, mapCoverTargetToSource, mergeLearnedPartHints, recoverEnclosedTargetRegion, recoverTargetSilhouette, scoreDrawingCandidate } from "../app/lib/drawing.ts";
import { decodeTopology } from "../app/lib/learned-parts.ts";
import { prepareNeuralCharacter, remapDominantHuePixels } from "../app/lib/rigged-model.ts";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the WallAlive product shell and security headers", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  assert.equal(response.headers.get("permissions-policy"), "tools=(self), camera=(self), xr-spatial-tracking=(self)");
  assert.equal(response.headers.get("x-content-type-options"), "nosniff");

  const html = await response.text();
  assert.match(html, /<title>WallAlive — Draw it\. Wake it\. Play\.<\/title>/i);
  assert.match(html, /Draw it/);
  assert.match(html, /Wake it/);
  assert.match(html, /START CAMERA/);
  assert.match(html, /PLAY JUDGE DEMO/);
  assert.match(html, /CAMERA-SAFE BY DESIGN/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project|CutRoom/i);
});

test("registers eight goal-level WebMCP collaboration tools with spatial direction and commerce", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const expectedTools = [
    "inspect_creative_scene",
    "inspect_character_capabilities",
    "request_rigged_3d_cast",
    "stage_magic_show",
    "direct_live_ensemble",
    "orchestrate_spatial_cinematics",
    "generate_shopify_merch_pipeline",
    "list_collaboration_history",
  ];

  for (const tool of expectedTools) assert.match(page, new RegExp(`name: ["']${tool}["']`));

  const registeredNames = [...page.matchAll(/name: ["']([^"']+)["']/g)].map((match) => match[1]);
  assert.equal(registeredNames.filter((name) => expectedTools.includes(name)).length, 8);
  assert.equal(registeredNames.some((name) => /camera|capture|upload/.test(name)), false);
  assert.match(page, /document\.modelContext/);
  assert.match(page, /registerTool\(tool, \{ signal: controller\.signal \}\)/);
  assert.match(page, /additionalProperties: false/);
  assert.match(page, /readOnlyHint/);
  assert.match(page, /Camera capture is human-only/);
  assert.match(page, /cameraFeedExposed: false/);
  assert.match(page, /neuralModelUsed: Boolean\(neuralAssetRef\.current\)/);
  assert.match(page, /full-volume neural mesh \+ skeleton skinning/);
  assert.match(page, /requiresHumanApproval: true/);
  assert.match(page, /approvalControlVisible: true/);
  assert.match(page, /awaiting-human-approval/);
  assert.match(page, /APPROVE &amp; PLAY/);
  assert.match(page, /validateCharacterMove/);
  assert.match(page, /ensembleActions=/);
  assert.match(page, /GENERATE REAL 3D/);
  assert.match(page, /externalUploadApproved/);
  assert.match(page, /FULL NEURAL RIG \+ DRAWING PARTS/);
  assert.match(page, /COLOR MATCHED/);
  assert.match(page, /viewableDegrees: neuralAssetRef\.current \? 360 : 0/);
  assert.match(page, /assessReconstructionReadiness/);
  assert.match(page, /did not substitute a rounded shell/);
  assert.doesNotMatch(page, /localFallbackRef/);
  assert.match(page, /topologyRecognition/);
  assert.match(page, /variable graph decoded from learned centerline, endpoints, and junction fields/);
  assert.match(page, /world: worldRef\.current/);
  assert.match(page, /DRAW ON WALL/);
  assert.match(page, /recognizeDrawingsFromImageUrl\(dataUrl, target, 6\)/);
  assert.match(page, /COPY PERFECT JUDGE DEMO PROMPT/);
  assert.match(page, /cyberpunk-neon/);
  assert.match(page, /SHOPIFY <span>MERCH/);
  assert.match(page, /OPEN SAFE MOCK CHECKOUT/);
  assert.match(page, /No address, card, order, or Shopify account/);
});

test("ships a pressure-aware drawing wall and real switchable Three.js worlds", async () => {
  const wall = await readFile(new URL("../app/components/DrawingWall.tsx", import.meta.url), "utf8");
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(wall, /getCoalescedEvents/);
  assert.match(wall, /event\.pressure/);
  assert.match(wall, /floodFill/);
  for (const tool of ["pencil", "brush", "marker", "spray", "eraser", "fill", "line", "rectangle", "circle", "triangle", "star"]) assert.match(wall, new RegExp(`id: ["']${tool}["']`));
  assert.match(wall, /canvas\.toDataURL\("image\/png"\)/);
  assert.doesNotMatch(css, /wallalive-worlds\.png/);
  assert.match(stage, /buildWorldEnvironment/);
  for (const geometry of ["RoundedBoxGeometry", "LatheGeometry", "ExtrudeGeometry", "TubeGeometry", "OctahedronGeometry", "TorusKnotGeometry"]) assert.match(stage, new RegExp(geometry));
  assert.match(stage, /wallalive-3d-world-\$\{world\}/);
  for (const landmark of ["studio-window", "storybook-castle", "wizard-column", "museum-frame"]) assert.match(stage, new RegExp(landmark));
  assert.match(stage, /perspective, lighting, occlusion, and shadows/);
  assert.match(stage, /MeshPhysicalMaterial/);
  assert.match(stage, /DataTexture/);
  assert.match(stage, /PointLight/);
  assert.match(stage, /worldMotion/);
  assert.match(stage, /wallalivePortalShader/);
  assert.match(stage, /new THREE\.ShaderMaterial/);
  assert.match(stage, /mount\.dataset\.worldMeshes/);
  assert.match(page, /world=\{world\}/);
  assert.match(stage, /new OrbitControls/);
  assert.match(stage, /controls\.enableZoom = true/);
  assert.match(stage, /controls\.enablePan = true/);
  assert.match(stage, /placement\.z -= delta \* 0\.42/);
  assert.doesNotMatch(stage, /camera\.position\.x = Math\.sin/);
});

test("implements local drawing extraction and real WebXR hit testing", async () => {
  const drawing = await readFile(new URL("../app/lib/drawing.ts", import.meta.url), "utf8");
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");
  const riggedModel = await readFile(new URL("../app/lib/rigged-model.ts", import.meta.url), "utf8");

  assert.match(drawing, /getImageData/);
  assert.match(drawing, /toDataURL/);
  assert.match(drawing, /connectedComponents/);
  assert.match(drawing, /recoverSilhouette/);
  assert.match(drawing, /ramerDouglasPeucker/);
  assert.match(drawing, /scoreDrawingCandidate/);
  assert.match(drawing, /extractMedialSkeleton/);
  assert.match(drawing, /Tap the drawing/);
  assert.match(drawing, /Float32Array/);
  assert.match(drawing, /Math\.SQRT2/);
  assert.doesNotMatch(drawing, /fetch\(|XMLHttpRequest|WebSocket/);
  assert.match(stage, /buildArtworkShellGeometry/);
  assert.match(stage, /silhouette-distance-lens/);
  assert.doesNotMatch(stage, /DecalGeometry/);
  assert.match(stage, /contour-preserving rounded 3D puppet/);
  assert.match(stage, /frontMaterial/);
  assert.match(stage, /backMaterial/);
  assert.doesNotMatch(stage, /MarchingCubes|volume\.field\.fill|volume\.blur/);
  assert.match(stage, /wallalive-semantic-character/);
  assert.match(stage, /SkinnedMesh/);
  assert.match(stage, /skinIndex/);
  assert.match(stage, /skinWeight/);
  assert.doesNotMatch(stage, /CapsuleGeometry/);
  assert.match(stage, /rig-\$\{part\.id\}-tip/);
  assert.match(stage, /verified two-joint chains/);
  assert.match(stage, /smoothstep\(bestProgress/);
  assert.match(stage, /mount\.dataset\.articulatedChains/);
  assert.match(drawing, /if \(part\.reviewed\) return true/);
  assert.match(stage, /texturePlane: false/);
  assert.match(stage, /viewableDegrees: 360/);
  assert.match(stage, /GLTFLoader/);
  assert.match(riggedModel, /SkinnedMesh/);
  assert.match(stage, /wallalive-neural-character/);
  assert.match(riggedModel, /wallaliveBaseQuaternion/);
  assert.match(riggedModel, /generated full 3D surface/);
  assert.doesNotMatch(stage, /PlaneGeometry|drawing-curved-over-inflated-front/);
  assert.match(stage, /isSessionSupported\("immersive-ar"\)/);
  assert.match(stage, /requestSession\("immersive-ar"/);
  assert.match(stage, /requiredFeatures: \["hit-test"\]/);
  assert.match(stage, /getHitTestResults/);
  assert.doesNotMatch(stage, /raised-lens|raised-pupil|addInkFeature/);
  assert.match(stage, /projectedSemanticFeatures: false/);
});

test("ships validation-calibrated same-origin drawing, pose, and variable-topology models", async () => {
  const modelUrl = new URL("../public/models/wallalive-parts-v3.onnx", import.meta.url);
  const faceModelUrl = new URL("../public/models/wallalive-face-v3.onnx", import.meta.url);
  const faceV4ModelUrl = new URL("../public/models/wallalive-face-v4.onnx", import.meta.url);
  const poseModelUrl = new URL("../public/models/wallalive-amateur-pose-v6.onnx", import.meta.url);
  const topologyModelUrl = new URL("../public/models/wallalive-topology-v10.onnx", import.meta.url);
  const faceV4Report = JSON.parse(await readFile(new URL("../public/models/wallalive-face-v4.json", import.meta.url), "utf8"));
  const ensembleReport = JSON.parse(await readFile(new URL("../public/models/wallalive-face-ensemble-v4.json", import.meta.url), "utf8"));
  const componentGateReport = JSON.parse(await readFile(new URL("../public/models/wallalive-component-gate-v5.json", import.meta.url), "utf8"));
  const poseReport = JSON.parse(await readFile(new URL("../public/models/wallalive-amateur-pose-v6.json", import.meta.url), "utf8"));
  const topologyReport = JSON.parse(await readFile(new URL("../public/models/wallalive-topology-v10.json", import.meta.url), "utf8"));
  const report = JSON.parse(await readFile(new URL("../public/models/wallalive-parts-v3.json", import.meta.url), "utf8"));
  const model = await stat(modelUrl);
  const faceModel = await stat(faceModelUrl);
  const faceV4Model = await stat(faceV4ModelUrl);
  const poseModel = await stat(poseModelUrl);
  const topologyModel = await stat(topologyModelUrl);
  const recognizer = await readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8");
  const componentGate = await readFile(new URL("../app/lib/face-component-gate.ts", import.meta.url), "utf8");

  assert.ok(model.size > 1_100_000 && model.size < 1_250_000, `expected a compact substantive body ONNX model, got ${model.size} bytes`);
  assert.ok(faceModel.size > 400_000 && faceModel.size < 500_000, `expected a compact substantive face ONNX model, got ${faceModel.size} bytes`);
  assert.ok(faceV4Model.size > 1_700_000 && faceV4Model.size < 1_900_000, `expected a substantive high-resolution face ONNX model, got ${faceV4Model.size} bytes`);
  assert.ok(poseModel.size > 600_000 && poseModel.size < 700_000, `expected a compact substantive pose ONNX model, got ${poseModel.size} bytes`);
  assert.ok(topologyModel.size > 1_550_000 && topologyModel.size < 1_700_000, `expected a compact spatial topology ONNX model, got ${topologyModel.size} bytes`);
  assert.equal(faceV4Report.architecture, "WallAlive ChildlikeSHAPES FaceUNet v4");
  assert.deepEqual(faceV4Report.input, [1, 3, 128, 128]);
  assert.equal(faceV4Report.parameters, 435_624);
  assert.equal(faceV4Report.best_epoch, 8);
  assert.equal(faceV4Report.test_split_used_for_selection, false);
  assert.equal(report.architecture, "WallAlive ChildlikeSHAPES PartUNet v3");
  assert.deepEqual(report.coarse_channels, ["foreground", "head", "torso", "upper_appendage", "lower_appendage"]);
  assert.deepEqual(report.face_parts, ["eye", "cheek", "mouth", "ear"]);
  assert.deepEqual(report.input, [1, 3, 96, 96]);
  assert.deepEqual(report.face_input, [1, 3, 96, 96]);
  assert.equal(report.parameters, 288_109);
  assert.equal(report.face_parameters, 109_832);
  assert.equal(report.official_training_drawings, 12_992);
  assert.equal(report.validation_drawings, 1_000);
  assert.equal(report.official_test_drawings, 1_986);
  assert.equal(report.synthetic_training_drawings, 3_000);
  assert.equal(report.best_epoch, 7);
  assert.deepEqual(report.parts, ["body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot"]);
  const officialFloors = { body: 0.88, eye: 0.55, cheek: 0.05, mouth: 0.45, ear: 0.38, arm: 0.55, hand: 0.48, leg: 0.63, foot: 0.60 };
  for (const [kind, floor] of Object.entries(officialFloors)) {
    assert.ok(report.official_test_part_iou[kind] >= floor, `${kind} official-test IoU should stay above ${floor}, got ${report.official_test_part_iou[kind]}`);
  }
  const faceFloors = { eye: 0.60, cheek: 0.12, mouth: 0.52, ear: 0.40 };
  for (const [kind, floor] of Object.entries(faceFloors)) {
    assert.ok(report.official_test_face_iou[kind] >= floor, `${kind} face-crop official-test IoU should stay above ${floor}, got ${report.official_test_face_iou[kind]}`);
  }
  assert.deepEqual(report.face_thresholds, { eye: 0.72, cheek: 0.24, mouth: 0.72, ear: 0.64 });
  assert.deepEqual(ensembleReport.blend_weight_v4, { eye: 0.5, cheek: 0.3, mouth: 0.6, ear: 0.5 });
  assert.deepEqual(ensembleReport.thresholds, { eye: 0.5664, cheek: 0.1836, mouth: 0.7578, ear: 0.6875 });
  assert.equal(ensembleReport.test_split_used_for_selection, false);
  const ensembleFloors = { eye: 0.64, cheek: 0.18, mouth: 0.56, ear: 0.46 };
  for (const [kind, floor] of Object.entries(ensembleFloors)) {
    assert.ok(ensembleReport.official_test_face_iou[kind] >= floor, `${kind} ensemble official-test IoU should stay above ${floor}, got ${ensembleReport.official_test_face_iou[kind]}`);
    assert.ok(ensembleReport.official_test_face_iou[kind] > report.official_test_face_iou[kind], `${kind} ensemble should outperform the v3 face crop`);
  }
  assert.equal(componentGateReport.test_split_used_for_selection, false);
  assert.equal(componentGateReport.segmentation_configuration_frozen, true);
  assert.equal(componentGateReport.configuration.eye.threshold, 0);
  assert.equal(componentGateReport.configuration.mouth.threshold, 0);
  assert.equal(componentGateReport.configuration.cheek.threshold, 0.24);
  assert.equal(componentGateReport.configuration.ear.threshold, 0.26);
  for (const kind of ["cheek", "ear"]) {
    const metrics = componentGateReport.official_test_browser_metrics[kind];
    assert.ok(metrics.selected.precision > metrics.baseline.precision, `${kind} gate must improve official-test precision`);
    assert.ok(metrics.selected.f1 > metrics.baseline.f1, `${kind} gate must improve official-test component F1`);
    assert.ok(metrics.selected.false_positive_rate_on_absent < metrics.baseline.false_positive_rate_on_absent, `${kind} gate must reduce absent-image false positives`);
    assert.ok(metrics.selected.count_exact_rate > metrics.baseline.count_exact_rate, `${kind} gate must improve count accuracy`);
  }
  assert.deepEqual(componentGateReport.official_test_browser_metrics.eye.selected, componentGateReport.official_test_browser_metrics.eye.baseline);
  assert.deepEqual(componentGateReport.official_test_browser_metrics.mouth.selected, componentGateReport.official_test_browser_metrics.mouth.baseline);
  assert.equal(poseReport.architecture, "WallAlive AmateurPoseNet v6");
  assert.deepEqual(poseReport.input, [1, 3, 96, 96]);
  assert.deepEqual(poseReport.output, [1, 17, 48, 48]);
  assert.equal(poseReport.parameters, 161_133);
  assert.deepEqual(
    [poseReport.training_drawings, poseReport.validation_drawings, poseReport.test_drawings],
    [352, 63, 75],
  );
  assert.equal(poseReport.best_epoch, 24);
  assert.equal(poseReport.test_split_used_for_selection, false);
  assert.equal(poseReport.official_test.pck_0_05, 0.7969);
  assert.equal(poseReport.official_test.pck_0_10, 0.8643);
  assert.equal(poseReport.official_test.mean_error_input_pixels, 5.578);
  assert.deepEqual(poseReport.onnx_official_test, poseReport.official_test);
  assert.equal(poseReport.onnx_export_verified, true);
  assert.equal(topologyReport.architecture, "WallAlive TopologyNet v10");
  assert.deepEqual(topologyReport.input, [1, 3, 96, 96]);
  assert.deepEqual(topologyReport.outputs.topology_fields, [1, 4, 48, 48]);
  assert.deepEqual(topologyReport.field_names, ["foreground", "centerline", "endpoint", "junction"]);
  assert.deepEqual(topologyReport.topology_classes, ["biped", "quadruped", "winged", "aquatic", "radial", "branched", "machine", "chain"]);
  assert.equal(topologyReport.parameters, 402_052);
  assert.equal(topologyReport.real_training_samples, 10_241);
  assert.equal(topologyReport.test_split_used_for_selection, false);
  assert.ok(topologyReport.official_test.centerline_f1_tolerance_1px >= 0.99);
  assert.ok(topologyReport.official_test.field_iou.endpoint >= 0.55);
  assert.ok(topologyReport.official_test.field_iou.junction >= 0.64);
  assert.ok(topologyReport.quickdraw_test_accuracy >= 0.95);
  assert.deepEqual(topologyReport.onnx_official_test, topologyReport.official_test);
  assert.equal(topologyReport.onnx_quickdraw_test_accuracy, topologyReport.quickdraw_test_accuracy);
  assert.equal(topologyReport.onnx_export_verified, true);
  assert.match(componentGate, /faceComponentGateScore/);
  assert.match(componentGate, /model_disagreement|disagreement/);
  assert.match(componentGate, /cheek:[\s\S]*threshold: 0\.24/);
  assert.match(componentGate, /ear:[\s\S]*threshold: 0\.26/);
  assert.match(recognizer, /acceptFaceComponent/);
  assert.doesNotMatch(recognizer, /anchorKinds[^\n]*cheek/);
  assert.match(recognizer, /const BODY_MODEL_PATH = ["']\/models\/wallalive-parts-v3\.onnx["']/);
  assert.match(recognizer, /const FACE_V3_MODEL_PATH = ["']\/models\/wallalive-face-v3\.onnx["']/);
  assert.match(recognizer, /const FACE_V4_MODEL_PATH = ["']\/models\/wallalive-face-v4\.onnx["']/);
  assert.match(recognizer, /const POSE_MODEL_PATH = ["']\/models\/wallalive-amateur-pose-v6\.onnx["']/);
  assert.match(recognizer, /const TOPOLOGY_MODEL_PATH = ["']\/models\/wallalive-topology-v10\.onnx["']/);
  assert.match(recognizer, /FALLBACK_MODEL_PATHS = \[["']\/models\/wallalive-parts-v2\.onnx/);
  assert.match(recognizer, /const BODY_SIZE = 96/);
  assert.match(recognizer, /const FACE_V3_SIZE = 96/);
  assert.match(recognizer, /const FACE_SIZE = 128/);
  assert.match(recognizer, /const POSE_HEATMAP_SIZE = 48/);
  assert.match(recognizer, /const TOPOLOGY_FIELD_SIZE = 48/);
  assert.match(recognizer, /Promise\.all\(\[/);
  assert.match(recognizer, /orderedHumanoid/);
  assert.match(recognizer, /topology\.kind === ["']biped["']/);
  assert.match(recognizer, /function decodeTopology/);
  assert.match(recognizer, /arms >= 1 && arms <= 2 && legs >= 1 && legs <= 2/);
  assert.match(recognizer, /blendFaceLogits/);
  assert.match(recognizer, /const rawX0 = Math\.floor\(sourceX\)/);
  assert.match(recognizer, /const x1 = clamp\(rawX0 \+ 1, 0, sourceSize - 1\)/);
  assert.match(recognizer, /locateHead/);
  assert.match(recognizer, /supplementFallbackHints/);
  assert.match(recognizer, /loadFallbackSessions/);
  assert.match(recognizer, /hints\.filter\(\(hint\) => hint\.kind === ["']eye["']\)\.length < 2/);
  assert.doesNotMatch(recognizer, /fullHints\.filter\(\(hint\) => hint\.kind === ["']eye["'] \|\| hint\.kind === ["']mouth["']\)/);
  assert.match(recognizer, /import\(["']onnxruntime-web\/wasm["']\)/);
  assert.doesNotMatch(recognizer, /https?:\/\//);
});

test("decodes a variable learned topology graph without a fixed human joint count", () => {
  const size = 48;
  const area = size * size;
  const values = new Float32Array(area * 4).fill(-8);
  const set = (channel, x, y, value = 8) => { values[channel * area + y * size + x] = value; };
  const line = (x0, y0, x1, y1) => {
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
    for (let index = 0; index <= steps; index += 1) {
      const x = Math.round(x0 + (x1 - x0) * index / steps);
      const y = Math.round(y0 + (y1 - y0) * index / steps);
      set(0, x, y);
      set(1, x, y);
    }
  };
  line(24, 25, 9, 7);
  line(24, 25, 39, 7);
  line(24, 25, 24, 43);
  for (const [x, y] of [[9, 7], [39, 7], [24, 43]]) {
    set(2, x, y);
    set(2, Math.min(47, x + 1), y, 6);
  }
  set(3, 24, 25);
  set(3, 25, 25, 6);
  const classes = new Float32Array(8).fill(-4);
  classes[5] = 8;
  const topology = decodeTopology(
    { data: values },
    { data: classes },
    { mapPoint: (x, y) => ({ x: x / 96, y: y / 96 }), contentRect: { x: 0, y: 0, width: 96, height: 96 } },
    12,
  );
  assert.equal(topology.kind, "branched");
  assert.equal(topology.applicable, true);
  assert.equal(topology.nodes.filter((node) => node.role === "root").length, 1);
  assert.equal(topology.nodes.filter((node) => node.role === "endpoint").length, 3);
  assert.equal(topology.edges.length, topology.nodes.length - 1);
  assert.ok(topology.edges.every((edge) => edge.path.length >= 2));
});

test("learned semantics snap to exact drawing pixels for position, outline, and color", () => {
  const outline = [
    { x: -0.19, y: 0.23 }, { x: -0.11, y: 0.23 },
    { x: -0.11, y: 0.13 }, { x: -0.19, y: 0.13 },
  ];
  const extraction = {
    previewUrl: "data:image/png;base64,preview",
    textureUrl: "data:image/png;base64,texture",
    contour: [{ x: -0.5, y: -0.6 }, { x: 0.5, y: -0.6 }, { x: 0.5, y: 0.6 }, { x: -0.5, y: 0.6 }],
    skeleton: [{ x: 0, y: 0, radius: 0.3 }],
    analysis: { shapeHint: "round", dominantColor: "#f4eee2", lineColor: "#9c3450", confidence: 0.9 },
    semanticRegions: [{ id: "exact-eye", x: -0.15, y: 0.18, width: 0.08, height: 0.1, color: "#1764a7", pixelCount: 62, density: 0.48, outline }],
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f4eee2",
      lineColor: "#9c3450",
      joints: [],
      detectedKinds: ["body"],
      parts: [{ id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 1, y: 1.2, z: 0.5 }, rotation: 0, color: "#f4eee2", confidence: 1, source: "silhouette-branch" }],
    },
  };
  const result = mergeLearnedPartHints(extraction, [{ kind: "eye", center: { x: 0.4, y: 0.38 }, size: { x: 0.07, y: 0.09 }, rotation: 0.1, confidence: 0.91 }], 24);
  const eye = result.rig.parts.find((part) => part.kind === "eye");
  assert.ok(eye);
  assert.deepEqual(eye.center, { x: -0.15, y: 0.18, z: 0 });
  assert.equal(eye.color, "#1764a7");
  assert.equal(eye.outline, outline);
  assert.equal(eye.source, "learned-model");
  assert.equal(result.learnedRecognition?.latencyMs, 24);
});

test("learned pose bends real limb paths without inventing face parts", () => {
  const parts = [
    { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.8, y: 1, z: 0.42 }, rotation: 0, color: "#f3d48b", confidence: 1, source: "silhouette-branch" },
    { id: "arm-left", kind: "arm", side: "left", parentId: "body", center: { x: -0.38, y: -0.02, z: 0 }, anchor: { x: -0.22, y: 0.18, z: 0 }, size: { x: 0.07, y: 0.32, z: 0.07 }, rotation: 0.2, color: "#f3d48b", confidence: 0.72, source: "silhouette-branch" },
    { id: "hand-left", kind: "hand", side: "left", parentId: "arm-left", center: { x: -0.38, y: -0.02, z: 0 }, size: { x: 0.09, y: 0.09, z: 0.07 }, rotation: 0, color: "#f3d48b", confidence: 0.66, source: "silhouette-branch" },
    { id: "leg-left", kind: "leg", side: "left", parentId: "body", center: { x: -0.19, y: -0.48, z: 0 }, anchor: { x: -0.13, y: -0.28, z: 0 }, size: { x: 0.08, y: 0.32, z: 0.08 }, rotation: 0.1, color: "#f3d48b", confidence: 0.7, source: "silhouette-branch" },
    { id: "foot-left", kind: "foot", side: "left", parentId: "leg-left", center: { x: -0.19, y: -0.48, z: 0 }, size: { x: 0.11, y: 0.08, z: 0.07 }, rotation: 0, color: "#f3d48b", confidence: 0.64, source: "silhouette-branch" },
  ];
  const extraction = {
    previewUrl: "data:image/png;base64,preview",
    textureUrl: "data:image/png;base64,texture",
    contour: [],
    skeleton: [],
    analysis: { shapeHint: "tall", dominantColor: "#f3d48b", secondaryColor: "#96344e", coveragePercent: 42, aspectRatio: 0.8, edgeEnergy: "bold", sourceWidth: 512, sourceHeight: 512, skeletonPoints: 18 },
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f3d48b",
      lineColor: "#96344e",
      parts,
      joints: [],
      detectedKinds: [...new Set(parts.map((part) => part.kind))],
    },
  };
  const joints = {
    nose: [0.5, 0.23], left_eye: [0.44, 0.22], right_eye: [0.56, 0.22], left_ear: [0.4, 0.25], right_ear: [0.6, 0.25],
    left_shoulder: [0.38, 0.34], right_shoulder: [0.62, 0.34], left_elbow: [0.25, 0.42], right_elbow: [0.74, 0.42],
    left_wrist: [0.18, 0.56], right_wrist: [0.82, 0.56], left_hip: [0.43, 0.56], right_hip: [0.57, 0.56],
    left_knee: [0.39, 0.72], right_knee: [0.61, 0.72], left_ankle: [0.34, 0.89], right_ankle: [0.66, 0.89],
  };
  const pose = {
    model: "wallalive-amateur-pose-v6",
    latencyMs: 31,
    applicable: true,
    joints: Object.entries(joints).map(([name, [x, y]]) => ({ name, x, y, confidence: 0.9 })),
  };
  const hints = [
    { kind: "arm", center: { x: 0.22, y: 0.47 }, size: { x: 0.06, y: 0.28 }, rotation: 0.2, confidence: 0.84 },
    { kind: "hand", center: { x: 0.18, y: 0.56 }, size: { x: 0.07, y: 0.07 }, rotation: 0, confidence: 0.78 },
    { kind: "leg", center: { x: 0.38, y: 0.73 }, size: { x: 0.07, y: 0.3 }, rotation: 0.1, confidence: 0.86 },
    { kind: "foot", center: { x: 0.34, y: 0.89 }, size: { x: 0.1, y: 0.06 }, rotation: 0, confidence: 0.8 },
  ];

  const result = mergeLearnedPartHints(extraction, hints, 47, pose);
  const arm = result.rig.parts.find((part) => part.kind === "arm" && part.side === "left");
  const hand = result.rig.parts.find((part) => part.kind === "hand" && part.side === "left");
  assert.equal(arm?.source, "learned-pose");
  assert.equal(arm?.path?.length, 3);
  assert.notEqual(arm?.path?.[1].x, (arm.path[0].x + arm.path[2].x) / 2, "elbow bend must survive as a real path point");
  assert.deepEqual(hand?.center, arm?.path?.[2]);
  assert.equal(hand?.source, "learned-pose");
  assert.equal(result.poseRecognition, pose);
  assert.equal(result.rig.parts.some((part) => ["eye", "ear", "mouth", "cheek"].includes(part.kind)), false);
});

test("an empty learned face result removes heuristic face inventions", () => {
  const parts = [
    { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 0.8, y: 0.9, z: 0.4 }, rotation: 0, color: "#f5efe7", confidence: 1, source: "silhouette-branch" },
    { id: "eye-left", kind: "eye", side: "left", parentId: "body", center: { x: -0.12, y: 0.16, z: 0 }, size: { x: 0.08, y: 0.1, z: 0.02 }, rotation: 0, color: "#7d3040", confidence: 0.62, source: "image-region" },
    { id: "pupil-left", kind: "pupil", side: "left", parentId: "eye-left", center: { x: -0.12, y: 0.16, z: 0.02 }, size: { x: 0.03, y: 0.04, z: 0.01 }, rotation: 0, color: "#241419", confidence: 0.58, source: "image-region" },
    { id: "cheek-left", kind: "cheek", side: "left", parentId: "body", center: { x: -0.18, y: -0.02, z: 0 }, size: { x: 0.07, y: 0.04, z: 0.01 }, rotation: 0, color: "#d9788c", confidence: 0.55, source: "image-region" },
    { id: "leg-left", kind: "leg", side: "left", parentId: "body", center: { x: -0.16, y: -0.5, z: 0 }, size: { x: 0.08, y: 0.3, z: 0.08 }, rotation: 0, color: "#f5efe7", confidence: 0.7, source: "skeleton-branch" },
  ];
  const extraction = {
    previewUrl: "data:image/png;base64,preview",
    textureUrl: "data:image/png;base64,texture",
    contour: [],
    skeleton: [],
    analysis: { shapeHint: "round", dominantColor: "#f5efe7", secondaryColor: "#7d3040", coveragePercent: 40, aspectRatio: 0.9, edgeEnergy: "soft", sourceWidth: 512, sourceHeight: 512, skeletonPoints: 0 },
    rig: {
      version: "wallalive-semantic-rig-v2",
      bodyColor: "#f5efe7",
      lineColor: "#7d3040",
      parts,
      joints: parts.filter((part) => part.parentId).map((part) => ({ id: `joint-${part.id}`, parentId: part.parentId, childId: part.id, x: part.center.x, y: part.center.y })),
      detectedKinds: [...new Set(parts.map((part) => part.kind))],
    },
  };

  const result = mergeLearnedPartHints(extraction, [], 18);
  assert.deepEqual(result.rig.parts.map((part) => part.kind), ["body", "leg"]);
  assert.deepEqual(result.rig.joints.map((joint) => joint.childId), ["leg-left"]);
  assert.deepEqual(result.rig.detectedKinds, ["body", "leg"]);
  assert.deepEqual(result.learnedRecognition.detectedKinds, []);
});

test("ships a verified colored AniGen SkinnedMesh instead of a 2D extrusion", async () => {
  const assetUrl = new URL("../public/anigen-demo.glb", import.meta.url);
  const metadata = await stat(assetUrl);
  assert.ok(metadata.size > 5_000_000, `expected a substantive generated GLB, got ${metadata.size} bytes`);
  const buffer = await readFile(assetUrl);
  const gltf = await new Promise((resolve, reject) => {
    new GLTFLoader().parse(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength), "", resolve, reject);
  });
  let meshes = 0;
  let skinnedMeshes = 0;
  let bones = 0;
  let vertices = 0;
  gltf.scene.traverse((object) => {
    if (object.isMesh) {
      meshes += 1;
      vertices += object.geometry.getAttribute("position")?.count ?? 0;
    }
    if (object.isSkinnedMesh) skinnedMeshes += 1;
    if (object.isBone) bones += 1;
  });
  assert.equal(meshes, 1);
  assert.equal(skinnedMeshes, 1);
  assert.equal(bones, 20);
  assert.ok(vertices > 100_000, `expected detailed generated geometry, got ${vertices} vertices`);
  const semanticRig = {
    version: "wallalive-semantic-rig-v2",
    bodyColor: "#f4eee2",
    lineColor: "#9c3450",
    joints: [],
    detectedKinds: ["body", "eye", "cheek", "mouth"],
    parts: [
      { id: "body", kind: "body", side: "center", parentId: null, center: { x: 0, y: 0, z: 0 }, size: { x: 1.1, y: 1.3, z: 0.6 }, rotation: 0, color: "#f4eee2", confidence: 1, source: "silhouette-branch" },
      { id: "eye-left", kind: "eye", side: "left", parentId: "body", center: { x: -0.14, y: 0.18, z: 0 }, size: { x: 0.1, y: 0.12, z: 0.02 }, rotation: 0, color: "#9c3450", confidence: 0.9, source: "learned-model" },
      { id: "eye-right", kind: "eye", side: "right", parentId: "body", center: { x: 0.14, y: 0.18, z: 0 }, size: { x: 0.1, y: 0.12, z: 0.02 }, rotation: 0, color: "#9c3450", confidence: 0.9, source: "image-region" },
      { id: "cheek-left", kind: "cheek", side: "left", parentId: "body", center: { x: -0.19, y: -0.01, z: 0 }, size: { x: 0.08, y: 0.05, z: 0.02 }, rotation: 0, color: "#dd6f87", confidence: 0.8, source: "image-region" },
      { id: "cheek-right", kind: "cheek", side: "right", parentId: "body", center: { x: 0.19, y: -0.01, z: 0 }, size: { x: 0.08, y: 0.05, z: 0.02 }, rotation: 0, color: "#dd6f87", confidence: 0.8, source: "image-region" },
      { id: "mouth", kind: "mouth", side: "center", parentId: "body", center: { x: 0, y: -0.09, z: 0 }, size: { x: 0.17, y: 0.05, z: 0.02 }, rotation: 0, color: "#9c3450", confidence: 0.84, source: "image-region" },
    ],
  };
  const prepared = prepareNeuralCharacter(gltf.scene, semanticRig);
  assert.deepEqual(prepared.info, {
    meshes: 1,
    skinnedMeshes: 1,
    bones: 20,
    vertices,
    semanticParts: 5,
    detectedKinds: ["eye", "cheek", "mouth"],
  });
  assert.ok(prepared.rigMap.armLeft, "expected a generated left-arm bone branch");
  assert.ok(prepared.rigMap.armRight, "expected a generated right-arm bone branch");
  assert.ok(prepared.rigMap.legLeft, "expected a generated left-leg bone branch");
  assert.ok(prepared.rigMap.legRight, "expected a generated right-leg bone branch");
  assert.ok(prepared.rigMap.arms.length >= 2, "expected all generated arm branches to remain animatable");
  assert.ok(prepared.rigMap.legs.length >= 2, "expected all generated leg branches to remain animatable");
  assert.ok(prepared.semanticMap.eyeLeft, "expected the detected left eye projected onto the neural mesh");
  assert.ok(prepared.semanticMap.eyeRight, "expected the detected right eye projected onto the neural mesh");
  assert.ok(prepared.semanticMap.cheekLeft, "expected the detected left cheek projected onto the neural mesh");
  assert.ok(prepared.semanticMap.cheekRight, "expected the detected right cheek projected onto the neural mesh");
  assert.ok(prepared.semanticMap.mouth, "expected the detected mouth projected onto the neural mesh");
  assert.equal(prepared.character.userData.reconstruction.semanticParts, 5);
  const skinned = gltf.scene.getObjectByProperty("isSkinnedMesh", true);
  assert.ok(skinned?.geometry.getAttribute("skinIndex"), "expected per-vertex bone indices");
  assert.ok(skinned?.geometry.getAttribute("skinWeight"), "expected per-vertex skinning weights");
});

test("parses AniGen preview mesh and skeleton files from nested Gradio output", () => {
  const parsed = parseAniGenPreview([
    { url: "https://example.test/preview_mesh.glb", mime_type: "model/gltf-binary" },
    [{ path: "/tmp/preview_skeleton.glb" }],
  ]);
  assert.equal(parsed.meshUrl, "https://example.test/preview_mesh.glb");
  assert.match(parsed.skeletonUrl ?? "", /preview_skeleton\.glb$/);
});

test("matches the generated dominant body hue while preserving white, black, and accent colors", () => {
  const original = new Uint8ClampedArray([
    247, 205, 41, 255,
    231, 186, 29, 255,
    255, 255, 255, 255,
    12, 15, 18, 255,
    91, 63, 174, 255,
  ]);
  const result = remapDominantHuePixels(original, "#ea783f");
  assert.ok(result.changedPixels >= 2, "expected the yellow generated body cluster to be color-matched");
  assert.ok(result.pixels[0] > result.pixels[1] && result.pixels[1] > result.pixels[2], "expected the body hue to move toward orange");
  assert.deepEqual([...result.pixels.slice(8, 12)], [255, 255, 255, 255], "white eye areas must stay white");
  assert.deepEqual([...result.pixels.slice(12, 16)], [12, 15, 18, 255], "dark line art must stay dark");
  assert.deepEqual([...result.pixels.slice(16, 20)], [91, 63, 174, 255], "unrelated clothing accents must keep their color");
});

test("prefers a targeted line-art character over dense lower-frame clutter", () => {
  const redCharacter = {
    pixelCount: 1800,
    minX: 105,
    minY: 150,
    maxX: 285,
    maxY: 385,
    averageChroma: 92,
  };
  const darkForegroundObject = {
    pixelCount: 15000,
    minX: 145,
    minY: 365,
    maxX: 335,
    maxY: 605,
    averageChroma: 7,
  };
  const characterScore = scoreDrawingCandidate(redCharacter, 480, 640, { x: 0.42, y: 0.47 });
  const clutterScore = scoreDrawingCandidate(darkForegroundObject, 480, 640, { x: 0.42, y: 0.47 });
  assert.ok(characterScore > clutterScore * 20, `character ${characterScore} should decisively beat clutter ${clutterScore}`);
});

test("prefers the character over a rectangular paper border", () => {
  const characterScore = scoreDrawingCandidate({
    pixelCount: 2250,
    minX: 118,
    minY: 146,
    maxX: 304,
    maxY: 405,
    averageChroma: 104,
    edgeFraction: 0.18,
  }, 480, 640, { x: 0.43, y: 0.45 });
  const paperBorderScore = scoreDrawingCandidate({
    pixelCount: 4300,
    minX: 43,
    minY: 74,
    maxX: 422,
    maxY: 468,
    averageChroma: 22,
    edgeFraction: 0.91,
  }, 480, 640, { x: 0.43, y: 0.45 });
  assert.ok(characterScore > paperBorderScore * 20, `character ${characterScore} should decisively beat paper border ${paperBorderScore}`);
});

test("extracts a radius-bearing medial skeleton from a filled character body", () => {
  const size = 41;
  const mask = new Uint8Array(size * size);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (((x - 20) / 15) ** 2 + ((y - 20) / 11) ** 2 <= 1) mask[y * size + x] = 1;
    }
  }
  const skeleton = extractMedialSkeleton(mask, size, size);
  assert.ok(skeleton.length >= 3);
  assert.ok(Math.max(...skeleton.map((point) => point.radius)) > 10);
  assert.ok(skeleton.every((point) => mask[point.y * size + point.x] === 1));
});

test("maps a visible tap through object-fit cover into raw camera coordinates", () => {
  const leftEdge = mapCoverTargetToSource({ x: 0, y: 0.5 }, 1280, 720, 400, 600);
  const center = mapCoverTargetToSource({ x: 0.5, y: 0.5 }, 1280, 720, 400, 600);
  assert.ok(leftEdge.x > 0.3 && leftEdge.x < 0.34, `expected the cropped raw-camera x coordinate, got ${leftEdge.x}`);
  assert.equal(center.x, 0.5);
  assert.equal(center.y, 0.5);
});

test("recovers one upright target silhouette from fragmented colored line art", () => {
  const width = 121;
  const height = 121;
  const mask = new Uint8Array(width * height);
  const center = { x: 60, y: 61 };
  for (let angleIndex = 0; angleIndex < 360; angleIndex += 1) {
    if ((angleIndex > 82 && angleIndex < 96) || (angleIndex > 248 && angleIndex < 264)) continue;
    const angle = angleIndex / 180 * Math.PI;
    const radiusX = 35 + (angleIndex > 55 && angleIndex < 78 ? 7 : 0);
    const radiusY = 43;
    const x = Math.round(center.x + Math.cos(angle) * radiusX);
    const y = Math.round(center.y + Math.sin(angle) * radiusY);
    mask[y * width + x] = 1;
  }
  for (const offset of [-13, 13]) {
    for (let angleIndex = 0; angleIndex < 360; angleIndex += 12) {
      const angle = angleIndex / 180 * Math.PI;
      const x = Math.round(center.x + offset + Math.cos(angle) * 5);
      const y = Math.round(center.y - 10 + Math.sin(angle) * 7);
      mask[y * width + x] = 1;
    }
  }
  const recovered = recoverTargetSilhouette(mask, width, height, { x: center.x / width, y: center.y / height });
  assert.equal(recovered[center.y * width + center.x], 1);
  const pixels = [...recovered.keys()].filter((index) => recovered[index]);
  const xs = pixels.map((index) => index % width);
  const ys = pixels.map((index) => Math.floor(index / width));
  const aspect = (Math.max(...xs) - Math.min(...xs)) / (Math.max(...ys) - Math.min(...ys));
  assert.ok(aspect > 0.72 && aspect < 1.08, `expected an upright body, got aspect ${aspect}`);
});

test("target silhouette excludes disconnected same-color writing and neighboring drawings", () => {
  const width = 161;
  const height = 141;
  const mask = new Uint8Array(width * height);
  const center = { x: 62, y: 72 };
  const plot = (x, y) => {
    if (x >= 0 && x < width && y >= 0 && y < height) mask[y * width + x] = 1;
  };

  // The tapped character: one mostly closed, round outline with two small gaps.
  for (let angleIndex = 0; angleIndex < 360; angleIndex += 1) {
    if ((angleIndex > 72 && angleIndex < 82) || (angleIndex > 250 && angleIndex < 260)) continue;
    const angle = angleIndex / 180 * Math.PI;
    plot(
      Math.round(center.x + Math.cos(angle) * 26),
      Math.round(center.y + Math.sin(angle) * 33),
    );
  }

  // Separate red writing and another sketch surround the character on the paper.
  // They occupy many viewing angles, which used to inflate the global radial fill.
  for (let angleIndex = -105; angleIndex <= 112; angleIndex += 3) {
    const angle = angleIndex / 180 * Math.PI;
    const wobble = angleIndex % 12 === 0 ? 7 : 0;
    plot(
      Math.round(center.x + Math.cos(angle) * (61 + wobble)),
      Math.round(center.y + Math.sin(angle) * (53 + wobble)),
    );
  }
  for (let x = 111; x <= 151; x += 8) {
    for (let y = 37; y <= 57; y += 1) plot(x, y);
  }

  const recovered = recoverTargetSilhouette(mask, width, height, { x: center.x / width, y: center.y / height });
  assert.equal(recovered[center.y * width + center.x], 1, "the tapped character body should be filled");
  assert.equal(recovered[center.y * width + 121], 0, "neighboring drawing must not become part of the target");
  assert.equal(recovered[47 * width + 135], 0, "writing must not become part of the target");

  const selected = [...recovered.keys()].filter((index) => recovered[index]);
  const maxX = Math.max(...selected.map((index) => index % width));
  assert.ok(maxX <= 94, `target mask leaked into neighboring content (max x ${maxX})`);
});

test("target-seeded flood fill closes compression gaps without crossing exterior ink", () => {
  const width = 151;
  const height = 131;
  const mask = new Uint8Array(width * height);
  const center = { x: 58, y: 67 };
  const plot = (x, y) => {
    if (x >= 0 && x < width && y >= 0 && y < height) mask[y * width + x] = 1;
  };
  for (let angleIndex = 0; angleIndex < 360; angleIndex += 1) {
    if ((angleIndex > 86 && angleIndex < 91) || (angleIndex > 267 && angleIndex < 272)) continue;
    const angle = angleIndex / 180 * Math.PI;
    plot(Math.round(center.x + Math.cos(angle) * 29), Math.round(center.y + Math.sin(angle) * 38));
  }
  // An exterior label nearly touches the top-left edge but is not part of the body.
  for (let x = 9; x <= 47; x += 1) {
    plot(x, 17);
    plot(x, 30);
  }
  for (let y = 17; y <= 30; y += 1) {
    plot(9, y);
    plot(47, y);
  }

  const recovered = recoverEnclosedTargetRegion(mask, width, height, { x: center.x / width, y: center.y / height });
  assert.ok(recovered, "expected the tap to remain enclosed after closing small outline gaps");
  assert.equal(recovered[center.y * width + center.x], 1);
  assert.equal(recovered[23 * width + 25], 0, "an exterior label must remain outside the target region");
});

test("localizes candidate strokes to the enclosed character instead of a rectangular paper crop", () => {
  const width = 121;
  const height = 111;
  const region = new Uint8Array(width * height);
  const ink = new Uint8Array(width * height);
  const center = { x: 51, y: 58 };
  for (let y = 32; y <= 84; y += 1) {
    for (let x = 30; x <= 72; x += 1) {
      if (((x - center.x) / 21) ** 2 + ((y - center.y) / 26) ** 2 <= 1) region[y * width + x] = 1;
    }
  }
  // Target outline and a close ear remain eligible.
  for (let angle = 0; angle < 360; angle += 1) {
    const radians = angle / 180 * Math.PI;
    const x = Math.round(center.x + Math.cos(radians) * 24);
    const y = Math.round(center.y + Math.sin(radians) * 29);
    ink[y * width + x] = 1;
  }
  ink[27 * width + 41] = 1;
  // These would be inside the old padded rectangle, but are disconnected
  // handwriting / a neighbouring sketch rather than target anatomy.
  ink[16 * width + 17] = 1;
  ink[51 * width + 104] = 1;

  const localized = inkAroundEnclosedRegion(ink, region, width, height);
  assert.equal(localized[27 * width + 41], 1, "a nearby ear stroke should be preserved");
  assert.equal(localized[16 * width + 17], 0, "diagonal handwriting must be excluded");
  assert.equal(localized[51 * width + 104], 0, "a neighbouring drawing must be excluded");
});

test("builds a semantic joint rig from facial regions and silhouette branches", () => {
  const skeleton = [
    { x: 0, y: 0, radius: 0.3 },
    { x: -0.27, y: 0.4, radius: 0.075 },
    { x: 0.27, y: 0.4, radius: 0.075 },
    { x: -0.45, y: 0.01, radius: 0.055 },
    { x: 0.45, y: 0.01, radius: 0.055 },
    { x: -0.14, y: -0.43, radius: 0.06 },
    { x: 0.14, y: -0.43, radius: 0.06 },
  ];
  const contour = [
    { x: -0.58, y: -0.68 }, { x: 0.58, y: -0.68 },
    { x: 0.58, y: 0.68 }, { x: -0.58, y: 0.68 },
  ];
  const regions = [
    { id: "left-eye", x: -0.13, y: 0.17, width: 0.1, height: 0.12, color: "#9c3450", pixelCount: 80, density: 0.42 },
    { id: "right-eye", x: 0.13, y: 0.17, width: 0.1, height: 0.12, color: "#9c3450", pixelCount: 82, density: 0.43 },
    { id: "left-pupil", x: -0.13, y: 0.17, width: 0.025, height: 0.035, color: "#251019", pixelCount: 18, density: 0.72 },
    { id: "right-pupil", x: 0.13, y: 0.17, width: 0.025, height: 0.035, color: "#251019", pixelCount: 19, density: 0.76 },
    { id: "left-cheek", x: -0.18, y: -0.01, width: 0.09, height: 0.06, color: "#9c3450", pixelCount: 48, density: 0.38 },
    { id: "right-cheek", x: 0.18, y: -0.01, width: 0.09, height: 0.06, color: "#9c3450", pixelCount: 46, density: 0.36 },
    { id: "mouth-line", x: 0, y: -0.08, width: 0.18, height: 0.05, color: "#9c3450", pixelCount: 60, density: 0.35 },
  ];
  const rig = inferSemanticRig(skeleton, contour, regions, "#f4eee2", "#9c3450");
  for (const kind of ["body", "eye", "pupil", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot"]) {
    assert.ok(rig.detectedKinds.includes(kind), `expected semantic ${kind}`);
  }
  assert.equal(rig.parts.filter((part) => part.kind === "pupil").length, 2, "must preserve pupil regions drawn inside the eyes");
  assert.ok(rig.parts.length >= 15);
  assert.ok(rig.joints.some((joint) => joint.childId === "arm-right"));
  assert.ok(rig.parts.some((part) => part.kind === "eye" && part.source === "image-region"));
  assert.ok(rig.parts.some((part) => part.kind === "leg" && part.source === "silhouette-branch"));
});

test("recovers one merged side hand from a real contour notch without inventing the smooth-side arm", () => {
  const skeleton = [
    { x: 0, y: 0.12, radius: 0.32 },
    { x: -0.12, y: 0.46, radius: 0.04 }, { x: 0.22, y: 0.47, radius: 0.04 },
    { x: -0.04, y: -0.43, radius: 0.05 }, { x: 0.16, y: -0.48, radius: 0.04 },
  ];
  const contour = [
    { x: -0.2, y: -0.25 }, { x: -0.28, y: -0.17 }, { x: -0.32, y: -0.02 },
    { x: -0.32, y: 0.13 }, { x: -0.29, y: 0.29 }, { x: -0.22, y: 0.44 },
    { x: 0.19, y: 0.5 }, { x: 0.26, y: 0.36 }, { x: 0.32, y: 0.2 },
    { x: 0.32, y: 0.02 }, { x: 0.31, y: -0.05 }, { x: 0.29, y: -0.09 },
    { x: 0.32, y: -0.15 }, { x: 0.32, y: -0.19 }, { x: 0.3, y: -0.24 },
    { x: 0.3, y: -0.36 }, { x: 0.2, y: -0.48 }, { x: -0.04, y: -0.45 },
  ];
  const regions = [
    { id: "eye-l", x: -0.13, y: 0.2, width: 0.09, height: 0.11, color: "#9c3450", pixelCount: 70, density: 0.5 },
    { id: "eye-r", x: 0.13, y: 0.2, width: 0.09, height: 0.11, color: "#9c3450", pixelCount: 70, density: 0.5 },
  ];
  const rig = inferSemanticRig(skeleton, contour, regions, "#f4eee2", "#9c3450");
  assert.ok(rig.parts.some((part) => part.id === "arm-right" && part.source === "structural-inference"));
  assert.ok(rig.parts.some((part) => part.id === "hand-right"));
  assert.equal(rig.parts.some((part) => part.id === "arm-left"), false);
});

test("handles paired, one-eyed, animal-like, and faceless drawing anatomy without inventing parts", () => {
  const contour = (halfWidth, halfHeight) => [
    { x: -halfWidth, y: -halfHeight }, { x: halfWidth, y: -halfHeight },
    { x: halfWidth, y: halfHeight }, { x: -halfWidth, y: halfHeight },
  ];
  const cases = [
    {
      name: "round two-eyed blob",
      skeleton: [{ x: 0, y: 0, radius: 0.34 }],
      contour: contour(0.56, 0.62),
      regions: [
        { id: "eye-l", x: -0.14, y: 0.2, width: 0.11, height: 0.13, color: "#164b79", pixelCount: 70, density: 0.5 },
        { id: "eye-r", x: 0.14, y: 0.2, width: 0.11, height: 0.13, color: "#164b79", pixelCount: 70, density: 0.5 },
        { id: "mouth", x: 0, y: -0.08, width: 0.16, height: 0.04, color: "#e54166", pixelCount: 45, density: 0.4 },
      ],
      present: ["eye", "mouth"],
      absent: ["pupil", "ear", "arm", "leg"],
    },
    {
      name: "one-eyed asymmetric creature",
      skeleton: [{ x: 0, y: 0, radius: 0.31 }],
      contour: contour(0.5, 0.65),
      regions: [
        { id: "eye", x: 0.02, y: 0.22, width: 0.18, height: 0.2, color: "#f5e7a4", pixelCount: 110, density: 0.62 },
        { id: "pupil", x: 0.03, y: 0.22, width: 0.05, height: 0.07, color: "#172120", pixelCount: 28, density: 0.8 },
        { id: "mouth", x: -0.01, y: -0.13, width: 0.2, height: 0.045, color: "#6f233e", pixelCount: 42, density: 0.38 },
      ],
      present: ["eye", "pupil", "mouth"],
      absent: ["cheek", "ear", "arm", "leg"],
    },
    {
      name: "wide animal-like body",
      skeleton: [
        { x: 0, y: 0, radius: 0.25 },
        { x: -0.38, y: 0.32, radius: 0.07 }, { x: 0.38, y: 0.32, radius: 0.07 },
        { x: -0.3, y: -0.36, radius: 0.06 }, { x: 0.3, y: -0.36, radius: 0.06 },
        { x: -0.65, y: 0, radius: 0.055 }, { x: 0.65, y: 0, radius: 0.055 },
      ],
      contour: contour(0.72, 0.42),
      regions: [
        { id: "eye-l", x: -0.15, y: 0.13, width: 0.08, height: 0.09, color: "#2d241d", pixelCount: 48, density: 0.6 },
        { id: "eye-r", x: 0.15, y: 0.13, width: 0.08, height: 0.09, color: "#2d241d", pixelCount: 47, density: 0.59 },
      ],
      present: ["eye", "ear", "arm", "leg"],
      absent: ["pupil", "cheek"],
    },
    {
      name: "faceless articulated machine",
      skeleton: [
        { x: 0, y: 0, radius: 0.28 },
        { x: -0.55, y: 0.02, radius: 0.08 }, { x: 0.55, y: 0.02, radius: 0.08 },
        { x: -0.2, y: -0.45, radius: 0.08 }, { x: 0.2, y: -0.45, radius: 0.08 },
      ],
      contour: contour(0.68, 0.58),
      regions: [],
      present: ["arm", "leg"],
      absent: ["eye", "pupil", "cheek", "mouth", "ear"],
    },
  ];

  for (const fixture of cases) {
    const rig = inferSemanticRig(fixture.skeleton, fixture.contour, fixture.regions, "#d7c7a4", "#342d29");
    for (const kind of fixture.present) assert.ok(rig.detectedKinds.includes(kind), `${fixture.name} should preserve ${kind}`);
    for (const kind of fixture.absent) assert.equal(rig.detectedKinds.includes(kind), false, `${fixture.name} must not invent ${kind}`);
    assert.equal(rig.parts[0].color, "#d7c7a4", `${fixture.name} should preserve its sampled body color`);
    assert.ok(rig.parts.every((part) => Number.isFinite(part.center.x) && Number.isFinite(part.center.y)), `${fixture.name} should keep finite part positions`);
  }
});
