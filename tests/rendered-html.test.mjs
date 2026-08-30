import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { parseAniGenPreview } from "../app/lib/anigen.ts";
import { extractMedialSkeleton, inferSemanticRig, inkAroundEnclosedRegion, mapCoverTargetToSource, mergeLearnedPartHints, recoverEnclosedTargetRegion, recoverTargetSilhouette, scoreDrawingCandidate } from "../app/lib/drawing.ts";
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
  assert.match(html, /What if their drawing/);
  assert.match(html, /jumped off the wall/);
  assert.match(html, /START CAMERA/);
  assert.match(html, /PLAY JUDGE DEMO/);
  assert.match(html, /CAMERA-SAFE BY DESIGN/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project|CutRoom/i);
});

test("registers eight strict WebMCP tools without camera authority", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const expectedTools = [
    "inspect_wall_scene",
    "reconstruct_rigged_3d_character",
    "set_character_personality",
    "place_character",
    "animate_character",
    "recolor_character",
    "tell_character_story",
    "list_activity",
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
  assert.match(page, /AniGen joint mesh-skeleton-skinning reconstruction/);
  assert.match(page, /requiresHumanApproval: true/);
  assert.match(page, /UPLOAD APPROVED DRAWING/);
  assert.match(page, /externalUploadApproved/);
  assert.match(page, /ANIGEN RIG \+ DRAWING PARTS/);
  assert.match(page, /COLOR MATCHED/);
  assert.match(page, /viewableDegrees: 360/);
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
  assert.match(stage, /MarchingCubes/);
  assert.match(stage, /silhouette-distance-lens/);
  assert.match(stage, /pointInsideContour/);
  assert.match(stage, /distanceToContour/);
  assert.match(stage, /DecalGeometry/);
  assert.match(stage, /curved-artwork-skin/);
  assert.match(stage, /volume\.field\.fill/);
  assert.match(stage, /wallalive-semantic-character/);
  assert.match(stage, /SphereGeometry/);
  assert.match(stage, /CapsuleGeometry/);
  assert.match(stage, /TubeGeometry/);
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
  assert.match(stage, /raised-lens/);
  assert.match(stage, /raised-pupil/);
  assert.match(stage, /TubeGeometry/);
});

test("ships a compact same-origin learned drawing-part model", async () => {
  const modelUrl = new URL("../public/models/wallalive-parts-v2.onnx", import.meta.url);
  const report = JSON.parse(await readFile(new URL("../public/models/wallalive-parts-v2.json", import.meta.url), "utf8"));
  const model = await stat(modelUrl);
  const recognizer = await readFile(new URL("../app/lib/learned-parts.ts", import.meta.url), "utf8");

  assert.ok(model.size > 950_000 && model.size < 1_100_000, `expected a compact substantive ONNX model, got ${model.size} bytes`);
  assert.equal(report.architecture, "WallAlive Hierarchical PartUNet v2");
  assert.deepEqual(report.coarse_channels, ["foreground", "face", "upper_appendage", "lower_appendage"]);
  assert.equal(report.parameters, 246_508);
  assert.equal(report.training_samples, 6_000);
  assert.equal(report.validation_samples, 700);
  assert.equal(report.real_training_drawings, 392);
  assert.equal(report.real_validation_drawings, 98);
  assert.deepEqual(report.parts, ["body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot"]);
  for (const [kind, iou] of Object.entries(report.validation_iou)) {
    assert.ok(iou >= 0.55, `${kind} held-out IoU should stay above the regression floor, got ${iou}`);
  }
  assert.ok(report.real_validation_iou.body >= 0.7, `real drawing foreground IoU should stay above 0.7, got ${report.real_validation_iou.body}`);
  assert.match(recognizer, /const MODEL_PATH = ["']\/models\/wallalive-parts-v2\.onnx["']/);
  assert.match(recognizer, /const FALLBACK_MODEL_PATH = ["']\/models\/wallalive-parts-v1\.onnx["']/);
  assert.match(recognizer, /supplementMissingHints/);
  assert.match(recognizer, /import\(["']onnxruntime-web\/wasm["']\)/);
  assert.doesNotMatch(recognizer, /https?:\/\//);
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
      { id: "eye-left", kind: "eye", side: "left", parentId: "body", center: { x: -0.14, y: 0.18, z: 0 }, size: { x: 0.1, y: 0.12, z: 0.02 }, rotation: 0, color: "#9c3450", confidence: 0.9, source: "image-region" },
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
