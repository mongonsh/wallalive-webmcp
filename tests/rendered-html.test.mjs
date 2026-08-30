import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import test from "node:test";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { parseAniGenPreview } from "../app/lib/anigen.ts";
import { extractMedialSkeleton, inferSemanticRig, recoverTargetSilhouette, scoreDrawingCandidate } from "../app/lib/drawing.ts";
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
