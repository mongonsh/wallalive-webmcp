"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { selectAnimatableRigParts, type CharacterRig, type ContourPoint, type DrawingExtraction, type LearnedDepthField, type SkeletonPoint } from "../lib/drawing";
import { buildArtworkShellGeometry } from "../lib/artwork-shell";
import { hasRecognizableArtworkSurface } from "../lib/mesh-materials";
import { disposeObject, prepareNeuralCharacter, type NeuralRigMap, type NeuralSemanticMap, type RiggedAssetInfo } from "../lib/rigged-model";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin" | "float";
export type ARWorld = "studio" | "storybook" | "wizard" | "museum";
export type LightingMood = "cyberpunk-neon" | "sunset-warm" | "moonlight";
export type CameraPreset = "cinematic-orbit" | "low-angle-hero" | "overhead";

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
  rotateBy: (yaw: number, pitch: number) => void;
  moveBy: (x: number, z: number) => void;
};

type ARStageProps = {
  characters: DrawingExtraction[] | null;
  contour: ContourPoint[] | null;
  skeleton: SkeletonPoint[] | null;
  textureUrl: string | null;
  rig: CharacterRig | null;
  depth: LearnedDepthField | null;
  action: CharacterAction;
  ensembleActions?: CharacterAction[] | null;
  world: ARWorld;
  lightingMood: LightingMood;
  cameraPreset: CameraPreset;
  accent: string;
  inflation: number;
  neuralAssetUrl: string | null;
  visible: boolean;
  onCapability: (supported: boolean) => void;
  onPlaced: (surface: "screen" | "world", x: number, y: number) => void;
  onNeuralAssetInfo: (info: RiggedAssetInfo | null) => void;
};

type SceneHandles = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  character: THREE.Group;
  reticle: THREE.Mesh;
  setWorld: (world: ARWorld) => void;
  setLightingMood: (mood: LightingMood) => void;
  setCameraPreset: (preset: CameraPreset) => void;
  controls: OrbitControls;
  dispose: () => void;
};

type WorldEnvironment = {
  group: THREE.Group;
  background: THREE.Color;
  fog: THREE.Fog;
};

const cameraPresets: Record<CameraPreset, { position: [number, number, number]; target: [number, number, number] }> = {
  "cinematic-orbit": { position: [3.1, 1.15, 4.9], target: [0, -0.12, -1.75] },
  "low-angle-hero": { position: [2.4, -0.62, 3.45], target: [0, 0.05, -1.65] },
  overhead: { position: [0.35, 5.25, 2.25], target: [0, -0.62, -2.15] },
};

const moodPalettes: Record<LightingMood, {
  hemisphere: [number, number, number];
  key: [number, number];
  fill: [number, number];
  rim: [number, number];
  grid: number;
  exposure: number;
}> = {
  "cyberpunk-neon": {
    hemisphere: [0x7bf7ff, 0x160c2d, 0.58], key: [0xff4fd8, 2.7], fill: [0x2ff3ff, 2.15], rim: [0xa5ff48, 1.25], grid: 0x42f5e9, exposure: 1.02,
  },
  "sunset-warm": {
    hemisphere: [0xffe3b0, 0x2f4258, 0.88], key: [0xffc178, 2.35], fill: [0x8ccce4, 0.82], rim: [0xff6b4d, 0.76], grid: 0xffb85c, exposure: 0.94,
  },
  moonlight: {
    hemisphere: [0xb8ccff, 0x0b1626, 0.48], key: [0xc7d8ff, 1.45], fill: [0x6f7cff, 1.05], rim: [0x58e1ff, 1.18], grid: 0x6c8cff, exposure: 0.78,
  },
};

const worldMaterial = (color: number, roughness = 0.82, emissive = 0x000000) => new THREE.MeshStandardMaterial({
  color,
  roughness,
  metalness: 0.02,
  emissive,
  emissiveIntensity: emissive ? 0.5 : 0,
});

/** Builds actual perspective geometry. No world is a bitmap or CSS plate. */
export function buildWorldEnvironment(world: ARWorld): WorldEnvironment {
  const group = new THREE.Group();
  group.name = `wallalive-3d-world-${world}`;
  group.userData.world = world;
  group.userData.rendering = "procedural Three.js geometry with perspective, lighting, occlusion, and shadows";

  const add = (name: string, geometry: THREE.BufferGeometry, material: THREE.Material, x: number, y: number, z: number, rotationY = 0) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = name;
    mesh.position.set(x, y, z);
    mesh.rotation.y = rotationY;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  };
  const box = (name: string, size: [number, number, number], color: number, position: [number, number, number], rotationY = 0) => (
    add(name, new THREE.BoxGeometry(...size), worldMaterial(color), ...position, rotationY)
  );
  const cylinder = (name: string, radius: number, height: number, color: number, position: [number, number, number], sides = 20) => (
    add(name, new THREE.CylinderGeometry(radius, radius * 1.04, height, sides), worldMaterial(color), ...position)
  );
  const cone = (name: string, radius: number, height: number, color: number, position: [number, number, number], sides = 20) => (
    add(name, new THREE.ConeGeometry(radius, height, sides), worldMaterial(color), ...position)
  );
  const sphere = (name: string, radius: number, color: number, position: [number, number, number], emissive = 0) => (
    add(name, new THREE.SphereGeometry(radius, 20, 14), worldMaterial(color, 0.68, emissive), ...position)
  );

  const settings: Record<ARWorld, { background: number; fog: number; floor: number; wall: number }> = {
    studio: { background: 0xf4ead5, fog: 0xe8dcc4, floor: 0xd9b98f, wall: 0xf1dfbd },
    storybook: { background: 0x9fd6dc, fog: 0xb8d9cf, floor: 0x74a96c, wall: 0xb7cfc3 },
    wizard: { background: 0x172433, fog: 0x223346, floor: 0x34364f, wall: 0x242a40 },
    museum: { background: 0xe9dfcb, fog: 0xd8ccb8, floor: 0x8e6c52, wall: 0xe8dcc4 },
  };
  const palette = settings[world];
  box(`${world}-floor`, [12, 0.12, 11], palette.floor, [0, -1.22, -2.2]);
  box(`${world}-back-wall`, [12, 6.4, 0.16], palette.wall, [0, 1.25, -6.4]);
  box(`${world}-left-wall`, [0.16, 6.4, 8.4], palette.wall, [-5.4, 1.25, -2.25]);
  box(`${world}-right-wall`, [0.16, 6.4, 8.4], palette.wall, [5.4, 1.25, -2.25]);

  if (world === "studio") {
    box("studio-window", [2.4, 1.65, 0.12], 0x9dcbd2, [-2.45, 1.3, -6.26]);
    box("studio-window-cross-v", [0.07, 1.65, 0.16], 0xf8f0dc, [-2.45, 1.3, -6.15]);
    box("studio-window-cross-h", [2.4, 0.07, 0.16], 0xf8f0dc, [-2.45, 1.3, -6.15]);
    box("studio-shelf", [2.1, 0.12, 0.48], 0x6c4836, [2.55, 0.25, -5.95]);
    [-0.72, 0, 0.72].forEach((offset, index) => box(`studio-book-${index}`, [0.32, 0.78 - index * 0.08, 0.28], [0xe66551, 0x4ea9aa, 0xe0af45][index], [1.9 + offset, 0.68, -5.78]));
    cylinder("studio-pot", 0.42, 0.48, 0xd2644c, [-3.6, -0.94, -5.2]);
    cone("studio-plant-a", 0.56, 1.55, 0x47785a, [-3.75, 0.04, -5.15], 8);
    cone("studio-plant-b", 0.46, 1.28, 0x5b9569, [-3.3, -0.02, -5.05], 8);
  }

  if (world === "storybook") {
    box("storybook-path", [1.6, 0.06, 7.4], 0xe7c883, [0, -1.12, -2.72]);
    box("storybook-castle", [3.15, 2.2, 0.72], 0xf1c5ad, [0, -0.05, -5.72]);
    [-1.8, 1.8].forEach((x, index) => {
      cylinder(`storybook-tower-${index}`, 0.62, 2.8, 0xe8a9a5, [x, 0.08, -5.42], 16);
      cone(`storybook-roof-${index}`, 0.86, 1.34, 0x6f79b7, [x, 2.14, -5.42], 16);
    });
    box("storybook-gate", [0.78, 1.3, 0.24], 0x725044, [0, -0.46, -5.28]);
    [-3.55, 3.55].forEach((x, index) => {
      cylinder(`storybook-tree-trunk-${index}`, 0.18, 1.45, 0x744b34, [x, -0.46, -4.72], 10);
      sphere(`storybook-tree-crown-${index}`, 0.88, 0x5a9c68, [x, 0.65, -4.72]);
      sphere(`storybook-tree-crown-small-${index}`, 0.57, 0x82b86a, [x + (index ? -0.45 : 0.45), 0.5, -4.62]);
    });
    sphere("storybook-cloud-a", 0.6, 0xffffff, [-2.4, 2.5, -5.9]);
    sphere("storybook-cloud-b", 0.43, 0xffffff, [-1.8, 2.54, -5.85]);
  }

  if (world === "wizard") {
    [-3.55, -2.1, 2.1, 3.55].forEach((x, index) => {
      cylinder(`wizard-column-${index}`, 0.3, 4.6, 0x646079, [x, 0.7, -5.55], 12);
      add(`wizard-arch-${index}`, new THREE.TorusGeometry(0.72, 0.14, 10, 24, Math.PI), worldMaterial(0x716b88), x, 2.9, -5.45);
    });
    box("wizard-dais", [3.15, 0.22, 1.55], 0x4c4265, [0, -0.96, -4.82]);
    sphere("wizard-orb-a", 0.19, 0x9cefff, [-2.5, 1.25, -4.6], 0x35c9ff);
    sphere("wizard-orb-b", 0.14, 0xd3a4ff, [2.55, 1.85, -4.9], 0xb56cff);
    sphere("wizard-orb-c", 0.11, 0xffd77b, [0.7, 2.45, -5.3], 0xffb52b);
    [-1.2, 0, 1.2].forEach((x, index) => box(`wizard-rune-step-${index}`, [0.82, 0.08, 0.55], [0x57627f, 0x62577f, 0x4f6c7c][index], [x, -1.08 + index * 0.015, -3.0 - index * 0.8]));
  }

  if (world === "museum") {
    [-3.75, 3.75].forEach((x, index) => {
      cylinder(`museum-column-${index}`, 0.38, 4.8, 0xf0e7d3, [x, 0.82, -5.45], 20);
      cylinder(`museum-column-base-${index}`, 0.55, 0.24, 0xd5c7ad, [x, -1.0, -5.45], 20);
    });
    [-2.25, 0, 2.25].forEach((x, index) => {
      box(`museum-frame-${index}`, [1.55, 1.75, 0.18], 0x9e7441, [x, 1.05, -6.18]);
      box(`museum-art-${index}`, [1.24, 1.42, 0.2], [0x83a8a8, 0xc88371, 0xd5ad55][index], [x, 1.05, -6.05]);
    });
    box("museum-plinth", [1.42, 1.05, 1.15], 0xe3d9c5, [0, -0.65, -4.7]);
    sphere("museum-sculpture", 0.55, 0x93a5a8, [0, 0.2, -4.7]);
  }

  return {
    group,
    background: new THREE.Color(palette.background),
    fog: new THREE.Fog(palette.fog, 5.5, 15),
  };
}

function segmentProjection(x: number, y: number, start: ContourPoint, end: ContourPoint) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const amount = lengthSquared ? Math.min(1, Math.max(0, ((x - start.x) * dx + (y - start.y) * dy) / lengthSquared)) : 0;
  return {
    amount,
    distance: Math.hypot(x - (start.x + dx * amount), y - (start.y + dy * amount)),
  };
}

export function buildCharacter(
  contour: ContourPoint[],
  skeleton: SkeletonPoint[],
  rig: CharacterRig,
  depth: LearnedDepthField | null,
  textureUrl: string | null,
  accent: string,
  inflation: number,
  applicability = { poseApplicable: false, topologyApplicable: false },
) {
  const character = new THREE.Group();
  character.name = "wallalive-semantic-character";

  const artworkTexture = textureUrl ? new THREE.TextureLoader().load(textureUrl) : null;
  if (artworkTexture) {
    artworkTexture.colorSpace = THREE.SRGBColorSpace;
    artworkTexture.anisotropy = 8;
  }
  const sideMaterial = new THREE.MeshPhysicalMaterial({
    color: rig.bodyColor,
    roughness: 0.76,
    metalness: 0,
    clearcoat: 0.08,
    clearcoatRoughness: 0.8,
    side: THREE.DoubleSide,
  });
  const frontMaterial = artworkTexture ? new THREE.MeshStandardMaterial({
    map: artworkTexture,
    transparent: true,
    alphaTest: 0.035,
    roughness: 0.82,
    metalness: 0,
    side: THREE.FrontSide,
  }) : sideMaterial;
  const backMaterial = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rig.bodyColor).lerp(new THREE.Color(0xfff7e4), 0.12),
    roughness: 0.86,
    metalness: 0,
    side: THREE.DoubleSide,
  });
  const bodyPart = rig.parts.find((part) => part.kind === "body");
  // The exact segmented outline is the source of truth. Build a closed,
  // contour-preserving rounded 3D puppet instead of blurring the artwork into
  // an implicit Marching Cubes blob. Learned depth may shape the shell by a
  // small bounded amount, but it can never move the silhouette or remap art.
  // This replaces the old voxel bounds (`localFront - fieldZ` and
  // `localBack + fieldZ`) while retaining independent learned front/back depth.
  const requestedHalfDepth = Math.min(0.16, Math.max(0.075, (bodyPart?.size.z ?? 0.28) * 0.42));
  const shell = buildArtworkShellGeometry(contour, depth, requestedHalfDepth, inflation, 2);
  const compactGeometry = shell.geometry;
  const texturedFrontTriangles = shell.frontTriangleCount;
  const neutralBackTriangles = shell.backTriangleCount;
  const totalTriangles = shell.frontTriangleCount + shell.backTriangleCount + shell.sideTriangleCount;
  const artworkSurfaceCoverage = texturedFrontTriangles / Math.max(1, totalTriangles);
  if (!hasRecognizableArtworkSurface(texturedFrontTriangles, totalTriangles, Boolean(artworkTexture))) {
    throw new Error("The private 3D preview could not preserve the drawing texture, so WallAlive refused to show it.");
  }
  // Only pose/topology paths that passed the character gate may deform the
  // artwork. Heuristic labels stay visible in the editor but never become
  // bones. This keeps safety while restoring real arm and leg movement.
  const structuralParts = selectAnimatableRigParts(rig, applicability);
  // If no branch passes the gate, this degrades to one safe root bone over one continuous surface instead of inventing anatomy.
  const rootBone = new THREE.Bone();
  rootBone.name = "rig-body";
  const branchBones = structuralParts.map((part) => {
    const anchor = part.anchor ?? bodyPart?.center ?? { x: 0, y: 0, z: 0 };
    const bone = new THREE.Bone();
    bone.name = `rig-${part.id}`;
    bone.position.set(anchor.x, anchor.y, 0);
    bone.userData.baseRotationZ = 0;
    rootBone.add(bone);
    return { part, bone, anchor };
  });
  const positions = compactGeometry.getAttribute("position");
  const skinIndices = new Uint16Array(positions.count * 4);
  const skinWeights = new Float32Array(positions.count * 4);
  for (let vertex = 0; vertex < positions.count; vertex += 1) {
    const x = positions.getX(vertex);
    const y = positions.getY(vertex);
    let bestBone = 0;
    let bestWeight = 0;
    branchBones.forEach(({ part, anchor }, index) => {
      const path = part.path?.length && part.path.length >= 2
        ? part.path
        : [anchor, part.center];
      let minimumDistance = Infinity;
      let progress = 0;
      for (let segment = 0; segment < path.length - 1; segment += 1) {
        const projection = segmentProjection(x, y, path[segment], path[segment + 1]);
        if (projection.distance < minimumDistance) {
          minimumDistance = projection.distance;
          progress = (segment + projection.amount) / (path.length - 1);
        }
      }
      const radius = Math.max(0.035, part.size.x * 0.72);
      const radial = Math.max(0, 1 - minimumDistance / (radius * 1.8));
      const alongBranch = Math.min(1, Math.max(0, (progress - 0.02) / 0.34));
      const influence = radial * radial * alongBranch;
      if (influence > bestWeight) {
        bestWeight = influence;
        bestBone = index + 1;
      }
    });
    const branchWeight = Math.min(0.96, bestWeight);
    const offset = vertex * 4;
    skinIndices[offset] = 0;
    skinIndices[offset + 1] = bestBone;
    skinWeights[offset] = 1 - branchWeight;
    skinWeights[offset + 1] = branchWeight;
  }
  compactGeometry.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(skinIndices, 4));
  compactGeometry.setAttribute("skinWeight", new THREE.Float32BufferAttribute(skinWeights, 4));
  const skinnedVolume = new THREE.SkinnedMesh(compactGeometry, [sideMaterial, frontMaterial, backMaterial]);
  skinnedVolume.name = "silhouette-distance-lens";
  skinnedVolume.add(rootBone);
  skinnedVolume.bind(new THREE.Skeleton([rootBone, ...branchBones.map(({ bone }) => bone)]));
  skinnedVolume.castShadow = true;
  skinnedVolume.receiveShadow = true;
  skinnedVolume.userData.reconstruction = {
    method: "contour-preserving rounded 3D puppet",
    polygonizer: "deterministic triangulated artwork shell",
    subdivisions: 2,
    topology: "closed",
    contourPoints: contour.length,
    skeletonPoints: skeleton.length,
    semanticRig: rig.version,
    skinning: `${branchBones.length} verified branch bones over one continuous surface; unreviewed anatomy cannot deform geometry`,
    maximumHalfDepth: shell.maximumHalfDepth,
    texturedFrontTriangles,
    neutralBackTriangles,
    sideTriangles: shell.sideTriangleCount,
    silhouetteError: 0,
    artworkSurfaceCoverage,
    projectedSemanticFeatures: false,
    learnedDepth: depth ? {
      model: depth.model,
      meanThickness: depth.meanThickness,
      meanAsymmetry: depth.meanAsymmetry,
      frontBackMirrored: false,
    } : null,
  };
  character.add(skinnedVolume);

  character.userData.reconstruction = {
    method: "contour-preserving rounded 3D puppet",
    texturePlane: false,
    viewableDegrees: 360,
    bodyTopology: "closed",
    backPrior: "bounded hidden-surface relief; the original artwork appears only on the front",
    maximumHalfDepth: shell.maximumHalfDepth,
    texturedFrontTriangles,
    neutralBackTriangles,
    artworkSurfaceCoverage,
    artworkPreservedOnFront: true,
    projectedSemanticFeatures: false,
    unreviewedAnatomyDeformsGeometry: false,
    depthModel: depth?.model ?? null,
    meanDepthAsymmetry: depth?.meanAsymmetry ?? 0,
    semanticParts: rig.parts.map((part) => ({ id: part.id, kind: part.kind, confidence: part.confidence, source: part.source })),
    joints: rig.joints,
    topologyKind: rig.topologyKind ?? null,
    topologyConfidence: rig.topologyConfidence ?? null,
    accent,
  };
  character.userData.wallaliveRig = rig;
  character.userData.wallaliveMovableParts = structuralParts.map((part) => part.id);

  return character;
}

export const ARStage = forwardRef<ARStageHandle, ARStageProps>(function ARStage(
  { characters, contour, skeleton, textureUrl, rig, depth, action, ensembleActions = null, world, lightingMood, cameraPreset, accent, inflation, neuralAssetUrl, visible, onCapability, onPlaced, onNeuralAssetInfo },
  ref,
) {
  const mountRef = useRef<HTMLDivElement>(null);
  const handlesRef = useRef<SceneHandles | null>(null);
  const [rendererError, setRendererError] = useState(false);
  const actionRef = useRef(action);
  const ensembleActionsRef = useRef<CharacterAction[] | null>(ensembleActions);
  const worldStateRef = useRef<ARWorld>(world);
  const moodStateRef = useRef<LightingMood>(lightingMood);
  const cameraPresetRef = useRef<CameraPreset>(cameraPreset);
  const placementRef = useRef({ x: 0, y: -0.15, z: -0.5, scale: 1 });
  const rotationRef = useRef({ yaw: 0, pitch: 0 });
  const xrHitSourceRef = useRef<XRHitTestSource | null>(null);
  const xrReferenceSpaceRef = useRef<XRReferenceSpace | null>(null);

  useEffect(() => { actionRef.current = action; }, [action]);
  useEffect(() => { ensembleActionsRef.current = ensembleActions; }, [ensembleActions]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;

    const width = Math.max(1, mount.clientWidth);
    const height = Math.max(1, mount.clientHeight);
    let renderer: THREE.WebGLRenderer | null = null;
    const rendererOptions: THREE.WebGLRendererParameters[] = [
      { antialias: true, alpha: true, powerPreference: "high-performance" },
      { antialias: false, alpha: true, powerPreference: "default" },
      { antialias: false, alpha: false, powerPreference: "low-power" },
    ];
    for (const options of rendererOptions) {
      try {
        renderer = new THREE.WebGLRenderer(options);
        break;
      } catch (error) {
        console.warn("WallAlive WebGL renderer profile was unavailable", options, error);
      }
    }
    if (!renderer) {
      setRendererError(true);
      onCapability(false);
      return;
    }
    setRendererError(false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.xr.enabled = true;
    renderer.domElement.setAttribute("aria-label", "Semantic articulated 3D drawing");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    let environment = buildWorldEnvironment(worldStateRef.current);
    scene.add(environment.group);
    scene.background = environment.background;
    scene.fog = environment.fog;
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.01, 40);
    const firstPreset = cameraPresets[cameraPresetRef.current];
    camera.position.set(...firstPreset.position);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...firstPreset.target);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enableRotate = true;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.screenSpacePanning = false;
    controls.minDistance = 1.45;
    controls.maxDistance = 11;
    controls.minPolarAngle = 0.08;
    controls.maxPolarAngle = Math.PI * 0.91;
    controls.zoomToCursor = true;
    controls.update();

    const pmrem = new THREE.PMREMGenerator(renderer);
    const roomEnvironment = new RoomEnvironment();
    const environmentMap = pmrem.fromScene(roomEnvironment, 0.04).texture;
    scene.environment = environmentMap;
    roomEnvironment.dispose();
    pmrem.dispose();

    const ambient = new THREE.HemisphereLight(0xfff4dc, 0x253d42, 0.85);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xfff7e8, 2.1);
    key.position.set(-2.6, 4.2, 5.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 14;
    key.shadow.bias = -0.0005;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9edce5, 0.7);
    fill.position.set(3.5, 1.2, 2.3);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffc7a8, 0.5);
    rim.position.set(0.6, 2.5, -3.2);
    scene.add(rim);

    const atmosphere = new THREE.Group();
    atmosphere.name = "wallalive-cinematic-atmosphere";
    const grid = new THREE.GridHelper(12, 30, 0xffb85c, 0x58706c);
    grid.position.set(0, -1.145, -2.2);
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    gridMaterials.forEach((material) => { material.transparent = true; material.opacity = 0.2; });
    atmosphere.add(grid);
    const particles = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(210);
    for (let index = 0; index < particlePositions.length; index += 3) {
      const seed = index / 3;
      particlePositions[index] = Math.sin(seed * 12.9898) * 4.7;
      particlePositions[index + 1] = 0.15 + ((seed * 37) % 29) / 8;
      particlePositions[index + 2] = -2.4 - ((seed * 19) % 37) / 9;
    }
    particles.setAttribute("position", new THREE.BufferAttribute(particlePositions, 3));
    const particleMaterial = new THREE.PointsMaterial({ color: 0xffd59a, size: 0.035, transparent: true, opacity: 0.52, depthWrite: false });
    const particleField = new THREE.Points(particles, particleMaterial);
    particleField.name = "cinematic-particles";
    atmosphere.add(particleField);
    scene.add(atmosphere);

    const applyLightingMood = (mood: LightingMood) => {
      const palette = moodPalettes[mood];
      ambient.color.setHex(palette.hemisphere[0]);
      ambient.groundColor.setHex(palette.hemisphere[1]);
      ambient.intensity = palette.hemisphere[2];
      key.color.setHex(palette.key[0]); key.intensity = palette.key[1];
      fill.color.setHex(palette.fill[0]); fill.intensity = palette.fill[1];
      rim.color.setHex(palette.rim[0]); rim.intensity = palette.rim[1];
      gridMaterials.forEach((material) => material.color.setHex(palette.grid));
      particleMaterial.color.setHex(palette.rim[0]);
      renderer.toneMappingExposure = palette.exposure;
      atmosphere.userData.lightingMood = mood;
    };
    applyLightingMood(moodStateRef.current);

    let cameraTransition: { start: number; fromPosition: THREE.Vector3; fromTarget: THREE.Vector3; toPosition: THREE.Vector3; toTarget: THREE.Vector3 } | null = null;
    const applyCameraPreset = (preset: CameraPreset) => {
      const next = cameraPresets[preset];
      cameraTransition = {
        start: performance.now(),
        fromPosition: camera.position.clone(),
        fromTarget: controls.target.clone(),
        toPosition: new THREE.Vector3(...next.position),
        toTarget: new THREE.Vector3(...next.target),
      };
    };

    const characterRoot = new THREE.Group();
    scene.add(characterRoot);
    if (neuralAssetUrl) {
      new GLTFLoader().load(
        neuralAssetUrl,
        (gltf) => {
          if (disposed) {
            disposeObject(gltf.scene);
            return;
          }
          const prepared = prepareNeuralCharacter(gltf.scene, rig ?? undefined);
          characterRoot.add(prepared.character);
          onNeuralAssetInfo(prepared.info);
        },
        undefined,
        () => { if (!disposed) onNeuralAssetInfo(null); },
      );
    } else if (characters?.length) {
      const targets = characters.map((drawing, index) => drawing.sourceTarget ?? {
        x: characters.length === 1 ? 0.5 : (index + 1) / (characters.length + 1),
        y: 0.5,
      });
      const centerX = targets.reduce((sum, target) => sum + target.x, 0) / targets.length;
      const centerY = targets.reduce((sum, target) => sum + target.y, 0) / targets.length;
      const ensembleScale = characters.length === 1 ? 1 : Math.max(0.48, Math.min(0.72, 1.18 / Math.sqrt(characters.length)));
      characters.forEach((drawing, index) => {
        const instance = buildCharacter(
          drawing.contour,
          drawing.skeleton,
          drawing.rig,
          drawing.depthRecognition ?? null,
          drawing.textureUrl,
          accent,
          inflation,
          {
            poseApplicable: Boolean(drawing.poseRecognition?.applicable),
            topologyApplicable: Boolean(drawing.topologyRecognition?.applicable),
          },
        );
        instance.position.set((targets[index].x - centerX) * 2.7, (centerY - targets[index].y) * 1.75, index * 0.025);
        instance.scale.setScalar(ensembleScale);
        instance.userData.wallaliveBasePosition = instance.position.clone();
        instance.userData.wallaliveBaseScale = ensembleScale;
        instance.userData.wallalivePhase = index * 0.73;
        instance.userData.wallaliveInstance = index;
        characterRoot.add(instance);
      });
      onNeuralAssetInfo(null);
    } else if (contour?.length && skeleton?.length && rig) {
      characterRoot.add(buildCharacter(contour, skeleton, rig, depth, textureUrl, accent, inflation, {
        poseApplicable: false,
        topologyApplicable: false,
      }));
      onNeuralAssetInfo(null);
    }

    const shadowMaterial = new THREE.MeshBasicMaterial({ color: 0x102927, transparent: true, opacity: 0.2, depthWrite: false });
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.64, 64), shadowMaterial);
    shadow.scale.y = 0.2;
    shadow.position.set(0, -1.03, -0.18);
    scene.add(shadow);

    const reticle = new THREE.Mesh(
      new THREE.RingGeometry(0.09, 0.12, 32).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xc8f15a }),
    );
    reticle.matrixAutoUpdate = false;
    reticle.visible = false;
    scene.add(reticle);

    const clock = new THREE.Clock();
    let previousElapsed = 0;
    const pressedKeys = new Set<string>();
    const onKeyDown = (event: KeyboardEvent) => {
      const keyName = event.key.toLowerCase();
      if (["arrowup", "arrowdown", "arrowleft", "arrowright", "w", "a", "s", "d"].includes(keyName)) pressedKeys.add(keyName);
    };
    const onKeyUp = (event: KeyboardEvent) => { pressedKeys.delete(event.key.toLowerCase()); };
    renderer.domElement.tabIndex = 0;
    renderer.domElement.addEventListener("keydown", onKeyDown);
    renderer.domElement.addEventListener("keyup", onKeyUp);
    const render = (_time?: number, frame?: XRFrame) => {
      const elapsed = clock.getElapsedTime();
      const delta = Math.min(0.05, Math.max(0, elapsed - previousElapsed));
      previousElapsed = elapsed;
      const currentAction = actionRef.current;
      const directedActions = ensembleActionsRef.current;
      const placement = placementRef.current;
      const root = characterRoot;
      const syntheticWorldVisible = !renderer.xr.isPresenting;
      environment.group.visible = syntheticWorldVisible;
      scene.background = syntheticWorldVisible ? environment.background : null;
      scene.fog = syntheticWorldVisible ? environment.fog : null;
      atmosphere.visible = syntheticWorldVisible;
      controls.enabled = syntheticWorldVisible;
      if (cameraTransition && syntheticWorldVisible) {
        const amount = Math.min(1, (performance.now() - cameraTransition.start) / 820);
        const eased = 1 - Math.pow(1 - amount, 3);
        camera.position.lerpVectors(cameraTransition.fromPosition, cameraTransition.toPosition, eased);
        controls.target.lerpVectors(cameraTransition.fromTarget, cameraTransition.toTarget, eased);
        if (amount >= 1) cameraTransition = null;
      }
      if (syntheticWorldVisible) controls.update();
      const navigationSpeed = delta * 1.35;
      if (pressedKeys.has("arrowleft") || pressedKeys.has("a")) placement.x -= navigationSpeed;
      if (pressedKeys.has("arrowright") || pressedKeys.has("d")) placement.x += navigationSpeed;
      if (pressedKeys.has("arrowup") || pressedKeys.has("w")) placement.z -= navigationSpeed;
      if (pressedKeys.has("arrowdown") || pressedKeys.has("s")) placement.z += navigationSpeed;
      if (!directedActions && currentAction === "walk") placement.z -= delta * 0.42;
      placement.x = THREE.MathUtils.clamp(placement.x, -3.8, 3.8);
      placement.z = THREE.MathUtils.clamp(placement.z, -4.9, 1.25);
      root.visible = visible;
      root.position.x = placement.x;
      root.position.y = placement.y + Math.sin(elapsed * 1.65) * 0.018;
      root.position.z = placement.z;
      root.scale.setScalar(placement.scale);
      root.rotation.x = rotationRef.current.pitch + Math.sin(elapsed * 1.1) * 0.018;
      root.rotation.y = rotationRef.current.yaw - 0.07 + Math.sin(elapsed * 0.72) * 0.035;
      root.rotation.z = Math.sin(elapsed * 0.9) * 0.012;

      const articulatedCharacters = root.getObjectsByProperty("name", "wallalive-semantic-character");
      const neural = root.getObjectByName("wallalive-neural-character");
      const neuralRig = neural?.userData.wallaliveRig as NeuralRigMap | undefined;
      const neuralSemantic = neural?.userData.wallaliveSemantic as NeuralSemanticMap | undefined;
      neuralRig?.all.forEach((bone) => {
        const base = bone.userData.wallaliveBaseQuaternion as THREE.Quaternion | undefined;
        if (base) bone.quaternion.copy(base);
      });
      const blink = Math.pow(Math.max(0, Math.sin(elapsed * 0.78)), 34);
      articulatedCharacters.forEach((articulated, instanceIndex) => {
        const instanceAction = directedActions?.[instanceIndex] ?? currentAction;
        const localRig = articulated.userData.wallaliveRig as CharacterRig | undefined;
        const movableIds = new Set<string>(articulated.userData.wallaliveMovableParts ?? []);
        const phase = Number(articulated.userData.wallalivePhase ?? instanceIndex * 0.73);
        const localElapsed = elapsed + phase;
        const basePosition = articulated.userData.wallaliveBasePosition as THREE.Vector3 | undefined;
        if (basePosition) {
          articulated.position.copy(basePosition);
          articulated.rotation.set(0, 0, 0);
          const baseScale = Number(articulated.userData.wallaliveBaseScale ?? 1);
          articulated.scale.setScalar(baseScale);
        }
        const movable = localRig?.parts.filter((part) => movableIds.has(part.id)) ?? [];
        movable.forEach((part) => {
          const node = articulated.getObjectByName(`rig-${part.id}`);
          if (!node) return;
          node.rotation.x = 0;
          node.rotation.y = 0;
          node.rotation.z = Number(node.userData.baseRotationZ ?? 0);
        });
        if (instanceAction === "wave") {
          const quadruped = localRig?.topologyKind === "quadruped";
          const part = (quadruped ? movable.find((candidate) => candidate.kind === "tail") : null)
            ?? movable.find((candidate) => candidate.kind === "arm" && candidate.side === "right")
            ?? movable.find((candidate) => candidate.kind === "arm")
            ?? movable.find((candidate) => candidate.kind === "wing" || candidate.kind === "tentacle" || candidate.kind === "tail");
          const node = part ? articulated.getObjectByName(`rig-${part.id}`) : null;
          if (node) {
            const amplitude = quadruped || part?.kind !== "arm" ? 0.2 : 0.3;
            node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + amplitude + Math.sin(localElapsed * 7.2) * amplitude;
          }
        }
        if (instanceAction === "dance") {
          movable.forEach((part, index) => {
            const node = articulated.getObjectByName(`rig-${part.id}`);
            if (!node) return;
            const direction = part.side === "right" || index % 2 ? -1 : 1;
            node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + direction * Math.sin(localElapsed * 5.2 + index * 0.45) * 0.24;
            node.rotation.x = Math.sin(localElapsed * 3.8 + index) * 0.08;
          });
        }
        if (instanceAction === "walk") {
          movable.filter((part) => part.kind === "leg").forEach((part, index) => {
            const node = articulated.getObjectByName(`rig-${part.id}`);
            if (!node) return;
            const direction = part.side === "right" || index % 2 ? -1 : 1;
            node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + direction * Math.sin(localElapsed * 7) * 0.26;
            node.rotation.x = direction * Math.sin(localElapsed * 7) * 0.12;
          });
          movable.filter((part) => part.kind === "arm").forEach((part, index) => {
            const node = articulated.getObjectByName(`rig-${part.id}`);
            if (node) node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + (index % 2 ? 1 : -1) * Math.sin(localElapsed * 7) * 0.24;
          });
        }
        if (directedActions) {
          if (instanceAction === "hop") articulated.position.y = (basePosition?.y ?? articulated.position.y) + Math.abs(Math.sin(localElapsed * 4.6)) * 0.34;
          if (instanceAction === "walk") articulated.position.x = (basePosition?.x ?? articulated.position.x) + Math.sin(localElapsed * 1.9) * 0.24;
          if (instanceAction === "hide") {
            articulated.position.x = (basePosition?.x ?? articulated.position.x) + (instanceIndex % 2 ? 0.68 : -0.68);
            articulated.scale.multiplyScalar(0.72);
          }
          if (instanceAction === "spin") articulated.rotation.y = localElapsed * 2.15;
          if (instanceAction === "dance") articulated.rotation.z = Math.sin(localElapsed * 5.2) * 0.15;
          if (instanceAction === "float") articulated.position.y = (basePosition?.y ?? articulated.position.y) + 0.18 + Math.sin(localElapsed * 2.1) * 0.22;
        }
      });
      [neuralSemantic?.eyeLeft, neuralSemantic?.eyeRight, neuralSemantic?.eyeCenter, neuralSemantic?.pupilLeft, neuralSemantic?.pupilRight, neuralSemantic?.pupilCenter]
        .forEach((node) => { if (node) node.scale.y = Math.max(0.12, 1 - blink * 0.88); });
      if (neuralSemantic?.mouth) neuralSemantic.mouth.scale.y = currentAction === "idle" ? 1 : 1 + Math.abs(Math.sin(elapsed * 5)) * 0.42;

      if (!directedActions && currentAction === "wave") {
        const neuralArm = neuralRig?.armRight ?? neuralRig?.armLeft;
        neuralArm?.rotateZ(0.72 + Math.sin(elapsed * 7.2) * 0.42);
        neuralArm?.rotateX(Math.sin(elapsed * 4.1) * 0.22);
        root.rotation.y = rotationRef.current.yaw - 0.18 + Math.sin(elapsed * 4.8) * 0.08;
        root.rotation.z = -0.035 + Math.sin(elapsed * 5.6) * 0.035;
        root.position.y += Math.sin(elapsed * 5.6) * 0.025;
      }
      if (!directedActions && currentAction === "dance") {
        neuralRig?.arms.forEach((arm, index) => {
          const direction = index % 2 ? -1 : 1;
          arm.rotateZ(direction * (0.52 + Math.sin(elapsed * 5.2 + index * 0.6) * 0.45));
        });
        neuralRig?.legs.forEach((leg, index) => leg.rotateX((index % 2 ? -1 : 1) * Math.sin(elapsed * 5.2 + index * 0.4) * 0.22));
        root.rotation.z = Math.sin(elapsed * 5.2) * 0.18;
        root.rotation.y = rotationRef.current.yaw + Math.sin(elapsed * 2.6) * 0.18;
        root.position.x = placement.x + Math.sin(elapsed * 3.4) * 0.15;
      }
      if (!directedActions && currentAction === "hop") {
        root.position.y = placement.y + Math.abs(Math.sin(elapsed * 4.6)) * 0.52;
        root.rotation.x = Math.sin(elapsed * 4.6) * 0.08;
      }
      if (!directedActions && currentAction === "walk") {
        neuralRig?.legs.forEach((leg, index) => leg.rotateX((index % 2 ? -1 : 1) * Math.sin(elapsed * 7 + index * 0.25) * 0.48));
        neuralRig?.arms.forEach((arm, index) => arm.rotateX((index % 2 ? 1 : -1) * Math.sin(elapsed * 7 + index * 0.25) * 0.24));
        root.position.x = placement.x + Math.sin(elapsed * 5.5) * 0.035;
        root.rotation.z = Math.sin(elapsed * 6) * 0.055;
        root.rotation.y = rotationRef.current.yaw + Math.sin(elapsed * 3) * 0.06;
      }
      if (!directedActions && currentAction === "hide") {
        root.position.x = placement.x + 1.08;
        root.rotation.y = -0.35;
        root.rotation.z = -0.16;
      }
      if (!directedActions && currentAction === "spin") root.rotation.y = rotationRef.current.yaw + elapsed * 2.15;
      if (!directedActions && currentAction === "float") {
        root.position.y = placement.y + 0.25 + Math.sin(elapsed * 2.15) * 0.25;
        root.rotation.z = Math.sin(elapsed * 1.55) * 0.055;
      }

      shadow.position.x = root.position.x;
      shadow.position.y = placement.y - 0.92;
      shadow.position.z = root.position.z - 0.18;
      shadow.scale.x = placement.scale * (currentAction === "hop" ? 1 - Math.abs(Math.sin(elapsed * 4.6)) * 0.35 : 1);
      shadow.visible = visible && !renderer.xr.isPresenting;

      if (frame && xrReferenceSpaceRef.current && xrHitSourceRef.current) {
        const results = frame.getHitTestResults(xrHitSourceRef.current);
        if (results.length) {
          const pose = results[0].getPose(xrReferenceSpaceRef.current);
          if (pose) {
            reticle.visible = true;
            reticle.matrix.fromArray(pose.transform.matrix);
          }
        } else reticle.visible = false;
      }
      renderer.render(scene, camera);
    };
    renderer.setAnimationLoop(render);

    const resize = () => {
      if (!mountRef.current || renderer.xr.isPresenting) return;
      const nextWidth = Math.max(1, mountRef.current.clientWidth);
      const nextHeight = Math.max(1, mountRef.current.clientHeight);
      camera.aspect = nextWidth / nextHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(nextWidth, nextHeight);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(mount);

    const setStageWorld = (nextWorld: ARWorld) => {
      scene.remove(environment.group);
      disposeObject(environment.group);
      environment = buildWorldEnvironment(nextWorld);
      scene.add(environment.group);
      if (!renderer.xr.isPresenting) {
        scene.background = environment.background;
        scene.fog = environment.fog;
      }
    };

    const dispose = () => {
      disposed = true;
      observer.disconnect();
      renderer.domElement.removeEventListener("keydown", onKeyDown);
      renderer.domElement.removeEventListener("keyup", onKeyUp);
      controls.dispose();
      environmentMap.dispose();
      renderer.setAnimationLoop(null);
      disposeObject(scene);
      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };

    handlesRef.current = { renderer, scene, camera, character: characterRoot, reticle, setWorld: setStageWorld, setLightingMood: applyLightingMood, setCameraPreset: applyCameraPreset, controls, dispose };
    if (navigator.xr) navigator.xr.isSessionSupported("immersive-ar").then(onCapability).catch(() => onCapability(false));
    else onCapability(false);

    return () => {
      handlesRef.current = null;
      dispose();
    };
  }, [accent, characters, contour, depth, inflation, neuralAssetUrl, onCapability, onNeuralAssetInfo, rig, skeleton, textureUrl, visible]);

  useEffect(() => {
    worldStateRef.current = world;
    handlesRef.current?.setWorld(world);
  }, [world]);

  useEffect(() => {
    moodStateRef.current = lightingMood;
    handlesRef.current?.setLightingMood(lightingMood);
  }, [lightingMood]);

  useEffect(() => {
    cameraPresetRef.current = cameraPreset;
    handlesRef.current?.setCameraPreset(cameraPreset);
  }, [cameraPreset]);

  useImperativeHandle(ref, () => ({
    placeNormalized(x: number, y: number, scale = placementRef.current.scale) {
      placementRef.current = { x: (x - 0.5) * 2.3, y: (0.5 - y) * 1.65 - 0.1, z: placementRef.current.z, scale: Math.min(1.55, Math.max(0.55, scale)) };
      onPlaced("screen", Number(x.toFixed(2)), Number(y.toFixed(2)));
    },
    rotateBy(yaw: number, pitch: number) {
      rotationRef.current = {
        yaw: rotationRef.current.yaw + yaw,
        // Yaw stays unbounded for a genuine full turn. Keep vertical drag to a
        // presentation-safe tilt so a tall character cannot be tumbled into a
        // confusing horizontal blob on a phone screen.
        pitch: Math.min(0.24, Math.max(-0.24, rotationRef.current.pitch + pitch)),
      };
    },
    moveBy(x: number, z: number) {
      placementRef.current.x = THREE.MathUtils.clamp(placementRef.current.x + x, -3.8, 3.8);
      placementRef.current.z = THREE.MathUtils.clamp(placementRef.current.z + z, -4.9, 1.25);
    },
    async enterImmersiveAR() {
      const handles = handlesRef.current;
      if (!handles || !navigator.xr) return { ok: false, error: "Immersive AR is not supported in this browser. Camera-overlay mode remains available." };
      try {
        const session = await navigator.xr.requestSession("immersive-ar", {
          requiredFeatures: ["hit-test"],
          optionalFeatures: ["dom-overlay", "anchors", "light-estimation"],
          domOverlay: { root: document.body },
        });
        const referenceSpace = await session.requestReferenceSpace("local");
        const viewerSpace = await session.requestReferenceSpace("viewer");
        const hitSource = await session.requestHitTestSource({ space: viewerSpace });
        xrReferenceSpaceRef.current = referenceSpace;
        xrHitSourceRef.current = hitSource;
        handles.renderer.xr.setReferenceSpaceType("local");
        await handles.renderer.xr.setSession(session);
        session.addEventListener("select", () => {
          if (!handles.reticle.visible) return;
          const position = new THREE.Vector3();
          position.setFromMatrixPosition(handles.reticle.matrix);
          handles.character.position.copy(position);
          placementRef.current = { x: position.x, y: position.y, z: position.z, scale: placementRef.current.scale };
          onPlaced("world", Number(position.x.toFixed(2)), Number(position.y.toFixed(2)));
        });
        session.addEventListener("end", () => {
          xrHitSourceRef.current?.cancel();
          xrHitSourceRef.current = null;
          xrReferenceSpaceRef.current = null;
          handles.reticle.visible = false;
          handles.character.position.z = 0;
        });
        return { ok: true };
      } catch (error) {
        return { ok: false, error: error instanceof Error ? error.message : "AR session could not start." };
      }
    },
  }), [onPlaced]);

  return <div className="three-layer" ref={mountRef} aria-hidden={!visible}>{rendererError ? <div className="renderer-fallback"><b>3D NEEDS WEBGL</b><span>The drawing and semantic rig are safe. Open this page in Safari or Chrome with hardware acceleration.</span></div> : null}</div>;
});
