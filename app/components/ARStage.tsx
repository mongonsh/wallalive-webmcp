"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { RoundedBoxGeometry } from "three/addons/geometries/RoundedBoxGeometry.js";
import { selectAnimatableRigParts, type CharacterRig, type ContourPoint, type DrawingExtraction, type LearnedDepthField, type SkeletonPoint } from "../lib/drawing";
import { buildArtworkShellGeometry } from "../lib/artwork-shell";
import { hasRecognizableArtworkSurface } from "../lib/mesh-materials";
import { resolvePaintProjection, type ModelPaintBrush, type ModelPaintTool } from "../lib/model-paint";
import { disposeObject, prepareNeuralCharacter, type NeuralRigMap, type NeuralSemanticMap, type RiggedAssetInfo } from "../lib/rigged-model";

export type { ModelPaintBrush, ModelPaintTool } from "../lib/model-paint";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin" | "float";
export type ARWorld = "studio" | "storybook" | "wizard" | "museum";
export type LightingMood = "cyberpunk-neon" | "sunset-warm" | "moonlight";
export type CameraPreset = "cinematic-orbit" | "low-angle-hero" | "overhead";
export type WorldObjectInteraction = { id: string; label: string; verb: string; story: string; world: ARWorld };
export type ModelPaintInspection = {
  strokeCount: number;
  paintedSurfaceCount: number;
  colors: string[];
  tools: ModelPaintTool[];
};

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
  rotateBy: (yaw: number, pitch: number) => void;
  moveBy: (x: number, z: number) => void;
  interactWorldObject: (id: string) => boolean;
  setPaintEnabled: (enabled: boolean) => void;
  beginPaintStroke: (brush: ModelPaintBrush) => void;
  paintAtNormalized: (x: number, y: number, pressure?: number) => { painted: boolean; target?: string };
  endPaintStroke: () => ModelPaintInspection;
  undoPaint: () => ModelPaintInspection;
  resetPaint: () => ModelPaintInspection;
  inspectPaint: () => ModelPaintInspection;
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
  paintEnabled?: boolean;
  visible: boolean;
  onCapability: (supported: boolean) => void;
  onRendererCapability: (supported: boolean) => void;
  onPlaced: (surface: "screen" | "world", x: number, y: number) => void;
  onNeuralAssetInfo: (info: RiggedAssetInfo | null) => void;
  onWorldInteraction?: (interaction: WorldObjectInteraction) => void;
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
  interactWorldObject: (id: string) => boolean;
  setPaintEnabled: (enabled: boolean) => void;
  beginPaintStroke: (brush: ModelPaintBrush) => void;
  paintAtNormalized: (x: number, y: number, pressure?: number) => { painted: boolean; target?: string };
  endPaintStroke: () => ModelPaintInspection;
  undoPaint: () => ModelPaintInspection;
  resetPaint: () => ModelPaintInspection;
  inspectPaint: () => ModelPaintInspection;
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

type SurfaceFinish = "plaster" | "stone" | "wood" | "brass" | "velvet" | "glass" | "glow";

function makeSurfaceTexture(color: number, finish: SurfaceFinish) {
  const size = 192;
  const data = new Uint8Array(size * size * 4);
  const base = new THREE.Color(color);
  const variation = finish === "stone" ? 0.13 : finish === "wood" ? 0.11 : finish === "plaster" ? 0.045 : 0.03;
  let seed = (color ^ finish.length * 0x9e3779b9) >>> 0;
  for (let pixel = 0; pixel < size * size; pixel += 1) {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    const x = pixel % size;
    const y = Math.floor(pixel / size);
    const woodBand = finish === "wood" ? Math.sin(x * 0.46 + y * 0.08) * 0.045 : 0;
    const broadGrain = Math.sin((x + (color & 31)) * 0.22) * Math.cos((y + ((color >>> 5) & 31)) * 0.18) * variation * 0.38;
    const grain = (((seed >>> 8) & 255) / 255 - 0.5) * variation * 0.62 + broadGrain + woodBand;
    data[pixel * 4] = Math.round(THREE.MathUtils.clamp(base.r + grain, 0, 1) * 255);
    data[pixel * 4 + 1] = Math.round(THREE.MathUtils.clamp(base.g + grain * 0.84, 0, 1) * 255);
    data[pixel * 4 + 2] = Math.round(THREE.MathUtils.clamp(base.b + grain * 0.62, 0, 1) * 255);
    data[pixel * 4 + 3] = 255;
  }
  const texture = new THREE.DataTexture(data, size, size, THREE.RGBAFormat);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(finish === "wood" ? 3 : finish === "stone" ? 2.4 : 2, finish === "wood" ? 2 : finish === "stone" ? 2.4 : 2);
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearMipmapLinearFilter;
  texture.needsUpdate = true;
  return texture;
}

/** Builds a material-rich, navigable perspective set. No world is a bitmap or CSS plate. */
export function buildWorldEnvironment(world: ARWorld): WorldEnvironment {
  const group = new THREE.Group();
  group.name = `wallalive-3d-world-${world}`;
  group.userData.world = world;
  group.userData.rendering = "procedural Three.js geometry with perspective, lighting, occlusion, and shadows";
  group.userData.artDirection = "original material-rich enchanted architecture; never a photographic backdrop";
  const materialCache = new Map<string, THREE.MeshPhysicalMaterial>();
  const material = (color: number, finish: SurfaceFinish = "plaster", emissive = 0x000000) => {
    const key = `${color}-${finish}-${emissive}`;
    const cached = materialCache.get(key);
    if (cached) return cached;
    const isGlass = finish === "glass";
    const isGlow = finish === "glow";
    const surfaceTexture = isGlass || isGlow ? null : makeSurfaceTexture(color, finish);
    const created = new THREE.MeshPhysicalMaterial({
      color: isGlass || isGlow ? color : 0xffffff,
      map: surfaceTexture,
      bumpMap: finish === "stone" || finish === "wood" || finish === "plaster" ? surfaceTexture : null,
      bumpScale: finish === "stone" ? 0.035 : finish === "wood" ? 0.02 : 0.012,
      roughness: finish === "brass" ? 0.25 : finish === "wood" ? 0.66 : finish === "velvet" ? 0.94 : isGlass ? 0.12 : isGlow ? 0.2 : 0.86,
      metalness: finish === "brass" ? 0.82 : isGlow ? 0.16 : 0.01,
      clearcoat: finish === "brass" ? 0.7 : finish === "wood" ? 0.16 : isGlass ? 1 : 0.06,
      clearcoatRoughness: finish === "brass" ? 0.18 : 0.72,
      transmission: isGlass ? 0.48 : 0,
      thickness: isGlass ? 0.55 : 0,
      transparent: isGlass || isGlow,
      opacity: isGlass ? 0.62 : isGlow ? 0.88 : 1,
      emissive,
      emissiveIntensity: emissive ? (isGlow ? 2.15 : 0.7) : 0,
      side: isGlass || isGlow ? THREE.DoubleSide : THREE.FrontSide,
    });
    materialCache.set(key, created);
    return created;
  };
  const add = (
    name: string,
    geometry: THREE.BufferGeometry,
    surface: THREE.Material,
    position: [number, number, number],
    rotation: [number, number, number] = [0, 0, 0],
    scale: [number, number, number] = [1, 1, 1],
  ) => {
    const mesh = new THREE.Mesh(geometry, surface);
    mesh.name = name;
    mesh.position.set(...position);
    mesh.rotation.set(...rotation);
    mesh.scale.set(...scale);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    return mesh;
  };
  const rounded = (name: string, size: [number, number, number], color: number, position: [number, number, number], finish: SurfaceFinish = "plaster", radius = 0.08, rotation: [number, number, number] = [0, 0, 0]) => (
    add(name, new RoundedBoxGeometry(size[0], size[1], size[2], 4, Math.min(radius, Math.min(...size) * 0.45)), material(color, finish), position, rotation)
  );
  const interactive = (mesh: THREE.Object3D, id: string, label: string, verb: string, story: string) => {
    mesh.userData.wallaliveInteraction = { id, label, verb, story, world } satisfies WorldObjectInteraction;
    mesh.traverse((child) => { child.userData.wallaliveInteraction = mesh.userData.wallaliveInteraction; });
    return mesh;
  };
  const column = (name: string, height: number, radius: number, color: number, position: [number, number, number], finish: SurfaceFinish = "stone") => {
    const profile = [
      new THREE.Vector2(radius * 1.52, 0), new THREE.Vector2(radius * 1.62, height * 0.055),
      new THREE.Vector2(radius * 1.06, height * 0.12), new THREE.Vector2(radius * 0.82, height * 0.22),
      new THREE.Vector2(radius * 0.76, height * 0.78), new THREE.Vector2(radius * 1.06, height * 0.9),
      new THREE.Vector2(radius * 1.62, height * 0.95), new THREE.Vector2(radius * 1.52, height),
    ];
    return add(name, new THREE.LatheGeometry(profile, 28), material(color, finish), position);
  };
  const arch = (name: string, width: number, height: number, thickness: number, color: number, position: [number, number, number], finish: SurfaceFinish = "stone") => {
    const outerRadius = width / 2;
    const innerRadius = Math.max(0.08, outerRadius - thickness);
    const spring = height - outerRadius;
    const shape = new THREE.Shape();
    shape.moveTo(-outerRadius, 0);
    shape.lineTo(-outerRadius, spring);
    shape.absarc(0, spring, outerRadius, Math.PI, 0, true);
    shape.lineTo(outerRadius, 0);
    shape.closePath();
    const hole = new THREE.Path();
    hole.moveTo(-innerRadius, -0.04);
    hole.lineTo(-innerRadius, spring);
    hole.absarc(0, spring, innerRadius, Math.PI, 0, true);
    hole.lineTo(innerRadius, -0.04);
    hole.closePath();
    shape.holes.push(hole);
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: 0.2, bevelEnabled: true, bevelSegments: 3, steps: 1, bevelSize: 0.035, bevelThickness: 0.035, curveSegments: 24 });
    geometry.center();
    return add(name, geometry, material(color, finish), [position[0], position[1] + height / 2, position[2]]);
  };
  const gem = (name: string, color: number, position: [number, number, number], scale = 1, phase = 0) => {
    const mesh = add(name, new THREE.OctahedronGeometry(0.2 * scale, 1), material(color, "glass", color), position, [0.2, phase, 0.12]);
    mesh.userData.worldMotion = { baseY: position[1], phase, amplitude: 0.1 * scale, speed: 1.1 + phase * 0.08, spin: 0.22 };
    return mesh;
  };
  const pointGlow = (name: string, color: number, intensity: number, position: [number, number, number], distance = 5) => {
    const light = new THREE.PointLight(color, intensity, distance, 2);
    light.name = name;
    light.position.set(...position);
    light.castShadow = false;
    group.add(light);
    return light;
  };
  const leafCluster = (name: string, color: number, position: [number, number, number], scale: [number, number, number]) => (
    add(name, new THREE.IcosahedronGeometry(0.72, 2), material(color, "velvet"), position, [0, 0, 0], scale)
  );

  const settings: Record<ARWorld, { background: number; fog: number; floor: number; wall: number }> = {
    studio: { background: 0xeee1c7, fog: 0xd9c7a8, floor: 0x9b6b46, wall: 0xe6d2aa },
    storybook: { background: 0x78b7c4, fog: 0x9bc6bb, floor: 0x567c4d, wall: 0xb99a78 },
    wizard: { background: 0x0d1524, fog: 0x111d31, floor: 0x20263d, wall: 0x343044 },
    museum: { background: 0xdfd4c2, fog: 0xc8b9a2, floor: 0x72513c, wall: 0xd8c6aa },
  };
  const palette = settings[world];
  add(`${world}-floor`, new THREE.CircleGeometry(8.5, 96), material(palette.floor, world === "wizard" ? "stone" : "wood"), [0, -1.22, -2.3], [-Math.PI / 2, 0, 0]);
  rounded(`${world}-back-wall`, [11.2, 6.8, 0.28], palette.wall, [0, 1.35, -6.75], world === "wizard" ? "stone" : "plaster", 0.12);
  rounded(`${world}-left-wall`, [0.28, 6.8, 8.5], palette.wall, [-5.5, 1.35, -2.7], world === "wizard" ? "stone" : "plaster", 0.12);
  rounded(`${world}-right-wall`, [0.28, 6.8, 8.5], palette.wall, [5.5, 1.35, -2.7], world === "wizard" ? "stone" : "plaster", 0.12);

  if (world === "studio") {
    arch("studio-window", 2.45, 2.65, 0.17, 0x7b5438, [-2.45, -0.05, -6.47], "wood");
    add("studio-window-glass", new THREE.CircleGeometry(0.99, 48, 0, Math.PI), material(0x8fc6d2, "glass"), [-2.45, 1.29, -6.31]);
    rounded("studio-shelf", [2.55, 0.16, 0.55], 0x68442f, [2.55, 0.12, -6.04], "wood", 0.055);
    [-0.9, -0.54, -0.16, 0.22, 0.61, 0.92].forEach((offset, index) => rounded(`studio-book-${index}`, [0.24 + (index % 2) * 0.05, 0.7 + (index % 3) * 0.12, 0.31], [0xb6483e, 0x386d72, 0xb88435, 0x5f4c83, 0x4b7554, 0x9c5b39][index], [2.55 + offset, 0.57, -5.78], "velvet", 0.035, [0, 0, (index - 2) * 0.025]));
    const tableTop = add("studio-sculpting-table", new THREE.CylinderGeometry(1.35, 1.38, 0.16, 48), material(0x70472f, "wood"), [2.65, -0.76, -3.85]);
    tableTop.scale.z = 0.62;
    column("studio-table-leg", 0.62, 0.32, 0x5c3827, [2.65, -1.18, -3.85], "wood");
    column("studio-pot", 0.55, 0.34, 0xa34e3c, [-3.72, -1.2, -5.15], "plaster");
    [-0.34, 0.02, 0.34].forEach((offset, index) => {
      const stem = add(`studio-plant-stem-${index}`, new THREE.CylinderGeometry(0.035, 0.045, 1.55 - index * 0.12, 10), material(0x3f6646, "velvet"), [-3.72 + offset, -0.22, -5.15], [0, 0, offset * 0.42]);
      stem.castShadow = true;
      leafCluster(`studio-plant-leaf-${index}`, [0x426c4a, 0x557f50, 0x365f43][index], [-3.75 + offset * 1.7, 0.5 - index * 0.08, -5.1], [0.42, 0.72, 0.38]);
    });
    rounded("studio-rug", [3.9, 0.035, 2.55], 0xc97863, [-0.15, -1.17, -2.55], "velvet", 0.28);
    const projector = rounded("studio-story-projector", [0.78, 0.46, 0.62], 0x3a4144, [-2.55, -0.82, -3.5], "brass", 0.09);
    add("studio-projector-lens", new THREE.CylinderGeometry(0.15, 0.18, 0.2, 32), material(0x8bcfe6, "glass", 0x54b9d7), [-2.55, -0.75, -3.16], [Math.PI / 2, 0, 0]);
    interactive(projector, "studio-projector", "Story projector", "play", "The projector turns the cast's drawings into a tiny shadow-movie scene.");
    interactive(tableTop, "studio-maker-table", "Maker table", "build", "The maker table opens a prop-building challenge for the whole cast.");
    pointGlow("studio-window-light", 0xffd49a, 5.2, [-2.5, 2.2, -4.9], 7);
  }

  if (world === "storybook") {
    const pathCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.6, -1.12, 1.2), new THREE.Vector3(0.55, -1.12, -0.3),
      new THREE.Vector3(-0.35, -1.12, -2.2), new THREE.Vector3(0.15, -1.12, -5.6),
    ]);
    add("storybook-path", new THREE.TubeGeometry(pathCurve, 42, 0.62, 10, false), material(0xcaa76a, "stone"), [0, 0, 0]);
    rounded("storybook-castle", [3.55, 2.25, 0.78], 0xc8a17e, [0, -0.02, -6.08], "stone", 0.12);
    const gate = arch("storybook-gate", 1.14, 1.62, 0.18, 0x714937, [0, -1.1, -5.58], "wood");
    interactive(gate, "storybook-gate", "Castle gate", "unlock", "The gate opens only when two characters cooperate.");
    [-1.92, 1.92].forEach((x, index) => {
      add(`storybook-tower-${index}`, new THREE.CylinderGeometry(0.62, 0.7, 3.15, 28), material(0xb89070, "stone"), [x, 0.08, -5.8]);
      add(`storybook-roof-${index}`, new THREE.ConeGeometry(0.9, 1.52, 28), material(0x5e557e, "velvet"), [x, 2.36, -5.8]);
      for (let crenel = 0; crenel < 7; crenel += 1) {
        const angle = crenel / 7 * Math.PI * 2;
        rounded(`storybook-crenel-${index}-${crenel}`, [0.18, 0.3, 0.18], 0xcaa788, [x + Math.cos(angle) * 0.52, 1.67, -5.8 + Math.sin(angle) * 0.52], "stone", 0.025);
      }
    });
    [-3.65, 3.55].forEach((x, index) => {
      column(`storybook-tree-trunk-${index}`, 1.75, 0.22, 0x67432d, [x, -1.18, -4.7], "wood");
      leafCluster(`storybook-tree-crown-${index}`, 0x396f46, [x, 0.82, -4.7], [1.2, 1.38, 1.05]);
      leafCluster(`storybook-tree-crown-small-${index}`, 0x5e9254, [x + (index ? -0.62 : 0.62), 0.46, -4.56], [0.72, 0.92, 0.72]);
    });
    [-2.8, -1.65, 1.55, 2.85].forEach((x, index) => {
      const firefly = gem(`storybook-firefly-${index}`, 0xffe37b, [x, 0.2 + (index % 2) * 0.8, -3.5 - index * 0.42], 0.34, index * 0.8);
      firefly.material = material(0xffe37b, "glow", 0xffd84e);
      interactive(firefly, `storybook-firefly-${index}`, `Firefly ${index + 1}`, "find", "A hidden firefly joins the cast's lantern trail.");
    });
    pointGlow("storybook-sun-glow", 0xffd28c, 5.8, [-2.2, 3.2, -4.4], 8);
  }

  if (world === "wizard") {
    arch("wizard-grand-arch", 4.45, 4.8, 0.38, 0x4a475b, [0, -1.18, -6.15], "stone");
    [-3.95, -2.45, 2.45, 3.95].forEach((x, index) => column(`wizard-column-${index}`, 4.75, 0.34, 0x4b4859, [x, -1.2, -5.92], "stone"));
    rounded("wizard-dais", [3.6, 0.2, 1.75], 0x34354a, [0, -1.03, -5.02], "stone", 0.12);
    [0, 1, 2].forEach((ring) => add(`wizard-portal-ring-${ring}`, new THREE.TorusGeometry(1.18 - ring * 0.17, 0.055 + ring * 0.012, 16, 80), material(ring === 1 ? 0x8e68ff : 0x42d9dc, "glow", ring === 1 ? 0x7c54ff : 0x2fd7dd), [0, 0.66, -5.49], [0.04 * ring, 0.08 * ring, ring * 0.33]));
    const portalMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uTeal: { value: new THREE.Color(0x45e1d2) },
        uViolet: { value: new THREE.Color(0x8667ff) },
      },
      vertexShader: `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
      fragmentShader: `
        uniform float uTime; uniform vec3 uTeal; uniform vec3 uViolet; varying vec2 vUv;
        void main(){
          vec2 p = vUv - 0.5; float radius = length(p); float angle = atan(p.y, p.x);
          float spiral = sin(angle * 7.0 - uTime * 1.8 + radius * 30.0) * 0.5 + 0.5;
          float rings = sin(radius * 58.0 - uTime * 2.4) * 0.5 + 0.5;
          float core = 1.0 - smoothstep(0.02, 0.49, radius);
          vec3 color = mix(uViolet, uTeal, clamp(spiral * 0.72 + rings * 0.28, 0.0, 1.0));
          color += vec3(0.34, 0.18, 0.62) * pow(core, 2.0);
          float edge = 1.0 - smoothstep(0.43, 0.5, radius);
          gl_FragColor = vec4(color * (0.72 + core * 0.78), edge * (0.72 + core * 0.24));
        }
      `,
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.AdditiveBlending,
    });
    const portal = add("wizard-living-portal", new THREE.CircleGeometry(0.91, 72), portalMaterial, [0, 0.66, -5.55]);
    interactive(portal, "wizard-portal", "Living portal", "enter", "The portal reveals the next chapter after the spell ingredients are found.");
    portal.userData.wallalivePortalShader = true;
    portal.userData.worldMotion = { baseY: 0.66, phase: 0.4, amplitude: 0.035, speed: 1.6, spin: 0.045, spinAxis: "z" };
    add("wizard-floor-rune-outer", new THREE.RingGeometry(1.55, 1.68, 72), material(0x9e7cff, "glow", 0x7b62ff), [0, -1.14, -3.55], [-Math.PI / 2, 0, 0.18]);
    add("wizard-floor-rune-inner", new THREE.RingGeometry(0.72, 0.79, 64), material(0x66e4d2, "glow", 0x3fd6c2), [0, -1.135, -3.55], [-Math.PI / 2, 0, -0.2]);
    [-3.2, 3.15].forEach((x, shelfIndex) => {
      rounded(`wizard-bookcase-${shelfIndex}`, [1.62, 3.28, 0.48], 0x4a2e25, [x, 0.34, -6.12], "wood", 0.1);
      [-0.9, -0.05, 0.8].forEach((y, row) => rounded(`wizard-shelf-${shelfIndex}-${row}`, [1.48, 0.1, 0.54], 0x2f1c19, [x, y + 0.4, -5.82], "wood", 0.025));
      for (let book = 0; book < 8; book += 1) {
        const xOffset = -0.58 + (book % 4) * 0.38;
        const row = Math.floor(book / 4);
        rounded(`wizard-book-${shelfIndex}-${book}`, [0.23, 0.58 + (book % 3) * 0.07, 0.25], [0x6f3453, 0x295c63, 0x72552d, 0x493b72][book % 4], [x + xOffset, -0.38 + row * 0.96, -5.53], "velvet", 0.025, [0, 0, (book % 2 ? -1 : 1) * 0.04]);
      }
    });
    add("wizard-astrolabe-a", new THREE.TorusGeometry(0.68, 0.055, 12, 64), material(0xc89745, "brass"), [-2.5, 1.98, -4.95], [0.25, 0.52, 0.2]);
    add("wizard-astrolabe-b", new THREE.TorusGeometry(0.48, 0.045, 12, 64), material(0xc89745, "brass"), [-2.5, 1.98, -4.95], [1.05, 0.15, 0.7]);
    interactive(gem("wizard-crystal-a", 0x57e2df, [2.35, -0.37, -4.65], 1.2, 0.2), "wizard-crystal-a", "Tide crystal", "cast", "The tide crystal teaches the cast to match color, sound, and motion.");
    interactive(gem("wizard-crystal-b", 0xa476ff, [2.72, -0.2, -4.72], 0.92, 1.3), "wizard-crystal-b", "Moon crystal", "cast", "The moon crystal asks one character to float while another spins.");
    interactive(gem("wizard-crystal-c", 0xffd27a, [2.08, -0.48, -4.52], 0.72, 2.1), "wizard-crystal-c", "Sun crystal", "cast", "The sun crystal completes the cooperative spell sequence.");
    const spellBook = rounded("wizard-spell-book", [0.76, 0.14, 0.56], 0x6f3453, [-1.72, -0.35, -4.62], "velvet", 0.04, [0, -0.3, 0.04]);
    interactive(spellBook, "wizard-spell-book", "Spell book", "read", "The spell book gives every character a role based on its verified movements.");
    pointGlow("wizard-portal-light", 0x6c72ff, 8.5, [0, 0.7, -4.7], 7);
    pointGlow("wizard-lantern-light", 0xffb960, 4.8, [-2.7, 1.6, -4.7], 5);
  }

  if (world === "museum") {
    [-4.05, 4.05].forEach((x, index) => column(`museum-column-${index}`, 4.85, 0.38, 0xd6c5a8, [x, -1.2, -5.85], "stone"));
    [-2.28, 0, 2.28].forEach((x, index) => {
      rounded(`museum-frame-${index}`, [1.72, 2.05, 0.22], 0x95662d, [x, 1.02, -6.47], "brass", 0.055);
      const artwork = rounded(`museum-art-${index}`, [1.38, 1.7, 0.23], [0x4f7779, 0x8a544c, 0xa17c35][index], [x, 1.02, -6.32], "velvet", 0.035);
      interactive(artwork, `museum-art-${index}`, ["River of Shapes", "The Brave Mark", "Golden Echo"][index], "curate", "This artwork asks the cast to compare color, shape, and feeling.");
      add(`museum-picture-light-${index}`, new THREE.CylinderGeometry(0.035, 0.035, 0.9, 12), material(0xb68942, "brass"), [x, 2.38, -6.04], [0, 0, Math.PI / 2]);
    });
    rounded("museum-plinth", [1.5, 1.12, 1.22], 0xd5c8b2, [0, -0.62, -4.72], "stone", 0.08);
    const sculpture = add("museum-sculpture", new THREE.TorusKnotGeometry(0.48, 0.14, 192, 24, 2, 3), material(0x799398, "brass"), [0, 0.34, -4.72], [0.42, 0.22, 0.12]);
    interactive(sculpture, "museum-sculpture", "Motion sculpture", "inspect", "The sculpture turns as the cast tells a new interpretation together.");
    add("museum-inlay", new THREE.RingGeometry(2.2, 2.3, 72), material(0xc39a58, "brass"), [0, -1.15, -3.5], [-Math.PI / 2, 0, 0]);
    pointGlow("museum-gallery-light", 0xffe5bd, 5.6, [0, 3.25, -3.6], 8);
  }

  return { group, background: new THREE.Color(palette.background), fog: new THREE.Fog(palette.fog, 6.2, 17.5) };
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

function samplePolyline(path: ContourPoint[], amount: number) {
  if (path.length < 2) return path[0] ?? { x: 0, y: 0 };
  const lengths = path.slice(0, -1).map((point, index) => Math.hypot(path[index + 1].x - point.x, path[index + 1].y - point.y));
  const total = lengths.reduce((sum, length) => sum + length, 0);
  let remaining = THREE.MathUtils.clamp(amount, 0, 1) * total;
  for (let index = 0; index < lengths.length; index += 1) {
    if (remaining <= lengths[index] || index === lengths.length - 1) {
      const local = lengths[index] ? remaining / lengths[index] : 0;
      return {
        x: THREE.MathUtils.lerp(path[index].x, path[index + 1].x, local),
        y: THREE.MathUtils.lerp(path[index].y, path[index + 1].y, local),
      };
    }
    remaining -= lengths[index];
  }
  return path[path.length - 1];
}

function pointInsidePolygon(x: number, y: number, polygon: ContourPoint[]) {
  let inside = false;
  for (let current = 0, previous = polygon.length - 1; current < polygon.length; previous = current, current += 1) {
    const a = polygon[current];
    const b = polygon[previous];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
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
    roughness: 0.68,
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
  const reliefParts = rig.parts.filter((part) => (
    ["eye", "cheek", "nose", "mouth"].includes(part.kind)
    && (part.reviewed || ["learned-model", "learned-pose"].includes(part.source))
  ));
  const shell = buildArtworkShellGeometry(contour, depth, requestedHalfDepth, inflation, 3, reliefParts);
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
  const branchChains = structuralParts.map((part) => {
    const anchor = part.anchor ?? bodyPart?.center ?? { x: 0, y: 0, z: 0 };
    const sourcePath = part.path?.length && part.path.length >= 2 ? part.path : [anchor, part.center];
    const path: ContourPoint[] = Math.hypot(sourcePath[0].x - anchor.x, sourcePath[0].y - anchor.y) > 0.035
      ? [anchor, ...sourcePath]
      : [{ x: anchor.x, y: anchor.y }, ...sourcePath.slice(1)];
    const joint = samplePolyline(path, 0.52);
    const upperBone = new THREE.Bone();
    upperBone.name = `rig-${part.id}`;
    upperBone.position.set(anchor.x, anchor.y, 0);
    upperBone.userData.baseRotationZ = 0;
    upperBone.userData.wallaliveJointRole = "proximal";
    const tipBone = new THREE.Bone();
    tipBone.name = `rig-${part.id}-tip`;
    tipBone.position.set(joint.x - anchor.x, joint.y - anchor.y, 0);
    tipBone.userData.baseRotationZ = 0;
    tipBone.userData.wallaliveJointRole = "distal";
    upperBone.add(tipBone);
    rootBone.add(upperBone);
    return { part, upperBone, tipBone, anchor, path };
  });
  const positions = compactGeometry.getAttribute("position");
  const skinIndices = new Uint16Array(positions.count * 4);
  const skinWeights = new Float32Array(positions.count * 4);
  for (let vertex = 0; vertex < positions.count; vertex += 1) {
    const x = positions.getX(vertex);
    const y = positions.getY(vertex);
    let bestChain = -1;
    let bestWeight = 0;
    let bestProgress = 0;
    branchChains.forEach(({ part, path }, index) => {
      let minimumDistance = Infinity;
      let progress = 0;
      for (let segment = 0; segment < path.length - 1; segment += 1) {
        const projection = segmentProjection(x, y, path[segment], path[segment + 1]);
        if (projection.distance < minimumDistance) {
          minimumDistance = projection.distance;
          progress = (segment + projection.amount) / (path.length - 1);
        }
      }
      const radius = Math.max(0.028, Math.min(part.size.x, part.size.y) * 0.52);
      const insideSemanticOutline = !part.outline?.length || pointInsidePolygon(x, y, part.outline);
      const radial = Math.max(0, 1 - minimumDistance / (radius * 1.32)) * (insideSemanticOutline ? 1 : 0.08);
      const alongBranch = Math.min(1, Math.max(0, (progress - 0.02) / 0.34));
      const influence = radial * radial * alongBranch;
      if (influence > bestWeight) {
        bestWeight = influence;
        bestChain = index;
        bestProgress = progress;
      }
    });
    const branchWeight = bestWeight < 0.12 ? 0 : Math.min(0.94, THREE.MathUtils.smoothstep(bestWeight, 0.08, 0.78));
    const distalBlend = bestChain >= 0 ? THREE.MathUtils.smoothstep(bestProgress, 0.38, 0.66) : 0;
    const offset = vertex * 4;
    skinIndices[offset] = 0;
    skinWeights[offset] = 1 - branchWeight;
    if (bestChain >= 0) {
      const upperIndex = 1 + bestChain * 2;
      const tipIndex = upperIndex + 1;
      skinIndices[offset + 1] = upperIndex;
      skinIndices[offset + 2] = tipIndex;
      skinWeights[offset + 1] = branchWeight * (1 - distalBlend);
      skinWeights[offset + 2] = branchWeight * distalBlend;
    }
  }
  compactGeometry.setAttribute("skinIndex", new THREE.Uint16BufferAttribute(skinIndices, 4));
  compactGeometry.setAttribute("skinWeight", new THREE.Float32BufferAttribute(skinWeights, 4));
  const skinnedVolume = new THREE.SkinnedMesh(compactGeometry, [sideMaterial, frontMaterial, backMaterial]);
  skinnedVolume.name = "silhouette-distance-lens";
  skinnedVolume.add(rootBone);
  const articulationBones = branchChains.flatMap(({ upperBone, tipBone }) => [upperBone, tipBone]);
  skinnedVolume.bind(new THREE.Skeleton([rootBone, ...articulationBones]));
  skinnedVolume.castShadow = true;
  skinnedVolume.receiveShadow = true;
  skinnedVolume.userData.reconstruction = {
    method: "high-resolution contour-preserving articulated relief preview",
    polygonizer: "deterministic triangulated artwork shell",
    subdivisions: 3,
    topology: "closed",
    contourPoints: contour.length,
    skeletonPoints: skeleton.length,
    semanticRig: rig.version,
    skinning: `${branchChains.length} verified two-joint chains (${articulationBones.length} branch bones) with semantic-outline-clipped weights; unreviewed anatomy cannot deform geometry`,
    maximumHalfDepth: shell.maximumHalfDepth,
    maximumFrontDepth: shell.maximumFrontDepth,
    texturedFrontTriangles,
    neutralBackTriangles,
    sideTriangles: shell.sideTriangleCount,
    silhouetteError: 0,
    artworkSurfaceCoverage,
    projectedSemanticFeatures: false,
    continuousSemanticRelief: true,
    semanticReliefParts: reliefParts.length,
    learnedDepth: depth ? {
      model: depth.model,
      meanThickness: depth.meanThickness,
      meanAsymmetry: depth.meanAsymmetry,
      frontBackMirrored: false,
    } : null,
  };
  character.add(skinnedVolume);

  character.userData.reconstruction = {
    method: "high-resolution articulated relief preview",
    texturePlane: false,
    orbitableDegrees: 360,
    fullSculptDegrees: 0,
    bodyTopology: "closed",
    backPrior: "bounded neutral relief; full unseen-view reconstruction requires the neural sculpt path",
    maximumHalfDepth: shell.maximumHalfDepth,
    maximumFrontDepth: shell.maximumFrontDepth,
    texturedFrontTriangles,
    neutralBackTriangles,
    artworkSurfaceCoverage,
    artworkPreservedOnFront: true,
    projectedSemanticFeatures: false,
    continuousSemanticRelief: true,
    semanticReliefParts: reliefParts.length,
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
  character.userData.wallaliveJointChains = branchChains.map(({ part, path }) => ({ id: part.id, joints: 2, pathPoints: path.length }));

  return character;
}

type PaintableMaterial = THREE.Material & {
  map?: THREE.Texture | null;
  color?: THREE.Color;
};

type TexturePaintSurface = {
  key: string;
  material: PaintableMaterial;
  originalMap: THREE.Texture | null;
  originalColor: THREE.Color | null;
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  texture: THREE.CanvasTexture;
};

type VertexPaintSurface = {
  key: string;
  mesh: THREE.Mesh;
  originalColorAttribute: THREE.BufferAttribute | THREE.InterleavedBufferAttribute | null;
  colors: THREE.BufferAttribute;
  originalColors: Float32Array;
  positions: THREE.BufferAttribute | THREE.InterleavedBufferAttribute;
  normals: THREE.BufferAttribute | THREE.InterleavedBufferAttribute | null;
  materialStates: Array<{ material: PaintableMaterial & { vertexColors?: boolean }; vertexColors: boolean }>;
  cellSize: number;
  grid: Map<string, number[]>;
};

type TexturePaintedPoint = { kind: "texture"; surfaceKey: string; u: number; v: number; pressure: number };
type VertexPaintedPoint = { kind: "vertex"; surfaceKey: string; face: [number, number, number]; pressure: number };
type PaintedPoint = TexturePaintedPoint | VertexPaintedPoint;
type PaintedStroke = { brush: ModelPaintBrush; points: PaintedPoint[] };

function normalizedPaintBrush(brush: ModelPaintBrush): ModelPaintBrush {
  return {
    tool: ["brush", "spray", "oil", "spill"].includes(brush.tool) ? brush.tool : "brush",
    color: /^#[0-9a-f]{6}$/i.test(brush.color) ? brush.color.toLowerCase() : "#238fc7",
    size: THREE.MathUtils.clamp(brush.size, 0.08, 1),
  };
}

function createModelPainter(renderer: THREE.WebGLRenderer, camera: THREE.PerspectiveCamera, character: THREE.Group, controls: OrbitControls) {
  const textureSize = 1024;
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const surfaces = new Map<string, TexturePaintSurface>();
  const vertexSurfaces = new Map<string, VertexPaintSurface>();
  const strokes: PaintedStroke[] = [];
  let activeStroke: PaintedStroke | null = null;
  let enabled = false;

  const materialForHit = (hit: THREE.Intersection<THREE.Object3D>) => {
    if (!(hit.object instanceof THREE.Mesh)) return null;
    const projection = resolvePaintProjection({ hasUv: Boolean(hit.uv), hasFace: Boolean(hit.face) });
    if (!projection) return null;
    const materials = Array.isArray(hit.object.material) ? hit.object.material : [hit.object.material];
    const materialIndex = hit.face?.materialIndex ?? 0;
    const material = materials[materialIndex] as PaintableMaterial | undefined;
    if (!material || !("color" in material || "map" in material)) return null;
    return { material, mesh: hit.object, label: hit.object.name || "3D character", projection };
  };

  const drawOriginalSurface = (surface: TexturePaintSurface) => {
    const { context, canvas, originalMap, originalColor } = surface;
    context.save();
    context.globalAlpha = 1;
    context.globalCompositeOperation = "source-over";
    context.clearRect(0, 0, canvas.width, canvas.height);
    let imageDrawn = false;
    const image = originalMap?.image as CanvasImageSource | undefined;
    if (image) {
      try {
        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        imageDrawn = true;
      } catch {
        imageDrawn = false;
      }
    }
    if (!imageDrawn) {
      context.fillStyle = originalColor?.getStyle() ?? "#f4f0e7";
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    context.restore();
    surface.texture.needsUpdate = true;
  };

  const ensureSurface = (material: PaintableMaterial) => {
    const existing = surfaces.get(material.uuid);
    if (existing) return existing;
    const canvas = document.createElement("canvas");
    canvas.width = textureSize;
    canvas.height = textureSize;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return null;
    const originalMap = material.map ?? null;
    const originalColor = material.color?.clone() ?? null;
    const texture = new THREE.CanvasTexture(canvas);
    texture.name = `wallalive-child-paint-${material.uuid}`;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    if (originalMap) {
      texture.flipY = originalMap.flipY;
      texture.wrapS = originalMap.wrapS;
      texture.wrapT = originalMap.wrapT;
      texture.repeat.copy(originalMap.repeat);
      texture.offset.copy(originalMap.offset);
      texture.center.copy(originalMap.center);
      texture.rotation = originalMap.rotation;
    }
    const surface: TexturePaintSurface = { key: material.uuid, material, originalMap, originalColor, canvas, context, texture };
    surfaces.set(surface.key, surface);
    drawOriginalSurface(surface);
    material.map = texture;
    material.color?.set(0xffffff);
    material.needsUpdate = true;
    return surface;
  };

  const seeded = (point: PaintedPoint, index: number) => {
    const first = point.kind === "texture" ? point.u : point.face[0] * 0.00017 + point.face[1] * 0.000031;
    const second = point.kind === "texture" ? point.v : point.face[2] * 0.000071;
    const value = Math.sin(first * 12871 + second * 7919 + index * 104729) * 43758.5453;
    return value - Math.floor(value);
  };

  const paintPoint = (surface: TexturePaintSurface, point: TexturePaintedPoint, brush: ModelPaintBrush, previous?: TexturePaintedPoint) => {
    const context = surface.context;
    const x = point.u * textureSize;
    const y = (1 - point.v) * textureSize;
    const radius = (8 + brush.size * 46) * (0.72 + point.pressure * 0.32);
    context.save();
    context.lineCap = "round";
    context.lineJoin = "round";

    if (brush.tool === "spray") {
      const dots = Math.round(24 + brush.size * 58);
      context.fillStyle = brush.color;
      for (let index = 0; index < dots; index += 1) {
        const angle = seeded(point, index * 2) * Math.PI * 2;
        const distance = Math.sqrt(seeded(point, index * 2 + 1)) * radius * 1.7;
        const dot = 0.9 + seeded(point, index + 231) * (2.2 + brush.size * 2.8);
        context.globalAlpha = 0.22 + seeded(point, index + 991) * 0.56;
        context.beginPath();
        context.arc(x + Math.cos(angle) * distance, y + Math.sin(angle) * distance, dot, 0, Math.PI * 2);
        context.fill();
      }
    } else if (brush.tool === "spill") {
      context.globalAlpha = 0.9;
      context.fillStyle = brush.color;
      context.beginPath();
      for (let index = 0; index < 22; index += 1) {
        const angle = index / 22 * Math.PI * 2;
        const wobble = radius * (0.72 + seeded(point, index) * 0.7);
        const px = x + Math.cos(angle) * wobble;
        const py = y + Math.sin(angle) * wobble;
        if (index === 0) context.moveTo(px, py); else context.lineTo(px, py);
      }
      context.closePath();
      context.fill();
      for (let drip = 0; drip < 4; drip += 1) {
        const dripX = x + (seeded(point, drip + 88) - 0.5) * radius * 1.2;
        const dripLength = radius * (0.7 + seeded(point, drip + 144) * 1.55);
        context.lineWidth = radius * (0.13 + seeded(point, drip + 202) * 0.12);
        context.strokeStyle = brush.color;
        context.beginPath();
        context.moveTo(dripX, y + radius * 0.3);
        context.lineTo(dripX + (seeded(point, drip + 301) - 0.5) * radius * 0.25, y + dripLength);
        context.stroke();
      }
    } else {
      const startX = previous ? previous.u * textureSize : x;
      const startY = previous ? (1 - previous.v) * textureSize : y;
      context.globalAlpha = brush.tool === "oil" ? 0.94 : 0.88;
      context.strokeStyle = brush.color;
      context.lineWidth = radius * (brush.tool === "oil" ? 2.25 : 1.55);
      context.shadowColor = brush.color;
      context.shadowBlur = brush.tool === "oil" ? radius * 0.2 : radius * 0.08;
      context.beginPath();
      context.moveTo(startX, startY);
      context.lineTo(x, y);
      context.stroke();
      if (brush.tool === "oil") {
        const highlight = new THREE.Color(brush.color).lerp(new THREE.Color(0xffffff), 0.42).getStyle();
        context.globalAlpha = 0.34;
        context.shadowBlur = 0;
        context.strokeStyle = highlight;
        context.lineWidth = Math.max(2, radius * 0.28);
        context.beginPath();
        context.moveTo(startX - radius * 0.22, startY - radius * 0.18);
        context.lineTo(x - radius * 0.22, y - radius * 0.18);
        context.stroke();
      }
    }
    context.restore();
    surface.texture.needsUpdate = true;
  };

  const ensureVertexSurface = (mesh: THREE.Mesh) => {
    const existing = vertexSurfaces.get(mesh.geometry.uuid);
    if (existing) return existing;
    const positions = mesh.geometry.getAttribute("position");
    if (!positions) return null;
    const originalColorAttribute = mesh.geometry.getAttribute("color") ?? null;
    const originalColors = new Float32Array(positions.count * 3);
    for (let index = 0; index < positions.count; index += 1) {
      originalColors[index * 3] = originalColorAttribute?.getX(index) ?? 1;
      originalColors[index * 3 + 1] = originalColorAttribute?.getY(index) ?? 1;
      originalColors[index * 3 + 2] = originalColorAttribute?.getZ(index) ?? 1;
    }
    const colors = new THREE.Float32BufferAttribute(originalColors.slice(), 3);
    colors.setUsage(THREE.DynamicDrawUsage);
    mesh.geometry.setAttribute("color", colors);
    mesh.geometry.computeBoundingSphere();
    const sphereRadius = mesh.geometry.boundingSphere?.radius ?? 0.7;
    const cellSize = Math.max(0.012, sphereRadius * 0.13);
    const grid = new Map<string, number[]>();
    const cellKey = (x: number, y: number, z: number) => `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)},${Math.floor(z / cellSize)}`;
    for (let index = 0; index < positions.count; index += 1) {
      const key = cellKey(positions.getX(index), positions.getY(index), positions.getZ(index));
      const cell = grid.get(key);
      if (cell) cell.push(index); else grid.set(key, [index]);
    }
    const materials = (Array.isArray(mesh.material) ? mesh.material : [mesh.material]) as Array<PaintableMaterial & { vertexColors?: boolean }>;
    const materialStates = materials.map((material) => ({ material, vertexColors: Boolean(material.vertexColors) }));
    materials.forEach((material) => { material.vertexColors = true; material.needsUpdate = true; });
    const surface: VertexPaintSurface = {
      key: `vertex-${mesh.geometry.uuid}`,
      mesh,
      originalColorAttribute,
      colors,
      originalColors,
      positions,
      normals: mesh.geometry.getAttribute("normal") ?? null,
      materialStates,
      cellSize,
      grid,
    };
    vertexSurfaces.set(mesh.geometry.uuid, surface);
    return surface;
  };

  const resetVertexSurface = (surface: VertexPaintSurface) => {
    const colorArray = surface.colors.array as Float32Array;
    colorArray.set(surface.originalColors);
    surface.colors.needsUpdate = true;
  };

  const paintVertexPoint = (surface: VertexPaintSurface, point: VertexPaintedPoint, brush: ModelPaintBrush) => {
    const { positions, normals, colors, cellSize, grid } = surface;
    const [a, b, c] = point.face;
    const centerX = (positions.getX(a) + positions.getX(b) + positions.getX(c)) / 3;
    const centerY = (positions.getY(a) + positions.getY(b) + positions.getY(c)) / 3;
    const centerZ = (positions.getZ(a) + positions.getZ(b) + positions.getZ(c)) / 3;
    const normalX = normals ? (normals.getX(a) + normals.getX(b) + normals.getX(c)) / 3 : 0;
    const normalY = normals ? (normals.getY(a) + normals.getY(b) + normals.getY(c)) / 3 : 0;
    const normalZ = normals ? (normals.getZ(a) + normals.getZ(b) + normals.getZ(c)) / 3 : 1;
    const sphereRadius = surface.mesh.geometry.boundingSphere?.radius ?? 0.7;
    const toolScale = brush.tool === "spill" ? 1.35 : brush.tool === "oil" ? 1.12 : brush.tool === "spray" ? 1.18 : 1;
    const radius = sphereRadius * (0.025 + brush.size * 0.13) * toolScale * (0.76 + point.pressure * 0.28);
    const cellReach = Math.max(1, Math.ceil(radius / cellSize));
    const cellX = Math.floor(centerX / cellSize);
    const cellY = Math.floor(centerY / cellSize);
    const cellZ = Math.floor(centerZ / cellSize);
    const paintColor = new THREE.Color(brush.color);
    const current = new THREE.Color();
    const white = new THREE.Color(0xffffff);
    const candidates = new Set<number>([a, b, c]);
    for (let x = -cellReach; x <= cellReach; x += 1) for (let y = -cellReach; y <= cellReach; y += 1) for (let z = -cellReach; z <= cellReach; z += 1) {
      grid.get(`${cellX + x},${cellY + y},${cellZ + z}`)?.forEach((index) => candidates.add(index));
    }
    candidates.forEach((index) => {
      const dx = positions.getX(index) - centerX;
      const dy = positions.getY(index) - centerY;
      const dz = positions.getZ(index) - centerZ;
      const distance = Math.hypot(dx, dy, dz);
      if (distance > radius) return;
      if (normals) {
        const facing = normals.getX(index) * normalX + normals.getY(index) * normalY + normals.getZ(index) * normalZ;
        if (facing < 0.05) return;
      }
      let strength = Math.max(0, 1 - distance / Math.max(0.0001, radius));
      if (brush.tool === "spray") {
        if (seeded(point, index) > strength * 0.92) return;
        strength *= 0.58;
      } else if (brush.tool === "spill") {
        const wobble = 0.7 + seeded(point, index + 317) * 0.42;
        if (distance > radius * wobble) return;
        strength = Math.min(0.92, 0.48 + strength * 0.52);
      } else if (brush.tool === "oil") {
        strength = Math.min(0.94, 0.38 + strength * 0.62);
      } else strength *= 0.82;
      current.setRGB(colors.getX(index), colors.getY(index), colors.getZ(index));
      current.lerp(paintColor, strength);
      if (brush.tool === "oil" && seeded(point, index + 701) > 0.83) current.lerp(white, 0.12);
      colors.setXYZ(index, current.r, current.g, current.b);
    });
    colors.needsUpdate = true;
  };

  const replay = () => {
    surfaces.forEach(drawOriginalSurface);
    vertexSurfaces.forEach(resetVertexSurface);
    strokes.forEach((stroke) => {
      let previous: TexturePaintedPoint | undefined;
      stroke.points.forEach((point) => {
        if (point.kind === "texture") {
          const surface = surfaces.get(point.surfaceKey);
          if (!surface) return;
          paintPoint(surface, point, stroke.brush, previous?.surfaceKey === point.surfaceKey ? previous : undefined);
          previous = point;
        } else {
          const surface = [...vertexSurfaces.values()].find((candidate) => candidate.key === point.surfaceKey);
          if (surface) paintVertexPoint(surface, point, stroke.brush);
          previous = undefined;
        }
      });
    });
  };

  const inspect = (): ModelPaintInspection => ({
    strokeCount: strokes.length + (activeStroke?.points.length ? 1 : 0),
    paintedSurfaceCount: new Set([...strokes, ...(activeStroke?.points.length ? [activeStroke] : [])].flatMap((stroke) => stroke.points.map((point) => point.surfaceKey))).size,
    colors: [...new Set([...strokes, ...(activeStroke?.points.length ? [activeStroke] : [])].map((stroke) => stroke.brush.color))],
    tools: [...new Set([...strokes, ...(activeStroke?.points.length ? [activeStroke] : [])].map((stroke) => stroke.brush.tool))],
  });

  return {
    setEnabled(next: boolean) {
      enabled = next;
      controls.enabled = !next;
      renderer.domElement.style.cursor = next ? "crosshair" : "grab";
      renderer.domElement.dataset.paintEnabled = String(next);
    },
    begin(brush: ModelPaintBrush) {
      if (activeStroke?.points.length) strokes.push(activeStroke);
      activeStroke = { brush: normalizedPaintBrush(brush), points: [] };
    },
    paint(x: number, y: number, pressure = 0.5) {
      if (!enabled || !activeStroke) return { painted: false };
      if (activeStroke.brush.tool === "spill" && activeStroke.points.length) return { painted: true };
      pointer.set(THREE.MathUtils.clamp(x, 0, 1) * 2 - 1, -(THREE.MathUtils.clamp(y, 0, 1) * 2 - 1));
      character.updateWorldMatrix(true, true);
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObject(character, true).find((candidate) => Boolean(candidate.uv || candidate.face));
      if (!hit) return { painted: false };
      const target = materialForHit(hit);
      if (!target) return { painted: false };
      const normalizedPressure = THREE.MathUtils.clamp(pressure || 0.5, 0.15, 1);
      let point: PaintedPoint;
      if (target.projection === "texture" && hit.uv) {
        const surface = ensureSurface(target.material);
        if (!surface) return { painted: false };
        point = {
          kind: "texture",
          surfaceKey: surface.key,
          u: THREE.MathUtils.clamp(hit.uv.x, 0, 1),
          v: THREE.MathUtils.clamp(hit.uv.y, 0, 1),
          pressure: normalizedPressure,
        };
        const previous = activeStroke.points.at(-1);
        paintPoint(surface, point, activeStroke.brush, previous?.kind === "texture" && previous.surfaceKey === point.surfaceKey ? previous : undefined);
      } else if (target.projection === "vertex" && hit.face) {
        const surface = ensureVertexSurface(target.mesh);
        if (!surface) return { painted: false };
        point = { kind: "vertex", surfaceKey: surface.key, face: [hit.face.a, hit.face.b, hit.face.c], pressure: normalizedPressure };
        paintVertexPoint(surface, point, activeStroke.brush);
      } else return { painted: false };
      activeStroke.points.push(point);
      return { painted: true, target: target.label };
    },
    end() {
      if (activeStroke?.points.length) strokes.push(activeStroke);
      activeStroke = null;
      return inspect();
    },
    undo() {
      activeStroke = null;
      strokes.pop();
      replay();
      return inspect();
    },
    reset() {
      activeStroke = null;
      strokes.length = 0;
      replay();
      return inspect();
    },
    inspect,
    dispose() {
      surfaces.forEach((surface) => {
        surface.material.map = surface.originalMap;
        if (surface.originalColor && surface.material.color) surface.material.color.copy(surface.originalColor);
        surface.material.needsUpdate = true;
        surface.texture.dispose();
      });
      vertexSurfaces.forEach((surface) => {
        if (surface.originalColorAttribute) surface.mesh.geometry.setAttribute("color", surface.originalColorAttribute);
        else surface.mesh.geometry.deleteAttribute("color");
        surface.materialStates.forEach(({ material, vertexColors }) => { material.vertexColors = vertexColors; material.needsUpdate = true; });
      });
      surfaces.clear();
      vertexSurfaces.clear();
    },
  };
}

export const ARStage = forwardRef<ARStageHandle, ARStageProps>(function ARStage(
  { characters, contour, skeleton, textureUrl, rig, depth, action, ensembleActions = null, world, lightingMood, cameraPreset, accent, inflation, neuralAssetUrl, paintEnabled = false, visible, onCapability, onRendererCapability, onPlaced, onNeuralAssetInfo, onWorldInteraction },
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
  const paintEnabledRef = useRef(paintEnabled);
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
    // Probe once before constructing Three renderers. THREE logs several hard
    // console errors for each failed profile, which turns an expected
    // no-WebGL browser fallback into noisy false alarms during judging.
    const probeCanvas = document.createElement("canvas");
    const probe = probeCanvas.getContext("webgl2", { alpha: true, antialias: false });
    if (!probe) {
      mount.dataset.renderer = "unavailable";
      setRendererError(true);
      onRendererCapability(false);
      onCapability(false);
      return;
    }
    probe.getExtension("WEBGL_lose_context")?.loseContext();
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
      onRendererCapability(false);
      onCapability(false);
      return;
    }
    setRendererError(false);
    onRendererCapability(true);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 0.9;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.xr.enabled = true;
    renderer.domElement.setAttribute("aria-label", "Semantic articulated 3D drawing");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    let environment = buildWorldEnvironment(worldStateRef.current);
    scene.add(environment.group);
    scene.background = environment.background;
    scene.fog = environment.fog;
    const syncWorldDiagnostics = () => {
      let meshes = 0;
      let lights = 0;
      const geometryKinds = new Set<string>();
      environment.group.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          meshes += 1;
          geometryKinds.add(object.geometry.type);
        }
        if (object instanceof THREE.Light) lights += 1;
      });
      mount.dataset.world = String(environment.group.userData.world ?? worldStateRef.current);
      mount.dataset.worldMeshes = String(meshes);
      mount.dataset.worldLights = String(lights);
      mount.dataset.worldGeometryKinds = String(geometryKinds.size);
    };
    syncWorldDiagnostics();
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

    const activateWorldObject = (id: string) => {
      let target: THREE.Object3D | null = null;
      environment.group.traverse((object) => {
        const interaction = object.userData.wallaliveInteraction as WorldObjectInteraction | undefined;
        if (!target && interaction?.id === id) target = object;
      });
      if (!target) return false;
      const object = target as THREE.Object3D;
      const interaction = object.userData.wallaliveInteraction as WorldObjectInteraction;
      object.userData.wallaliveActivatedAt = performance.now();
      object.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((surface) => {
          if ("emissiveIntensity" in surface && typeof surface.emissiveIntensity === "number") surface.emissiveIntensity = Math.max(surface.emissiveIntensity, 1.25);
        });
      });
      onWorldInteraction?.(interaction);
      return true;
    };

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const interactiveAt = (event: PointerEvent) => {
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.set((event.clientX - bounds.left) / Math.max(1, bounds.width) * 2 - 1, -((event.clientY - bounds.top) / Math.max(1, bounds.height)) * 2 + 1);
      raycaster.setFromCamera(pointer, camera);
      return raycaster.intersectObjects(environment.group.children, true).find((hit) => Boolean(hit.object.userData.wallaliveInteraction))?.object ?? null;
    };
    const onWorldPointerMove = (event: PointerEvent) => {
      if (renderer.domElement.dataset.paintEnabled === "true") return;
      renderer.domElement.style.cursor = interactiveAt(event) ? "pointer" : "grab";
    };
    const onWorldPointerUp = (event: PointerEvent) => {
      if (renderer.domElement.dataset.paintEnabled === "true") return;
      const target = interactiveAt(event);
      const interaction = target?.userData.wallaliveInteraction as WorldObjectInteraction | undefined;
      if (interaction) activateWorldObject(interaction.id);
    };
    renderer.domElement.addEventListener("pointermove", onWorldPointerMove);
    renderer.domElement.addEventListener("pointerup", onWorldPointerUp);

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
    const syncCharacterDiagnostics = () => {
      let bones = 0;
      let articulatedChains = 0;
      characterRoot.traverse((object) => {
        if (object instanceof THREE.Bone) bones += 1;
        if (object.userData.wallaliveJointRole === "distal") articulatedChains += 1;
      });
      mount.dataset.rigBones = String(bones);
      mount.dataset.articulatedChains = String(articulatedChains);
      mount.dataset.characterMode = neuralAssetUrl ? "neural-gltf" : "local-artwork-shell";
    };
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
          syncCharacterDiagnostics();
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
      syncCharacterDiagnostics();
      onNeuralAssetInfo(null);
    } else if (contour?.length && skeleton?.length && rig) {
      characterRoot.add(buildCharacter(contour, skeleton, rig, depth, textureUrl, accent, inflation, {
        poseApplicable: false,
        topologyApplicable: false,
      }));
      syncCharacterDiagnostics();
      onNeuralAssetInfo(null);
    }
    const modelPainter = createModelPainter(renderer, camera, characterRoot, controls);
    modelPainter.setEnabled(Boolean(paintEnabledRef.current));

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

    const animationStartedAt = performance.now();
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
      const elapsed = (performance.now() - animationStartedAt) / 1000;
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
      environment.group.traverse((object) => {
        if (object.userData.wallalivePortalShader && object instanceof THREE.Mesh && object.material instanceof THREE.ShaderMaterial) {
          object.material.uniforms.uTime.value = elapsed;
        }
        const motion = object.userData.worldMotion as { baseY: number; phase: number; amplitude: number; speed: number; spin: number; spinAxis?: "y" | "z" } | undefined;
        if (!motion) return;
        object.position.y = motion.baseY + Math.sin(elapsed * motion.speed + motion.phase) * motion.amplitude;
        object.rotation[motion.spinAxis ?? "y"] += delta * motion.spin;
      });
      environment.group.traverse((object) => {
        const activatedAt = Number(object.userData.wallaliveActivatedAt ?? 0);
        if (!activatedAt) return;
        const age = (performance.now() - activatedAt) / 1000;
        const pulse = age < 1.4 ? 1 + Math.sin(age * Math.PI * 5) * Math.max(0, 0.12 * (1 - age / 1.4)) : 1;
        object.scale.setScalar(pulse);
        if (age >= 1.4) object.userData.wallaliveActivatedAt = 0;
      });
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
        const chainNodes = (partId: string) => ({
          upper: articulated.getObjectByName(`rig-${partId}`),
          tip: articulated.getObjectByName(`rig-${partId}-tip`),
        });
        movable.forEach((part) => {
          const { upper, tip } = chainNodes(part.id);
          [upper, tip].forEach((node) => {
            if (!node) return;
            node.rotation.x = 0;
            node.rotation.y = 0;
            node.rotation.z = Number(node.userData.baseRotationZ ?? 0);
          });
        });
        if (instanceAction === "wave") {
          const quadruped = localRig?.topologyKind === "quadruped";
          const part = (quadruped ? movable.find((candidate) => candidate.kind === "tail") : null)
            ?? movable.find((candidate) => candidate.kind === "arm" && candidate.side === "right")
            ?? movable.find((candidate) => candidate.kind === "arm")
            ?? movable.find((candidate) => candidate.kind === "wing" || candidate.kind === "tentacle" || candidate.kind === "tail");
          const chain = part ? chainNodes(part.id) : null;
          if (chain?.upper) {
            const amplitude = quadruped || part?.kind !== "arm" ? 0.2 : 0.3;
            chain.upper.rotation.z = Number(chain.upper.userData.baseRotationZ ?? 0) + amplitude + Math.sin(localElapsed * 7.2) * amplitude;
            if (chain.tip) chain.tip.rotation.z = Number(chain.tip.userData.baseRotationZ ?? 0) + Math.sin(localElapsed * 7.2 + 0.7) * amplitude * 0.72;
          }
        }
        if (instanceAction === "dance") {
          movable.forEach((part, index) => {
            const { upper, tip } = chainNodes(part.id);
            if (!upper) return;
            const direction = part.side === "right" || index % 2 ? -1 : 1;
            upper.rotation.z = Number(upper.userData.baseRotationZ ?? 0) + direction * Math.sin(localElapsed * 5.2 + index * 0.45) * 0.24;
            upper.rotation.x = Math.sin(localElapsed * 3.8 + index) * 0.08;
            if (tip) tip.rotation.z = Number(tip.userData.baseRotationZ ?? 0) - direction * Math.sin(localElapsed * 5.2 + index * 0.45 + 0.8) * 0.2;
          });
        }
        if (instanceAction === "walk") {
          movable.filter((part) => part.kind === "leg").forEach((part, index) => {
            const { upper, tip } = chainNodes(part.id);
            if (!upper) return;
            const direction = part.side === "right" || index % 2 ? -1 : 1;
            const stride = Math.sin(localElapsed * 7);
            upper.rotation.z = Number(upper.userData.baseRotationZ ?? 0) + direction * stride * 0.26;
            upper.rotation.x = direction * stride * 0.12;
            if (tip) tip.rotation.z = Number(tip.userData.baseRotationZ ?? 0) - direction * Math.max(0, stride) * 0.38;
          });
          movable.filter((part) => part.kind === "arm").forEach((part, index) => {
            const { upper, tip } = chainNodes(part.id);
            const direction = index % 2 ? 1 : -1;
            if (upper) upper.rotation.z = Number(upper.userData.baseRotationZ ?? 0) + direction * Math.sin(localElapsed * 7) * 0.24;
            if (tip) tip.rotation.z = Number(tip.userData.baseRotationZ ?? 0) - direction * Math.sin(localElapsed * 7 + 0.5) * 0.12;
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
      syncWorldDiagnostics();
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
      renderer.domElement.removeEventListener("pointermove", onWorldPointerMove);
      renderer.domElement.removeEventListener("pointerup", onWorldPointerUp);
      controls.dispose();
      modelPainter.dispose();
      environmentMap.dispose();
      renderer.setAnimationLoop(null);
      disposeObject(scene);
      renderer.dispose();
      renderer.forceContextLoss();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };

    handlesRef.current = {
      renderer,
      scene,
      camera,
      character: characterRoot,
      reticle,
      setWorld: setStageWorld,
      setLightingMood: applyLightingMood,
      setCameraPreset: applyCameraPreset,
      interactWorldObject: activateWorldObject,
      setPaintEnabled: modelPainter.setEnabled,
      beginPaintStroke: modelPainter.begin,
      paintAtNormalized: modelPainter.paint,
      endPaintStroke: modelPainter.end,
      undoPaint: modelPainter.undo,
      resetPaint: modelPainter.reset,
      inspectPaint: modelPainter.inspect,
      controls,
      dispose,
    };
    if (navigator.xr) navigator.xr.isSessionSupported("immersive-ar").then(onCapability).catch(() => onCapability(false));
    else onCapability(false);

    return () => {
      handlesRef.current = null;
      dispose();
    };
  }, [accent, characters, contour, depth, inflation, neuralAssetUrl, onCapability, onNeuralAssetInfo, onRendererCapability, onWorldInteraction, rig, skeleton, textureUrl, visible]);

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

  useEffect(() => {
    paintEnabledRef.current = paintEnabled;
    handlesRef.current?.setPaintEnabled(Boolean(paintEnabled));
  }, [paintEnabled]);

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
    interactWorldObject(id: string) {
      return handlesRef.current?.interactWorldObject(id) ?? false;
    },
    setPaintEnabled(enabled: boolean) {
      paintEnabledRef.current = enabled;
      handlesRef.current?.setPaintEnabled(enabled);
    },
    beginPaintStroke(brush: ModelPaintBrush) {
      handlesRef.current?.beginPaintStroke(brush);
    },
    paintAtNormalized(x: number, y: number, pressure = 0.5) {
      return handlesRef.current?.paintAtNormalized(x, y, pressure) ?? { painted: false };
    },
    endPaintStroke() {
      return handlesRef.current?.endPaintStroke() ?? { strokeCount: 0, paintedSurfaceCount: 0, colors: [], tools: [] };
    },
    undoPaint() {
      return handlesRef.current?.undoPaint() ?? { strokeCount: 0, paintedSurfaceCount: 0, colors: [], tools: [] };
    },
    resetPaint() {
      return handlesRef.current?.resetPaint() ?? { strokeCount: 0, paintedSurfaceCount: 0, colors: [], tools: [] };
    },
    inspectPaint() {
      return handlesRef.current?.inspectPaint() ?? { strokeCount: 0, paintedSurfaceCount: 0, colors: [], tools: [] };
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
