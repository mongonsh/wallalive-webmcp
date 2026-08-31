"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { DecalGeometry } from "three/addons/geometries/DecalGeometry.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MarchingCubes } from "three/addons/objects/MarchingCubes.js";
import type { CharacterRig, ContourPoint, LearnedDepthField, SemanticPart, SkeletonPoint } from "../lib/drawing";
import { disposeObject, prepareNeuralCharacter, type NeuralRigMap, type NeuralSemanticMap, type RiggedAssetInfo } from "../lib/rigged-model";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin";

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
  rotateBy: (yaw: number, pitch: number) => void;
};

type ARStageProps = {
  contour: ContourPoint[] | null;
  skeleton: SkeletonPoint[] | null;
  textureUrl: string | null;
  rig: CharacterRig | null;
  depth: LearnedDepthField | null;
  action: CharacterAction;
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
  dispose: () => void;
};

const VOLUME_RESOLUTION = 64;
const GRAPH_APPENDAGE_KINDS = new Set([
  "arm", "leg", "wing", "fin", "tail", "tentacle", "trunk", "branch", "segment", "linkage",
]);

function pointInsideContour(x: number, y: number, contour: ContourPoint[]) {
  let inside = false;
  for (let current = 0, previous = contour.length - 1; current < contour.length; previous = current, current += 1) {
    const a = contour[current];
    const b = contour[previous];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function distanceToSegment(x: number, y: number, start: ContourPoint, end: ContourPoint) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const amount = lengthSquared ? Math.min(1, Math.max(0, ((x - start.x) * dx + (y - start.y) * dy) / lengthSquared)) : 0;
  return Math.hypot(x - (start.x + dx * amount), y - (start.y + dy * amount));
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

function distanceToContour(x: number, y: number, contour: ContourPoint[]) {
  let distance = Infinity;
  for (let index = 0; index < contour.length; index += 1) {
    distance = Math.min(distance, distanceToSegment(x, y, contour[index], contour[(index + 1) % contour.length]));
  }
  return distance;
}

function inkMaterial(color: string) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.48,
    metalness: 0,
    emissive: new THREE.Color(color),
    emissiveIntensity: 0.16,
  });
}

function sampleLearnedDepth(field: Float32Array, depth: LearnedDepthField, x: number, y: number) {
  const modelX = Math.min(depth.size - 1, Math.max(0, (x / 1.4 + 0.5) * (depth.size - 1)));
  const modelY = Math.min(depth.size - 1, Math.max(0, (0.5 - y / 1.4) * (depth.size - 1)));
  const x0 = Math.floor(modelX);
  const y0 = Math.floor(modelY);
  const x1 = Math.min(depth.size - 1, x0 + 1);
  const y1 = Math.min(depth.size - 1, y0 + 1);
  const amountX = modelX - x0;
  const amountY = modelY - y0;
  const top = field[y0 * depth.size + x0] * (1 - amountX) + field[y0 * depth.size + x1] * amountX;
  const bottom = field[y1 * depth.size + x0] * (1 - amountX) + field[y1 * depth.size + x1] * amountX;
  return (top * (1 - amountY) + bottom * amountY) * depth.depthScale;
}

export function buildCharacter(contour: ContourPoint[], skeleton: SkeletonPoint[], rig: CharacterRig, depth: LearnedDepthField | null, textureUrl: string | null, accent: string, inflation: number) {
  const character = new THREE.Group();
  character.name = "wallalive-semantic-character";

  const volumeMaterial = new THREE.MeshPhysicalMaterial({
    color: rig.bodyColor,
    roughness: 0.68,
    metalness: 0,
    clearcoat: 0.18,
    clearcoatRoughness: 0.72,
    side: THREE.DoubleSide,
  });
  const volume = new MarchingCubes(VOLUME_RESOLUTION, volumeMaterial, false, false, 200_000);
  volume.name = "silhouette-distance-lens";
  volume.isolation = 0;
  volume.field.fill(-1);
  const half = VOLUME_RESOLUTION / 2;
  const cell = 1 / half;
  const bodyPart = rig.parts.find((part) => part.kind === "body");
  const insideField = new Uint8Array(VOLUME_RESOLUTION * VOLUME_RESOLUTION);
  const edgeDistanceField = new Float32Array(VOLUME_RESOLUTION * VOLUME_RESOLUTION);
  let maximumInteriorDistance = cell;
  for (let y = 1; y < VOLUME_RESOLUTION - 1; y += 1) {
    const fieldY = (y - half) / half;
    for (let x = 1; x < VOLUME_RESOLUTION - 1; x += 1) {
      const fieldX = (x - half) / half;
      const index = y * VOLUME_RESOLUTION + x;
      const inside = pointInsideContour(fieldX, fieldY, contour);
      const edgeDistance = distanceToContour(fieldX, fieldY, contour);
      insideField[index] = inside ? 1 : 0;
      edgeDistanceField[index] = edgeDistance;
      if (inside) maximumInteriorDistance = Math.max(maximumInteriorDistance, edgeDistance);
    }
  }
  const bodyHalfDepth = Math.min(0.48, Math.max(0.12, (bodyPart?.size.z ?? 0.34) * 0.58 * inflation));
  const fallbackDepthAt = (x: number, y: number) => {
    if (!pointInsideContour(x, y, contour)) return 0;
    const normalizedDistance = Math.min(1, distanceToContour(x, y, contour) / maximumInteriorDistance);
    return Math.max(cell * 0.7, bodyHalfDepth * Math.sqrt(normalizedDistance));
  };
  const frontDepthAt = (x: number, y: number) => {
    if (!pointInsideContour(x, y, contour)) return 0;
    return depth
      ? Math.max(cell * 0.7, sampleLearnedDepth(depth.front, depth, x, y) * inflation)
      : fallbackDepthAt(x, y);
  };
  const backDepthAt = (x: number, y: number) => {
    if (!pointInsideContour(x, y, contour)) return 0;
    return depth
      ? Math.max(cell * 0.7, sampleLearnedDepth(depth.back, depth, x, y) * inflation)
      : fallbackDepthAt(x, y);
  };
  let maximumFrontDepth = cell;
  let maximumBackDepth = cell;
  const frontDepthField = new Float32Array(VOLUME_RESOLUTION * VOLUME_RESOLUTION);
  const backDepthField = new Float32Array(VOLUME_RESOLUTION * VOLUME_RESOLUTION);
  for (let y = 1; y < VOLUME_RESOLUTION - 1; y += 1) {
    const fieldY = (y - half) / half;
    for (let x = 1; x < VOLUME_RESOLUTION - 1; x += 1) {
      const planeIndex = y * VOLUME_RESOLUTION + x;
      if (!insideField[planeIndex]) continue;
      const fieldX = (x - half) / half;
      frontDepthField[planeIndex] = frontDepthAt(fieldX, fieldY);
      backDepthField[planeIndex] = backDepthAt(fieldX, fieldY);
      maximumFrontDepth = Math.max(maximumFrontDepth, frontDepthField[planeIndex]);
      maximumBackDepth = Math.max(maximumBackDepth, backDepthField[planeIndex]);
    }
  }
  for (let z = 1; z < VOLUME_RESOLUTION - 1; z += 1) {
    const fieldZ = (z - half) / half;
    for (let y = 1; y < VOLUME_RESOLUTION - 1; y += 1) {
      for (let x = 1; x < VOLUME_RESOLUTION - 1; x += 1) {
        const planeIndex = y * VOLUME_RESOLUTION + x;
        const edgeDistance = edgeDistanceField[planeIndex];
        const signedEdge = insideField[planeIndex] ? edgeDistance : -edgeDistance;
        const localFront = frontDepthField[planeIndex];
        const localBack = backDepthField[planeIndex];
        volume.setCell(x, y, z, Math.min(signedEdge, localFront - fieldZ, localBack + fieldZ));
      }
    }
  }
  volume.blur(0.08);
  volume.update();
  // MarchingCubes allocates its maximum vertex budget up front. DecalGeometry
  // iterates attribute.count rather than drawRange, so passing that oversized
  // buffer would project hundreds of thousands of unused zero vertices on a
  // phone. Compact to the actual polygonized surface before skinning/decaling.
  const activeVertexCount = volume.geometry.drawRange.count;
  const sourcePositions = volume.geometry.getAttribute("position") as THREE.BufferAttribute;
  const sourceNormals = volume.geometry.getAttribute("normal") as THREE.BufferAttribute;
  const compactGeometry = new THREE.BufferGeometry();
  compactGeometry.setAttribute("position", new THREE.Float32BufferAttribute(
    new Float32Array((sourcePositions.array as Float32Array).subarray(0, activeVertexCount * 3)), 3,
  ));
  compactGeometry.setAttribute("normal", new THREE.Float32BufferAttribute(
    new Float32Array((sourceNormals.array as Float32Array).subarray(0, activeVertexCount * 3)), 3,
  ));
  compactGeometry.setDrawRange(0, activeVertexCount);
  compactGeometry.computeBoundingBox();
  compactGeometry.computeBoundingSphere();
  volume.geometry.dispose();
  const structuralParts = rig.parts.filter((part) => GRAPH_APPENDAGE_KINDS.has(part.kind));
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
  const skinnedVolume = new THREE.SkinnedMesh(compactGeometry, volumeMaterial);
  skinnedVolume.name = "silhouette-distance-lens";
  skinnedVolume.add(rootBone);
  skinnedVolume.bind(new THREE.Skeleton([rootBone, ...branchBones.map(({ bone }) => bone)]));
  skinnedVolume.castShadow = true;
  skinnedVolume.receiveShadow = true;
  skinnedVolume.userData.reconstruction = {
    method: depth ? "learned distinct front/back depth fields" : "silhouette-preserving signed-distance lens fallback",
    polygonizer: "Marching Cubes",
    resolution: VOLUME_RESOLUTION,
    topology: "closed",
    contourPoints: contour.length,
    skeletonPoints: skeleton.length,
    semanticRig: rig.version,
    skinning: `${branchBones.length + 1} variable graph bones over one continuous surface`,
    learnedDepth: depth ? {
      model: depth.model,
      meanThickness: depth.meanThickness,
      meanAsymmetry: depth.meanAsymmetry,
      frontBackMirrored: false,
    } : null,
  };
  character.add(skinnedVolume);

  if (textureUrl) {
    skinnedVolume.updateMatrixWorld(true);
    const artworkTexture = new THREE.TextureLoader().load(textureUrl);
    artworkTexture.colorSpace = THREE.SRGBColorSpace;
    artworkTexture.anisotropy = 8;
    const decal = new THREE.Mesh(
      new DecalGeometry(
        skinnedVolume,
        new THREE.Vector3(0, 0, maximumFrontDepth * 0.82),
        new THREE.Euler(0, 0, 0),
        new THREE.Vector3(1.4, 1.4, Math.max(0.2, (maximumFrontDepth + maximumBackDepth) * 1.4)),
      ),
      new THREE.MeshStandardMaterial({
        map: artworkTexture,
        transparent: true,
        alphaTest: 0.04,
        roughness: 0.7,
        metalness: 0,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -4,
      }),
    );
    decal.name = "curved-artwork-skin";
    character.add(decal);
  }

  const addInkFeature = (part: SemanticPart) => {
    const group = new THREE.Group();
    group.name = `rig-${part.id}`;
    group.position.set(part.center.x, part.center.y, 0);
    const outline = part.outline?.length && part.outline.length >= 4
      ? part.outline
      : Array.from({ length: 32 }, (_, index) => {
        const angle = index / 32 * Math.PI * 2;
        const cos = Math.cos(part.rotation);
        const sin = Math.sin(part.rotation);
        const localX = Math.cos(angle) * part.size.x * 0.5;
        const localY = Math.sin(angle) * part.size.y * 0.5;
        return { x: part.center.x + localX * cos - localY * sin, y: part.center.y + localX * sin + localY * cos };
      });
    if (part.kind === "eye") {
      const lensColor = new THREE.Color(rig.bodyColor).lerp(new THREE.Color(0xfffdf4), 0.62);
      const localOutline = outline.map((point) => ({ x: point.x - part.center.x, y: point.y - part.center.y }));
      const signedArea = localOutline.reduce((total, point, index) => {
        const next = localOutline[(index + 1) % localOutline.length];
        return total + point.x * next.y - next.x * point.y;
      }, 0) * 0.5;
      const depth = Math.max(0.012, part.size.z * 1.35);
      let geometry: THREE.BufferGeometry;
      if (Math.abs(signedArea) >= Math.max(0.00005, part.size.x * part.size.y * 0.08)) {
        const ordered = signedArea > 0 ? localOutline : [...localOutline].reverse();
        const shape = new THREE.Shape();
        shape.moveTo(ordered[0].x, ordered[0].y);
        ordered.slice(1).forEach((point) => shape.lineTo(point.x, point.y));
        shape.closePath();
        geometry = new THREE.ExtrudeGeometry(shape, {
          depth,
          steps: 1,
          bevelEnabled: true,
          bevelSegments: 3,
          bevelSize: Math.min(part.size.x, part.size.y) * 0.08,
          bevelThickness: depth * 0.34,
        });
      } else {
        geometry = new THREE.SphereGeometry(0.5, 28, 18);
      }
      const lens = new THREE.Mesh(geometry, new THREE.MeshPhysicalMaterial({
        color: lensColor,
        roughness: 0.34,
        metalness: 0,
        clearcoat: 0.46,
        clearcoatRoughness: 0.36,
      }));
      lens.name = `${part.id}-raised-lens`;
      lens.position.z = frontDepthAt(part.center.x, part.center.y) + 0.004;
      if (geometry instanceof THREE.SphereGeometry) lens.scale.set(part.size.x * 0.94, part.size.y * 0.9, Math.max(0.024, part.size.z * 1.7));
      lens.castShadow = true;
      group.add(lens);
    } else if (part.kind === "pupil") {
      const pupil = new THREE.Mesh(new THREE.SphereGeometry(0.5, 20, 14), inkMaterial(part.color));
      pupil.name = `${part.id}-raised-pupil`;
      pupil.position.z = frontDepthAt(part.center.x, part.center.y) + Math.max(0.02, part.size.z * 0.8);
      pupil.scale.set(part.size.x, part.size.y, Math.max(0.018, part.size.z));
      pupil.castShadow = true;
      group.add(pupil);
    }
    const points = outline.map((point) => new THREE.Vector3(
      point.x - part.center.x,
      point.y - part.center.y,
      frontDepthAt(point.x, point.y) + 0.012,
    ));
    const curve = new THREE.CatmullRomCurve3(points, true, "centripetal", 0.3);
    const radius = Math.min(0.014, Math.max(0.0055, Math.min(part.size.x, part.size.y) * 0.075));
    const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(24, points.length * 2), radius, 8, true), inkMaterial(rig.lineColor));
    mesh.castShadow = true;
    group.add(mesh);
    character.add(group);
  };
  // Keep the authored silhouette readable even when the source paper and the
  // filled body are both pale. The front rim reinforces the exact captured
  // contour; the rear rim makes the separately learned hidden surface clear
  // during a 360° turn without copying facial features onto it.
  addInkFeature({
    id: "body-ink-outline",
    kind: "marking",
    side: "center",
    parentId: "body",
    center: { x: 0, y: 0, z: 0 },
    size: bodyPart?.size ?? { x: 1, y: 1, z: 0.3 },
    rotation: 0,
    color: rig.lineColor,
    confidence: 1,
    source: "image-region",
    outline: contour,
  });
  if (contour.length >= 6) {
    const rearPoints = contour.map((point) => new THREE.Vector3(point.x, point.y, -backDepthAt(point.x, point.y) - 0.009));
    const rearRim = new THREE.Mesh(
      new THREE.TubeGeometry(new THREE.CatmullRomCurve3(rearPoints, true, "centripetal", 0.3), Math.max(48, contour.length * 2), 0.0065, 8, true),
      inkMaterial(rig.lineColor),
    );
    rearRim.name = "inferred-back-silhouette-rim";
    rearRim.castShadow = true;
    character.add(rearRim);
  }
  // The texture preserves the child's exact artwork, while these shallow
  // semantic meshes give eyes, pupils, cheeks and mouths real parallax and
  // animation. They follow the detected outlines; they are not generic decals.
  rig.parts.filter((part) => part.kind === "eye" || part.kind === "pupil" || part.kind === "cheek" || part.kind === "nose" || part.kind === "mouth" || part.kind === "marking")
    .forEach(addInkFeature);

  character.userData.reconstruction = {
    method: depth ? "learned variable topology graph + learned asymmetric front/back depth volume" : "variable topology graph + symmetric emergency depth fallback",
    texturePlane: false,
    viewableDegrees: 360,
    bodyTopology: "closed",
    backPrior: depth ? "sketch-depth-v1 learned hidden surface; no copied face marks" : "symmetric emergency fallback; no invented face marks",
    depthModel: depth?.model ?? null,
    meanDepthAsymmetry: depth?.meanAsymmetry ?? 0,
    semanticParts: rig.parts.map((part) => ({ id: part.id, kind: part.kind, confidence: part.confidence, source: part.source })),
    joints: rig.joints,
    topologyKind: rig.topologyKind ?? null,
    topologyConfidence: rig.topologyConfidence ?? null,
    accent,
  };

  return character;
}

export const ARStage = forwardRef<ARStageHandle, ARStageProps>(function ARStage(
  { contour, skeleton, textureUrl, rig, depth, action, accent, inflation, neuralAssetUrl, visible, onCapability, onPlaced, onNeuralAssetInfo },
  ref,
) {
  const mountRef = useRef<HTMLDivElement>(null);
  const handlesRef = useRef<SceneHandles | null>(null);
  const [rendererError, setRendererError] = useState(false);
  const actionRef = useRef(action);
  const placementRef = useRef({ x: 0, y: -0.15, scale: 1 });
  const rotationRef = useRef({ yaw: 0, pitch: 0 });
  const xrHitSourceRef = useRef<XRHitTestSource | null>(null);
  const xrReferenceSpaceRef = useRef<XRReferenceSpace | null>(null);

  useEffect(() => { actionRef.current = action; }, [action]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;
    let disposed = false;

    const width = Math.max(1, mount.clientWidth);
    const height = Math.max(1, mount.clientHeight);
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
      setRendererError(false);
    } catch {
      setRendererError(true);
      onCapability(false);
      return;
    }
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
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.01, 40);
    camera.position.set(0, 0.05, 4.15);

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
    } else if (contour?.length && skeleton?.length && rig) {
      characterRoot.add(buildCharacter(contour, skeleton, rig, depth, textureUrl, accent, inflation));
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
    const render = (_time?: number, frame?: XRFrame) => {
      const elapsed = clock.getElapsedTime();
      const currentAction = actionRef.current;
      const placement = placementRef.current;
      const root = characterRoot;
      root.visible = visible;
      root.position.x = placement.x;
      root.position.y = placement.y + Math.sin(elapsed * 1.65) * 0.018;
      root.scale.setScalar(placement.scale);
      root.rotation.x = rotationRef.current.pitch + Math.sin(elapsed * 1.1) * 0.018;
      root.rotation.y = rotationRef.current.yaw - 0.07 + Math.sin(elapsed * 0.72) * 0.035;
      root.rotation.z = Math.sin(elapsed * 0.9) * 0.012;

      const articulated = root.getObjectByName("wallalive-semantic-character");
      const neural = root.getObjectByName("wallalive-neural-character");
      const neuralRig = neural?.userData.wallaliveRig as NeuralRigMap | undefined;
      const neuralSemantic = neural?.userData.wallaliveSemantic as NeuralSemanticMap | undefined;
      neuralRig?.all.forEach((bone) => {
        const base = bone.userData.wallaliveBaseQuaternion as THREE.Quaternion | undefined;
        if (base) bone.quaternion.copy(base);
      });
      rig?.parts.filter((part) => part.kind === "ear" || GRAPH_APPENDAGE_KINDS.has(part.kind)).forEach((part) => {
        const node = articulated?.getObjectByName(`rig-${part.id}`);
        if (node) {
          node.rotation.x = 0;
          node.rotation.z = Number(node.userData.baseRotationZ ?? part.rotation);
        }
      });
      const blink = Math.pow(Math.max(0, Math.sin(elapsed * 0.78)), 34);
      rig?.parts.filter((part) => part.kind === "eye").forEach((part) => {
        const node = articulated?.getObjectByName(`rig-${part.id}`);
        if (node) node.scale.y = Math.max(0.12, 1 - blink * 0.88);
      });
      [neuralSemantic?.eyeLeft, neuralSemantic?.eyeRight, neuralSemantic?.eyeCenter, neuralSemantic?.pupilLeft, neuralSemantic?.pupilRight, neuralSemantic?.pupilCenter]
        .forEach((node) => { if (node) node.scale.y = Math.max(0.12, 1 - blink * 0.88); });
      const mouth = articulated?.getObjectByName("rig-mouth");
      if (mouth) mouth.scale.y = currentAction === "idle" ? 1 : 1 + Math.abs(Math.sin(elapsed * 5)) * 0.42;
      if (neuralSemantic?.mouth) neuralSemantic.mouth.scale.y = currentAction === "idle" ? 1 : 1 + Math.abs(Math.sin(elapsed * 5)) * 0.42;

      if (currentAction === "wave") {
        const arm = articulated?.getObjectByName("rig-arm-right")
          ?? articulated?.getObjectByName("rig-arm-left")
          ?? rig?.parts.filter((part) => part.kind === "wing" || part.kind === "tentacle" || part.kind === "tail")
            .map((part) => articulated?.getObjectByName(`rig-${part.id}`)).find(Boolean);
        if (arm) arm.rotation.z = Number(arm.userData.baseRotationZ ?? 0) + 0.92 + Math.sin(elapsed * 7.2) * 0.42;
        const neuralArm = neuralRig?.armRight ?? neuralRig?.armLeft;
        neuralArm?.rotateZ(0.72 + Math.sin(elapsed * 7.2) * 0.42);
        neuralArm?.rotateX(Math.sin(elapsed * 4.1) * 0.22);
        root.rotation.y = rotationRef.current.yaw - 0.18 + Math.sin(elapsed * 4.8) * 0.08;
        root.rotation.z = -0.035 + Math.sin(elapsed * 5.6) * 0.035;
        root.position.y += Math.sin(elapsed * 5.6) * 0.025;
      }
      if (currentAction === "dance") {
        const leftArm = articulated?.getObjectByName("rig-arm-left");
        const rightArm = articulated?.getObjectByName("rig-arm-right");
        if (leftArm) leftArm.rotation.z = Number(leftArm.userData.baseRotationZ ?? 0) + Math.sin(elapsed * 5.2) * 0.55;
        if (rightArm) rightArm.rotation.z = Number(rightArm.userData.baseRotationZ ?? 0) - Math.sin(elapsed * 5.2) * 0.55;
        rig?.parts.filter((part) => GRAPH_APPENDAGE_KINDS.has(part.kind) && part.kind !== "arm" && part.kind !== "trunk")
          .forEach((part, index) => {
            const node = articulated?.getObjectByName(`rig-${part.id}`);
            if (node) node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + (index % 2 ? -1 : 1) * Math.sin(elapsed * 5.2 + index * 0.45) * 0.34;
          });
        neuralRig?.arms.forEach((arm, index) => {
          const direction = index % 2 ? -1 : 1;
          arm.rotateZ(direction * (0.52 + Math.sin(elapsed * 5.2 + index * 0.6) * 0.45));
        });
        neuralRig?.legs.forEach((leg, index) => leg.rotateX((index % 2 ? -1 : 1) * Math.sin(elapsed * 5.2 + index * 0.4) * 0.22));
        root.rotation.z = Math.sin(elapsed * 5.2) * 0.18;
        root.rotation.y = rotationRef.current.yaw + Math.sin(elapsed * 2.6) * 0.18;
        root.position.x = placement.x + Math.sin(elapsed * 3.4) * 0.15;
      }
      if (currentAction === "hop") {
        root.position.y = placement.y + Math.abs(Math.sin(elapsed * 4.6)) * 0.52;
        root.rotation.x = Math.sin(elapsed * 4.6) * 0.08;
      }
      if (currentAction === "walk") {
        rig?.parts.filter((part) => part.kind === "leg").forEach((part, index) => {
          const leg = articulated?.getObjectByName(`rig-${part.id}`);
          if (leg) leg.rotation.x = (index % 2 ? -1 : 1) * Math.sin(elapsed * 7 + index * 0.25) * 0.5;
        });
        neuralRig?.legs.forEach((leg, index) => leg.rotateX((index % 2 ? -1 : 1) * Math.sin(elapsed * 7 + index * 0.25) * 0.48));
        neuralRig?.arms.forEach((arm, index) => arm.rotateX((index % 2 ? 1 : -1) * Math.sin(elapsed * 7 + index * 0.25) * 0.24));
        root.position.x = placement.x + Math.sin(elapsed * 1.5) * 0.85;
        root.rotation.z = Math.sin(elapsed * 6) * 0.055;
        root.rotation.y = rotationRef.current.yaw + Math.sin(elapsed * 3) * 0.12;
      }
      if (currentAction === "hide") {
        root.position.x = placement.x + 1.08;
        root.rotation.y = -0.35;
        root.rotation.z = -0.16;
      }
      if (currentAction === "spin") root.rotation.y = rotationRef.current.yaw + elapsed * 2.15;

      shadow.position.x = root.position.x;
      shadow.position.y = placement.y - 0.92;
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

    const dispose = () => {
      disposed = true;
      observer.disconnect();
      renderer.setAnimationLoop(null);
      disposeObject(scene);
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };

    handlesRef.current = { renderer, scene, camera, character: characterRoot, reticle, dispose };
    if (navigator.xr) navigator.xr.isSessionSupported("immersive-ar").then(onCapability).catch(() => onCapability(false));
    else onCapability(false);

    return () => {
      handlesRef.current = null;
      dispose();
    };
  }, [accent, contour, depth, inflation, neuralAssetUrl, onCapability, onNeuralAssetInfo, rig, skeleton, textureUrl, visible]);

  useImperativeHandle(ref, () => ({
    placeNormalized(x: number, y: number, scale = placementRef.current.scale) {
      placementRef.current = { x: (x - 0.5) * 2.3, y: (0.5 - y) * 1.65 - 0.1, scale: Math.min(1.55, Math.max(0.55, scale)) };
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
          placementRef.current = { x: position.x, y: position.y, scale: placementRef.current.scale };
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
