"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { DecalGeometry } from "three/addons/geometries/DecalGeometry.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MarchingCubes } from "three/addons/objects/MarchingCubes.js";
import type { CharacterRig, ContourPoint, SemanticPart, SkeletonPoint } from "../lib/drawing";
import { disposeObject, prepareNeuralCharacter, type NeuralRigMap, type RiggedAssetInfo } from "../lib/rigged-model";

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

function distanceToContour(x: number, y: number, contour: ContourPoint[]) {
  let distance = Infinity;
  for (let index = 0; index < contour.length; index += 1) {
    distance = Math.min(distance, distanceToSegment(x, y, contour[index], contour[(index + 1) % contour.length]));
  }
  return distance;
}

function meshMaterial(color: string, roughness = 0.62) {
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness,
    metalness: 0,
    clearcoat: 0.22,
    clearcoatRoughness: 0.68,
  });
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

export function buildCharacter(contour: ContourPoint[], skeleton: SkeletonPoint[], rig: CharacterRig, textureUrl: string | null, accent: string, inflation: number) {
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
  const depthAt = (x: number, y: number) => {
    if (!pointInsideContour(x, y, contour)) return 0;
    const normalizedDistance = Math.min(1, distanceToContour(x, y, contour) / maximumInteriorDistance);
    return Math.max(cell * 0.7, bodyHalfDepth * Math.sqrt(normalizedDistance));
  };
  for (let z = 1; z < VOLUME_RESOLUTION - 1; z += 1) {
    const fieldZ = (z - half) / half;
    for (let y = 1; y < VOLUME_RESOLUTION - 1; y += 1) {
      for (let x = 1; x < VOLUME_RESOLUTION - 1; x += 1) {
        const planeIndex = y * VOLUME_RESOLUTION + x;
        const edgeDistance = edgeDistanceField[planeIndex];
        const signedEdge = insideField[planeIndex] ? edgeDistance : -edgeDistance;
        const localDepth = insideField[planeIndex]
          ? Math.max(cell * 0.7, bodyHalfDepth * Math.sqrt(Math.min(1, edgeDistance / maximumInteriorDistance)))
          : 0;
        volume.setCell(x, y, z, Math.min(signedEdge, localDepth - Math.abs(fieldZ)));
      }
    }
  }
  volume.blur(0.08);
  volume.update();
  volume.castShadow = true;
  volume.receiveShadow = true;
  volume.userData.reconstruction = {
    method: "silhouette-preserving signed-distance lens",
    polygonizer: "Marching Cubes",
    resolution: VOLUME_RESOLUTION,
    topology: "closed",
    contourPoints: contour.length,
    skeletonPoints: skeleton.length,
    semanticRig: rig.version,
  };
  character.add(volume);

  if (textureUrl) {
    volume.updateMatrixWorld(true);
    const artworkTexture = new THREE.TextureLoader().load(textureUrl);
    artworkTexture.colorSpace = THREE.SRGBColorSpace;
    artworkTexture.anisotropy = 8;
    const decal = new THREE.Mesh(
      new DecalGeometry(
        volume,
        new THREE.Vector3(0, 0, bodyHalfDepth * 0.82),
        new THREE.Euler(0, 0, 0),
        new THREE.Vector3(1.4, 1.4, bodyHalfDepth * 2.6),
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

  const frontDepthAt = depthAt;

  rig.parts.filter((part) => part.kind === "ear").forEach((part) => {
    const anchor = part.anchor ?? part.center;
    const pivot = new THREE.Group();
    pivot.name = `rig-${part.id}`;
    pivot.position.set(anchor.x, anchor.y, 0);
    pivot.rotation.z = part.rotation;
    pivot.userData.baseRotationZ = part.rotation;
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.5, 28, 18), meshMaterial(part.color));
    const distance = Math.hypot(part.center.x - anchor.x, part.center.y - anchor.y);
    mesh.position.y = distance;
    mesh.scale.set(part.size.x, part.size.y, part.size.z);
    mesh.castShadow = true;
    pivot.add(mesh);
    character.add(pivot);
  });

  rig.parts.filter((part) => part.kind === "arm" || part.kind === "leg").forEach((part) => {
    const anchor = part.anchor ?? bodyPart?.center ?? { x: 0, y: 0, z: 0 };
    const pivot = new THREE.Group();
    pivot.name = `rig-${part.id}`;
    pivot.position.set(anchor.x, anchor.y, 0);
    pivot.rotation.z = part.rotation;
    pivot.userData.baseRotationZ = part.rotation;
    const radius = Math.max(0.018, part.size.x * 0.5);
    const length = Math.max(radius * 2.1, part.size.y);
    const limb = new THREE.Mesh(new THREE.CapsuleGeometry(radius, Math.max(0.001, length - radius * 2), 8, 18), meshMaterial(part.color));
    limb.position.y = length * 0.5;
    limb.castShadow = true;
    limb.receiveShadow = true;
    pivot.add(limb);
    const endPart = rig.parts.find((candidate) => candidate.parentId === part.id && (candidate.kind === "hand" || candidate.kind === "foot"));
    if (endPart) {
      const end = new THREE.Mesh(new THREE.SphereGeometry(0.5, 24, 16), meshMaterial(endPart.color));
      end.name = `rig-${endPart.id}`;
      end.position.y = length;
      end.scale.set(endPart.size.x, endPart.size.y, endPart.size.z);
      end.castShadow = true;
      pivot.add(end);
    }
    character.add(pivot);
  });

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
  if (!textureUrl) {
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
    rig.parts.filter((part) => part.kind === "eye" || part.kind === "cheek" || part.kind === "mouth" || part.kind === "marking")
      .forEach(addInkFeature);
  }

  character.userData.reconstruction = {
    method: "color-isolated semantic ink over a silhouette-preserving distance-field body",
    texturePlane: false,
    viewableDegrees: 360,
    bodyTopology: "closed",
    semanticParts: rig.parts.map((part) => ({ id: part.id, kind: part.kind, confidence: part.confidence, source: part.source })),
    joints: rig.joints,
    accent,
  };

  return character;
}

export const ARStage = forwardRef<ARStageHandle, ARStageProps>(function ARStage(
  { contour, skeleton, textureUrl, rig, action, accent, inflation, neuralAssetUrl, visible, onCapability, onPlaced, onNeuralAssetInfo },
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
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.xr.enabled = true;
    renderer.domElement.setAttribute("aria-label", "Semantic articulated 3D drawing");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, width / height, 0.01, 40);
    camera.position.set(0, 0.05, 4.15);

    const ambient = new THREE.HemisphereLight(0xfff4dc, 0x253d42, 1.65);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xfff7e8, 3.7);
    key.position.set(-2.6, 4.2, 5.5);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 14;
    key.shadow.bias = -0.0005;
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x9edce5, 1.2);
    fill.position.set(3.5, 1.2, 2.3);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0xffc7a8, 0.85);
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
          const prepared = prepareNeuralCharacter(gltf.scene);
          characterRoot.add(prepared.character);
          onNeuralAssetInfo(prepared.info);
        },
        undefined,
        () => { if (!disposed) onNeuralAssetInfo(null); },
      );
    } else if (contour?.length && skeleton?.length && rig) {
      characterRoot.add(buildCharacter(contour, skeleton, rig, textureUrl, accent, inflation));
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
      neuralRig?.all.forEach((bone) => {
        const base = bone.userData.wallaliveBaseQuaternion as THREE.Quaternion | undefined;
        if (base) bone.quaternion.copy(base);
      });
      rig?.parts.filter((part) => part.kind === "ear" || part.kind === "arm" || part.kind === "leg").forEach((part) => {
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
      const mouth = articulated?.getObjectByName("rig-mouth");
      if (mouth) mouth.scale.y = currentAction === "idle" ? 1 : 1 + Math.abs(Math.sin(elapsed * 5)) * 0.42;

      if (currentAction === "wave") {
        const arm = articulated?.getObjectByName("rig-arm-right") ?? articulated?.getObjectByName("rig-arm-left");
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
        neuralRig?.armLeft?.rotateZ(0.52 + Math.sin(elapsed * 5.2) * 0.45);
        neuralRig?.armRight?.rotateZ(-0.52 - Math.sin(elapsed * 5.2) * 0.45);
        neuralRig?.legLeft?.rotateX(Math.sin(elapsed * 5.2) * 0.22);
        neuralRig?.legRight?.rotateX(-Math.sin(elapsed * 5.2) * 0.22);
        root.rotation.z = Math.sin(elapsed * 5.2) * 0.18;
        root.rotation.y = rotationRef.current.yaw + Math.sin(elapsed * 2.6) * 0.18;
        root.position.x = placement.x + Math.sin(elapsed * 3.4) * 0.15;
      }
      if (currentAction === "hop") {
        root.position.y = placement.y + Math.abs(Math.sin(elapsed * 4.6)) * 0.52;
        root.rotation.x = Math.sin(elapsed * 4.6) * 0.08;
      }
      if (currentAction === "walk") {
        const leftLeg = articulated?.getObjectByName("rig-leg-left");
        const rightLeg = articulated?.getObjectByName("rig-leg-right");
        if (leftLeg) leftLeg.rotation.x = Math.sin(elapsed * 7) * 0.5;
        if (rightLeg) rightLeg.rotation.x = -Math.sin(elapsed * 7) * 0.5;
        neuralRig?.legLeft?.rotateX(Math.sin(elapsed * 7) * 0.48);
        neuralRig?.legRight?.rotateX(-Math.sin(elapsed * 7) * 0.48);
        neuralRig?.armLeft?.rotateX(-Math.sin(elapsed * 7) * 0.24);
        neuralRig?.armRight?.rotateX(Math.sin(elapsed * 7) * 0.24);
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
  }, [accent, contour, inflation, neuralAssetUrl, onCapability, onNeuralAssetInfo, rig, skeleton, textureUrl, visible]);

  useImperativeHandle(ref, () => ({
    placeNormalized(x: number, y: number, scale = placementRef.current.scale) {
      placementRef.current = { x: (x - 0.5) * 2.3, y: (0.5 - y) * 1.65 - 0.1, scale: Math.min(1.55, Math.max(0.55, scale)) };
      onPlaced("screen", Number(x.toFixed(2)), Number(y.toFixed(2)));
    },
    rotateBy(yaw: number, pitch: number) {
      rotationRef.current = {
        yaw: rotationRef.current.yaw + yaw,
        pitch: Math.min(0.62, Math.max(-0.62, rotationRef.current.pitch + pitch)),
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
