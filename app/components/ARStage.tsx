"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { MarchingCubes } from "three/addons/objects/MarchingCubes.js";
import type { CharacterRig, SemanticPart, SkeletonPoint } from "../lib/drawing";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin";

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
  rotateBy: (yaw: number, pitch: number) => void;
};

type ARStageProps = {
  skeleton: SkeletonPoint[] | null;
  rig: CharacterRig | null;
  action: CharacterAction;
  accent: string;
  inflation: number;
  visible: boolean;
  onCapability: (supported: boolean) => void;
  onPlaced: (surface: "screen" | "world", x: number, y: number) => void;
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

function meshMaterial(color: string, roughness = 0.62) {
  return new THREE.MeshPhysicalMaterial({
    color,
    roughness,
    metalness: 0,
    clearcoat: 0.22,
    clearcoatRoughness: 0.68,
  });
}

function buildCharacter(skeleton: SkeletonPoint[], rig: CharacterRig, accent: string, inflation: number) {
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
  volume.name = "medial-skeleton-sphere-union";
  volume.isolation = 0;
  volume.field.fill(-0.12);
  const half = VOLUME_RESOLUTION / 2;
  const cell = 1 / half;
  const bodyPart = rig.parts.find((part) => part.kind === "body");
  const appendages = rig.parts.filter((part) => part.kind === "ear" || part.kind === "arm" || part.kind === "leg");
  const bodySkeleton = skeleton.filter((point) => !appendages.some((part) => {
    const reach = Math.max(part.size.x, part.size.y) * (part.kind === "ear" ? 0.58 : 0.42);
    const awayFromCore = bodyPart ? Math.hypot(point.x - bodyPart.center.x, point.y - bodyPart.center.y) > Math.min(bodyPart.size.x, bodyPart.size.y) * 0.16 : true;
    return awayFromCore && Math.hypot(point.x - part.center.x, point.y - part.center.y) < reach;
  }));
  const volumeSkeleton = bodySkeleton.length >= 3 ? bodySkeleton : skeleton;
  const inflatedSkeleton = volumeSkeleton.map((point) => ({ ...point, radius: Math.min(0.5, Math.max(0.032, point.radius * inflation)) }));
  for (const point of inflatedSkeleton) {
    const reach = point.radius + cell * 1.5;
    const minX = Math.max(1, Math.floor((point.x - reach) * half + half));
    const maxX = Math.min(VOLUME_RESOLUTION - 2, Math.ceil((point.x + reach) * half + half));
    const minY = Math.max(1, Math.floor((point.y - reach) * half + half));
    const maxY = Math.min(VOLUME_RESOLUTION - 2, Math.ceil((point.y + reach) * half + half));
    const minZ = Math.max(1, Math.floor(-reach * half + half));
    const maxZ = Math.min(VOLUME_RESOLUTION - 2, Math.ceil(reach * half + half));
    for (let z = minZ; z <= maxZ; z += 1) {
      const fieldZ = (z - half) / half;
      for (let y = minY; y <= maxY; y += 1) {
        const fieldY = (y - half) / half;
        for (let x = minX; x <= maxX; x += 1) {
          const fieldX = (x - half) / half;
          const value = point.radius - Math.hypot(fieldX - point.x, fieldY - point.y, fieldZ);
          if (value > volume.getCell(x, y, z)) volume.setCell(x, y, z, value);
        }
      }
    }
  }
  volume.blur(0.12);
  volume.update();
  volume.castShadow = true;
  volume.receiveShadow = true;
  volume.userData.reconstruction = {
    method: "semantic medial-skeleton sphere union",
    polygonizer: "Marching Cubes",
    resolution: VOLUME_RESOLUTION,
    topology: "closed",
    skeletonPoints: volumeSkeleton.length,
    semanticRig: rig.version,
  };
  character.add(volume);

  const frontDepthAt = (x: number, y: number) => {
    let depth = 0;
    for (const point of inflatedSkeleton) {
      const distanceSquared = (x - point.x) ** 2 + (y - point.y) ** 2;
      if (distanceSquared < point.radius ** 2) depth = Math.max(depth, Math.sqrt(point.radius ** 2 - distanceSquared));
    }
    return depth;
  };
  const addEllipsoid = (part: SemanticPart, materialColor = part.color) => {
    const group = new THREE.Group();
    group.name = `rig-${part.id}`;
    group.position.set(part.center.x, part.center.y, part.center.z);
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.5, 28, 18), meshMaterial(materialColor));
    mesh.scale.set(part.size.x, part.size.y, part.size.z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
    character.add(group);
    return group;
  };

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

  rig.parts.filter((part) => part.kind === "eye").forEach((part) => {
    const group = addEllipsoid({ ...part, center: { ...part.center, z: frontDepthAt(part.center.x, part.center.y) + part.size.z * 0.24 } }, "#fffaf0");
    const pupil = rig.parts.find((candidate) => candidate.parentId === part.id && candidate.kind === "pupil");
    if (pupil) {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.5, 22, 14), meshMaterial(pupil.color, 0.48));
      mesh.name = `rig-${pupil.id}`;
      mesh.position.z = part.size.z * 0.56;
      mesh.scale.set(pupil.size.x, pupil.size.y, pupil.size.z);
      group.add(mesh);
    }
  });

  rig.parts.filter((part) => part.kind === "mouth").forEach((part) => {
    const curve = new THREE.QuadraticBezierCurve3(
      new THREE.Vector3(-part.size.x * 0.5, part.size.y * 0.28, 0),
      new THREE.Vector3(0, -part.size.y * 0.62, part.size.z * 0.2),
      new THREE.Vector3(part.size.x * 0.5, part.size.y * 0.28, 0),
    );
    const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 24, Math.max(0.008, part.size.y * 0.16), 8, false), meshMaterial(part.color, 0.52));
    mesh.name = `rig-${part.id}`;
    mesh.position.set(part.center.x, part.center.y, frontDepthAt(part.center.x, part.center.y) + 0.02);
    mesh.castShadow = true;
    character.add(mesh);
  });

  rig.parts.filter((part) => part.kind === "marking").forEach((part) => {
    const raised = { ...part, center: { ...part.center, z: frontDepthAt(part.center.x, part.center.y) + part.size.z * 0.6 } };
    addEllipsoid(raised);
  });

  character.userData.reconstruction = {
    method: "hierarchical semantic regions plus articulated volumetric parts",
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
  { skeleton, rig, action, accent, inflation, visible, onCapability, onPlaced },
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
    if (skeleton?.length && rig) characterRoot.add(buildCharacter(skeleton, rig, accent, inflation));

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
      root.rotation.y = rotationRef.current.yaw - 0.24 + Math.sin(elapsed * 0.72) * 0.11;
      root.rotation.z = Math.sin(elapsed * 0.9) * 0.012;

      const articulated = root.getObjectByName("wallalive-semantic-character");
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
        root.rotation.y = rotationRef.current.yaw - 0.18 + Math.sin(elapsed * 4.8) * 0.08;
        root.rotation.z = -0.035 + Math.sin(elapsed * 5.6) * 0.035;
        root.position.y += Math.sin(elapsed * 5.6) * 0.025;
      }
      if (currentAction === "dance") {
        const leftArm = articulated?.getObjectByName("rig-arm-left");
        const rightArm = articulated?.getObjectByName("rig-arm-right");
        if (leftArm) leftArm.rotation.z = Number(leftArm.userData.baseRotationZ ?? 0) + Math.sin(elapsed * 5.2) * 0.55;
        if (rightArm) rightArm.rotation.z = Number(rightArm.userData.baseRotationZ ?? 0) - Math.sin(elapsed * 5.2) * 0.55;
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
      observer.disconnect();
      renderer.setAnimationLoop(null);
      scene.traverse((object) => {
        if (!(object instanceof THREE.Mesh)) return;
        object.geometry.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => {
          if ("map" in material && material.map instanceof THREE.Texture) material.map.dispose();
          if ("alphaMap" in material && material.alphaMap instanceof THREE.Texture && material.alphaMap !== material.map) material.alphaMap.dispose();
          if ("displacementMap" in material && material.displacementMap instanceof THREE.Texture) material.displacementMap.dispose();
          material.dispose();
        });
      });
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
  }, [accent, inflation, onCapability, rig, skeleton, visible]);

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
