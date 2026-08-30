"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";
import { MarchingCubes } from "three/addons/objects/MarchingCubes.js";
import type { SkeletonPoint } from "../lib/drawing";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin";

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
};

type ARStageProps = {
  textureUrl: string | null;
  skeleton: SkeletonPoint[] | null;
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

function buildCharacter(textureUrl: string, skeleton: SkeletonPoint[], accent: string, inflation: number) {
  const character = new THREE.Group();
  character.name = "wallalive-skeleton-volume";

  const volumeMaterial = new THREE.MeshPhysicalMaterial({
    color: accent,
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
  const inflatedSkeleton = skeleton.map((point) => ({ ...point, radius: Math.min(0.5, Math.max(0.032, point.radius * inflation)) }));
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
    method: "medial-skeleton sphere union",
    polygonizer: "Marching Cubes",
    resolution: VOLUME_RESOLUTION,
    topology: "closed",
    skeletonPoints: skeleton.length,
  };
  character.add(volume);

  const textureLoader = new THREE.TextureLoader();
  textureLoader.loadAsync(textureUrl).then((texture) => {
    if (!character.parent) {
      texture.dispose();
      return;
    }
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearMipmapLinearFilter;
    texture.magFilter = THREE.LinearFilter;
    const frontGeometry = new THREE.PlaneGeometry(1.4, 1.4, 72, 72);
    const positions = frontGeometry.attributes.position;
    for (let index = 0; index < positions.count; index += 1) {
      const x = positions.getX(index);
      const y = positions.getY(index);
      let frontDepth = 0;
      for (const point of inflatedSkeleton) {
        const distanceSquared = (x - point.x) ** 2 + (y - point.y) ** 2;
        if (distanceSquared < point.radius ** 2) frontDepth = Math.max(frontDepth, Math.sqrt(point.radius ** 2 - distanceSquared));
      }
      positions.setZ(index, 0.008 + frontDepth);
    }
    positions.needsUpdate = true;
    frontGeometry.computeVertexNormals();
    const frontMaterial = new THREE.MeshPhysicalMaterial({
      map: texture,
      transparent: true,
      alphaTest: 0.04,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
      roughness: 0.64,
      metalness: 0,
      clearcoat: 0.11,
      clearcoatRoughness: 0.78,
      side: THREE.FrontSide,
    });
    const front = new THREE.Mesh(frontGeometry, frontMaterial);
    front.name = "drawing-curved-over-inflated-front";
    front.renderOrder = 2;
    character.add(front);
  }).catch(() => undefined);

  return character;
}

export const ARStage = forwardRef<ARStageHandle, ARStageProps>(function ARStage(
  { textureUrl, skeleton, action, accent, inflation, visible, onCapability, onPlaced },
  ref,
) {
  const mountRef = useRef<HTMLDivElement>(null);
  const handlesRef = useRef<SceneHandles | null>(null);
  const actionRef = useRef(action);
  const placementRef = useRef({ x: 0, y: -0.15, scale: 1 });
  const xrHitSourceRef = useRef<XRHitTestSource | null>(null);
  const xrReferenceSpaceRef = useRef<XRReferenceSpace | null>(null);

  useEffect(() => { actionRef.current = action; }, [action]);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = Math.max(1, mount.clientWidth);
    const height = Math.max(1, mount.clientHeight);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.xr.enabled = true;
    renderer.domElement.setAttribute("aria-label", "Mathematically reconstructed 3D drawing");
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
    if (textureUrl && skeleton?.length) characterRoot.add(buildCharacter(textureUrl, skeleton, accent, inflation));

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
      root.rotation.x = Math.sin(elapsed * 1.1) * 0.018;
      root.rotation.y = -0.24 + Math.sin(elapsed * 0.72) * 0.11;
      root.rotation.z = Math.sin(elapsed * 0.9) * 0.012;

      if (currentAction === "wave") {
        root.rotation.y = -0.18 + Math.sin(elapsed * 4.8) * 0.34;
        root.rotation.z = -0.055 + Math.sin(elapsed * 5.6) * 0.075;
        root.position.y += Math.sin(elapsed * 5.6) * 0.025;
      }
      if (currentAction === "dance") {
        root.rotation.z = Math.sin(elapsed * 5.2) * 0.18;
        root.rotation.y = Math.sin(elapsed * 2.6) * 0.18;
        root.position.x = placement.x + Math.sin(elapsed * 3.4) * 0.15;
      }
      if (currentAction === "hop") {
        root.position.y = placement.y + Math.abs(Math.sin(elapsed * 4.6)) * 0.52;
        root.rotation.x = Math.sin(elapsed * 4.6) * 0.08;
      }
      if (currentAction === "walk") {
        root.position.x = placement.x + Math.sin(elapsed * 1.5) * 0.85;
        root.rotation.z = Math.sin(elapsed * 6) * 0.055;
        root.rotation.y = Math.sin(elapsed * 3) * 0.12;
      }
      if (currentAction === "hide") {
        root.position.x = placement.x + 1.08;
        root.rotation.y = -0.35;
        root.rotation.z = -0.16;
      }
      if (currentAction === "spin") root.rotation.y = elapsed * 2.15;

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
  }, [accent, inflation, onCapability, skeleton, textureUrl, visible]);

  useImperativeHandle(ref, () => ({
    placeNormalized(x: number, y: number, scale = placementRef.current.scale) {
      placementRef.current = { x: (x - 0.5) * 2.3, y: (0.5 - y) * 1.65 - 0.1, scale: Math.min(1.55, Math.max(0.55, scale)) };
      onPlaced("screen", Number(x.toFixed(2)), Number(y.toFixed(2)));
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

  return <div className="three-layer" ref={mountRef} aria-hidden={!visible} />;
});
