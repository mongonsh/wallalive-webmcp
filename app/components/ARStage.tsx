"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef } from "react";
import * as THREE from "three";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin";
export type BodyShape = "round" | "tall" | "wide" | "spiky";

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
};

type ARStageProps = {
  textureUrl: string | null;
  action: CharacterAction;
  accent: string;
  bodyShape: BodyShape;
  visible: boolean;
  onCapability: (supported: boolean) => void;
  onPlaced: (surface: "screen" | "world", x: number, y: number) => void;
};

type SceneHandles = {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  character: THREE.Group;
  rightArm: THREE.Group;
  leftArm: THREE.Group;
  leftLeg: THREE.Group;
  rightLeg: THREE.Group;
  reticle: THREE.Mesh;
  dispose: () => void;
};

function limb(material: THREE.Material, length: number, width: number) {
  const group = new THREE.Group();
  const geometry = new THREE.CapsuleGeometry(width, length, 5, 10);
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.y = -length / 2;
  group.add(mesh);
  return group;
}

function buildCharacter(textureUrl: string, accent: string, bodyShape: BodyShape, onReady: (group: THREE.Group) => void) {
  const character = new THREE.Group();
  character.name = "wallalive-character";
  const textureLoader = new THREE.TextureLoader();
  textureLoader.load(textureUrl, (texture) => {
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.minFilter = THREE.LinearFilter;
    const frontMaterial = new THREE.MeshStandardMaterial({ map: texture, transparent: true, roughness: 0.72, metalness: 0.02, side: THREE.DoubleSide });
    const edgeMaterial = new THREE.MeshStandardMaterial({ map: texture, color: accent, transparent: true, opacity: 0.95, roughness: 0.86, side: THREE.DoubleSide });
    const planeGeometry = new THREE.PlaneGeometry(1.4, 1.4, 1, 1);
    for (let index = 0; index < 7; index += 1) {
      const layer = new THREE.Mesh(planeGeometry, index === 6 ? frontMaterial : edgeMaterial);
      layer.position.z = (index - 3) * 0.026;
      character.add(layer);
    }

    const eyeWhite = new THREE.MeshStandardMaterial({ color: 0xfffbef, roughness: 0.35 });
    const ink = new THREE.MeshStandardMaterial({ color: 0x18312e, roughness: 0.72 });
    const eyeGeometry = new THREE.SphereGeometry(0.12, 24, 16);
    const pupilGeometry = new THREE.SphereGeometry(0.048, 18, 12);
    [-0.22, 0.22].forEach((x) => {
      const eye = new THREE.Mesh(eyeGeometry, eyeWhite);
      eye.scale.y = 1.25;
      eye.position.set(x, 0.16, 0.18);
      const pupil = new THREE.Mesh(pupilGeometry, ink);
      pupil.position.set(x + 0.025, 0.145, 0.285);
      character.add(eye, pupil);
    });

    const smile = new THREE.Mesh(new THREE.TorusGeometry(0.17, 0.026, 8, 22, Math.PI), ink);
    smile.rotation.z = Math.PI;
    smile.position.set(0, -0.2, 0.19);
    character.add(smile);

    const rightArm = limb(ink, 0.45, 0.036);
    rightArm.name = "right-arm";
    rightArm.position.set(0.7, 0.05, 0.02);
    rightArm.rotation.z = -0.72;
    const leftArm = limb(ink, 0.45, 0.036);
    leftArm.name = "left-arm";
    leftArm.position.set(-0.7, 0.04, 0.02);
    leftArm.rotation.z = 0.72;
    const leftLeg = limb(ink, 0.38, 0.045);
    leftLeg.name = "left-leg";
    leftLeg.position.set(-0.28, -0.62, 0);
    leftLeg.rotation.z = 0.12;
    const rightLeg = limb(ink, 0.38, 0.045);
    rightLeg.name = "right-leg";
    rightLeg.position.set(0.28, -0.62, 0);
    rightLeg.rotation.z = -0.12;
    character.add(rightArm, leftArm, leftLeg, rightLeg);

    const scale = bodyShape === "tall" ? [0.82, 1.18] : bodyShape === "wide" ? [1.18, 0.84] : bodyShape === "spiky" ? [0.92, 1.08] : [1, 1];
    character.scale.set(scale[0], scale[1], 1);
    onReady(character);
  });
  return character;
}

export const ARStage = forwardRef<ARStageHandle, ARStageProps>(function ARStage(
  { textureUrl, action, accent, bodyShape, visible, onCapability, onPlaced },
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
    renderer.shadowMap.enabled = true;
    renderer.xr.enabled = true;
    renderer.domElement.setAttribute("aria-label", "Animated 3D character overlay");
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.01, 40);
    camera.position.set(0, 0.05, 4.15);
    scene.add(new THREE.HemisphereLight(0xfff4d8, 0x385558, 2.4));
    const key = new THREE.DirectionalLight(0xffffff, 3.1);
    key.position.set(-2.5, 4, 5);
    key.castShadow = true;
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x5fc7df, 1.7);
    rim.position.set(4, 1, 2);
    scene.add(rim);

    const characterRoot = new THREE.Group();
    scene.add(characterRoot);
    let character = new THREE.Group();
    let rightArm = new THREE.Group();
    let leftArm = new THREE.Group();
    let leftLeg = new THREE.Group();
    let rightLeg = new THREE.Group();

    if (textureUrl) {
      character = buildCharacter(textureUrl, accent, bodyShape, (ready) => {
        rightArm = ready.getObjectByName("right-arm") as THREE.Group;
        leftArm = ready.getObjectByName("left-arm") as THREE.Group;
        leftLeg = ready.getObjectByName("left-leg") as THREE.Group;
        rightLeg = ready.getObjectByName("right-leg") as THREE.Group;
      });
      characterRoot.add(character);
    }

    const shadowMaterial = new THREE.MeshBasicMaterial({ color: 0x18312e, transparent: true, opacity: 0.18, depthWrite: false });
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(0.62, 40), shadowMaterial);
    shadow.scale.y = 0.2;
    shadow.position.set(0, -1.08, -0.2);
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
      root.position.y = placement.y + Math.sin(elapsed * 2.1) * 0.025;
      root.scale.setScalar(placement.scale);
      root.rotation.z *= 0.82;
      root.rotation.y *= 0.82;

      if (currentAction === "wave" && rightArm) rightArm.rotation.z = -0.8 + Math.sin(elapsed * 7) * 0.55;
      if (currentAction === "dance") {
        root.rotation.z = Math.sin(elapsed * 5.2) * 0.18;
        root.position.x = placement.x + Math.sin(elapsed * 3.4) * 0.15;
        if (leftArm) leftArm.rotation.z = 0.5 + Math.sin(elapsed * 6) * 0.3;
        if (rightArm) rightArm.rotation.z = -0.5 - Math.sin(elapsed * 6) * 0.3;
      }
      if (currentAction === "hop") root.position.y = placement.y + Math.abs(Math.sin(elapsed * 4.6)) * 0.52;
      if (currentAction === "walk") {
        root.position.x = placement.x + Math.sin(elapsed * 1.5) * 0.85;
        if (leftLeg) leftLeg.rotation.z = Math.sin(elapsed * 8) * 0.36;
        if (rightLeg) rightLeg.rotation.z = -Math.sin(elapsed * 8) * 0.36;
      }
      if (currentAction === "hide") {
        root.position.x = placement.x + 1.08;
        root.rotation.z = -0.22;
      }
      if (currentAction === "spin") root.rotation.y = elapsed * 4.2;

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
      renderer.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();
          const materials = Array.isArray(object.material) ? object.material : [object.material];
          materials.forEach((material) => {
            if (material instanceof THREE.MeshStandardMaterial && material.map) material.map.dispose();
            material.dispose();
          });
        }
      });
      if (renderer.domElement.parentNode === mount) mount.removeChild(renderer.domElement);
    };

    handlesRef.current = { renderer, scene, camera, character: characterRoot, rightArm, leftArm, leftLeg, rightLeg, reticle, dispose };
    if (navigator.xr) navigator.xr.isSessionSupported("immersive-ar").then(onCapability).catch(() => onCapability(false));
    else onCapability(false);

    return () => {
      handlesRef.current = null;
      dispose();
    };
  }, [accent, bodyShape, onCapability, textureUrl, visible]);

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
