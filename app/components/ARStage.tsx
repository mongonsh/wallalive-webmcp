"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { selectAnimatableRigParts, type CharacterRig, type ContourPoint, type DrawingExtraction, type LearnedDepthField, type SkeletonPoint } from "../lib/drawing";
import { buildArtworkShellGeometry } from "../lib/artwork-shell";
import { hasRecognizableArtworkSurface } from "../lib/mesh-materials";
import { disposeObject, prepareNeuralCharacter, type NeuralRigMap, type NeuralSemanticMap, type RiggedAssetInfo } from "../lib/rigged-model";

export type CharacterAction = "idle" | "wave" | "dance" | "hop" | "walk" | "hide" | "spin";

export type ARStageHandle = {
  enterImmersiveAR: () => Promise<{ ok: boolean; error?: string }>;
  placeNormalized: (x: number, y: number, scale?: number) => void;
  rotateBy: (yaw: number, pitch: number) => void;
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
  { characters, contour, skeleton, textureUrl, rig, depth, action, ensembleActions = null, accent, inflation, neuralAssetUrl, visible, onCapability, onPlaced, onNeuralAssetInfo },
  ref,
) {
  const mountRef = useRef<HTMLDivElement>(null);
  const handlesRef = useRef<SceneHandles | null>(null);
  const [rendererError, setRendererError] = useState(false);
  const actionRef = useRef(action);
  const ensembleActionsRef = useRef<CharacterAction[] | null>(ensembleActions);
  const placementRef = useRef({ x: 0, y: -0.15, scale: 1 });
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
    const render = (_time?: number, frame?: XRFrame) => {
      const elapsed = clock.getElapsedTime();
      const currentAction = actionRef.current;
      const directedActions = ensembleActionsRef.current;
      const placement = placementRef.current;
      const root = characterRoot;
      root.visible = visible;
      root.position.x = placement.x;
      root.position.y = placement.y + Math.sin(elapsed * 1.65) * 0.018;
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
          const part = movable.find((candidate) => candidate.kind === "arm" && candidate.side === "right")
            ?? movable.find((candidate) => candidate.kind === "arm")
            ?? movable.find((candidate) => candidate.kind === "wing" || candidate.kind === "tentacle" || candidate.kind === "tail");
          const node = part ? articulated.getObjectByName(`rig-${part.id}`) : null;
          if (node) node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + 0.76 + Math.sin(localElapsed * 7.2) * 0.48;
        }
        if (instanceAction === "dance") {
          movable.forEach((part, index) => {
            const node = articulated.getObjectByName(`rig-${part.id}`);
            if (!node) return;
            const direction = part.side === "right" || index % 2 ? -1 : 1;
            node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + direction * Math.sin(localElapsed * 5.2 + index * 0.45) * 0.48;
            node.rotation.x = Math.sin(localElapsed * 3.8 + index) * 0.12;
          });
        }
        if (instanceAction === "walk") {
          movable.filter((part) => part.kind === "leg").forEach((part, index) => {
            const node = articulated.getObjectByName(`rig-${part.id}`);
            if (!node) return;
            const direction = part.side === "right" || index % 2 ? -1 : 1;
            node.rotation.z = Number(node.userData.baseRotationZ ?? 0) + direction * Math.sin(localElapsed * 7) * 0.38;
            node.rotation.x = direction * Math.sin(localElapsed * 7) * 0.16;
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
        root.position.x = placement.x + Math.sin(elapsed * 1.5) * 0.85;
        root.rotation.z = Math.sin(elapsed * 6) * 0.055;
        root.rotation.y = rotationRef.current.yaw + Math.sin(elapsed * 3) * 0.12;
      }
      if (!directedActions && currentAction === "hide") {
        root.position.x = placement.x + 1.08;
        root.rotation.y = -0.35;
        root.rotation.z = -0.16;
      }
      if (!directedActions && currentAction === "spin") root.rotation.y = rotationRef.current.yaw + elapsed * 2.15;

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
  }, [accent, characters, contour, depth, inflation, neuralAssetUrl, onCapability, onNeuralAssetInfo, rig, skeleton, textureUrl, visible]);

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
