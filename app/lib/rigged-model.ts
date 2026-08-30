import * as THREE from "three";
import type { CharacterRig, SemanticPart, SemanticPartKind } from "./drawing";

export type RiggedAssetInfo = {
  meshes: number;
  skinnedMeshes: number;
  bones: number;
  vertices: number;
  semanticParts: number;
  detectedKinds: SemanticPartKind[];
  colorTransfer?: { sourceHue: number; targetHue: number; changedPixels: number };
};

export type NeuralRigMap = {
  all: THREE.Bone[];
  armLeft?: THREE.Bone;
  armRight?: THREE.Bone;
  legLeft?: THREE.Bone;
  legRight?: THREE.Bone;
};

export type NeuralSemanticMap = {
  all: THREE.Group[];
  byId: Record<string, THREE.Group>;
  eyeLeft?: THREE.Group;
  eyeRight?: THREE.Group;
  eyeCenter?: THREE.Group;
  pupilLeft?: THREE.Group;
  pupilRight?: THREE.Group;
  pupilCenter?: THREE.Group;
  cheekLeft?: THREE.Group;
  cheekRight?: THREE.Group;
  mouth?: THREE.Group;
};

const PROJECTED_KINDS = new Set<SemanticPartKind>(["eye", "pupil", "cheek", "mouth"]);

type Hsl = { h: number; s: number; l: number };

function rgbToHsl(r: number, g: number, b: number): Hsl {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  const l = (max + min) * 0.5;
  if (delta < 1e-5) return { h: 0, s: 0, l };
  const s = delta / (1 - Math.abs(2 * l - 1));
  const sector = max === r ? ((g - b) / delta) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
  return { h: ((sector / 6) + 1) % 1, s, l };
}

function hslToRgb({ h, s, l }: Hsl) {
  const chroma = (1 - Math.abs(2 * l - 1)) * s;
  const sector = h * 6;
  const x = chroma * (1 - Math.abs((sector % 2) - 1));
  const [r, g, b] = sector < 1 ? [chroma, x, 0]
    : sector < 2 ? [x, chroma, 0]
      : sector < 3 ? [0, chroma, x]
        : sector < 4 ? [0, x, chroma]
          : sector < 5 ? [x, 0, chroma] : [chroma, 0, x];
  const offset = l - chroma * 0.5;
  return { r: r + offset, g: g + offset, b: b + offset };
}

function hueDistance(a: number, b: number) {
  const distance = Math.abs(a - b);
  return Math.min(distance, 1 - distance);
}

function mixHue(from: number, to: number, amount: number) {
  let delta = to - from;
  if (delta > 0.5) delta -= 1;
  if (delta < -0.5) delta += 1;
  return (from + delta * amount + 1) % 1;
}

function hexHsl(hex: string) {
  const normalized = hex.replace("#", "");
  const expanded = normalized.length === 3 ? normalized.split("").map((value) => value + value).join("") : normalized;
  const value = Number.parseInt(expanded, 16);
  return rgbToHsl(((value >> 16) & 255) / 255, ((value >> 8) & 255) / 255, (value & 255) / 255);
}

export function remapDominantHuePixels(source: Uint8ClampedArray, targetHex: string) {
  const bins = 36;
  const histogram = new Float64Array(bins);
  for (let index = 0; index < source.length; index += 4) {
    if (source[index + 3] < 48) continue;
    const hsl = rgbToHsl(source[index] / 255, source[index + 1] / 255, source[index + 2] / 255);
    if (hsl.s < 0.24 || hsl.l < 0.09 || hsl.l > 0.91) continue;
    histogram[Math.min(bins - 1, Math.floor(hsl.h * bins))] += 0.25 + hsl.s;
  }
  let dominantBin = 0;
  for (let index = 1; index < histogram.length; index += 1) {
    if (histogram[index] > histogram[dominantBin]) dominantBin = index;
  }
  const sourceHue = (dominantBin + 0.5) / bins;
  const target = hexHsl(targetHex);
  const pixels = new Uint8ClampedArray(source);
  let changedPixels = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    if (pixels[index + 3] < 48) continue;
    const hsl = rgbToHsl(pixels[index] / 255, pixels[index + 1] / 255, pixels[index + 2] / 255);
    const distance = hueDistance(hsl.h, sourceHue);
    if (hsl.s < 0.2 || hsl.l < 0.08 || hsl.l > 0.92 || distance > 0.13) continue;
    const strength = Math.pow(1 - distance / 0.13, 1.35) * Math.min(1, hsl.s / 0.45);
    const adjusted = hslToRgb({
      h: mixHue(hsl.h, target.h, strength),
      s: hsl.s + (Math.max(hsl.s, target.s * 0.82) - hsl.s) * strength * 0.72,
      l: hsl.l + (target.l - 0.5) * strength * 0.16,
    });
    pixels[index] = Math.round(adjusted.r * 255);
    pixels[index + 1] = Math.round(adjusted.g * 255);
    pixels[index + 2] = Math.round(adjusted.b * 255);
    changedPixels += 1;
  }
  return { pixels, sourceHue, targetHue: target.h, changedPixels };
}

function applyDrawingPalette(source: THREE.Group, targetHex?: string) {
  if (!targetHex || typeof document === "undefined") return undefined;
  const replacements = new Map<THREE.Texture, { texture: THREE.CanvasTexture; metadata: { sourceHue: number; targetHue: number; changedPixels: number } }>();
  const processedGeometry = new Set<THREE.BufferGeometry>();
  let result: { sourceHue: number; targetHue: number; changedPixels: number } | undefined;
  source.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) return;
    const colors = object.geometry.getAttribute("color");
    if (colors && colors.itemSize >= 3 && !processedGeometry.has(object.geometry)) {
      processedGeometry.add(object.geometry);
      const pixels = new Uint8ClampedArray(colors.count * 4);
      const color = new THREE.Color();
      for (let index = 0; index < colors.count; index += 1) {
        color.setRGB(colors.getX(index), colors.getY(index), colors.getZ(index)).convertLinearToSRGB();
        pixels[index * 4] = Math.round(THREE.MathUtils.clamp(color.r, 0, 1) * 255);
        pixels[index * 4 + 1] = Math.round(THREE.MathUtils.clamp(color.g, 0, 1) * 255);
        pixels[index * 4 + 2] = Math.round(THREE.MathUtils.clamp(color.b, 0, 1) * 255);
        pixels[index * 4 + 3] = 255;
      }
      const remapped = remapDominantHuePixels(pixels, targetHex);
      if (remapped.changedPixels >= colors.count * 0.008) {
        for (let index = 0; index < colors.count; index += 1) {
          color.setRGB(
            remapped.pixels[index * 4] / 255,
            remapped.pixels[index * 4 + 1] / 255,
            remapped.pixels[index * 4 + 2] / 255,
            THREE.SRGBColorSpace,
          );
          colors.setXYZ(index, color.r, color.g, color.b);
        }
        colors.needsUpdate = true;
        result = { sourceHue: remapped.sourceHue, targetHue: remapped.targetHue, changedPixels: remapped.changedPixels };
      }
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      if (!("map" in material) || !(material.map instanceof THREE.Texture) || !material.map.image) return;
      const original = material.map;
      let replacement = replacements.get(original);
      if (!replacement) {
        const image = original.image as CanvasImageSource & { width?: number; height?: number; naturalWidth?: number; naturalHeight?: number };
        const width = image.naturalWidth ?? image.width ?? 0;
        const height = image.naturalHeight ?? image.height ?? 0;
        if (!width || !height) return;
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext("2d", { willReadFrequently: true });
        if (!context) return;
        context.drawImage(image, 0, 0, width, height);
        const imageData = context.getImageData(0, 0, width, height);
        const remapped = remapDominantHuePixels(imageData.data, targetHex);
        if (remapped.changedPixels < width * height * 0.008) return;
        imageData.data.set(remapped.pixels);
        context.putImageData(imageData, 0, 0);
        const texture = new THREE.CanvasTexture(canvas);
        texture.name = `${original.name || "generated"}-wallalive-color-match`;
        texture.colorSpace = original.colorSpace;
        texture.flipY = original.flipY;
        texture.wrapS = original.wrapS;
        texture.wrapT = original.wrapT;
        texture.magFilter = original.magFilter;
        texture.minFilter = original.minFilter;
        texture.anisotropy = original.anisotropy;
        texture.offset.copy(original.offset);
        texture.repeat.copy(original.repeat);
        texture.center.copy(original.center);
        texture.rotation = original.rotation;
        texture.needsUpdate = true;
        replacement = { texture, metadata: { sourceHue: remapped.sourceHue, targetHue: remapped.targetHue, changedPixels: remapped.changedPixels } };
        replacements.set(original, replacement);
      }
      material.map = replacement.texture;
      material.needsUpdate = true;
      result = result
        ? { ...replacement.metadata, changedPixels: result.changedPixels + replacement.metadata.changedPixels }
        : replacement.metadata;
    });
  });
  replacements.forEach((_replacement, original) => original.dispose());
  return result;
}

function semanticProperty(part: SemanticPart) {
  if (part.kind === "mouth") return "mouth" as const;
  if (part.side === "center" && part.kind === "eye") return "eyeCenter" as const;
  if (part.side === "center" && part.kind === "pupil") return "pupilCenter" as const;
  if (part.side === "center") return null;
  const suffix = part.side === "left" ? "Left" : "Right";
  if (part.kind === "eye") return `eye${suffix}` as "eyeLeft" | "eyeRight";
  if (part.kind === "pupil") return `pupil${suffix}` as "pupilLeft" | "pupilRight";
  if (part.kind === "cheek") return `cheek${suffix}` as "cheekLeft" | "cheekRight";
  return null;
}

function featurePoints(part: SemanticPart, body: SemanticPart, modelSize: THREE.Vector3) {
  const scaleX = modelSize.x * 0.92 / Math.max(0.001, body.size.x);
  const scaleY = modelSize.y * 0.92 / Math.max(0.001, body.size.y);
  const outline = part.outline?.length && part.outline.length >= 4
    ? part.outline
    : Array.from({ length: 32 }, (_, index) => {
      const angle = index / 32 * Math.PI * 2;
      const cos = Math.cos(part.rotation);
      const sin = Math.sin(part.rotation);
      const localX = Math.cos(angle) * part.size.x * 0.5;
      const localY = Math.sin(angle) * part.size.y * 0.5;
      return {
        x: part.center.x + localX * cos - localY * sin,
        y: part.center.y + localX * sin + localY * cos,
      };
    });
  return outline.map((point) => new THREE.Vector3(
    (point.x - part.center.x) * scaleX,
    (point.y - part.center.y) * scaleY,
    0,
  ));
}

function createSemanticFeature(part: SemanticPart, body: SemanticPart, modelSize: THREE.Vector3) {
  const group = new THREE.Group();
  group.name = `neural-semantic-${part.id}`;
  group.userData.kind = part.kind;
  group.userData.side = part.side;
  group.userData.confidence = part.confidence;
  const material = new THREE.MeshBasicMaterial({
    color: new THREE.Color(part.color),
    transparent: true,
    opacity: 0.97,
    depthWrite: false,
    toneMapped: false,
  });
  const points = featurePoints(part, body, modelSize);
  let feature: THREE.Mesh;
  if (part.kind === "pupil") {
    const width = Math.max(0.012, part.size.x / Math.max(0.001, body.size.x) * modelSize.x * 0.92);
    const height = Math.max(0.012, part.size.y / Math.max(0.001, body.size.y) * modelSize.y * 0.92);
    feature = new THREE.Mesh(new THREE.SphereGeometry(0.5, 18, 12), material);
    feature.scale.set(width, height, Math.min(width, height) * 0.26);
  } else {
    const curve = new THREE.CatmullRomCurve3(points, true, "centripetal", 0.3);
    const radius = Math.max(0.0045, Math.min(0.012, Math.min(modelSize.x, modelSize.y) * 0.0065));
    feature = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(28, points.length * 2), radius, 8, true), material);
  }
  feature.renderOrder = 8;
  group.add(feature);
  return group;
}

function projectSemanticRig(source: THREE.Group, bounds: THREE.Box3, rig?: CharacterRig): NeuralSemanticMap {
  const semanticMap: NeuralSemanticMap = { all: [], byId: {} };
  if (!rig) return semanticMap;
  const body = rig.parts.find((part) => part.kind === "body");
  if (!body) return semanticMap;
  const modelSize = bounds.getSize(new THREE.Vector3());
  const modelCenter = bounds.getCenter(new THREE.Vector3());
  const raycaster = new THREE.Raycaster();
  const front = new THREE.Vector3(0, 0, 1);

  rig.parts.filter((part) => part.source === "image-region" && PROJECTED_KINDS.has(part.kind)).forEach((part) => {
    const normalizedX = (part.center.x - body.center.x) / Math.max(0.001, body.size.x);
    const normalizedY = (part.center.y - body.center.y) / Math.max(0.001, body.size.y);
    const origin = new THREE.Vector3(
      modelCenter.x + normalizedX * modelSize.x * 0.92,
      modelCenter.y + normalizedY * modelSize.y * 0.92,
      bounds.max.z + Math.max(0.4, modelSize.z),
    );
    raycaster.set(origin, new THREE.Vector3(0, 0, -1));
    raycaster.far = Math.max(1, modelSize.z * 3);
    const hit = raycaster.intersectObject(source, true)[0];
    const normal = hit?.face
      ? hit.face.normal.clone().applyMatrix3(new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld)).normalize()
      : front.clone();
    if (normal.z < 0) normal.multiplyScalar(-1);
    const feature = createSemanticFeature(part, body, modelSize);
    feature.position.copy(hit?.point ?? new THREE.Vector3(origin.x, origin.y, bounds.max.z));
    feature.position.addScaledVector(normal, Math.max(0.004, Math.min(modelSize.x, modelSize.y) * 0.004));
    feature.quaternion.setFromUnitVectors(front, normal);
    source.parent?.add(feature);
    semanticMap.all.push(feature);
    semanticMap.byId[part.id] = feature;
    const property = semanticProperty(part);
    if (property) semanticMap[property] = feature;
  });
  return semanticMap;
}

export function disposeObject(root: THREE.Object3D) {
  root.traverse((object) => {
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
}

export function prepareNeuralCharacter(source: THREE.Group, rig?: CharacterRig) {
  const character = new THREE.Group();
  character.name = "wallalive-neural-character";
  source.rotation.y = Math.PI;
  character.add(source);
  character.updateMatrixWorld(true);

  let bounds = new THREE.Box3().setFromObject(source);
  const size = bounds.getSize(new THREE.Vector3());
  const scale = 1.62 / Math.max(0.001, size.x, size.y, size.z);
  source.scale.setScalar(scale);
  character.updateMatrixWorld(true);
  bounds = new THREE.Box3().setFromObject(source);
  const center = bounds.getCenter(new THREE.Vector3());
  source.position.x -= center.x;
  source.position.z -= center.z;
  source.position.y += -0.82 - bounds.min.y;
  character.updateMatrixWorld(true);

  const info: RiggedAssetInfo = { meshes: 0, skinnedMeshes: 0, bones: 0, vertices: 0, semanticParts: 0, detectedKinds: [] };
  const bones: THREE.Bone[] = [];
  source.traverse((object) => {
    if (object instanceof THREE.Mesh) {
      info.meshes += 1;
      info.vertices += object.geometry.getAttribute("position")?.count ?? 0;
      object.castShadow = true;
      object.receiveShadow = true;
    }
    if (object instanceof THREE.SkinnedMesh) info.skinnedMeshes += 1;
    if (object instanceof THREE.Bone) {
      info.bones += 1;
      bones.push(object);
      object.userData.wallaliveBaseQuaternion = object.quaternion.clone();
    }
  });
  const colorTransfer = applyDrawingPalette(source, rig?.bodyColor);
  if (colorTransfer) info.colorTransfer = colorTransfer;

  const root = bones.find((bone) => !(bone.parent instanceof THREE.Bone)) ?? bones[0];
  const rigMap: NeuralRigMap = { all: bones };
  if (root) {
    const rootPosition = root.getWorldPosition(new THREE.Vector3());
    const boneBounds = new THREE.Box3();
    bones.forEach((bone) => boneBounds.expandByPoint(bone.getWorldPosition(new THREE.Vector3())));
    const boneSize = boneBounds.getSize(new THREE.Vector3());
    const branches = root.children.filter((child): child is THREE.Bone => child instanceof THREE.Bone);
    for (const branch of branches) {
      const branchBones: THREE.Bone[] = [];
      branch.traverse((node) => { if (node instanceof THREE.Bone) branchBones.push(node); });
      const endpoint = branchBones.reduce((furthest, bone) => {
        const position = bone.getWorldPosition(new THREE.Vector3());
        return position.distanceTo(rootPosition) > furthest.distanceTo(rootPosition) ? position : furthest;
      }, branch.getWorldPosition(new THREE.Vector3()));
      const dx = endpoint.x - rootPosition.x;
      const dy = endpoint.y - rootPosition.y;
      if (dy < -boneSize.y * 0.13) {
        if (dx < 0) rigMap.legLeft = branch;
        else rigMap.legRight = branch;
      } else if (Math.abs(dx) > boneSize.x * 0.2) {
        if (dx < 0) rigMap.armLeft = branch;
        else rigMap.armRight = branch;
      }
    }
  }
  character.updateMatrixWorld(true);
  bounds = new THREE.Box3().setFromObject(source);
  const semanticMap = projectSemanticRig(source, bounds, rig);
  info.semanticParts = semanticMap.all.length;
  info.detectedKinds = rig?.detectedKinds.filter((kind) => kind !== "body") ?? [];
  character.userData.wallaliveRig = rigMap;
  character.userData.wallaliveSemantic = semanticMap;
  character.userData.reconstruction = {
    method: "AniGen joint mesh-skeleton-skinning reconstruction",
    assetType: "glTF SkinnedMesh",
    topology: "generated full 3D surface",
    viewableDegrees: 360,
    ...info,
  };
  return { character, info, rigMap, semanticMap };
}
