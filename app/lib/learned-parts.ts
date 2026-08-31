import {
  mergeLearnedPartHints,
  POSE_JOINT_NAMES,
  TOPOLOGY_CLASSES,
  type DrawingExtraction,
  type CaptureTarget,
  type LearnedPartHint,
  type LearnedDepthField,
  type LearnedPose,
  type LearnedTopology,
  type TopologyNode,
} from "./drawing.ts";
import { acceptFaceComponent } from "./face-component-gate.ts";
import { averageLogitConfidence, sameSemanticInstance, selectDominantInkColor } from "./model-math.ts";
import { isolateDrawingFromImageUrl, isolateDrawingFromVideo } from "./target-cutout.ts";

const BODY_MODEL_PATH = "/models/wallalive-parts-v3.onnx";
const FACE_V3_MODEL_PATH = "/models/wallalive-face-v3.onnx";
const FACE_V4_MODEL_PATH = "/models/wallalive-face-v4.onnx";
const POSE_MODEL_PATH = "/models/wallalive-amateur-pose-v6.onnx";
const TOPOLOGY_MODEL_PATH = "/models/wallalive-topology-v10.onnx";
const DEPTH_MODEL_PATH = "/models/wallalive-sketch-depth-v1.onnx";
const FALLBACK_MODEL_PATHS = ["/models/wallalive-parts-v2.onnx", "/models/wallalive-parts-v1.onnx"] as const;
const BODY_SIZE = 96;
const FACE_V3_SIZE = 96;
const FACE_SIZE = 128;
const POSE_HEATMAP_SIZE = 48;
const TOPOLOGY_FIELD_SIZE = 48;
const DEPTH_SIZE = 64;
const FALLBACK_SIZE = 64;
const PARTS = ["body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot"] as const;
const FACE_PARTS = ["eye", "cheek", "mouth", "ear"] as const;
const FACE_KINDS = new Set<LearnedPartHint["kind"]>(FACE_PARTS);

const BODY_THRESHOLDS: Record<(typeof PARTS)[number], number> = {
  body: 0.54,
  eye: 0.72,
  cheek: 0.24,
  mouth: 0.70,
  ear: 0.72,
  arm: 0.72,
  hand: 0.60,
  leg: 0.72,
  foot: 0.72,
};
const FACE_THRESHOLDS: Record<(typeof FACE_PARTS)[number], number> = {
  eye: 0.5664,
  cheek: 0.1836,
  mouth: 0.7578,
  ear: 0.6875,
};
const FACE_V4_BLEND: Record<(typeof FACE_PARTS)[number], number> = {
  eye: 0.5,
  cheek: 0.3,
  mouth: 0.6,
  ear: 0.5,
};
const FALLBACK_THRESHOLDS = {
  body: 0.48,
  eye: 0.54,
  cheek: 0.52,
  mouth: 0.50,
  ear: 0.52,
  arm: 0.49,
  hand: 0.52,
  leg: 0.49,
  foot: 0.52,
} satisfies Record<(typeof PARTS)[number], number>;

type OrtRuntime = typeof import("onnxruntime-web/wasm");
type OrtSession = import("onnxruntime-web/wasm").InferenceSession;
type Point = { x: number; y: number };
type PointMap = (x: number, y: number) => Point;
type PreparedImage = {
  values: Float32Array;
  mapPoint: PointMap;
  contentRect: { x: number; y: number; width: number; height: number };
};

let runtimePromise: Promise<OrtRuntime> | null = null;
let sessionPromise: Promise<{ ort: OrtRuntime; body: OrtSession; faceV3: OrtSession; faceV4: OrtSession; pose: OrtSession; topology: OrtSession; depth: OrtSession }> | null = null;
let fallbackSessionPromise: Promise<OrtSession[]> | null = null;

const sessionOptions = {
  executionProviders: ["wasm"] as const,
  graphOptimizationLevel: "all" as const,
  executionMode: "sequential" as const,
};

function loadSessions() {
  if (!sessionPromise) {
    runtimePromise ??= import("onnxruntime-web/wasm");
    sessionPromise = runtimePromise.then(async (ort) => {
      ort.env.wasm.numThreads = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated ? 2 : 1;
      ort.env.wasm.proxy = false;
      const [body, faceV3, faceV4, pose, topology, depth] = await Promise.all([
        ort.InferenceSession.create(BODY_MODEL_PATH, sessionOptions),
        ort.InferenceSession.create(FACE_V3_MODEL_PATH, sessionOptions),
        ort.InferenceSession.create(FACE_V4_MODEL_PATH, sessionOptions),
        ort.InferenceSession.create(POSE_MODEL_PATH, sessionOptions),
        ort.InferenceSession.create(TOPOLOGY_MODEL_PATH, sessionOptions),
        ort.InferenceSession.create(DEPTH_MODEL_PATH, sessionOptions),
      ]);
      return { ort, body, faceV3, faceV4, pose, topology, depth };
    });
  }
  return sessionPromise;
}

function loadFallbackSessions(ort: OrtRuntime) {
  fallbackSessionPromise ??= Promise.all(FALLBACK_MODEL_PATHS.map((path) => ort.InferenceSession.create(path, sessionOptions)));
  return fallbackSessionPromise;
}

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The isolated drawing could not be prepared for local recognition."));
    image.src = url;
  });
}

const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

function canvasValues(canvas: HTMLCanvasElement) {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
  const area = canvas.width * canvas.height;
  const values = new Float32Array(area * 3);
  for (let index = 0; index < area; index += 1) {
    values[index] = rgba[index * 4] / 255;
    values[area + index] = rgba[index * 4 + 1] / 255;
    values[area * 2 + index] = rgba[index * 4 + 2] / 255;
  }
  return values;
}

function prepareImage(image: HTMLImageElement, size: number, crop?: { x: number; y: number; width: number; height: number }, stretch = false): PreparedImage {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, size, size);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";

  const normalizedCrop = crop ?? { x: 0, y: 0, width: 1, height: 1 };
  const sourceX = normalizedCrop.x * image.naturalWidth;
  const sourceY = normalizedCrop.y * image.naturalHeight;
  const sourceWidth = Math.max(1, normalizedCrop.width * image.naturalWidth);
  const sourceHeight = Math.max(1, normalizedCrop.height * image.naturalHeight);
  const scale = stretch ? 1 : Math.min(size / sourceWidth, size / sourceHeight);
  const drawWidth = stretch ? size : sourceWidth * scale;
  const drawHeight = stretch ? size : sourceHeight * scale;
  const offsetX = (size - drawWidth) / 2;
  const offsetY = (size - drawHeight) / 2;
  context.drawImage(image, sourceX, sourceY, sourceWidth, sourceHeight, offsetX, offsetY, drawWidth, drawHeight);

  const mapPoint = (x: number, y: number): Point => ({
    x: clamp(normalizedCrop.x + ((x - offsetX) / drawWidth) * normalizedCrop.width, 0, 1),
    y: clamp(normalizedCrop.y + ((y - offsetY) / drawHeight) * normalizedCrop.height, 0, 1),
  });
  return {
    values: canvasValues(canvas),
    mapPoint,
    contentRect: { x: offsetX, y: offsetY, width: drawWidth, height: drawHeight },
  };
}

function prepareDepthImage(image: HTMLImageElement) {
  const canvas = document.createElement("canvas");
  canvas.width = DEPTH_SIZE;
  canvas.height = DEPTH_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.clearRect(0, 0, DEPTH_SIZE, DEPTH_SIZE);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, DEPTH_SIZE, DEPTH_SIZE);
  const rgba = context.getImageData(0, 0, DEPTH_SIZE, DEPTH_SIZE).data;
  const area = DEPTH_SIZE * DEPTH_SIZE;
  const mask = new Uint8Array(area);
  for (let index = 0; index < area; index += 1) mask[index] = rgba[index * 4 + 3] > 38 ? 1 : 0;

  const distance = new Float32Array(area);
  const far = DEPTH_SIZE * 2;
  const diagonal = Math.SQRT2;
  for (let index = 0; index < area; index += 1) distance[index] = mask[index] ? far : 0;
  for (let y = 0; y < DEPTH_SIZE; y += 1) {
    for (let x = 0; x < DEPTH_SIZE; x += 1) {
      const index = y * DEPTH_SIZE + x;
      if (!distance[index]) continue;
      if (x) distance[index] = Math.min(distance[index], distance[index - 1] + 1);
      if (y) distance[index] = Math.min(distance[index], distance[index - DEPTH_SIZE] + 1);
      if (x && y) distance[index] = Math.min(distance[index], distance[index - DEPTH_SIZE - 1] + diagonal);
      if (x + 1 < DEPTH_SIZE && y) distance[index] = Math.min(distance[index], distance[index - DEPTH_SIZE + 1] + diagonal);
    }
  }
  let maximumDistance = 1;
  for (let y = DEPTH_SIZE - 1; y >= 0; y -= 1) {
    for (let x = DEPTH_SIZE - 1; x >= 0; x -= 1) {
      const index = y * DEPTH_SIZE + x;
      if (!distance[index]) continue;
      if (x + 1 < DEPTH_SIZE) distance[index] = Math.min(distance[index], distance[index + 1] + 1);
      if (y + 1 < DEPTH_SIZE) distance[index] = Math.min(distance[index], distance[index + DEPTH_SIZE] + 1);
      if (x + 1 < DEPTH_SIZE && y + 1 < DEPTH_SIZE) distance[index] = Math.min(distance[index], distance[index + DEPTH_SIZE + 1] + diagonal);
      if (x && y + 1 < DEPTH_SIZE) distance[index] = Math.min(distance[index], distance[index + DEPTH_SIZE - 1] + diagonal);
      maximumDistance = Math.max(maximumDistance, distance[index]);
    }
  }

  const ink = new Uint8Array(area);
  const colorDifference = (left: number, right: number) => {
    const leftOffset = left * 4;
    const rightOffset = right * 4;
    return Math.max(
      Math.abs(rgba[leftOffset] - rgba[rightOffset]),
      Math.abs(rgba[leftOffset + 1] - rgba[rightOffset + 1]),
      Math.abs(rgba[leftOffset + 2] - rgba[rightOffset + 2]),
      Math.abs(rgba[leftOffset + 3] - rgba[rightOffset + 3]),
    );
  };
  for (let y = 0; y < DEPTH_SIZE; y += 1) {
    for (let x = 0; x < DEPTH_SIZE; x += 1) {
      const index = y * DEPTH_SIZE + x;
      if (!mask[index]) continue;
      const neighbors = [
        x ? index - 1 : -1,
        x + 1 < DEPTH_SIZE ? index + 1 : -1,
        y ? index - DEPTH_SIZE : -1,
        y + 1 < DEPTH_SIZE ? index + DEPTH_SIZE : -1,
      ];
      ink[index] = neighbors.some((neighbor) => neighbor < 0 || !mask[neighbor] || colorDifference(index, neighbor) >= 30) ? 1 : 0;
    }
  }
  // Match the one-pixel analytic-contour dilation used while training.
  const dilatedInk = Uint8Array.from(ink);
  for (let y = 1; y < DEPTH_SIZE - 1; y += 1) {
    for (let x = 1; x < DEPTH_SIZE - 1; x += 1) {
      const index = y * DEPTH_SIZE + x;
      if (!mask[index] || ink[index]) continue;
      for (let offsetY = -1; offsetY <= 1 && !dilatedInk[index]; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (ink[(y + offsetY) * DEPTH_SIZE + x + offsetX]) {
            dilatedInk[index] = 1;
            break;
          }
        }
      }
    }
  }
  const values = new Float32Array(area * 3);
  for (let index = 0; index < area; index += 1) {
    values[index] = mask[index];
    values[area + index] = distance[index] / maximumDistance;
    values[area * 2 + index] = dilatedInk[index];
  }
  return values;
}

function decodeDepth(output: import("onnxruntime-web/wasm").Tensor, latencyMs: number): LearnedDepthField {
  const values = output.data as Float32Array;
  const area = DEPTH_SIZE * DEPTH_SIZE;
  const front = Float32Array.from(values.subarray(0, area));
  const back = Float32Array.from(values.subarray(area, area * 2));
  let occupied = 0;
  let thickness = 0;
  let asymmetry = 0;
  for (let index = 0; index < area; index += 1) {
    if (front[index] + back[index] <= 0.001) continue;
    occupied += 1;
    thickness += (front[index] + back[index]) * 0.525;
    asymmetry += Math.abs(front[index] - back[index]) * 0.525;
  }
  return {
    model: "wallalive-sketch-depth-v1",
    latencyMs,
    size: DEPTH_SIZE,
    depthScale: 0.525,
    front,
    back,
    meanThickness: thickness / Math.max(1, occupied),
    meanAsymmetry: asymmetry / Math.max(1, occupied),
  };
}

type PixelComponent = {
  pixels: number[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  centerX: number;
  centerY: number;
};

function components(mask: Uint8Array, size: number) {
  const visited = new Uint8Array(mask.length);
  const found: PixelComponent[] = [];
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;
    const queue = [start];
    const pixels: number[] = [];
    visited[start] = 1;
    let minX = size;
    let minY = size;
    let maxX = 0;
    let maxY = 0;
    let sumX = 0;
    let sumY = 0;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const index = queue[cursor];
      const x = index % size;
      const y = Math.floor(index / size);
      pixels.push(index);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      sumX += x;
      sumY += y;
      for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
        for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
          if (!offsetX && !offsetY) continue;
          const nextX = x + offsetX;
          const nextY = y + offsetY;
          if (nextX < 0 || nextY < 0 || nextX >= size || nextY >= size) continue;
          const next = nextY * size + nextX;
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            queue.push(next);
          }
        }
      }
    }
    found.push({ pixels, minX, minY, maxX, maxY, centerX: sumX / pixels.length, centerY: sumY / pixels.length });
  }
  return found;
}

// Zhang-Suen thinning turns the learned centerline probability band into a
// one-pixel graph. Endpoint heatmaps often merge nearby tentacles or tree
// branches into one blob; graph degree remains separable after thinning.
export function thinTopologyMask(input: Uint8Array, size: number) {
  const mask = Uint8Array.from(input);
  const neighborValues = (x: number, y: number) => [
    mask[(y - 1) * size + x],
    mask[(y - 1) * size + x + 1],
    mask[y * size + x + 1],
    mask[(y + 1) * size + x + 1],
    mask[(y + 1) * size + x],
    mask[(y + 1) * size + x - 1],
    mask[y * size + x - 1],
    mask[(y - 1) * size + x - 1],
  ];
  for (let iteration = 0; iteration < size * 2; iteration += 1) {
    let changed = false;
    for (const phase of [0, 1]) {
      const remove: number[] = [];
      for (let y = 1; y < size - 1; y += 1) {
        for (let x = 1; x < size - 1; x += 1) {
          const index = y * size + x;
          if (!mask[index]) continue;
          const neighbors = neighborValues(x, y);
          const count = neighbors.reduce((total, value) => total + value, 0);
          if (count < 2 || count > 6) continue;
          let transitions = 0;
          for (let neighbor = 0; neighbor < 8; neighbor += 1) {
            if (!neighbors[neighbor] && neighbors[(neighbor + 1) % 8]) transitions += 1;
          }
          if (transitions !== 1) continue;
          const [north, , east, , south, , west] = neighbors;
          const firstTriplet = phase === 0 ? north * east * south : north * east * west;
          const secondTriplet = phase === 0 ? east * south * west : north * south * west;
          if (!firstTriplet && !secondTriplet) remove.push(index);
        }
      }
      if (remove.length) changed = true;
      for (const index of remove) mask[index] = 0;
    }
    if (!changed) break;
  }
  return mask;
}

function componentOutline(component: PixelComponent, size: number, mapPoint: PointMap) {
  const mask = new Uint8Array(size * size);
  for (const index of component.pixels) mask[index] = 1;
  const directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]] as const;
  let startX = -1;
  let startY = -1;
  for (let y = component.minY; y <= component.maxY && startX < 0; y += 1) {
    for (let x = component.minX; x <= component.maxX; x += 1) {
      if (mask[y * size + x] && (x === 0 || !mask[y * size + x - 1])) {
        startX = x;
        startY = y;
        break;
      }
    }
  }
  if (startX < 0) return [];
  const points: Point[] = [];
  let x = startX;
  let y = startY;
  let backX = x - 1;
  let backY = y;
  for (let step = 0; step < component.pixels.length * 10; step += 1) {
    points.push({ x, y });
    let backIndex = directions.findIndex(([offsetX, offsetY]) => x + offsetX === backX && y + offsetY === backY);
    if (backIndex < 0) backIndex = 4;
    let found = false;
    for (let offset = 1; offset <= directions.length; offset += 1) {
      const directionIndex = (backIndex + offset) % directions.length;
      const [offsetX, offsetY] = directions[directionIndex];
      const nextX = x + offsetX;
      const nextY = y + offsetY;
      if (nextX < 0 || nextY < 0 || nextX >= size || nextY >= size || !mask[nextY * size + nextX]) continue;
      const before = directions[(directionIndex + directions.length - 1) % directions.length];
      backX = x + before[0];
      backY = y + before[1];
      x = nextX;
      y = nextY;
      found = true;
      break;
    }
    if (!found || (x === startX && y === startY && points.length > 2)) break;
  }
  const sampleEvery = Math.max(1, Math.ceil(points.length / 48));
  return points.filter((_, index) => index % sampleEvery === 0).map((point) => mapPoint(point.x + 0.5, point.y + 0.5));
}

function componentColor(component: PixelComponent, imageValues: Float32Array, size: number) {
  const area = size * size;
  const samples = component.pixels.map((index) => ({
    r: imageValues[index] * 255,
    g: imageValues[area + index] * 255,
    b: imageValues[area * 2 + index] * 255,
  })).filter((sample) => (sample.r + sample.g + sample.b) / 3 < 248 || Math.max(sample.r, sample.g, sample.b) - Math.min(sample.r, sample.g, sample.b) > 4);
  const color = selectDominantInkColor(samples);
  if (!color) return undefined;
  return `#${[color.r, color.g, color.b].map((channel) => Math.round(channel).toString(16).padStart(2, "0")).join("")}`;
}

function describeComponent(kind: LearnedPartHint["kind"], component: PixelComponent, logits: Float32Array, imageValues: Float32Array, channelOffset: number, size: number, mapPoint: PointMap): LearnedPartHint {
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const index of component.pixels) {
    const x = index % size - component.centerX;
    const y = Math.floor(index / size) - component.centerY;
    xx += x * x;
    yy += y * y;
    xy += x * y;
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const axis = { x: Math.cos(angle), y: Math.sin(angle) };
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const index of component.pixels) {
    const x = index % size - component.centerX;
    const y = Math.floor(index / size) - component.centerY;
    const projection = x * axis.x + y * axis.y;
    minimum = Math.min(minimum, projection);
    maximum = Math.max(maximum, projection);
  }
  const center = mapPoint(component.centerX + 0.5, component.centerY + 0.5);
  const topLeft = mapPoint(component.minX, component.minY);
  const bottomRight = mapPoint(component.maxX + 1, component.maxY + 1);
  return {
    kind,
    center,
    size: { x: Math.abs(bottomRight.x - topLeft.x), y: Math.abs(bottomRight.y - topLeft.y) },
    endpoints: [
      mapPoint(component.centerX + axis.x * minimum, component.centerY + axis.y * minimum),
      mapPoint(component.centerX + axis.x * maximum, component.centerY + axis.y * maximum),
    ],
    outline: componentOutline(component, size, mapPoint),
    rotation: angle,
    confidence: averageLogitConfidence(logits, component.pixels, channelOffset),
    color: componentColor(component, imageValues, size),
  };
}

function maximumInstances(kind: LearnedPartHint["kind"]) {
  if (kind === "mouth") return 3;
  if (kind === "eye" || kind === "cheek" || kind === "ear") return 6;
  return 10;
}

function decodeHints<K extends readonly LearnedPartHint["kind"][]>(
  output: import("onnxruntime-web/wasm").Tensor,
  size: number,
  kinds: K,
  thresholds: Record<K[number], number>,
  imageValues: Float32Array,
  mapPoint: PointMap,
  options: { skipBody: boolean; faceCrop?: boolean; componentGate?: { v3Logits: Float32Array; v4Logits: Float32Array } },
) {
  const probabilities = output.data as Float32Array;
  const area = size * size;
  const hints: LearnedPartHint[] = [];
  for (let channel = options.skipBody ? 1 : 0; channel < kinds.length; channel += 1) {
    const kind = kinds[channel];
    const offset = channel * area;
    const mask = new Uint8Array(area);
    for (let index = 0; index < area; index += 1) mask[index] = sigmoid(probabilities[offset + index]) >= thresholds[kind] ? 1 : 0;
    const minimumArea = Math.max(kind === "eye" || kind === "mouth" ? 3 : 4, Math.round(area * (kind === "arm" || kind === "leg" ? 0.00055 : 0.00035)));
    const maximumFraction = options.faceCrop ? 0.34 : kind === "arm" || kind === "leg" ? 0.18 : kind === "hand" || kind === "foot" ? 0.14 : kind === "ear" ? 0.13 : 0.09;
    const candidates = components(mask, size)
      .filter((component) => component.pixels.length >= minimumArea && component.pixels.length <= area * maximumFraction)
      .sort((a, b) => b.pixels.length - a.pixels.length)
      .slice(0, maximumInstances(kind));
    const accepted = options.componentGate
      ? candidates.filter((component, rank) => acceptFaceComponent(
        kind as (typeof FACE_PARTS)[number],
        component,
        probabilities,
        options.componentGate!.v3Logits,
        options.componentGate!.v4Logits,
        offset,
        size,
        thresholds[kind],
        rank,
        candidates.length,
      ))
      : candidates;
    for (const component of accepted) {
      const hint = describeComponent(kind, component, probabilities, imageValues, offset, size, mapPoint);
      if (!hints.some((candidate) => sameSemanticInstance(candidate, hint))) hints.push(hint);
    }
  }
  return hints;
}

function locateHead(output: import("onnxruntime-web/wasm").Tensor, prepared: PreparedImage) {
  const values = output.data as Float32Array;
  const area = BODY_SIZE * BODY_SIZE;
  const mask = new Uint8Array(area);
  const offset = area;
  for (let index = 0; index < area; index += 1) mask[index] = sigmoid(values[offset + index]) >= 0.42 ? 1 : 0;
  const candidates = components(mask, BODY_SIZE).filter((component) => component.pixels.length >= area * 0.004);
  if (!candidates.length) {
    const rect = prepared.contentRect;
    return { x: rect.x, y: rect.y, width: rect.width, height: rect.height * 0.52 };
  }
  const minX = Math.min(...candidates.map((candidate) => candidate.minX));
  const minY = Math.min(...candidates.map((candidate) => candidate.minY));
  const maxX = Math.max(...candidates.map((candidate) => candidate.maxX));
  const maxY = Math.max(...candidates.map((candidate) => candidate.maxY));
  const span = Math.max(maxX - minX + 1, maxY - minY + 1);
  const margin = Math.max(3, span * 0.18);
  const content = prepared.contentRect;
  const x = clamp(minX - margin, content.x, content.x + content.width);
  const y = clamp(minY - margin, content.y, content.y + content.height);
  const right = clamp(maxX + margin + 1, content.x, content.x + content.width);
  const bottom = clamp(maxY + margin + 1, content.y, content.y + content.height);
  return { x, y, width: Math.max(1, right - x), height: Math.max(1, bottom - y) };
}

function softmax(values: Float32Array) {
  const maximum = Math.max(...values);
  const exponentials = Array.from(values, (value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / Math.max(1e-9, total));
}

export function decodeTopology(
  fieldsOutput: import("onnxruntime-web/wasm").Tensor,
  classOutput: import("onnxruntime-web/wasm").Tensor,
  prepared: PreparedImage,
  latencyMs: number,
): LearnedTopology {
  const values = fieldsOutput.data as Float32Array;
  const area = TOPOLOGY_FIELD_SIZE * TOPOLOGY_FIELD_SIZE;
  const probability = (channel: number, index: number) => sigmoid(values[channel * area + index]);
  const maskFor = (channel: number, threshold: number) => {
    const mask = new Uint8Array(area);
    for (let index = 0; index < area; index += 1) mask[index] = probability(channel, index) >= threshold ? 1 : 0;
    return mask;
  };
  const centerlineMask = maskFor(1, 0.46);
  const thinnedCenterline = thinTopologyMask(centerlineMask, TOPOLOGY_FIELD_SIZE);
  const describe = (channel: number, role: "endpoint" | "junction", threshold: number, maximum: number) => components(maskFor(channel, threshold), TOPOLOGY_FIELD_SIZE)
    .filter((component) => component.pixels.length <= area * 0.06)
    .map((component) => {
      let total = 0;
      let weightedX = 0;
      let weightedY = 0;
      let confidence = 0;
      for (const index of component.pixels) {
        const weight = probability(channel, index);
        total += weight;
        weightedX += (index % TOPOLOGY_FIELD_SIZE) * weight;
        weightedY += Math.floor(index / TOPOLOGY_FIELD_SIZE) * weight;
        confidence = Math.max(confidence, weight);
      }
      return { role, x: weightedX / Math.max(1e-6, total), y: weightedY / Math.max(1e-6, total), confidence };
    })
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, maximum);
  const graphCandidates = (role: "endpoint" | "junction", minimumNeighbors: number, maximumNeighbors: number, maximum: number) => {
    const found: Array<{ role: "endpoint" | "junction"; x: number; y: number; confidence: number }> = [];
    for (let y = 1; y < TOPOLOGY_FIELD_SIZE - 1; y += 1) {
      for (let x = 1; x < TOPOLOGY_FIELD_SIZE - 1; x += 1) {
        const index = y * TOPOLOGY_FIELD_SIZE + x;
        if (!thinnedCenterline[index]) continue;
        let neighbors = 0;
        for (let offsetY = -1; offsetY <= 1; offsetY += 1) {
          for (let offsetX = -1; offsetX <= 1; offsetX += 1) {
            if (offsetX || offsetY) neighbors += thinnedCenterline[(y + offsetY) * TOPOLOGY_FIELD_SIZE + x + offsetX];
          }
        }
        if (neighbors < minimumNeighbors || neighbors > maximumNeighbors) continue;
        const confidence = Math.max(probability(role === "endpoint" ? 2 : 3, index), probability(1, index) * 0.82);
        found.push({ role, x, y, confidence });
      }
    }
    return found.sort((a, b) => b.confidence - a.confidence).filter((candidate, index, all) => !all.slice(0, index).some((other) => (
      Math.hypot(other.x - candidate.x, other.y - candidate.y) < (role === "endpoint" ? 3.2 : 4.2)
    ))).slice(0, maximum);
  };
  const endpointCandidates = [...describe(2, "endpoint", 0.42, 14), ...graphCandidates("endpoint", 0, 1, 14)]
    .sort((a, b) => b.confidence - a.confidence)
    .filter((candidate, index, all) => !all.slice(0, index).some((other) => Math.hypot(other.x - candidate.x, other.y - candidate.y) < 2.5))
    .slice(0, 14);
  const junctionCandidates = [...describe(3, "junction", 0.42, 8), ...graphCandidates("junction", 3, 8, 8)]
    .sort((a, b) => b.confidence - a.confidence)
    .filter((candidate, index, all) => !all.slice(0, index).some((other) => Math.hypot(other.x - candidate.x, other.y - candidate.y) < 3.2))
    .slice(0, 8);
  const candidates = [...junctionCandidates, ...endpointCandidates].filter((candidate, index, all) => !all.slice(0, index).some((other) => (
    other.role === candidate.role && Math.hypot(other.x - candidate.x, other.y - candidate.y) < 2.5
  )));

  const centerlinePixels: number[] = [];
  for (let index = 0; index < area; index += 1) if (centerlineMask[index]) centerlinePixels.push(index);
  const contentCenter = {
    x: (prepared.contentRect.x + prepared.contentRect.width / 2) * TOPOLOGY_FIELD_SIZE / BODY_SIZE,
    y: (prepared.contentRect.y + prepared.contentRect.height / 2) * TOPOLOGY_FIELD_SIZE / BODY_SIZE,
  };
  const centralPixel = centerlinePixels.reduce((best, index) => {
    const score = Math.hypot(index % TOPOLOGY_FIELD_SIZE - contentCenter.x, Math.floor(index / TOPOLOGY_FIELD_SIZE) - contentCenter.y)
      - probability(0, index) * 5;
    return score < best.score ? { index, score } : best;
  }, { index: Math.round(contentCenter.y) * TOPOLOGY_FIELD_SIZE + Math.round(contentCenter.x), score: Infinity }).index;
  if (!junctionCandidates.length) candidates.unshift({
    role: "junction",
    x: centralPixel % TOPOLOGY_FIELD_SIZE,
    y: Math.floor(centralPixel / TOPOLOGY_FIELD_SIZE),
    confidence: probability(1, centralPixel),
  });

  const nearestLinePixel = (x: number, y: number) => centerlinePixels.reduce((best, index) => {
    const distance = Math.hypot(index % TOPOLOGY_FIELD_SIZE - x, Math.floor(index / TOPOLOGY_FIELD_SIZE) - y);
    return distance < best.distance ? { index, distance } : best;
  }, { index: Math.round(y) * TOPOLOGY_FIELD_SIZE + Math.round(x), distance: Infinity }).index;
  const anchors = candidates.map((candidate) => nearestLinePixel(candidate.x, candidate.y));
  const rootIndex = candidates.reduce((best, candidate, index) => {
    if (candidate.role !== "junction") return best;
    const distance = Math.hypot(candidate.x - contentCenter.x, candidate.y - contentCenter.y);
    return distance < best.distance ? { index, distance } : best;
  }, { index: 0, distance: Infinity }).index;
  const nodes: TopologyNode[] = candidates.map((candidate, index) => {
    const point = prepared.mapPoint(
      (candidate.x + 0.5) * BODY_SIZE / TOPOLOGY_FIELD_SIZE,
      (candidate.y + 0.5) * BODY_SIZE / TOPOLOGY_FIELD_SIZE,
    );
    return { id: `topology-${index}`, role: index === rootIndex ? "root" : candidate.role, ...point, confidence: candidate.confidence };
  });

  type CandidatePath = { from: number; to: number; distance: number; confidence: number; indices: number[] };
  const paths: CandidatePath[] = [];
  const directions = [-TOPOLOGY_FIELD_SIZE - 1, -TOPOLOGY_FIELD_SIZE, -TOPOLOGY_FIELD_SIZE + 1, -1, 1, TOPOLOGY_FIELD_SIZE - 1, TOPOLOGY_FIELD_SIZE, TOPOLOGY_FIELD_SIZE + 1];
  for (let from = 0; from < anchors.length; from += 1) {
    const distances = new Int16Array(area);
    distances.fill(-1);
    const previous = new Int16Array(area);
    previous.fill(-1);
    const queue = [anchors[from]];
    distances[anchors[from]] = 0;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const current = queue[cursor];
      const x = current % TOPOLOGY_FIELD_SIZE;
      const y = Math.floor(current / TOPOLOGY_FIELD_SIZE);
      for (const offset of directions) {
        const next = current + offset;
        if (next < 0 || next >= area || distances[next] >= 0 || !centerlineMask[next]) continue;
        const nextX = next % TOPOLOGY_FIELD_SIZE;
        const nextY = Math.floor(next / TOPOLOGY_FIELD_SIZE);
        if (Math.abs(nextX - x) > 1 || Math.abs(nextY - y) > 1) continue;
        distances[next] = distances[current] + 1;
        previous[next] = current;
        queue.push(next);
      }
    }
    for (let to = from + 1; to < anchors.length; to += 1) {
      let current = anchors[to];
      const indices: number[] = [];
      if (distances[current] >= 0) {
        while (current >= 0 && current !== anchors[from] && indices.length < area) {
          indices.push(current);
          current = previous[current];
        }
        indices.push(anchors[from]);
        indices.reverse();
      } else {
        indices.push(anchors[from], anchors[to]);
      }
      paths.push({
        from,
        to,
        distance: distances[anchors[to]] >= 0 ? distances[anchors[to]] : Math.hypot(candidates[from].x - candidates[to].x, candidates[from].y - candidates[to].y) * 4,
        confidence: (candidates[from].confidence + candidates[to].confidence) / 2 * (distances[anchors[to]] >= 0 ? 1 : 0.55),
        indices,
      });
    }
  }
  paths.sort((a, b) => a.distance - b.distance);
  const parents = nodes.map((_, index) => index);
  const find = (index: number): number => parents[index] === index ? index : (parents[index] = find(parents[index]));
  const edges: LearnedTopology["edges"] = [];
  for (const candidate of paths) {
    const left = find(candidate.from);
    const right = find(candidate.to);
    if (left === right) continue;
    parents[left] = right;
    edges.push({
      id: `topology-edge-${edges.length}`,
      from: nodes[candidate.from].id,
      to: nodes[candidate.to].id,
      confidence: candidate.confidence,
      path: candidate.indices.filter((_, index) => index % 2 === 0 || index === candidate.indices.length - 1).map((pixel) => prepared.mapPoint(
        (pixel % TOPOLOGY_FIELD_SIZE + 0.5) * BODY_SIZE / TOPOLOGY_FIELD_SIZE,
        (Math.floor(pixel / TOPOLOGY_FIELD_SIZE) + 0.5) * BODY_SIZE / TOPOLOGY_FIELD_SIZE,
      )),
    });
    if (edges.length === nodes.length - 1) break;
  }
  const classProbabilities = softmax(classOutput.data as Float32Array);
  const kindIndex = classProbabilities.indexOf(Math.max(...classProbabilities));
  const fieldConfidence = centerlinePixels.length
    ? centerlinePixels.reduce((total, index) => total + probability(1, index), 0) / centerlinePixels.length
    : 0;
  return {
    model: "wallalive-topology-v10",
    latencyMs,
    kind: TOPOLOGY_CLASSES[Math.max(0, kindIndex)],
    kindConfidence: classProbabilities[Math.max(0, kindIndex)] ?? 0,
    fieldConfidence,
    applicable: nodes.length >= 2 && edges.length >= 1 && fieldConfidence >= 0.48,
    nodes,
    edges,
  };
}

function decodePose(
  output: import("onnxruntime-web/wasm").Tensor,
  prepared: PreparedImage,
  limbHints: LearnedPartHint[],
  latencyMs: number,
  topology: LearnedTopology,
): LearnedPose {
  const values = output.data as Float32Array;
  const area = POSE_HEATMAP_SIZE * POSE_HEATMAP_SIZE;
  const joints = POSE_JOINT_NAMES.map((name, channel) => {
    const offset = channel * area;
    let bestIndex = 0;
    let bestLogit = -Infinity;
    for (let index = 0; index < area; index += 1) {
      if (values[offset + index] > bestLogit) {
        bestLogit = values[offset + index];
        bestIndex = index;
      }
    }
    const x = bestIndex % POSE_HEATMAP_SIZE;
    const y = Math.floor(bestIndex / POSE_HEATMAP_SIZE);
    const point = prepared.mapPoint((x + 0.5) * 2, (y + 0.5) * 2);
    return { name, ...point, confidence: sigmoid(bestLogit) };
  });
  const byName = new Map(joints.map((joint) => [joint.name, joint]));
  const averageY = (...names: Array<(typeof POSE_JOINT_NAMES)[number]>) => names.reduce((total, name) => total + (byName.get(name)?.y ?? 0.5), 0) / names.length;
  const arms = limbHints.filter((hint) => hint.kind === "arm").length;
  const legs = limbHints.filter((hint) => hint.kind === "leg").length;
  const orderedHumanoid = averageY("left_shoulder", "right_shoulder") < averageY("left_hip", "right_hip")
    && averageY("left_hip", "right_hip") < averageY("left_knee", "right_knee")
    && averageY("left_knee", "right_knee") < averageY("left_ankle", "right_ankle");
  const applicable = topology.kind === "biped" && topology.kindConfidence >= 0.42
    && arms >= 1 && arms <= 2 && legs >= 1 && legs <= 2 && orderedHumanoid;
  return { model: "wallalive-amateur-pose-v6", latencyMs, applicable, joints };
}

function modelRectToImageCrop(rect: { x: number; y: number; width: number; height: number }, mapPoint: PointMap) {
  const topLeft = mapPoint(rect.x, rect.y);
  const bottomRight = mapPoint(rect.x + rect.width, rect.y + rect.height);
  return {
    x: topLeft.x,
    y: topLeft.y,
    width: Math.max(0.01, bottomRight.x - topLeft.x),
    height: Math.max(0.01, bottomRight.y - topLeft.y),
  };
}

function resizeLogitsBilinear(values: Float32Array, sourceSize: number, targetSize: number, channels: number) {
  if (sourceSize === targetSize) return values;
  const resized = new Float32Array(channels * targetSize * targetSize);
  const sourceArea = sourceSize * sourceSize;
  const targetArea = targetSize * targetSize;
  for (let channel = 0; channel < channels; channel += 1) {
    const sourceOffset = channel * sourceArea;
    const targetOffset = channel * targetArea;
    for (let targetY = 0; targetY < targetSize; targetY += 1) {
      const sourceY = (targetY + 0.5) * sourceSize / targetSize - 0.5;
      const rawY0 = Math.floor(sourceY);
      const y0 = clamp(rawY0, 0, sourceSize - 1);
      const y1 = clamp(rawY0 + 1, 0, sourceSize - 1);
      const fractionY = sourceY - rawY0;
      for (let targetX = 0; targetX < targetSize; targetX += 1) {
        const sourceX = (targetX + 0.5) * sourceSize / targetSize - 0.5;
        const rawX0 = Math.floor(sourceX);
        const x0 = clamp(rawX0, 0, sourceSize - 1);
        const x1 = clamp(rawX0 + 1, 0, sourceSize - 1);
        const fractionX = sourceX - rawX0;
        const top = values[sourceOffset + y0 * sourceSize + x0] * (1 - fractionX) + values[sourceOffset + y0 * sourceSize + x1] * fractionX;
        const bottom = values[sourceOffset + y1 * sourceSize + x0] * (1 - fractionX) + values[sourceOffset + y1 * sourceSize + x1] * fractionX;
        resized[targetOffset + targetY * targetSize + targetX] = top * (1 - fractionY) + bottom * fractionY;
      }
    }
  }
  return resized;
}

function blendFaceLogits(v3Output: import("onnxruntime-web/wasm").Tensor, v4Output: import("onnxruntime-web/wasm").Tensor) {
  const v3 = resizeLogitsBilinear(v3Output.data as Float32Array, FACE_V3_SIZE, FACE_SIZE, FACE_PARTS.length);
  const v4 = v4Output.data as Float32Array;
  const area = FACE_SIZE * FACE_SIZE;
  const blended = new Float32Array(FACE_PARTS.length * area);
  for (let channel = 0; channel < FACE_PARTS.length; channel += 1) {
    const weight = FACE_V4_BLEND[FACE_PARTS[channel]];
    const offset = channel * area;
    for (let index = 0; index < area; index += 1) {
      blended[offset + index] = v3[offset + index] * (1 - weight) + v4[offset + index] * weight;
    }
  }
  return { blended, v3, v4 };
}

function supplementHints(primary: LearnedPartHint[], fallback: LearnedPartHint[], onlyMissingKinds: boolean) {
  const combined = [...primary];
  const missingKinds = new Set(fallback.map((hint) => hint.kind).filter((kind) => !primary.some((candidate) => candidate.kind === kind)));
  for (const hint of fallback) {
    if (onlyMissingKinds && !missingKinds.has(hint.kind)) continue;
    const sameKind = combined.filter((candidate) => candidate.kind === hint.kind);
    if (sameKind.length >= maximumInstances(hint.kind)) continue;
    if (sameKind.some((candidate) => sameSemanticInstance(candidate, hint))) continue;
    combined.push({ ...hint, confidence: hint.confidence * 0.90 });
  }
  return combined;
}

function supplementFallbackHints(primary: LearnedPartHint[], fallback: LearnedPartHint[]) {
  const minimumInstances: Partial<Record<LearnedPartHint["kind"], number>> = {
    eye: 2,
    mouth: 1,
    arm: 2,
    hand: 2,
    leg: 2,
    foot: 2,
  };
  let combined = [...primary];
  for (const kind of ["eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot"] as const) {
    const needed = minimumInstances[kind] ?? 1;
    if (combined.filter((hint) => hint.kind === kind).length >= needed) continue;
    combined = supplementHints(combined, fallback.filter((hint) => hint.kind === kind), false);
  }
  return combined;
}

export async function recognizeDrawingParts(extraction: DrawingExtraction): Promise<DrawingExtraction> {
  const started = performance.now();
  const [{ ort, body, faceV3, faceV4, pose, topology, depth }, image] = await Promise.all([loadSessions(), loadImage(extraction.textureUrl)]);
  const prepared = prepareImage(image, BODY_SIZE);
  const bodyTensor = new ort.Tensor("float32", prepared.values, [1, 3, BODY_SIZE, BODY_SIZE]);
  const depthValues = prepareDepthImage(image);
  const depthTensor = new ort.Tensor("float32", depthValues, [1, 3, DEPTH_SIZE, DEPTH_SIZE]);
  const poseStarted = performance.now();
  const depthStarted = performance.now();
  const [bodyResults, poseResults, topologyResults, depthResults] = await Promise.all([
    body.run({ pixel_values: bodyTensor }),
    pose.run({ pose_values: bodyTensor }),
    topology.run({ topology_values: bodyTensor }),
    depth.run({ sketch_values: depthTensor }),
  ]);
  const learnedDepth = decodeDepth(
    depthResults.front_back_depth ?? Object.values(depthResults)[0],
    Math.round(performance.now() - depthStarted),
  );
  const partOutput = bodyResults.part_logits ?? Object.values(bodyResults)[0];
  const coarseOutput = bodyResults.coarse_logits ?? Object.values(bodyResults)[1];
  const fullHints = decodeHints(partOutput, BODY_SIZE, PARTS, BODY_THRESHOLDS, prepared.values, prepared.mapPoint, { skipBody: true });
  let learnedTopology = decodeTopology(
    topologyResults.topology_fields ?? Object.values(topologyResults)[0],
    topologyResults.topology_logits ?? Object.values(topologyResults)[1],
    prepared,
    Math.round(performance.now() - poseStarted),
  );
  let learnedPose = decodePose(
    poseResults.joint_heatmaps ?? Object.values(poseResults)[0],
    prepared,
    fullHints,
    Math.round(performance.now() - poseStarted),
    learnedTopology,
  );

  const faceCrop = modelRectToImageCrop(locateHead(coarseOutput, prepared), prepared.mapPoint);
  const preparedFaceV3 = prepareImage(image, FACE_V3_SIZE, faceCrop);
  const preparedFace = prepareImage(image, FACE_SIZE, faceCrop);
  const [faceV3Results, faceV4Results] = await Promise.all([
    faceV3.run({ face_values: new ort.Tensor("float32", preparedFaceV3.values, [1, 3, FACE_V3_SIZE, FACE_V3_SIZE]) }),
    faceV4.run({ face_values: new ort.Tensor("float32", preparedFace.values, [1, 3, FACE_SIZE, FACE_SIZE]) }),
  ]);
  const alignedFaceLogits = blendFaceLogits(
    faceV3Results.face_logits ?? Object.values(faceV3Results)[0],
    faceV4Results.face_logits ?? Object.values(faceV4Results)[0],
  );
  const faceLogits = new ort.Tensor("float32", alignedFaceLogits.blended, [1, FACE_PARTS.length, FACE_SIZE, FACE_SIZE]);
  const faceHints = decodeHints(faceLogits, FACE_SIZE, FACE_PARTS, FACE_THRESHOLDS, preparedFace.values, preparedFace.mapPoint, {
    skipBody: false,
    faceCrop: true,
    componentGate: { v3Logits: alignedFaceLogits.v3, v4Logits: alignedFaceLogits.v4 },
  });
  let hints = [...fullHints.filter((hint) => !FACE_KINDS.has(hint.kind)), ...faceHints];
  // The enlarged crop has better detail. The whole-character pass may restore
  // a missing bilateral eye or missing mouth, but must not append a third eye
  // after the face crop already found a pair (striped cheeks can look eye-like
  // at 96 px). Uncalibrated whole-image cheek/ear masks never bypass the gate.
  if (hints.filter((hint) => hint.kind === "eye").length < 2) {
    hints = supplementHints(hints, fullHints.filter((hint) => hint.kind === "eye"), false);
  }
  if (!hints.some((hint) => hint.kind === "mouth")) {
    hints = supplementHints(hints, fullHints.filter((hint) => hint.kind === "mouth"), false);
  }

  // Keep the legacy ensemble off the mobile critical path. It is downloaded
  // and executed only when v3 cannot find a basic face/limb anchor.
  const anchorKinds: LearnedPartHint["kind"][] = ["eye", "mouth", "arm", "leg"];
  if (anchorKinds.some((kind) => !hints.some((hint) => hint.kind === kind))) {
    const fallbackPrepared = prepareImage(image, FALLBACK_SIZE, undefined, true);
    const fallbackTensor = new ort.Tensor("float32", fallbackPrepared.values, [1, 3, FALLBACK_SIZE, FALLBACK_SIZE]);
    const fallbacks = await loadFallbackSessions(ort);
    const fallbackResults = await Promise.all(fallbacks.map((session) => session.run({ pixel_values: fallbackTensor })));
    let oldHints: LearnedPartHint[] = [];
    for (const results of fallbackResults) {
      oldHints = supplementHints(oldHints, decodeHints(
        results.part_logits ?? Object.values(results)[0],
        FALLBACK_SIZE,
        PARTS,
        FALLBACK_THRESHOLDS,
        fallbackPrepared.values,
        fallbackPrepared.mapPoint,
        { skipBody: true },
      ), false);
    }
    // Older models can restore a missing bilateral mate, but stop at the normal
    // pair count so they cannot overwhelm unusual v3 multi-part predictions.
    hints = supplementFallbackHints(hints, oldHints);
  }

  // Avoid a self-reinforcing topology error: a few false vertical limb blobs
  // can make an upright round person read as a quadruped, which would then
  // authorize four legs and erase its arms. A compact upright silhouette with
  // a conventional two-eye face and mouth is stronger posture evidence.
  const uprightFace = extraction.analysis.aspectRatio <= 1.06
    && hints.filter((hint) => hint.kind === "eye").length >= 2
    && hints.some((hint) => hint.kind === "mouth");
  if (learnedTopology.kind === "quadruped" && uprightFace) {
    learnedTopology = {
      ...learnedTopology,
      kind: "biped",
      kindConfidence: Math.min(learnedTopology.kindConfidence, 0.78),
      applicable: true,
    };
    learnedPose = decodePose(
      poseResults.joint_heatmaps ?? Object.values(poseResults)[0],
      prepared,
      hints,
      Math.round(performance.now() - poseStarted),
      learnedTopology,
    );
  }

  const instanceLimits: Partial<Record<LearnedPartHint["kind"], number>> = learnedTopology.kind === "biped"
    ? { eye: 2, cheek: 2, mouth: 1, ear: 2, arm: 2, hand: 2, leg: 2, foot: 2 }
    : { eye: 2, cheek: 2, mouth: 1, ear: 2 };
  hints = hints
    .sort((left, right) => right.confidence - left.confidence)
    .filter((hint, index, all) => {
      const limit = instanceLimits[hint.kind];
      return limit === undefined || all.slice(0, index).filter((candidate) => candidate.kind === hint.kind).length < limit;
    });
  return {
    ...mergeLearnedPartHints(extraction, hints, Math.round(performance.now() - started), learnedPose, learnedTopology),
    depthRecognition: learnedDepth,
  };
}

export async function recognizeDrawingFromVideo(video: HTMLVideoElement, target: CaptureTarget): Promise<DrawingExtraction> {
  return recognizeDrawingParts(await isolateDrawingFromVideo(video, target));
}

export async function recognizeDrawingFromImageUrl(imageUrl: string, target: CaptureTarget = { x: 0.5, y: 0.5 }): Promise<DrawingExtraction> {
  return recognizeDrawingParts(await isolateDrawingFromImageUrl(imageUrl, target));
}
