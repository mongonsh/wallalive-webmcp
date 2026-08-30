import { mergeLearnedPartHints, type DrawingExtraction, type LearnedPartHint } from "./drawing";
import { averageLogitConfidence } from "./model-math";

const BODY_MODEL_PATH = "/models/wallalive-parts-v3.onnx";
const FACE_MODEL_PATH = "/models/wallalive-face-v3.onnx";
const FALLBACK_MODEL_PATHS = ["/models/wallalive-parts-v2.onnx", "/models/wallalive-parts-v1.onnx"] as const;
const BODY_SIZE = 96;
const FACE_SIZE = 96;
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
  eye: 0.72,
  cheek: 0.24,
  mouth: 0.72,
  ear: 0.64,
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
let sessionPromise: Promise<{ ort: OrtRuntime; body: OrtSession; face: OrtSession }> | null = null;
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
      const [body, face] = await Promise.all([
        ort.InferenceSession.create(BODY_MODEL_PATH, sessionOptions),
        ort.InferenceSession.create(FACE_MODEL_PATH, sessionOptions),
      ]);
      return { ort, body, face };
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

function describeComponent(kind: LearnedPartHint["kind"], component: PixelComponent, logits: Float32Array, channelOffset: number, size: number, mapPoint: PointMap): LearnedPartHint {
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
    rotation: angle,
    confidence: averageLogitConfidence(logits, component.pixels, channelOffset),
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
  mapPoint: PointMap,
  options: { skipBody: boolean; faceCrop?: boolean },
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
    for (const component of candidates) hints.push(describeComponent(kind, component, probabilities, offset, size, mapPoint));
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

function supplementHints(primary: LearnedPartHint[], fallback: LearnedPartHint[], onlyMissingKinds: boolean) {
  const combined = [...primary];
  const missingKinds = new Set(fallback.map((hint) => hint.kind).filter((kind) => !primary.some((candidate) => candidate.kind === kind)));
  for (const hint of fallback) {
    if (onlyMissingKinds && !missingKinds.has(hint.kind)) continue;
    const sameKind = combined.filter((candidate) => candidate.kind === hint.kind);
    if (sameKind.length >= maximumInstances(hint.kind)) continue;
    if (sameKind.some((candidate) => {
      const separation = Math.hypot(candidate.center.x - hint.center.x, candidate.center.y - hint.center.y);
      const sameInstanceReach = Math.max(0.055, Math.min(0.11, Math.max(candidate.size.x, candidate.size.y, hint.size.x, hint.size.y) * 0.58));
      return separation < sameInstanceReach;
    })) continue;
    combined.push({ ...hint, confidence: hint.confidence * 0.90 });
  }
  return combined;
}

function supplementFallbackHints(primary: LearnedPartHint[], fallback: LearnedPartHint[]) {
  const minimumInstances: Partial<Record<LearnedPartHint["kind"], number>> = {
    eye: 2,
    cheek: 2,
    mouth: 1,
    ear: 2,
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
  const [{ ort, body, face }, image] = await Promise.all([loadSessions(), loadImage(extraction.textureUrl)]);
  const prepared = prepareImage(image, BODY_SIZE);
  const bodyTensor = new ort.Tensor("float32", prepared.values, [1, 3, BODY_SIZE, BODY_SIZE]);
  const bodyResults = await body.run({ pixel_values: bodyTensor });
  const partOutput = bodyResults.part_logits ?? Object.values(bodyResults)[0];
  const coarseOutput = bodyResults.coarse_logits ?? Object.values(bodyResults)[1];
  const fullHints = decodeHints(partOutput, BODY_SIZE, PARTS, BODY_THRESHOLDS, prepared.mapPoint, { skipBody: true });

  const faceCrop = modelRectToImageCrop(locateHead(coarseOutput, prepared), prepared.mapPoint);
  const preparedFace = prepareImage(image, FACE_SIZE, faceCrop);
  const faceResults = await face.run({ face_values: new ort.Tensor("float32", preparedFace.values, [1, 3, FACE_SIZE, FACE_SIZE]) });
  const faceHints = decodeHints(faceResults.face_logits ?? Object.values(faceResults)[0], FACE_SIZE, FACE_PARTS, FACE_THRESHOLDS, preparedFace.mapPoint, { skipBody: false, faceCrop: true });
  let hints = [...fullHints.filter((hint) => !FACE_KINDS.has(hint.kind)), ...faceHints];
  // The enlarged crop has better detail; the full-character pass can still
  // contribute a separate second eye/ear/cheek that the crop missed.
  hints = supplementHints(hints, fullHints, false);

  // Keep the legacy ensemble off the mobile critical path. It is downloaded
  // and executed only when v3 cannot find a basic face/limb anchor.
  const anchorKinds: LearnedPartHint["kind"][] = ["eye", "cheek", "mouth", "arm", "leg"];
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
        fallbackPrepared.mapPoint,
        { skipBody: true },
      ), false);
    }
    // Older models can restore a missing bilateral mate, but stop at the normal
    // pair count so they cannot overwhelm unusual v3 multi-part predictions.
    hints = supplementFallbackHints(hints, oldHints);
  }
  return mergeLearnedPartHints(extraction, hints, Math.round(performance.now() - started));
}
