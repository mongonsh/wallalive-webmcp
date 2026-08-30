import { mergeLearnedPartHints, type DrawingExtraction, type LearnedPartHint } from "./drawing";

const MODEL_PATH = "/models/wallalive-parts-v1.onnx";
const MODEL_SIZE = 64;
const PARTS = ["body", "eye", "cheek", "mouth", "ear", "arm", "hand", "leg", "foot"] as const;
const THRESHOLDS: Record<(typeof PARTS)[number], number> = {
  body: 0.48,
  eye: 0.54,
  cheek: 0.52,
  mouth: 0.50,
  ear: 0.52,
  arm: 0.49,
  hand: 0.52,
  leg: 0.49,
  foot: 0.52,
};

type OrtRuntime = typeof import("onnxruntime-web/wasm");
let runtimePromise: Promise<OrtRuntime> | null = null;
let sessionPromise: Promise<{ ort: OrtRuntime; session: import("onnxruntime-web/wasm").InferenceSession }> | null = null;

function loadSession() {
  if (!sessionPromise) {
    runtimePromise ??= import("onnxruntime-web/wasm");
    sessionPromise = runtimePromise.then(async (ort) => {
      ort.env.wasm.numThreads = typeof crossOriginIsolated !== "undefined" && crossOriginIsolated ? 2 : 1;
      ort.env.wasm.proxy = false;
      const session = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
        executionMode: "sequential",
      });
      return { ort, session };
    });
  }
  return sessionPromise;
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

function sigmoid(value: number) {
  return 1 / (1 + Math.exp(-value));
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

function components(mask: Uint8Array) {
  const visited = new Uint8Array(mask.length);
  const found: PixelComponent[] = [];
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;
    const queue = [start];
    const pixels: number[] = [];
    visited[start] = 1;
    let minX = MODEL_SIZE;
    let minY = MODEL_SIZE;
    let maxX = 0;
    let maxY = 0;
    let sumX = 0;
    let sumY = 0;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const index = queue[cursor];
      const x = index % MODEL_SIZE;
      const y = Math.floor(index / MODEL_SIZE);
      pixels.push(index);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      sumX += x;
      sumY += y;
      for (let oy = -1; oy <= 1; oy += 1) {
        for (let ox = -1; ox <= 1; ox += 1) {
          if (!ox && !oy) continue;
          const nx = x + ox;
          const ny = y + oy;
          if (nx < 0 || ny < 0 || nx >= MODEL_SIZE || ny >= MODEL_SIZE) continue;
          const next = ny * MODEL_SIZE + nx;
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

function describeComponent(kind: LearnedPartHint["kind"], component: PixelComponent, probabilities: Float32Array, channelOffset: number): LearnedPartHint {
  let confidence = 0;
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const index of component.pixels) {
    confidence += probabilities[channelOffset + index];
    const x = index % MODEL_SIZE - component.centerX;
    const y = Math.floor(index / MODEL_SIZE) - component.centerY;
    xx += x * x;
    yy += y * y;
    xy += x * y;
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  const axis = { x: Math.cos(angle), y: Math.sin(angle) };
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const index of component.pixels) {
    const x = index % MODEL_SIZE - component.centerX;
    const y = Math.floor(index / MODEL_SIZE) - component.centerY;
    const projection = x * axis.x + y * axis.y;
    minimum = Math.min(minimum, projection);
    maximum = Math.max(maximum, projection);
  }
  const normalize = (x: number, y: number) => ({ x: (x + 0.5) / MODEL_SIZE, y: (y + 0.5) / MODEL_SIZE });
  return {
    kind,
    center: normalize(component.centerX, component.centerY),
    size: {
      x: (component.maxX - component.minX + 1) / MODEL_SIZE,
      y: (component.maxY - component.minY + 1) / MODEL_SIZE,
    },
    endpoints: [
      normalize(component.centerX + axis.x * minimum, component.centerY + axis.y * minimum),
      normalize(component.centerX + axis.x * maximum, component.centerY + axis.y * maximum),
    ],
    rotation: angle,
    confidence: confidence / component.pixels.length,
  };
}

function decodeHints(output: import("onnxruntime-web/wasm").Tensor) {
  const probabilities = output.data as Float32Array;
  const area = MODEL_SIZE * MODEL_SIZE;
  const hints: LearnedPartHint[] = [];
  for (let channel = 1; channel < PARTS.length; channel += 1) {
    const kind = PARTS[channel] as LearnedPartHint["kind"];
    const offset = channel * area;
    const mask = new Uint8Array(area);
    for (let index = 0; index < area; index += 1) mask[index] = sigmoid(probabilities[offset + index]) >= THRESHOLDS[kind] ? 1 : 0;
    const minimumArea = kind === "arm" || kind === "leg" ? 5 : kind === "eye" || kind === "mouth" ? 3 : 4;
    const maximumArea = kind === "ear" ? area * 0.12 : area * 0.07;
    const candidates = components(mask)
      .filter((component) => component.pixels.length >= minimumArea && component.pixels.length <= maximumArea)
      .sort((a, b) => b.pixels.length - a.pixels.length)
      .slice(0, kind === "eye" ? 3 : kind === "mouth" ? 1 : 2);
    for (const component of candidates) hints.push(describeComponent(kind, component, probabilities, offset));
  }
  return hints;
}

export async function recognizeDrawingParts(extraction: DrawingExtraction): Promise<DrawingExtraction> {
  const started = performance.now();
  const [{ ort, session }, image] = await Promise.all([loadSession(), loadImage(extraction.textureUrl)]);
  const canvas = document.createElement("canvas");
  canvas.width = MODEL_SIZE;
  canvas.height = MODEL_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(image, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const rgba = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const area = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(area * 3);
  for (let index = 0; index < area; index += 1) {
    input[index] = rgba[index * 4] / 255;
    input[area + index] = rgba[index * 4 + 1] / 255;
    input[area * 2 + index] = rgba[index * 4 + 2] / 255;
  }
  const results = await session.run({ pixel_values: new ort.Tensor("float32", input, [1, 3, MODEL_SIZE, MODEL_SIZE]) });
  const output = results.part_logits ?? Object.values(results)[0];
  const hints = decodeHints(output);
  return mergeLearnedPartHints(extraction, hints, Math.round(performance.now() - started));
}
