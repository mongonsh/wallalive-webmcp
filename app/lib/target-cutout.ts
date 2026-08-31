import {
  extractDrawingFromCanvas,
  hasMeaningfulSelectedAlpha,
  mapCoverTargetToSource,
  type CaptureTarget,
  type DrawingExtraction,
  type ExtractionScope,
} from "./drawing.ts";

const MODEL_PATH = "/models/wallalive-target-cutout-v2.onnx";
const MODEL_SIZE = 128;
const MASK_THRESHOLD = 0.54;
const CROP_SCALES = [0.46, 0.62, 0.8, 1] as const;

type SourceFrame = {
  canvas: HTMLCanvasElement;
  target: CaptureTarget;
  scope: ExtractionScope;
};

type Candidate = {
  crop: { x: number; y: number; size: number };
  scale: number;
  prompt: CaptureTarget;
  input: Float32Array;
};

type DecodedMask = {
  candidate: Candidate;
  mask: Uint8Array;
  confidence: number;
  areaPercent: number;
  score: number;
};

type OrtRuntime = typeof import("onnxruntime-web/wasm");
type OrtSession = import("onnxruntime-web/wasm").InferenceSession;

let runtimePromise: Promise<OrtRuntime> | null = null;
let sessionPromise: Promise<{ ort: OrtRuntime; session: OrtSession }> | null = null;

const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

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

function sourceCanvas(source: CanvasImageSource, width: number, height: number) {
  const maximum = 1280;
  const scale = Math.min(1, maximum / Math.max(width, height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(width * scale));
  canvas.height = Math.max(1, Math.round(height * scale));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function loadImage(url: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.decoding = "async";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("That image could not be opened."));
    image.src = url;
  });
}

function cropFor(canvas: HTMLCanvasElement, target: CaptureTarget, scale: number) {
  const base = Math.min(canvas.width, canvas.height);
  const size = Math.min(Math.max(96, base * scale), Math.max(canvas.width, canvas.height));
  const targetX = target.x * canvas.width;
  const targetY = target.y * canvas.height;
  const x = clamp(targetX - size / 2, 0, Math.max(0, canvas.width - size));
  const y = clamp(targetY - size / 2, 0, Math.max(0, canvas.height - size));
  return { x, y, size };
}

function prepareCandidate(canvas: HTMLCanvasElement, target: CaptureTarget, scale: number): Candidate {
  const crop = cropFor(canvas, target, scale);
  const prepared = document.createElement("canvas");
  prepared.width = MODEL_SIZE;
  prepared.height = MODEL_SIZE;
  const context = prepared.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(canvas, crop.x, crop.y, crop.size, crop.size, 0, 0, MODEL_SIZE, MODEL_SIZE);
  const rgba = context.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
  const area = MODEL_SIZE * MODEL_SIZE;
  const input = new Float32Array(area * 4);
  const prompt = {
    x: clamp((target.x * canvas.width - crop.x) / crop.size, 0, 1),
    y: clamp((target.y * canvas.height - crop.y) / crop.size, 0, 1),
  };
  const sigma = MODEL_SIZE * 0.055;
  const promptX = prompt.x * (MODEL_SIZE - 1);
  const promptY = prompt.y * (MODEL_SIZE - 1);
  for (let index = 0; index < area; index += 1) {
    input[index] = rgba[index * 4] / 255;
    input[area + index] = rgba[index * 4 + 1] / 255;
    input[area * 2 + index] = rgba[index * 4 + 2] / 255;
    const x = index % MODEL_SIZE;
    const y = Math.floor(index / MODEL_SIZE);
    input[area * 3 + index] = Math.exp(-((x - promptX) ** 2 + (y - promptY) ** 2) / (2 * sigma ** 2));
  }
  return { crop, scale, prompt, input };
}

function promptedComponent(probabilities: Float32Array, prompt: CaptureTarget) {
  const area = MODEL_SIZE * MODEL_SIZE;
  const active = new Uint8Array(area);
  for (let index = 0; index < area; index += 1) active[index] = probabilities[index] >= MASK_THRESHOLD ? 1 : 0;
  const promptX = Math.round(prompt.x * (MODEL_SIZE - 1));
  const promptY = Math.round(prompt.y * (MODEL_SIZE - 1));
  let seed = promptY * MODEL_SIZE + promptX;
  if (!active[seed]) {
    let bestDistance = Infinity;
    let bestProbability = 0;
    for (let y = Math.max(0, promptY - 15); y <= Math.min(MODEL_SIZE - 1, promptY + 15); y += 1) {
      for (let x = Math.max(0, promptX - 15); x <= Math.min(MODEL_SIZE - 1, promptX + 15); x += 1) {
        const index = y * MODEL_SIZE + x;
        if (!active[index]) continue;
        const distance = Math.hypot(x - promptX, y - promptY);
        if (distance < bestDistance || (distance === bestDistance && probabilities[index] > bestProbability)) {
          seed = index;
          bestDistance = distance;
          bestProbability = probabilities[index];
        }
      }
    }
    if (bestDistance === Infinity) return null;
  }
  const mask = new Uint8Array(area);
  const queue = new Int32Array(area);
  let head = 0;
  let tail = 0;
  queue[tail++] = seed;
  mask[seed] = 1;
  let sum = 0;
  let pixels = 0;
  let edgePixels = 0;
  let leftEdge = false;
  let rightEdge = false;
  let topEdge = false;
  let bottomEdge = false;
  while (head < tail) {
    const index = queue[head++];
    const x = index % MODEL_SIZE;
    const y = Math.floor(index / MODEL_SIZE);
    pixels += 1;
    sum += probabilities[index];
    if (x <= 1 || y <= 1 || x >= MODEL_SIZE - 2 || y >= MODEL_SIZE - 2) edgePixels += 1;
    if (x <= 1) leftEdge = true;
    if (x >= MODEL_SIZE - 2) rightEdge = true;
    if (y <= 1) topEdge = true;
    if (y >= MODEL_SIZE - 2) bottomEdge = true;
    for (const next of [index - 1, index + 1, index - MODEL_SIZE, index + MODEL_SIZE]) {
      if (next < 0 || next >= area || mask[next] || !active[next]) continue;
      const nextX = next % MODEL_SIZE;
      if (Math.abs(nextX - x) > 1) continue;
      mask[next] = 1;
      queue[tail++] = next;
    }
  }
  return {
    mask,
    pixels,
    mean: sum / Math.max(1, pixels),
    edgeFraction: edgePixels / Math.max(1, pixels),
    edgeSides: [leftEdge, rightEdge, topEdge, bottomEdge].filter(Boolean).length,
  };
}

function decodeCandidate(candidate: Candidate, logits: Float32Array, offset: number): DecodedMask | null {
  const area = MODEL_SIZE * MODEL_SIZE;
  const probabilities = new Float32Array(area);
  for (let index = 0; index < area; index += 1) probabilities[index] = sigmoid(logits[offset + index]);
  const component = promptedComponent(probabilities, candidate.prompt);
  if (!component) return null;
  const areaFraction = component.pixels / area;
  if (areaFraction < 0.025 || areaFraction > 0.78) return null;
  // Torn paper and notebook borders form a dominant closed component that
  // commonly spans three sides of a crop. Refuse it even at high confidence;
  // a complete prompted character should fit inside at least one crop scale.
  if (component.edgeSides >= 3 && areaFraction > 0.12) return null;
  const centerIndex = Math.round(candidate.prompt.y * (MODEL_SIZE - 1)) * MODEL_SIZE + Math.round(candidate.prompt.x * (MODEL_SIZE - 1));
  const promptConfidence = probabilities[centerIndex];
  const areaScore = 1 - Math.min(1, Math.abs(areaFraction - 0.3) / 0.48);
  const edgePenalty = Math.min(0.36, component.edgeFraction * 4.5 + component.edgeSides * 0.035);
  const score = component.mean * 0.48 + promptConfidence * 0.28 + areaScore * 0.24 - edgePenalty;
  return {
    candidate,
    mask: component.mask,
    confidence: component.mean * 0.65 + promptConfidence * 0.35,
    areaPercent: areaFraction * 100,
    score,
  };
}

function sourceCrop(frame: SourceFrame, decoded: DecodedMask, applyMask: boolean) {
  const { crop } = decoded.candidate;
  const size = Math.min(720, Math.max(256, Math.round(crop.size)));
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.drawImage(frame.canvas, crop.x, crop.y, crop.size, crop.size, 0, 0, size, size);
  if (applyMask) {
    const image = context.getImageData(0, 0, size, size);
    for (let y = 0; y < size; y += 1) {
      const maskY = Math.min(MODEL_SIZE - 1, Math.floor(y / size * MODEL_SIZE));
      for (let x = 0; x < size; x += 1) {
        const maskX = Math.min(MODEL_SIZE - 1, Math.floor(x / size * MODEL_SIZE));
        image.data[(y * size + x) * 4 + 3] = decoded.mask[maskY * MODEL_SIZE + maskX] ? 255 : 0;
      }
    }
    context.putImageData(image, 0, 0);
  }
  return canvas;
}

async function isolate(frame: SourceFrame): Promise<DrawingExtraction> {
  const started = performance.now();
  if (frame.scope === "selected-image") {
    const context = frame.canvas.getContext("2d", { willReadFrequently: true });
    if (context && hasMeaningfulSelectedAlpha(context.getImageData(0, 0, frame.canvas.width, frame.canvas.height).data, frame.scope)) {
      return extractDrawingFromCanvas(frame.canvas, frame.target, "selected-image");
    }
  }
  const candidates = CROP_SCALES.map((scale) => prepareCandidate(frame.canvas, frame.target, scale));
  const area = MODEL_SIZE * MODEL_SIZE;
  const values = new Float32Array(candidates.length * area * 4);
  candidates.forEach((candidate, batch) => values.set(candidate.input, batch * area * 4));
  const { ort, session } = await loadSession();
  const results = await session.run({ prompted_image: new ort.Tensor("float32", values, [candidates.length, 4, MODEL_SIZE, MODEL_SIZE]) });
  const logits = (results.target_mask ?? Object.values(results)[0]).data as Float32Array;
  const decoded = candidates
    .map((candidate, index) => decodeCandidate(candidate, logits, index * area))
    .filter((candidate): candidate is DecodedMask => Boolean(candidate))
    .sort((left, right) => right.score - left.score)[0];
  if (!decoded || decoded.confidence < 0.56 || decoded.score < 0.5) {
    throw new Error("I can’t separate one complete character yet. Move closer, tap inside its body, and keep other drawings outside the frame.");
  }
  // The learned mask decides which scale contains the prompted character. On
  // line drawings, rerun the exact closed-outline extractor inside that tight
  // crop first: disconnected labels and neighboring doodles then cannot ride
  // a soft neural bridge into the silhouette. Filled/painted characters fall
  // back to the learned alpha mask.
  let extraction: DrawingExtraction;
  try {
    extraction = extractDrawingFromCanvas(sourceCrop(frame, decoded, false), decoded.candidate.prompt, "camera");
  } catch {
    extraction = extractDrawingFromCanvas(sourceCrop(frame, decoded, true), decoded.candidate.prompt, "selected-image");
  }
  return {
    ...extraction,
    previewUrl: frame.canvas.toDataURL("image/jpeg", 0.86),
    sourceTarget: frame.target,
    sourceScope: frame.scope,
    cutoutRecognition: {
      model: "wallalive-target-cutout-v2",
      latencyMs: Math.round(performance.now() - started),
      confidence: Number(decoded.confidence.toFixed(3)),
      areaPercent: Number(decoded.areaPercent.toFixed(1)),
      cropScale: decoded.candidate.scale,
    },
  };
}

export async function isolateDrawingFromVideo(video: HTMLVideoElement, target: CaptureTarget): Promise<DrawingExtraction> {
  if (!video.videoWidth || !video.videoHeight) throw new Error("The camera is still focusing. Try again in a moment.");
  const bounds = video.getBoundingClientRect();
  const mappedTarget = mapCoverTargetToSource(target, video.videoWidth, video.videoHeight, bounds.width || video.clientWidth, bounds.height || video.clientHeight);
  return isolate({ canvas: sourceCanvas(video, video.videoWidth, video.videoHeight), target: mappedTarget, scope: "camera" });
}

export async function isolateDrawingFromImageUrl(imageUrl: string, target: CaptureTarget = { x: 0.5, y: 0.5 }): Promise<DrawingExtraction> {
  const image = await loadImage(imageUrl);
  return isolate({ canvas: sourceCanvas(image, image.naturalWidth, image.naturalHeight), target, scope: "selected-image" });
}
