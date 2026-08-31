import {
  extractDrawingFromCanvas,
  hasMeaningfulSelectedAlpha,
  mapCoverTargetToSource,
  type CaptureTarget,
  type DrawingExtraction,
  type ExtractionScope,
} from "./drawing.ts";

const MODEL_PATH = "/models/wallalive-target-cutout-v2.onnx";
const MAGIC_TOUCH_MODEL = "https://storage.googleapis.com/mediapipe-models/interactive_segmenter_v2/magic_touch/int8/1/interactive_segmentation.task";
const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
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
let interactiveSegmenterPromise: Promise<{
  segmenter: import("@mediapipe/tasks-vision").InteractiveSegmenter;
}> | null = null;

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

function loadInteractiveSegmenter() {
  if (!interactiveSegmenterPromise) {
    interactiveSegmenterPromise = import("@mediapipe/tasks-vision").then(async (vision) => {
      const files = await vision.FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      const segmenter = await vision.InteractiveSegmenter.createFromModelPath(files, MAGIC_TOUCH_MODEL);
      return { segmenter };
    }).catch((error) => {
      interactiveSegmenterPromise = null;
      throw error;
    });
  }
  return interactiveSegmenterPromise;
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

type MagicMask = {
  mask: Uint8Array;
  width: number;
  height: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  pixels: number;
  quality: number;
};

function promptedMagicComponent(values: Uint8Array, width: number, height: number, target: CaptureTarget): MagicMask | null {
  const area = width * height;
  if (!area || values.length < area) return null;
  // MediaPipe builds have returned both binary 0/1 masks and 0/255 masks.
  // Detect the scale so a valid 0/1 mask is not silently interpreted as empty.
  let maximumValue = 0;
  for (let index = 0; index < area; index += 1) maximumValue = Math.max(maximumValue, values[index]);
  const activeThreshold = maximumValue <= 1 ? 1 : 128;
  const active = new Uint8Array(area);
  for (let index = 0; index < area; index += 1) active[index] = values[index] >= activeThreshold ? 1 : 0;
  const promptX = Math.round(clamp(target.x, 0, 1) * (width - 1));
  const promptY = Math.round(clamp(target.y, 0, 1) * (height - 1));
  let seed = promptY * width + promptX;
  if (!active[seed]) {
    let bestDistance = Infinity;
    const radius = Math.max(12, Math.round(Math.min(width, height) * 0.09));
    for (let y = Math.max(0, promptY - radius); y <= Math.min(height - 1, promptY + radius); y += 1) {
      for (let x = Math.max(0, promptX - radius); x <= Math.min(width - 1, promptX + radius); x += 1) {
        if (!active[y * width + x]) continue;
        const distance = Math.hypot(x - promptX, y - promptY);
        if (distance < bestDistance) {
          bestDistance = distance;
          seed = y * width + x;
        }
      }
    }
    if (!Number.isFinite(bestDistance)) return null;
  }
  const component = new Uint8Array(area);
  const queue = new Int32Array(area);
  let head = 0;
  let tail = 0;
  let pixels = 0;
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let edgePixels = 0;
  let leftEdge = false;
  let rightEdge = false;
  let topEdge = false;
  let bottomEdge = false;
  queue[tail++] = seed;
  component[seed] = 1;
  while (head < tail) {
    const index = queue[head++];
    const x = index % width;
    const y = Math.floor(index / width);
    pixels += 1;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    if (x <= 1 || y <= 1 || x >= width - 2 || y >= height - 2) edgePixels += 1;
    if (x <= 1) leftEdge = true;
    if (x >= width - 2) rightEdge = true;
    if (y <= 1) topEdge = true;
    if (y >= height - 2) bottomEdge = true;
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (!dx && !dy) continue;
        const nextX = x + dx;
        const nextY = y + dy;
        if (nextX < 0 || nextY < 0 || nextX >= width || nextY >= height) continue;
        const next = nextY * width + nextX;
        if (component[next] || !active[next]) continue;
        component[next] = 1;
        queue[tail++] = next;
      }
    }
  }
  const areaFraction = pixels / area;
  const boxArea = Math.max(1, (maxX - minX + 1) * (maxY - minY + 1));
  const fill = pixels / boxArea;
  const edgeSides = [leftEdge, rightEdge, topEdge, bottomEdge].filter(Boolean).length;
  if (areaFraction < 0.0025 || areaFraction > 0.72 || pixels < 180) return null;
  if (edgeSides >= 3 && areaFraction > 0.08) return null;
  if (maxX - minX < width * 0.035 || maxY - minY < height * 0.035) return null;
  const edgePenalty = Math.min(0.28, edgePixels / Math.max(1, pixels) * 5 + edgeSides * 0.035);
  const sizeScore = 1 - Math.min(1, Math.abs(areaFraction - 0.23) / 0.52);
  const quality = clamp(0.66 + sizeScore * 0.18 + Math.min(0.12, fill * 0.16) - edgePenalty, 0, 0.98);
  return { mask: component, width, height, minX, minY, maxX, maxY, pixels, quality };
}

function cropMagicMask(frame: SourceFrame, decoded: MagicMask) {
  const boxWidth = decoded.maxX - decoded.minX + 1;
  const boxHeight = decoded.maxY - decoded.minY + 1;
  const padding = Math.max(8, Math.round(Math.max(boxWidth, boxHeight) * 0.08));
  const extent = Math.min(Math.min(decoded.width, decoded.height), Math.max(boxWidth, boxHeight) + padding * 2);
  const centerX = (decoded.minX + decoded.maxX) / 2;
  const centerY = (decoded.minY + decoded.maxY) / 2;
  const cropX = clamp(centerX - extent / 2, 0, Math.max(0, decoded.width - extent));
  const cropY = clamp(centerY - extent / 2, 0, Math.max(0, decoded.height - extent));
  const size = Math.min(720, Math.max(256, Math.round(extent)));
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = "high";
  context.drawImage(
    frame.canvas,
    cropX / decoded.width * frame.canvas.width,
    cropY / decoded.height * frame.canvas.height,
    extent / decoded.width * frame.canvas.width,
    extent / decoded.height * frame.canvas.height,
    0,
    0,
    size,
    size,
  );
  const image = context.getImageData(0, 0, size, size);
  for (let y = 0; y < size; y += 1) {
    const maskY = Math.min(decoded.height - 1, Math.max(0, Math.floor(cropY + y / size * extent)));
    for (let x = 0; x < size; x += 1) {
      const maskX = Math.min(decoded.width - 1, Math.max(0, Math.floor(cropX + x / size * extent)));
      const selected = decoded.mask[maskY * decoded.width + maskX];
      let alpha = selected ? 255 : 0;
      if (!selected) {
        let neighbors = 0;
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            const nextX = maskX + dx;
            const nextY = maskY + dy;
            if (nextX >= 0 && nextY >= 0 && nextX < decoded.width && nextY < decoded.height) neighbors += decoded.mask[nextY * decoded.width + nextX];
          }
        }
        if (neighbors) alpha = Math.round(24 + neighbors / 9 * 96);
      }
      image.data[(y * size + x) * 4 + 3] = alpha;
    }
  }
  context.putImageData(image, 0, 0);
  return {
    canvas,
    target: {
      x: clamp((frame.target.x * decoded.width - cropX) / extent, 0, 1),
      y: clamp((frame.target.y * decoded.height - cropY) / extent, 0, 1),
    },
    cropScale: extent / Math.min(decoded.width, decoded.height),
  };
}

async function isolateWithMagicTouch(frame: SourceFrame): Promise<DrawingExtraction> {
  const started = performance.now();
  const { segmenter } = await loadInteractiveSegmenter();
  segmenter.setImage(frame.canvas);
  const result = segmenter.segment([
    // @mediapipe/tasks-vision 1.0.1 declares BrushMode but its ESM bundle does
    // not export the enum object. These are the documented enum wire values.
    { brushMode: 1 as import("@mediapipe/tasks-vision").BrushMode, point: [frame.target], isCompleted: true },
    {
      brushMode: 2 as import("@mediapipe/tasks-vision").BrushMode,
      point: [{ x: 0.01, y: 0.01 }, { x: 0.99, y: 0.01 }, { x: 0.99, y: 0.99 }, { x: 0.01, y: 0.99 }],
      isCompleted: true,
    },
  ]);
  try {
    const decoded = promptedMagicComponent(result.getAsUint8Array(), result.width, result.height, frame.target);
    if (!decoded || decoded.quality < 0.64) throw new Error("The selected character mask is uncertain.");
    const cropped = cropMagicMask(frame, decoded);
    const extraction = extractDrawingFromCanvas(cropped.canvas, cropped.target, "selected-image");
    return {
      ...extraction,
      previewUrl: frame.canvas.toDataURL("image/jpeg", 0.86),
      sourceTarget: frame.target,
      sourceScope: frame.scope,
      cutoutRecognition: {
        model: "mediapipe-magic-touch-v2",
        latencyMs: Math.round(performance.now() - started),
        confidence: Number(decoded.quality.toFixed(3)),
        areaPercent: Number((decoded.pixels / (decoded.width * decoded.height) * 100).toFixed(1)),
        cropScale: Number(cropped.cropScale.toFixed(3)),
      },
    };
  } finally {
    result.close();
  }
}

function isolateWithTargetedLocalExtraction(frame: SourceFrame, started: number): DrawingExtraction {
  let lastError: unknown;
  // This path is intentionally point-local. It rescues faint pencil and thin
  // line art that general object segmentation may consider background, while
  // preventing a remote paper edge or enclosing circle from winning by size.
  for (const scale of [0.42, 0.56] as const) {
    const crop = cropFor(frame.canvas, frame.target, scale);
    const size = Math.min(720, Math.max(256, Math.round(crop.size)));
    const canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Canvas processing is unavailable in this browser.");
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = "high";
    context.drawImage(frame.canvas, crop.x, crop.y, crop.size, crop.size, 0, 0, size, size);
    const prompt = {
      x: clamp((frame.target.x * frame.canvas.width - crop.x) / crop.size, 0, 1),
      y: clamp((frame.target.y * frame.canvas.height - crop.y) / crop.size, 0, 1),
    };
    try {
      const extraction = extractDrawingFromCanvas(canvas, prompt, "camera");
      if (extraction.analysis.coveragePercent <= 1) {
        throw new Error("The tapped drawing is too faint or too far away for a safe 3D reconstruction.");
      }
      return {
        ...extraction,
        previewUrl: frame.canvas.toDataURL("image/jpeg", 0.86),
        sourceTarget: frame.target,
        sourceScope: frame.scope,
        cutoutRecognition: {
          model: "targeted-local-extraction-v3",
          latencyMs: Math.round(performance.now() - started),
          confidence: 0.62,
          areaPercent: extraction.analysis.coveragePercent,
          cropScale: scale,
        },
      };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error("The tapped line art could not be isolated locally.");
}

async function isolateWithCompactDrawingModel(frame: SourceFrame, started: number): Promise<DrawingExtraction> {
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
  if (extraction.analysis.coveragePercent <= 1) {
    throw new Error("That drawing is too faint or too far away for a safe reconstruction. Move closer so the character fills more of the frame.");
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

async function isolate(frame: SourceFrame): Promise<DrawingExtraction> {
  const started = performance.now();
  if (frame.scope === "selected-image") {
    const context = frame.canvas.getContext("2d", { willReadFrequently: true });
    if (context && hasMeaningfulSelectedAlpha(context.getImageData(0, 0, frame.canvas.width, frame.canvas.height).data, frame.scope)) {
      return extractDrawingFromCanvas(frame.canvas, frame.target, "selected-image");
    }
  }
  // Drawing-aware extraction goes first. General object segmentation is a
  // last resort because it can confidently select paper, a monitor, or an
  // arbitrary coherent camera patch instead of the character drawn on it.
  try {
    return isolateWithTargetedLocalExtraction(frame, started);
  } catch (error) {
    console.warn("WallAlive point-local line extraction fell back to the compact drawing model", error);
  }
  try {
    return await isolateWithCompactDrawingModel(frame, started);
  } catch (error) {
    console.warn("WallAlive compact drawing segmentation fell back to MagicTouch", error);
  }
  try {
    return await isolateWithMagicTouch(frame);
  } catch (error) {
    console.warn("WallAlive general segmentation was safely rejected", error);
    throw new Error("I can’t verify one complete character yet. Move closer, tap inside its body, and keep other drawings outside the guide.");
  }
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
