export type ShapeHint = "round" | "tall" | "wide" | "spiky";

export type ContourPoint = { x: number; y: number };
export type SkeletonPoint = { x: number; y: number; radius: number };
export type CaptureTarget = { x: number; y: number };

export type DrawingCandidateFeatures = {
  pixelCount: number;
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  averageChroma: number;
  edgeFraction?: number;
};

export type DrawingAnalysis = {
  dominantColor: string;
  secondaryColor: string;
  coveragePercent: number;
  aspectRatio: number;
  shapeHint: ShapeHint;
  edgeEnergy: "soft" | "scribbly" | "bold";
  sourceWidth: number;
  sourceHeight: number;
  skeletonPoints: number;
};

export type DrawingExtraction = {
  textureUrl: string;
  previewUrl: string;
  contour: ContourPoint[];
  skeleton: SkeletonPoint[];
  analysis: DrawingAnalysis;
};

type RGB = { r: number; g: number; b: number };
type Component = { pixels: number[]; minX: number; minY: number; maxX: number; maxY: number; centerX: number; centerY: number };

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const colorDistance = (a: RGB, b: RGB) => Math.sqrt((a.r - b.r) ** 2 + (a.g - b.g) ** 2 + (a.b - b.b) ** 2);
const toHex = ({ r, g, b }: RGB) => `#${[r, g, b].map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0")).join("")}`;

function averageBorder(data: Uint8ClampedArray, width: number, height: number): RGB {
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  const inset = Math.max(2, Math.round(Math.min(width, height) * 0.025));
  const stride = Math.max(1, Math.floor(Math.min(width, height) / 90));
  const take = (x: number, y: number) => {
    const index = (y * width + x) * 4;
    r += data[index];
    g += data[index + 1];
    b += data[index + 2];
    count += 1;
  };
  for (let x = inset; x < width - inset; x += stride) {
    take(x, inset);
    take(x, height - inset - 1);
  }
  for (let y = inset + stride; y < height - inset - stride; y += stride) {
    take(inset, y);
    take(width - inset - 1, y);
  }
  return { r: r / count, g: g / count, b: b / count };
}

function inkScore(pixel: RGB, background: RGB) {
  const maxChannel = Math.max(pixel.r, pixel.g, pixel.b);
  const minChannel = Math.min(pixel.r, pixel.g, pixel.b);
  const chroma = maxChannel - minChannel;
  const lightness = (maxChannel + minChannel) / 2;
  const backgroundLightness = (Math.max(background.r, background.g, background.b) + Math.min(background.r, background.g, background.b)) / 2;
  const darkerThanSurface = Math.max(0, backgroundLightness - lightness);
  return chroma * 1.55 + darkerThanSurface * 1.2 + colorDistance(pixel, background) * 0.26;
}

function dilate(mask: Uint8Array, width: number, height: number, radius: number) {
  const result = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let found = false;
      for (let oy = -radius; oy <= radius && !found; oy += 1) {
        for (let ox = -radius; ox <= radius; ox += 1) {
          const nx = x + ox;
          const ny = y + oy;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height && mask[ny * width + nx]) {
            found = true;
            break;
          }
        }
      }
      if (found) result[y * width + x] = 1;
    }
  }
  return result;
}

function erode(mask: Uint8Array, width: number, height: number, radius: number) {
  const result = new Uint8Array(mask.length);
  for (let y = radius; y < height - radius; y += 1) {
    for (let x = radius; x < width - radius; x += 1) {
      let solid = true;
      for (let oy = -radius; oy <= radius && solid; oy += 1) {
        for (let ox = -radius; ox <= radius; ox += 1) {
          if (!mask[(y + oy) * width + x + ox]) {
            solid = false;
            break;
          }
        }
      }
      if (solid) result[y * width + x] = 1;
    }
  }
  return result;
}

function connectedComponents(mask: Uint8Array, width: number, height: number) {
  const visited = new Uint8Array(mask.length);
  const components: Component[] = [];
  const neighbors = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]];

  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;
    const queue = [start];
    const pixels: number[] = [];
    visited[start] = 1;
    let minX = width;
    let minY = height;
    let maxX = 0;
    let maxY = 0;
    let sumX = 0;
    let sumY = 0;
    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const index = queue[cursor];
      const x = index % width;
      const y = Math.floor(index / width);
      pixels.push(index);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      sumX += x;
      sumY += y;
      for (const [ox, oy] of neighbors) {
        const nx = x + ox;
        const ny = y + oy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const next = ny * width + nx;
        if (mask[next] && !visited[next]) {
          visited[next] = 1;
          queue.push(next);
        }
      }
    }
    components.push({ pixels, minX, minY, maxX, maxY, centerX: sumX / pixels.length, centerY: sumY / pixels.length });
  }
  return components;
}

export function scoreDrawingCandidate(candidate: DrawingCandidateFeatures, width: number, height: number, target: CaptureTarget = { x: 0.5, y: 0.48 }) {
  const boxWidth = candidate.maxX - candidate.minX + 1;
  const boxHeight = candidate.maxY - candidate.minY + 1;
  const boxArea = boxWidth * boxHeight;
  const frameArea = width * height;
  const areaRatio = boxArea / frameArea;
  const density = candidate.pixelCount / Math.max(1, boxArea);
  const aspect = boxWidth / Math.max(1, boxHeight);
  const centerX = (candidate.minX + candidate.maxX) / 2;
  const centerY = (candidate.minY + candidate.maxY) / 2;
  const focusX = target.x * width;
  const focusY = target.y * height;
  const outsideX = Math.max(candidate.minX - focusX, 0, focusX - candidate.maxX);
  const outsideY = Math.max(candidate.minY - focusY, 0, focusY - candidate.maxY);
  const focusDistance = Math.hypot(outsideX, outsideY) / Math.hypot(width, height);
  const centerDistance = Math.hypot(centerX - focusX, centerY - focusY) / Math.hypot(width, height);
  const lineStructure = density <= 0.38 ? clamp(density / 0.035, 0.35, 1) : candidate.averageChroma > 30 ? 0.62 : 0.06;
  const aspectPenalty = aspect > 3.2 || aspect < 0.24 ? 0.08 : aspect > 2.25 || aspect < 0.38 ? 0.45 : 1;
  const sizePenalty = areaRatio > 0.46 ? 0.08 : areaRatio > 0.32 ? 0.38 : areaRatio < 0.0025 ? 0.2 : 1;
  const lowerClutterPenalty = centerY / height > 0.82 ? 0.07 : 1;
  const frameEdgePenalty = candidate.minX < width * 0.045 || candidate.maxX > width * 0.955 || candidate.minY < height * 0.07 || candidate.maxY > height * 0.88 ? 0.22 : 1;
  const rectangularBorderPenalty = (candidate.edgeFraction ?? 0) > 0.78 ? 0.04 : (candidate.edgeFraction ?? 0) > 0.62 ? 0.24 : 1;
  const colorBoost = 0.82 + Math.min(1.25, candidate.averageChroma / 72);
  const targetAffinity = 1 / (0.18 + focusDistance * 7 + centerDistance * 0.85);
  return Math.pow(boxArea, 0.62) * lineStructure * aspectPenalty * sizePenalty * lowerClutterPenalty * frameEdgePenalty * rectangularBorderPenalty * colorBoost * targetAffinity;
}

function candidateFeatures(component: Component, pixels: Uint8ClampedArray, width: number): DrawingCandidateFeatures {
  let chroma = 0;
  let edgePixels = 0;
  const edgeBand = Math.max(2, Math.round(Math.min(component.maxX - component.minX, component.maxY - component.minY) * 0.025));
  for (const index of component.pixels) {
    const rgba = index * 4;
    const red = pixels[rgba];
    const green = pixels[rgba + 1];
    const blue = pixels[rgba + 2];
    chroma += Math.max(red, green, blue) - Math.min(red, green, blue);
    const x = index % width;
    const y = Math.floor(index / width);
    if (x - component.minX <= edgeBand || component.maxX - x <= edgeBand || y - component.minY <= edgeBand || component.maxY - y <= edgeBand) edgePixels += 1;
  }
  return {
    pixelCount: component.pixels.length,
    minX: component.minX,
    minY: component.minY,
    maxX: component.maxX,
    maxY: component.maxY,
    averageChroma: chroma / Math.max(1, component.pixels.length),
    edgeFraction: edgePixels / Math.max(1, component.pixels.length),
  };
}

function chooseDrawing(components: Component[], pixels: Uint8ClampedArray, width: number, height: number, target: CaptureTarget) {
  const viable = components.filter((component) => {
    const boxWidth = component.maxX - component.minX + 1;
    const boxHeight = component.maxY - component.minY + 1;
    return component.pixels.length >= Math.max(18, width * height * 0.00018)
      && boxWidth >= width * 0.035
      && boxHeight >= height * 0.035;
  });
  if (!viable.length) return null;
  return viable.sort((a, b) => scoreDrawingCandidate(candidateFeatures(b, pixels, width), width, height, target)
    - scoreDrawingCandidate(candidateFeatures(a, pixels, width), width, height, target))[0];
}

function recoverSilhouette(mask: Uint8Array, width: number, height: number) {
  const outside = new Uint8Array(mask.length);
  const queue: number[] = [];
  const add = (x: number, y: number) => {
    const index = y * width + x;
    if (!mask[index] && !outside[index]) {
      outside[index] = 1;
      queue.push(index);
    }
  };
  for (let x = 0; x < width; x += 1) {
    add(x, 0);
    add(x, height - 1);
  }
  for (let y = 1; y < height - 1; y += 1) {
    add(0, y);
    add(width - 1, y);
  }
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const index = queue[cursor];
    const x = index % width;
    const y = Math.floor(index / width);
    if (x > 0) add(x - 1, y);
    if (x + 1 < width) add(x + 1, y);
    if (y > 0) add(x, y - 1);
    if (y + 1 < height) add(x, y + 1);
  }
  const silhouette = new Uint8Array(mask.length);
  for (let index = 0; index < silhouette.length; index += 1) silhouette[index] = outside[index] ? 0 : 1;
  return silhouette;
}

function pointLineDistance(point: ContourPoint, start: ContourPoint, end: ContourPoint) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (!dx && !dy) return Math.hypot(point.x - start.x, point.y - start.y);
  const t = clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy), 0, 1);
  return Math.hypot(point.x - (start.x + t * dx), point.y - (start.y + t * dy));
}

function ramerDouglasPeucker(points: ContourPoint[], tolerance: number): ContourPoint[] {
  if (points.length <= 2) return points;
  let farthest = 0;
  let index = 0;
  for (let cursor = 1; cursor < points.length - 1; cursor += 1) {
    const distance = pointLineDistance(points[cursor], points[0], points[points.length - 1]);
    if (distance > farthest) {
      farthest = distance;
      index = cursor;
    }
  }
  if (farthest <= tolerance) return [points[0], points[points.length - 1]];
  const left = ramerDouglasPeucker(points.slice(0, index + 1), tolerance);
  const right = ramerDouglasPeucker(points.slice(index), tolerance);
  return [...left.slice(0, -1), ...right];
}

function traceContour(mask: Uint8Array, width: number, height: number) {
  const directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  let startX = -1;
  let startY = -1;
  for (let y = 0; y < height && startX < 0; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (mask[y * width + x] && (x === 0 || !mask[y * width + x - 1])) {
        startX = x;
        startY = y;
        break;
      }
    }
  }
  if (startX < 0) return [];
  const points: ContourPoint[] = [];
  let x = startX;
  let y = startY;
  let backX = x - 1;
  let backY = y;
  const firstNext = { x: -1, y: -1 };
  for (let step = 0; step < width * height * 3; step += 1) {
    points.push({ x, y });
    let backIndex = directions.findIndex(([ox, oy]) => x + ox === backX && y + oy === backY);
    if (backIndex < 0) backIndex = 4;
    let found = false;
    for (let offset = 1; offset <= 8; offset += 1) {
      const directionIndex = (backIndex + offset) % 8;
      const [ox, oy] = directions[directionIndex];
      const nx = x + ox;
      const ny = y + oy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height || !mask[ny * width + nx]) continue;
      const beforeIndex = (directionIndex + 7) % 8;
      backX = x + directions[beforeIndex][0];
      backY = y + directions[beforeIndex][1];
      x = nx;
      y = ny;
      found = true;
      if (firstNext.x < 0) {
        firstNext.x = x;
        firstNext.y = y;
      }
      break;
    }
    if (!found) break;
    if (x === startX && y === startY && points.length > 2) break;
  }
  const simplified = ramerDouglasPeucker([...points, points[0]], Math.max(width, height) * 0.005).slice(0, -1);
  const sampled = simplified.length > 220 ? simplified.filter((_, index) => index % Math.ceil(simplified.length / 220) === 0) : simplified;
  return sampled.map((point) => ({
    x: Number((((point.x + 0.5) / width - 0.5) * 1.4).toFixed(4)),
    y: Number(((0.5 - (point.y + 0.5) / height) * 1.4).toFixed(4)),
  }));
}

function contourFromCanvas(canvas: HTMLCanvasElement) {
  const sampleSize = 128;
  const sample = document.createElement("canvas");
  sample.width = sampleSize;
  sample.height = sampleSize;
  const context = sample.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.drawImage(canvas, 0, 0, sampleSize, sampleSize);
  const pixels = context.getImageData(0, 0, sampleSize, sampleSize).data;
  const mask = new Uint8Array(sampleSize * sampleSize);
  for (let index = 0; index < mask.length; index += 1) mask[index] = pixels[index * 4 + 3] > 38 ? 1 : 0;
  return traceContour(mask, sampleSize, sampleSize);
}

function distanceTransform(mask: Uint8Array, width: number, height: number) {
  const distance = new Float32Array(width * height);
  const diagonal = Math.SQRT2;
  const far = width + height;
  for (let index = 0; index < distance.length; index += 1) distance[index] = mask[index] ? far : 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      if (!distance[index]) continue;
      if (x > 0) distance[index] = Math.min(distance[index], distance[index - 1] + 1);
      if (y > 0) distance[index] = Math.min(distance[index], distance[index - width] + 1);
      if (x > 0 && y > 0) distance[index] = Math.min(distance[index], distance[index - width - 1] + diagonal);
      if (x + 1 < width && y > 0) distance[index] = Math.min(distance[index], distance[index - width + 1] + diagonal);
    }
  }
  let maxDistance = 0;
  for (let y = height - 1; y >= 0; y -= 1) {
    for (let x = width - 1; x >= 0; x -= 1) {
      const index = y * width + x;
      if (!distance[index]) continue;
      if (x + 1 < width) distance[index] = Math.min(distance[index], distance[index + 1] + 1);
      if (y + 1 < height) distance[index] = Math.min(distance[index], distance[index + width] + 1);
      if (x + 1 < width && y + 1 < height) distance[index] = Math.min(distance[index], distance[index + width + 1] + diagonal);
      if (x > 0 && y + 1 < height) distance[index] = Math.min(distance[index], distance[index + width - 1] + diagonal);
      maxDistance = Math.max(maxDistance, distance[index]);
    }
  }
  return { distance, maxDistance: Math.max(1, maxDistance) };
}

export function extractMedialSkeleton(mask: Uint8Array, width: number, height: number) {
  const { distance } = distanceTransform(mask, width, height);
  const candidates: Array<{ x: number; y: number; radius: number }> = [];
  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const index = y * width + x;
      const radius = distance[index];
      if (radius < 1.15) continue;
      const horizontal = radius >= distance[index - 1] && radius >= distance[index + 1]
        && (radius > distance[index - 1] || radius > distance[index + 1]);
      const vertical = radius >= distance[index - width] && radius >= distance[index + width]
        && (radius > distance[index - width] || radius > distance[index + width]);
      const diagonalA = radius >= distance[index - width - 1] && radius >= distance[index + width + 1];
      const diagonalB = radius >= distance[index - width + 1] && radius >= distance[index + width - 1];
      if (horizontal || vertical || (diagonalA && diagonalB)) candidates.push({ x, y, radius });
    }
  }
  candidates.sort((a, b) => b.radius - a.radius);
  const skeleton: typeof candidates = [];
  for (const candidate of candidates) {
    const contained = skeleton.some((kept) => Math.hypot(candidate.x - kept.x, candidate.y - kept.y) + candidate.radius <= kept.radius + 0.42);
    const duplicate = skeleton.some((kept) => Math.hypot(candidate.x - kept.x, candidate.y - kept.y) < 1.35 && Math.abs(candidate.radius - kept.radius) < 1.1);
    if (!contained && !duplicate) skeleton.push(candidate);
    if (skeleton.length >= 180) break;
  }
  if (!skeleton.length) {
    let best = 0;
    for (let index = 1; index < distance.length; index += 1) if (distance[index] > distance[best]) best = index;
    if (distance[best] > 0) skeleton.push({ x: best % width, y: Math.floor(best / width), radius: distance[best] });
  }
  return skeleton;
}

function skeletonFromTexture(canvas: HTMLCanvasElement): SkeletonPoint[] {
  const size = 96;
  const sample = document.createElement("canvas");
  sample.width = size;
  sample.height = size;
  const context = sample.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.drawImage(canvas, 0, 0, size, size);
  const pixels = context.getImageData(0, 0, size, size).data;
  const mask = new Uint8Array(size * size);
  for (let index = 0; index < mask.length; index += 1) mask[index] = pixels[index * 4 + 3] > 38 ? 1 : 0;
  return extractMedialSkeleton(mask, size, size).map((point) => ({
    x: Number((((point.x + 0.5) / size - 0.5) * 1.4).toFixed(4)),
    y: Number(((0.5 - (point.y + 0.5) / size) * 1.4).toFixed(4)),
    radius: Number(((point.radius / size) * 1.4).toFixed(4)),
  }));
}

function classifyShape(width: number, height: number, coverage: number): ShapeHint {
  const ratio = width / Math.max(1, height);
  if (coverage < 0.12) return "spiky";
  if (ratio > 1.24) return "wide";
  if (ratio < 0.78) return "tall";
  return "round";
}

function extractDrawingFromSource(sourceImage: CanvasImageSource, sourceWidth: number, sourceHeight: number, target: CaptureTarget): DrawingExtraction {
  const width = 480;
  const height = Math.round(width * (sourceHeight / sourceWidth));
  const source = document.createElement("canvas");
  source.width = width;
  source.height = height;
  const context = source.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.drawImage(sourceImage, 0, 0, width, height);
  const frame = context.getImageData(0, 0, width, height);
  const background = averageBorder(frame.data, width, height);
  const backgroundChroma = Math.max(background.r, background.g, background.b) - Math.min(background.r, background.g, background.b);
  const backgroundLightness = (Math.max(background.r, background.g, background.b) + Math.min(background.r, background.g, background.b)) / 2;
  const rawInk = new Uint8Array(width * height);
  const scanInsetX = Math.round(width * 0.08);
  const scanInsetTop = Math.round(height * 0.09);
  const scanInsetBottom = Math.round(height * 0.16);

  for (let y = scanInsetTop; y < height - scanInsetBottom; y += 1) {
    for (let x = scanInsetX; x < width - scanInsetX; x += 1) {
      const pixelIndex = (y * width + x) * 4;
      const pixel = { r: frame.data[pixelIndex], g: frame.data[pixelIndex + 1], b: frame.data[pixelIndex + 2] };
      const maxChannel = Math.max(pixel.r, pixel.g, pixel.b);
      const minChannel = Math.min(pixel.r, pixel.g, pixel.b);
      const chroma = maxChannel - minChannel;
      const lightness = (maxChannel + minChannel) / 2;
      const sampleDistance = 3;
      let localContrast = 0;
      for (const [offsetX, offsetY] of [[-sampleDistance, 0], [sampleDistance, 0], [0, -sampleDistance], [0, sampleDistance]]) {
        const neighborIndex = ((y + offsetY) * width + x + offsetX) * 4;
        localContrast = Math.max(localContrast, colorDistance(pixel, {
          r: frame.data[neighborIndex],
          g: frame.data[neighborIndex + 1],
          b: frame.data[neighborIndex + 2],
        }));
      }
      const vividInk = chroma >= Math.max(58, backgroundChroma + 28);
      const darkInk = lightness <= Math.min(112, backgroundLightness - 30);
      const contrastedStroke = localContrast >= 20 && inkScore(pixel, background) > 43;
      if (inkScore(pixel, background) > 54 && (vividInk || darkInk || contrastedStroke)) rawInk[y * width + x] = 1;
    }
  }

  const connectedInk = erode(dilate(rawInk, width, height, 2), width, height, 1);
  const components = connectedComponents(connectedInk, width, height);
  const anchor = chooseDrawing(components, frame.data, width, height, target);
  if (!anchor) throw new Error("I couldn't find one clear character outline. Tap the drawing, move closer, and capture again.");

  const span = Math.max(anchor.maxX - anchor.minX, anchor.maxY - anchor.minY);
  const anchorArea = Math.max(1, (anchor.maxX - anchor.minX + 1) * (anchor.maxY - anchor.minY + 1));
  const selected = components.filter((component) => {
    if (component === anchor) return true;
    const componentArea = (component.maxX - component.minX + 1) * (component.maxY - component.minY + 1);
    const withinCharacterBounds = component.centerX >= anchor.minX - span * 0.06
      && component.centerX <= anchor.maxX + span * 0.06
      && component.centerY >= anchor.minY - span * 0.06
      && component.centerY <= anchor.maxY + span * 0.06;
    return withinCharacterBounds && componentArea <= anchorArea * 0.24 && component.pixels.length >= 8;
  });
  let minX = Math.min(...selected.map((component) => component.minX));
  let minY = Math.min(...selected.map((component) => component.minY));
  let maxX = Math.max(...selected.map((component) => component.maxX));
  let maxY = Math.max(...selected.map((component) => component.maxY));
  const padding = Math.max(10, Math.round(Math.max(maxX - minX, maxY - minY) * 0.12));
  minX = clamp(minX - padding, 0, width - 2);
  minY = clamp(minY - padding, 0, height - 2);
  maxX = clamp(maxX + padding, minX + 1, width - 1);
  maxY = clamp(maxY + padding, minY + 1, height - 1);
  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;
  const selectedMask = new Uint8Array(cropWidth * cropHeight);
  const selectedIndices = new Set(selected.flatMap((component) => component.pixels));
  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      if (selectedIndices.has((y + minY) * width + x + minX)) selectedMask[y * cropWidth + x] = 1;
    }
  }
  const sealed = erode(dilate(selectedMask, cropWidth, cropHeight, 3), cropWidth, cropHeight, 1);
  const silhouette = recoverSilhouette(sealed, cropWidth, cropHeight);
  const silhouettePixels = silhouette.reduce((sum, value) => sum + value, 0);
  if (silhouettePixels < cropWidth * cropHeight * 0.03) throw new Error("The outline is too open to become a solid. Move closer or use a bolder drawing.");

  const cropped = context.getImageData(minX, minY, cropWidth, cropHeight);
  const cutout = document.createElement("canvas");
  cutout.width = cropWidth;
  cutout.height = cropHeight;
  const cutoutContext = cutout.getContext("2d");
  if (!cutoutContext) throw new Error("Canvas processing is unavailable in this browser.");
  const transparent = cutoutContext.createImageData(cropWidth, cropHeight);
  let inkPixels = 0;
  let colorR = 0;
  let colorG = 0;
  let colorB = 0;
  for (let index = 0; index < silhouette.length; index += 1) {
    if (!silhouette[index]) continue;
    const rgba = index * 4;
    transparent.data[rgba] = cropped.data[rgba];
    transparent.data[rgba + 1] = cropped.data[rgba + 1];
    transparent.data[rgba + 2] = cropped.data[rgba + 2];
    transparent.data[rgba + 3] = 255;
    if (selectedMask[index]) {
      inkPixels += 1;
      colorR += cropped.data[rgba];
      colorG += cropped.data[rgba + 1];
      colorB += cropped.data[rgba + 2];
    }
  }
  cutoutContext.putImageData(transparent, 0, 0);

  const output = document.createElement("canvas");
  output.width = 512;
  output.height = 512;
  const outputContext = output.getContext("2d");
  if (!outputContext) throw new Error("Canvas processing is unavailable in this browser.");
  const scale = Math.min(448 / cropWidth, 448 / cropHeight);
  const drawWidth = cropWidth * scale;
  const drawHeight = cropHeight * scale;
  outputContext.imageSmoothingEnabled = true;
  outputContext.imageSmoothingQuality = "high";
  outputContext.drawImage(cutout, (512 - drawWidth) / 2, (512 - drawHeight) / 2, drawWidth, drawHeight);

  const dominant = inkPixels ? { r: colorR / inkPixels, g: colorG / inkPixels, b: colorB / inkPixels } : background;
  const coverage = inkPixels / (width * height);
  const contour = contourFromCanvas(output);
  if (contour.length < 6) {
    const diagnostic = candidateFeatures(anchor, frame.data, width);
    throw new Error(`The selected component was not a closed character (box ${cropWidth}×${cropHeight}, density ${(diagnostic.pixelCount / ((diagnostic.maxX - diagnostic.minX + 1) * (diagnostic.maxY - diagnostic.minY + 1))).toFixed(2)}, chroma ${diagnostic.averageChroma.toFixed(0)}, border ${(diagnostic.edgeFraction ?? 0).toFixed(2)}). Tap inside the character body and capture again.`);
  }
  const skeleton = skeletonFromTexture(output);
  if (!skeleton.length) throw new Error("The drawing body could not be skeletonized. Use one closed, bold character outline.");

  return {
    textureUrl: output.toDataURL("image/png"),
    previewUrl: source.toDataURL("image/jpeg", 0.82),
    contour,
    skeleton,
    analysis: {
      dominantColor: toHex(dominant),
      secondaryColor: toHex(background),
      coveragePercent: Math.max(1, Math.round(coverage * 100)),
      aspectRatio: Number((cropWidth / cropHeight).toFixed(2)),
      shapeHint: classifyShape(cropWidth, cropHeight, silhouettePixels / (cropWidth * cropHeight)),
      edgeEnergy: coverage < 0.025 ? "scribbly" : coverage > 0.09 ? "bold" : "soft",
      sourceWidth,
      sourceHeight,
      skeletonPoints: skeleton.length,
    },
  };
}

export function extractDrawingFromVideo(video: HTMLVideoElement, target: CaptureTarget = { x: 0.5, y: 0.48 }): DrawingExtraction {
  if (!video.videoWidth || !video.videoHeight) throw new Error("The camera is still focusing. Try capture again in a moment.");
  return extractDrawingFromSource(video, video.videoWidth, video.videoHeight, target);
}

export function extractDrawingFromCanvas(canvas: HTMLCanvasElement, target: CaptureTarget = { x: 0.5, y: 0.48 }): DrawingExtraction {
  return extractDrawingFromSource(canvas, canvas.width, canvas.height, target);
}

export function createDemoDoodle(): DrawingExtraction {
  const output = document.createElement("canvas");
  output.width = 512;
  output.height = 512;
  const context = output.getContext("2d");
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");

  context.lineCap = "round";
  context.lineJoin = "round";
  context.fillStyle = "#ff674d";
  context.strokeStyle = "#18312e";
  context.lineWidth = 15;
  context.beginPath();
  context.moveTo(115, 205);
  context.quadraticCurveTo(92, 92, 196, 82);
  context.quadraticCurveTo(262, 50, 320, 98);
  context.quadraticCurveTo(426, 107, 409, 228);
  context.quadraticCurveTo(432, 356, 318, 405);
  context.quadraticCurveTo(196, 438, 103, 349);
  context.quadraticCurveTo(65, 278, 115, 205);
  context.closePath();
  context.fill();
  context.stroke();

  context.fillStyle = "#5fc7df";
  context.beginPath();
  context.moveTo(135, 132);
  context.lineTo(104, 42);
  context.lineTo(198, 104);
  context.closePath();
  context.fill();
  context.stroke();
  context.beginPath();
  context.moveTo(320, 101);
  context.lineTo(404, 49);
  context.lineTo(374, 144);
  context.closePath();
  context.fill();
  context.stroke();

  const eye = (x: number, y: number) => {
    context.fillStyle = "#fffaf0";
    context.beginPath();
    context.ellipse(x, y, 39, 51, 0, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = "#18312e";
    context.beginPath();
    context.arc(x + 8, y + 8, 13, 0, Math.PI * 2);
    context.fill();
  };
  eye(205, 228);
  eye(317, 221);
  context.beginPath();
  context.arc(266, 307, 47, 0.12, Math.PI - 0.12);
  context.stroke();
  context.fillStyle = "#c8f15a";
  context.beginPath();
  context.arc(378, 303, 24, 0, Math.PI * 2);
  context.fill();
  context.stroke();

  const preview = document.createElement("canvas");
  preview.width = 768;
  preview.height = 480;
  const previewContext = preview.getContext("2d");
  if (!previewContext) throw new Error("Canvas processing is unavailable in this browser.");
  previewContext.fillStyle = "#ded0b7";
  previewContext.fillRect(0, 0, preview.width, preview.height);
  previewContext.fillStyle = "rgba(91, 72, 53, 0.12)";
  previewContext.fillRect(382, 0, 2, 414);
  previewContext.fillStyle = "#98785c";
  previewContext.fillRect(0, 414, preview.width, 66);
  previewContext.fillStyle = "#6b5440";
  previewContext.fillRect(0, 406, preview.width, 8);
  previewContext.fillStyle = "#755944";
  previewContext.fillRect(545, 316, 165, 13);
  previewContext.fillStyle = "#ff674d";
  previewContext.fillRect(574, 275, 28, 41);
  previewContext.fillStyle = "#5fc7df";
  previewContext.fillRect(621, 258, 34, 58);
  previewContext.save();
  previewContext.translate(208, 234);
  previewContext.rotate(-0.035);
  previewContext.fillStyle = "#fffaf0";
  previewContext.shadowColor = "rgba(24, 49, 46, 0.2)";
  previewContext.shadowBlur = 13;
  previewContext.shadowOffsetX = 7;
  previewContext.shadowOffsetY = 9;
  previewContext.fillRect(-132, -154, 264, 308);
  previewContext.shadowColor = "transparent";
  previewContext.strokeStyle = "#816a52";
  previewContext.lineWidth = 7;
  previewContext.strokeRect(-132, -154, 264, 308);
  previewContext.drawImage(output, -113, -132, 226, 226);
  previewContext.fillStyle = "#18312e";
  previewContext.font = "700 13px sans-serif";
  previewContext.textAlign = "center";
  previewContext.fillText("PIP · AGE 7", 0, 127);
  previewContext.restore();

  // The no-camera demo deliberately goes through the same segmentation and
  // skeleton pipeline as a live capture; it is not a pre-cut 3D asset.
  return extractDrawingFromCanvas(preview, { x: 208 / preview.width, y: 234 / preview.height });
}
