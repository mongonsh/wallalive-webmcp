export type ShapeHint = "round" | "tall" | "wide" | "spiky";

export type ContourPoint = { x: number; y: number };
export type SkeletonPoint = { x: number; y: number; radius: number };
export type CaptureTarget = { x: number; y: number };

export type SemanticPartKind = "body" | "eye" | "pupil" | "cheek" | "mouth" | "ear" | "arm" | "hand" | "leg" | "foot" | "marking";
export type SemanticSide = "left" | "right" | "center";
export type SemanticPartSource = "image-region" | "silhouette-branch" | "structural-inference" | "learned-model";

export type LearnedPartHint = {
  kind: Exclude<SemanticPartKind, "body" | "pupil" | "marking">;
  center: { x: number; y: number };
  size: { x: number; y: number };
  endpoints?: [{ x: number; y: number }, { x: number; y: number }];
  rotation: number;
  confidence: number;
};

export type SemanticRegionCandidate = {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color: string;
  pixelCount: number;
  density: number;
  rotation?: number;
  outline?: ContourPoint[];
};

export type SemanticPart = {
  id: string;
  kind: SemanticPartKind;
  side: SemanticSide;
  parentId: string | null;
  center: { x: number; y: number; z: number };
  anchor?: { x: number; y: number; z: number };
  size: { x: number; y: number; z: number };
  rotation: number;
  color: string;
  confidence: number;
  source: SemanticPartSource;
  outline?: ContourPoint[];
};

export type CharacterRig = {
  version: "wallalive-semantic-rig-v2";
  bodyColor: string;
  lineColor: string;
  parts: SemanticPart[];
  joints: Array<{ id: string; parentId: string; childId: string; x: number; y: number }>;
  detectedKinds: SemanticPartKind[];
};

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
  rig: CharacterRig;
  analysis: DrawingAnalysis;
  semanticRegions?: SemanticRegionCandidate[];
  learnedRecognition?: {
    model: "wallalive-parts-v2-ensemble";
    latencyMs: number;
    detectedKinds: SemanticPartKind[];
  };
};

type RGB = { r: number; g: number; b: number };
type Component = { pixels: number[]; minX: number; minY: number; maxX: number; maxY: number; centerX: number; centerY: number };

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const colorDistance = (a: RGB, b: RGB) => Math.sqrt((a.r - b.r) ** 2 + (a.g - b.g) ** 2 + (a.b - b.b) ** 2);
const toHex = ({ r, g, b }: RGB) => `#${[r, g, b].map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0")).join("")}`;
const mixColor = (a: RGB, b: RGB, amount: number): RGB => ({
  r: a.r + (b.r - a.r) * amount,
  g: a.g + (b.g - a.g) * amount,
  b: a.b + (b.b - a.b) * amount,
});
const boostInkColor = (color: RGB): RGB => {
  const average = (color.r + color.g + color.b) / 3;
  return {
    r: clamp(average + (color.r - average) * 2.35, 0, 255),
    g: clamp(average + (color.g - average) * 2.35, 0, 255),
    b: clamp(average + (color.b - average) * 2.35, 0, 255),
  };
};

export function mapCoverTargetToSource(
  target: CaptureTarget,
  sourceWidth: number,
  sourceHeight: number,
  viewportWidth: number,
  viewportHeight: number,
): CaptureTarget {
  if (!sourceWidth || !sourceHeight || !viewportWidth || !viewportHeight) return target;
  const scale = Math.max(viewportWidth / sourceWidth, viewportHeight / sourceHeight);
  const renderedWidth = sourceWidth * scale;
  const renderedHeight = sourceHeight * scale;
  const cropX = Math.max(0, (renderedWidth - viewportWidth) / 2);
  const cropY = Math.max(0, (renderedHeight - viewportHeight) / 2);
  return {
    x: clamp((target.x * viewportWidth + cropX) / renderedWidth, 0, 1),
    y: clamp((target.y * viewportHeight + cropY) / renderedHeight, 0, 1),
  };
}

function hueOf(color: RGB) {
  const r = color.r / 255;
  const g = color.g / 255;
  const b = color.b / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const delta = max - min;
  if (delta < 0.001) return 0;
  const sector = max === r ? ((g - b) / delta) % 6 : max === g ? (b - r) / delta + 2 : (r - g) / delta + 4;
  return (sector * 60 + 360) % 360;
}

function hueDistance(a: RGB, b: RGB) {
  const distance = Math.abs(hueOf(a) - hueOf(b));
  return Math.min(distance, 360 - distance);
}

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
  if (radius <= 0) return new Uint8Array(mask);
  const stride = width + 1;
  const integral = new Uint32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      rowSum += mask[y * width + x];
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + rowSum;
    }
  }
  const result = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const minX = Math.max(0, x - radius);
      const minY = Math.max(0, y - radius);
      const maxX = Math.min(width - 1, x + radius);
      const maxY = Math.min(height - 1, y + radius);
      const sum = integral[(maxY + 1) * stride + maxX + 1]
        - integral[minY * stride + maxX + 1]
        - integral[(maxY + 1) * stride + minX]
        + integral[minY * stride + minX];
      if (sum) result[y * width + x] = 1;
    }
  }
  return result;
}

function erode(mask: Uint8Array, width: number, height: number, radius: number) {
  if (radius <= 0) return new Uint8Array(mask);
  const stride = width + 1;
  const integral = new Uint32Array((width + 1) * (height + 1));
  for (let y = 0; y < height; y += 1) {
    let rowSum = 0;
    for (let x = 0; x < width; x += 1) {
      rowSum += mask[y * width + x];
      integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + rowSum;
    }
  }
  const result = new Uint8Array(mask.length);
  for (let y = radius; y < height - radius; y += 1) {
    for (let x = radius; x < width - radius; x += 1) {
      const minX = x - radius;
      const minY = y - radius;
      const maxX = x + radius;
      const maxY = y + radius;
      const sum = integral[(maxY + 1) * stride + maxX + 1]
        - integral[minY * stride + maxX + 1]
        - integral[(maxY + 1) * stride + minX]
        + integral[minY * stride + minX];
      if (sum === (radius * 2 + 1) ** 2) result[y * width + x] = 1;
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

function componentColor(component: Component, pixels: Uint8ClampedArray): RGB {
  let r = 0;
  let g = 0;
  let b = 0;
  for (const index of component.pixels) {
    const rgba = index * 4;
    r += pixels[rgba];
    g += pixels[rgba + 1];
    b += pixels[rgba + 2];
  }
  return { r: r / component.pixels.length, g: g / component.pixels.length, b: b / component.pixels.length };
}

function componentRotation(component: Component, width: number) {
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (const index of component.pixels) {
    const x = index % width - component.centerX;
    const y = Math.floor(index / width) - component.centerY;
    xx += x * x;
    yy += y * y;
    xy += x * y;
  }
  return Number((0.5 * Math.atan2(2 * xy, xx - yy)).toFixed(4));
}

function componentOutline(component: Component, width: number, height: number) {
  const mask = new Uint8Array(width * height);
  for (const index of component.pixels) mask[index] = 1;
  return traceContour(mask, width, height);
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

function isolateTargetInk(mask: Uint8Array, width: number, height: number, target: CaptureTarget) {
  const components = connectedComponents(mask, width, height);
  if (components.length <= 1) return mask;

  const centerX = target.x * width;
  const centerY = target.y * height;
  const angleBins = 72;
  const describe = (component: Component) => {
    const occupiedAngles = new Uint8Array(angleBins);
    const radii: number[] = [];
    for (const index of component.pixels) {
      const x = index % width;
      const y = Math.floor(index / width);
      const dx = x - centerX;
      const dy = y - centerY;
      const angle = (Math.atan2(dy, dx) + Math.PI * 2) % (Math.PI * 2);
      occupiedAngles[Math.floor(angle / (Math.PI * 2) * angleBins) % angleBins] = 1;
      radii.push(Math.hypot(dx, dy));
    }
    radii.sort((a, b) => a - b);
    const outsideX = Math.max(component.minX - centerX, 0, centerX - component.maxX);
    const outsideY = Math.max(component.minY - centerY, 0, centerY - component.maxY);
    const boxWidth = component.maxX - component.minX + 1;
    const boxHeight = component.maxY - component.minY + 1;
    return {
      component,
      boxWidth,
      boxHeight,
      boxArea: boxWidth * boxHeight,
      boxDistance: Math.hypot(outsideX, outsideY),
      containsTarget: outsideX === 0 && outsideY === 0,
      angularCoverage: occupiedAngles.reduce((sum, value) => sum + value, 0),
      medianRadius: radii[Math.floor(radii.length / 2)] ?? 0,
      maxRadius: radii[radii.length - 1] ?? 0,
    };
  };
  const descriptions = components.map(describe);
  const frameArea = width * height;
  const frameSpan = Math.min(width, height);
  const seed = descriptions.sort((a, b) => {
    const score = (value: typeof a) => value.angularCoverage * 6
      + (value.containsTarget ? 180 : 0)
      + Math.log1p(value.component.pixels.length) * 9
      + Math.min(42, Math.max(value.boxWidth, value.boxHeight) / frameSpan * 42)
      - value.boxDistance / frameSpan * 190
      - value.boxArea / frameArea * 55;
    return score(b) - score(a);
  })[0];
  if (!seed) return mask;

  const seedSpan = Math.max(seed.boxWidth, seed.boxHeight);
  const expansionLimit = seedSpan * 1.24;
  const proximityLimit = Math.max(4, seedSpan * 0.18);
  const seedAlreadyWrapsTarget = seed.angularCoverage >= angleBins * 0.45;
  const selected = descriptions.filter((value) => {
    if (value === seed) return true;
    if (seedAlreadyWrapsTarget) return false;
    const insideSeedBounds = value.component.centerX >= seed.component.minX - proximityLimit
      && value.component.centerX <= seed.component.maxX + proximityLimit
      && value.component.centerY >= seed.component.minY - proximityLimit
      && value.component.centerY <= seed.component.maxY + proximityLimit;
    const unionWidth = Math.max(seed.component.maxX, value.component.maxX) - Math.min(seed.component.minX, value.component.minX) + 1;
    const unionHeight = Math.max(seed.component.maxY, value.component.maxY) - Math.min(seed.component.minY, value.component.minY) + 1;
    const sameOutlineRadius = value.medianRadius >= seed.medianRadius * 0.68
      && value.medianRadius <= seed.medianRadius * 1.2
      && value.maxRadius <= seed.maxRadius * 1.2;
    return insideSeedBounds
      && sameOutlineRadius
      && value.angularCoverage >= 2
      && unionWidth <= expansionLimit
      && unionHeight <= expansionLimit;
  });

  const isolated = new Uint8Array(mask.length);
  for (const { component } of selected) {
    for (const index of component.pixels) isolated[index] = 1;
  }
  return isolated;
}

export function recoverEnclosedTargetRegion(mask: Uint8Array, width: number, height: number, target: CaptureTarget) {
  const targetX = clamp(Math.round(target.x * width), 0, width - 1);
  const targetY = clamp(Math.round(target.y * height), 0, height - 1);
  const frameArea = width * height;

  for (let radius = 1; radius <= 14; radius += 1) {
    const barrier = dilate(mask, width, height, radius);
    const openSpace = new Uint8Array(mask.length);
    for (let index = 0; index < openSpace.length; index += 1) openSpace[index] = barrier[index] ? 0 : 1;
    const candidate = connectedComponents(openSpace, width, height).filter((component) => {
      const touchesFrame = component.minX === 0 || component.minY === 0 || component.maxX === width - 1 || component.maxY === height - 1;
      if (touchesFrame || component.pixels.length < frameArea * 0.0025 || component.pixels.length > frameArea * 0.3) return false;
      const boxWidth = component.maxX - component.minX + 1;
      const boxHeight = component.maxY - component.minY + 1;
      const outsideX = Math.max(component.minX - targetX, 0, targetX - component.maxX);
      const outsideY = Math.max(component.minY - targetY, 0, targetY - component.maxY);
      return boxWidth >= width * 0.035
        && boxHeight >= height * 0.035
        && Math.hypot(outsideX, outsideY) <= Math.min(width, height) * 0.09;
    }).sort((a, b) => {
      const score = (component: Component) => {
        const outsideX = Math.max(component.minX - targetX, 0, targetX - component.maxX);
        const outsideY = Math.max(component.minY - targetY, 0, targetY - component.maxY);
        const centerDistance = Math.hypot(component.centerX - targetX, component.centerY - targetY);
        return Math.sqrt(component.pixels.length)
          / (1 + Math.hypot(outsideX, outsideY) * 0.18 + centerDistance / Math.min(width, height) * 1.4);
      };
      return score(b) - score(a);
    })[0];
    if (!candidate) continue;

    const region = new Uint8Array(mask.length);
    for (const index of candidate.pixels) region[index] = 1;

    // The temporary barrier grows inward to close camera-compression gaps.
    // Expanding the enclosed region by the same amount restores the drawn edge
    // without crossing into unrelated marks outside that boundary.
    return dilate(region, width, height, radius);
  }
  return null;
}

export function recoverTargetSilhouette(mask: Uint8Array, width: number, height: number, target: CaptureTarget, isolateInk = true) {
  const targetInk = isolateInk ? isolateTargetInk(mask, width, height, target) : mask;
  const centerX = target.x * width;
  const centerY = target.y * height;
  const closedSilhouette = recoverSilhouette(targetInk, width, height);
  const inkIndices = [...targetInk.keys()].filter((index) => targetInk[index]);
  if (inkIndices.length) {
    const inkXs = inkIndices.map((index) => index % width);
    const inkYs = inkIndices.map((index) => Math.floor(index / width));
    const inkBoundsArea = (Math.max(...inkXs) - Math.min(...inkXs) + 1) * (Math.max(...inkYs) - Math.min(...inkYs) + 1);
    const closedPixels = closedSilhouette.reduce((sum, value) => sum + value, 0);
    const targetIndex = clamp(Math.round(centerY), 0, height - 1) * width + clamp(Math.round(centerX), 0, width - 1);
    if (closedSilhouette[targetIndex] && closedPixels >= inkBoundsArea * 0.2) return closedSilhouette;
  }
  const binCount = 96;
  const radii = new Float32Array(binCount);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!targetInk[y * width + x]) continue;
      const dx = x - centerX;
      const dy = y - centerY;
      const angle = (Math.atan2(dy, dx) + Math.PI * 2) % (Math.PI * 2);
      const bin = Math.round(angle / (Math.PI * 2) * binCount) % binCount;
      radii[bin] = Math.max(radii[bin], Math.hypot(dx, dy));
    }
  }
  const present = [...radii].filter((radius) => radius > 2).sort((a, b) => a - b);
  if (present.length < binCount * 0.24) return recoverSilhouette(targetInk, width, height);
  const medianRadius = present[Math.floor(present.length / 2)];
  for (let index = 0; index < binCount; index += 1) {
    if (radii[index] > 2) continue;
    for (let offset = 1; offset < binCount / 2; offset += 1) {
      const before = radii[(index - offset + binCount) % binCount];
      const after = radii[(index + offset) % binCount];
      if (before > 2 || after > 2) {
        radii[index] = before > 2 && after > 2 ? (before + after) / 2 : Math.max(before, after);
        break;
      }
    }
  }
  for (let pass = 0; pass < 3; pass += 1) {
    const next = new Float32Array(binCount);
    for (let index = 0; index < binCount; index += 1) {
      const window = [-4, -3, -2, -1, 0, 1, 2, 3, 4].map((offset) => radii[(index + offset + binCount) % binCount]).sort((a, b) => a - b);
      const localMedian = window[4];
      const locallyClamped = Math.min(radii[index], localMedian * 1.22);
      next[index] = clamp(locallyClamped * 0.28 + localMedian * 0.72, medianRadius * 0.64, medianRadius * 1.38);
    }
    radii.set(next);
  }
  const polygon = [...radii].map((radius, index) => {
    const angle = index / binCount * Math.PI * 2;
    return { x: centerX + Math.cos(angle) * radius, y: centerY + Math.sin(angle) * radius };
  });
  const bounds = polygon.reduce((value, point) => ({
    minX: Math.min(value.minX, point.x), minY: Math.min(value.minY, point.y),
    maxX: Math.max(value.maxX, point.x), maxY: Math.max(value.maxY, point.y),
  }), { minX: width, minY: height, maxX: 0, maxY: 0 });
  const result = new Uint8Array(width * height);
  for (let y = Math.max(0, Math.floor(bounds.minY)); y <= Math.min(height - 1, Math.ceil(bounds.maxY)); y += 1) {
    for (let x = Math.max(0, Math.floor(bounds.minX)); x <= Math.min(width - 1, Math.ceil(bounds.maxX)); x += 1) {
      let inside = false;
      for (let current = 0, previous = polygon.length - 1; current < polygon.length; previous = current, current += 1) {
        const a = polygon[current];
        const b = polygon[previous];
        if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
      }
      if (inside) result[y * width + x] = 1;
    }
  }
  return result;
}

export function inkAroundEnclosedRegion(mask: Uint8Array, region: Uint8Array, width: number, height: number) {
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  for (let index = 0; index < region.length; index += 1) {
    if (!region[index]) continue;
    const x = index % width;
    const y = Math.floor(index / width);
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  if (minX > maxX || minY > maxY) return mask;
  const span = Math.max(maxX - minX + 1, maxY - minY + 1);
  // The enclosed negative-space region is a cheap local equivalent of a
  // detector box. Pad that box just enough to recover its contour, ears, and
  // short limbs; the former 70% padding also admitted adjacent wall art.
  const horizontalPadding = Math.max(7, Math.round(span * 0.11));
  const topPadding = Math.max(7, Math.round(span * 0.2));
  const bottomPadding = Math.max(7, Math.round(span * 0.1));
  const windowMinX = Math.max(0, minX - horizontalPadding);
  const windowMinY = Math.max(0, minY - topPadding);
  const windowMaxX = Math.min(width - 1, maxX + horizontalPadding);
  const windowMaxY = Math.min(height - 1, maxY + bottomPadding);
  const localized = new Uint8Array(mask.length);
  for (let y = windowMinY; y <= windowMaxY; y += 1) {
    for (let x = windowMinX; x <= windowMaxX; x += 1) localized[y * width + x] = mask[y * width + x];
  }
  return localized;
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

function analyzeSemanticCanvas(canvas: HTMLCanvasElement, preferredLine: RGB, enableGuidedFace: boolean) {
  const width = canvas.width;
  const height = canvas.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  const pixels = context.getImageData(0, 0, width, height).data;
  const opaque = new Uint8Array(width * height);
  const palette = new Map<string, { r: number; g: number; b: number; count: number }>();
  let opaqueCount = 0;
  for (let index = 0; index < opaque.length; index += 1) {
    if (pixels[index * 4 + 3] <= 38) continue;
    opaque[index] = 1;
    opaqueCount += 1;
    const rgba = index * 4;
    const key = `${pixels[rgba] >> 4}:${pixels[rgba + 1] >> 4}:${pixels[rgba + 2] >> 4}`;
    const entry = palette.get(key) ?? { r: 0, g: 0, b: 0, count: 0 };
    entry.r += pixels[rgba];
    entry.g += pixels[rgba + 1];
    entry.b += pixels[rgba + 2];
    entry.count += 1;
    palette.set(key, entry);
  }
  const colors = [...palette.values()].map((entry) => ({
    r: entry.r / entry.count,
    g: entry.g / entry.count,
    b: entry.b / entry.count,
    count: entry.count,
  })).sort((a, b) => b.count - a.count);
  const body = colors[0] ?? { r: 247, g: 240, b: 223, count: 1 };
  const bodyRgb = { r: body.r, g: body.g, b: body.b };
  const paletteLine = colors.slice(1).filter((color) => colorDistance(color, bodyRgb) > 34).sort((a, b) => {
    const score = (color: typeof a) => color.count * (1 + colorDistance(color, bodyRgb) / 180) * (1 + (255 - (color.r + color.g + color.b) / 3) / 255);
    return score(b) - score(a);
  })[0] ?? { r: 24, g: 49, b: 46, count: 1 };
  const line = colorDistance(preferredLine, bodyRgb) > 18 ? preferredLine : paletteLine;
  const surface = mixColor(bodyRgb, line, 0.14);

  // Removing a narrow alpha-border band prevents the outside contour from
  // swallowing the smaller eye, mouth, and marking regions inside it.
  const interior = erode(opaque, width, height, Math.max(2, Math.round(width / 128)));
  const { distance: bodyInteriorDistance } = distanceTransform(opaque, width, height);
  const featureInset = Math.max(7, Math.round(width * 0.026));
  const featureMask = new Uint8Array(opaque.length);
  const lineChroma = Math.max(line.r, line.g, line.b) - Math.min(line.r, line.g, line.b);
  for (let index = 0; index < featureMask.length; index += 1) {
    if (!interior[index] || bodyInteriorDistance[index] < featureInset) continue;
    const rgba = index * 4;
    const pixel = { r: pixels[rgba], g: pixels[rgba + 1], b: pixels[rgba + 2] };
    const pixelChroma = Math.max(pixel.r, pixel.g, pixel.b) - Math.min(pixel.r, pixel.g, pixel.b);
    const inkMatches = lineChroma < 12 || (pixelChroma > 6 && hueDistance(pixel, line) < 64);
    if (colorDistance(pixel, bodyRgb) > 18 && inkMatches) featureMask[index] = 1;
  }
  const connectedFeatures = erode(dilate(featureMask, width, height, 2), width, height, 1);
  const components = connectedComponents(connectedFeatures, width, height)
    .filter((component) => component.pixels.length >= 8 && component.pixels.length <= Math.max(24, opaqueCount * 0.16));
  const enclosedSpace = new Uint8Array(opaque.length);
  const featureBarrier = dilate(featureMask, width, height, 4);
  for (let index = 0; index < enclosedSpace.length; index += 1) enclosedSpace[index] = interior[index] && !featureBarrier[index] ? 1 : 0;
  const enclosedFeatures = connectedComponents(enclosedSpace, width, height).filter((component) => {
    const boxWidth = component.maxX - component.minX + 1;
    const boxHeight = component.maxY - component.minY + 1;
    if (component.pixels.length < 28 || component.pixels.length > opaqueCount * 0.035 || boxWidth < 6 || boxHeight < 6) return false;
    return !component.pixels.some((pixelIndex) => {
      const x = pixelIndex % width;
      const y = Math.floor(pixelIndex / width);
      return x <= 0 || y <= 0 || x >= width - 1 || y >= height - 1
        || !interior[pixelIndex - 1] || !interior[pixelIndex + 1]
        || !interior[pixelIndex - width] || !interior[pixelIndex + width];
    });
  });
  const enclosedRegions: SemanticRegionCandidate[] = enclosedFeatures.map((component, index) => {
    const boxWidth = component.maxX - component.minX + 1;
    const boxHeight = component.maxY - component.minY + 1;
    return {
      id: `enclosed-region-${index + 1}`,
      x: Number((((component.centerX + 0.5) / width - 0.5) * 1.4).toFixed(4)),
      y: Number(((0.5 - (component.centerY + 0.5) / height) * 1.4).toFixed(4)),
      width: Number(((boxWidth / width) * 1.4).toFixed(4)),
      height: Number(((boxHeight / height) * 1.4).toFixed(4)),
      rotation: componentRotation(component, width),
      outline: componentOutline(component, width, height),
      color: toHex(line),
      pixelCount: component.pixels.length,
      density: 1,
    };
  });
  const strokeRegions: SemanticRegionCandidate[] = components.map((component, index) => {
    let r = 0;
    let g = 0;
    let b = 0;
    for (const pixelIndex of component.pixels) {
      const rgba = pixelIndex * 4;
      r += pixels[rgba];
      g += pixels[rgba + 1];
      b += pixels[rgba + 2];
    }
    const boxWidth = component.maxX - component.minX + 1;
    const boxHeight = component.maxY - component.minY + 1;
    return {
      id: `region-${index + 1}`,
      x: Number((((component.centerX + 0.5) / width - 0.5) * 1.4).toFixed(4)),
      y: Number(((0.5 - (component.centerY + 0.5) / height) * 1.4).toFixed(4)),
      width: Number(((boxWidth / width) * 1.4).toFixed(4)),
      height: Number(((boxHeight / height) * 1.4).toFixed(4)),
      rotation: componentRotation(component, width),
      outline: componentOutline(component, width, height),
      color: toHex({ r: r / component.pixels.length, g: g / component.pixels.length, b: b / component.pixels.length }),
      pixelCount: component.pixels.length,
      density: Number((component.pixels.length / (boxWidth * boxHeight)).toFixed(3)),
    };
  });
  const bodyComponent = connectedComponents(opaque, width, height).sort((a, b) => b.pixels.length - a.pixels.length)[0];
  const guidedRegions: SemanticRegionCandidate[] = [];
  if (bodyComponent && enableGuidedFace) {
    const bodyPixelWidth = bodyComponent.maxX - bodyComponent.minX + 1;
    const bodyPixelHeight = bodyComponent.maxY - bodyComponent.minY + 1;
    const findWindowRegion = (id: string, left: number, top: number, right: number, bottom: number) => {
      const minX = Math.round(bodyComponent.minX + bodyPixelWidth * left);
      const minY = Math.round(bodyComponent.minY + bodyPixelHeight * top);
      const maxX = Math.round(bodyComponent.minX + bodyPixelWidth * right);
      const maxY = Math.round(bodyComponent.minY + bodyPixelHeight * bottom);
      const windowMask = new Uint8Array(featureMask.length);
      for (let y = Math.max(0, minY); y <= Math.min(height - 1, maxY); y += 1) {
        for (let x = Math.max(0, minX); x <= Math.min(width - 1, maxX); x += 1) {
          const index = y * width + x;
          windowMask[index] = connectedFeatures[index];
        }
      }
      const expectedX = (minX + maxX) / 2;
      const expectedY = (minY + maxY) / 2;
      const component = connectedComponents(windowMask, width, height).filter((candidate) => candidate.pixels.length >= 10).sort((a, b) => {
        const score = (candidate: Component) => Math.sqrt(candidate.pixels.length)
          - Math.hypot(candidate.centerX - expectedX, candidate.centerY - expectedY) * 0.08;
        return score(b) - score(a);
      })[0];
      if (!component) return null;
      const boxWidth = component.maxX - component.minX + 1;
      const boxHeight = component.maxY - component.minY + 1;
      let r = 0;
      let g = 0;
      let b = 0;
      for (const pixelIndex of component.pixels) {
        const rgba = pixelIndex * 4;
        r += pixels[rgba];
        g += pixels[rgba + 1];
        b += pixels[rgba + 2];
      }
      return {
        id,
        x: Number((((component.centerX + 0.5) / width - 0.5) * 1.4).toFixed(4)),
        y: Number(((0.5 - (component.centerY + 0.5) / height) * 1.4).toFixed(4)),
        width: Number(((boxWidth / width) * 1.4).toFixed(4)),
        height: Number(((boxHeight / height) * 1.4).toFixed(4)),
        rotation: componentRotation(component, width),
        outline: componentOutline(component, width, height),
        color: toHex({ r: r / component.pixels.length, g: g / component.pixels.length, b: b / component.pixels.length }),
        pixelCount: component.pixels.length,
        density: Number((component.pixels.length / (boxWidth * boxHeight)).toFixed(3)),
      } satisfies SemanticRegionCandidate;
    };
    const guidedEyes = [
      findWindowRegion("guided-eye-left", 0.16, 0.2, 0.5, 0.42),
      findWindowRegion("guided-eye-right", 0.5, 0.2, 0.84, 0.42),
    ].filter((region): region is SemanticRegionCandidate => Boolean(region));
    const pairedEyeSignal = guidedEyes.length === 2
      && Math.abs(guidedEyes[0].y - guidedEyes[1].y) < bodyPixelHeight / height * 1.4 * 0.14
      && Math.max(guidedEyes[0].pixelCount, guidedEyes[1].pixelCount) / Math.max(1, Math.min(guidedEyes[0].pixelCount, guidedEyes[1].pixelCount)) < 4;
    if (pairedEyeSignal) {
      guidedRegions.push(...guidedEyes);
      for (const region of [
        findWindowRegion("guided-cheek-left", 0.1, 0.36, 0.45, 0.58),
        findWindowRegion("guided-cheek-right", 0.55, 0.36, 0.9, 0.58),
        findWindowRegion("guided-mouth", 0.34, 0.44, 0.66, 0.62),
      ]) if (region) guidedRegions.push(region);
    }
  }
  const structuralRegions = [...guidedRegions, ...enclosedRegions];
  const regions = [...structuralRegions, ...strokeRegions.filter((region) => !structuralRegions.some((enclosed) => (
    Math.abs(region.x - enclosed.x) < Math.max(region.width, enclosed.width) * 0.32
      && Math.abs(region.y - enclosed.y) < Math.max(region.height, enclosed.height) * 0.32
      && region.width >= enclosed.width * 0.72
      && region.height >= enclosed.height * 0.72
  )))];
  return { bodyColor: toHex(surface), lineColor: toHex(line), regions };
}

export function inferSemanticRig(
  skeleton: SkeletonPoint[],
  contour: ContourPoint[],
  regions: SemanticRegionCandidate[],
  bodyColor: string,
  lineColor: string,
): CharacterRig {
  const root = skeleton.reduce((largest, point) => point.radius > largest.radius ? point : largest, skeleton[0] ?? { x: 0, y: 0, radius: 0.3 });
  const contourBounds = contour.reduce((bounds, point) => ({
    minX: Math.min(bounds.minX, point.x),
    minY: Math.min(bounds.minY, point.y),
    maxX: Math.max(bounds.maxX, point.x),
    maxY: Math.max(bounds.maxY, point.y),
  }), { minX: 0.7, minY: 0.7, maxX: -0.7, maxY: -0.7 });
  const bodyWidth = Math.max(root.radius * 2, contourBounds.maxX - contourBounds.minX);
  const bodyHeight = Math.max(root.radius * 2, contourBounds.maxY - contourBounds.minY);
  const bodyDepth = clamp(Math.min(bodyWidth, bodyHeight) * 0.58, 0.2, 0.72);
  const parts: SemanticPart[] = [{
    id: "body",
    kind: "body",
    side: "center",
    parentId: null,
    center: { x: root.x, y: root.y, z: 0 },
    size: { x: bodyWidth, y: bodyHeight, z: bodyDepth },
    rotation: 0,
    color: bodyColor,
    confidence: 1,
    source: "silhouette-branch",
  }];

  const relativeY = (region: SemanticRegionCandidate) => (region.y - contourBounds.minY) / bodyHeight;
  const faceCandidates = regions.filter((region) => {
    const y = relativeY(region);
    const area = region.width * region.height / Math.max(0.01, bodyWidth * bodyHeight);
    return y > 0.26 && y < 0.86 && area > 0.00035 && area < 0.09 && region.width < bodyWidth * 0.34;
  });
  const findPair = (candidates: SemanticRegionCandidate[], targetY?: number) => {
    let best: [SemanticRegionCandidate, SemanticRegionCandidate] | null = null;
    let bestScore = -Infinity;
    for (let leftIndex = 0; leftIndex < candidates.length; leftIndex += 1) {
      for (let rightIndex = leftIndex + 1; rightIndex < candidates.length; rightIndex += 1) {
        const pair = [candidates[leftIndex], candidates[rightIndex]].sort((a, b) => a.x - b.x) as [SemanticRegionCandidate, SemanticRegionCandidate];
        const separation = pair[1].x - pair[0].x;
        if (separation < bodyWidth * 0.09 || separation > bodyWidth * 0.68) continue;
        const averageY = (pair[0].y + pair[1].y) / 2;
        const yError = Math.abs(pair[0].y - pair[1].y) / bodyHeight;
        const symmetryError = Math.abs((pair[0].x + pair[1].x) / 2 - root.x) / bodyWidth;
        const sizeError = Math.abs(pair[0].width * pair[0].height - pair[1].width * pair[1].height)
          / Math.max(0.001, pair[0].width * pair[0].height + pair[1].width * pair[1].height);
        if (yError > 0.105 || symmetryError > 0.2 || sizeError > 0.68) continue;
        const verticalScore = targetY === undefined
          ? (relativeY(pair[0]) + relativeY(pair[1]))
          : -Math.abs(averageY - targetY) / bodyHeight * 3;
        const score = verticalScore - yError * 7 - symmetryError * 5 - sizeError * 1.8;
        if (score > bestScore) {
          best = pair;
          bestScore = score;
        }
      }
    }
    return { pair: best, score: bestScore };
  };
  const usedRegions = new Set<string>();
  const addPairedRegion = (kind: "eye" | "cheek", pair: [SemanticRegionCandidate, SemanticRegionCandidate], confidence: number) => {
    pair.forEach((region, index) => {
      const side: SemanticSide = index === 0 ? "left" : "right";
      const partWidth = clamp(region.width, bodyWidth * 0.035, bodyWidth * 0.2);
      const partHeight = clamp(region.height, bodyHeight * 0.025, bodyHeight * 0.18);
      parts.push({
        id: `${kind}-${side}`,
        kind,
        side,
        parentId: "body",
        center: { x: region.x, y: region.y, z: 0 },
        size: { x: partWidth, y: partHeight, z: Math.min(partWidth, partHeight) * 0.18 },
        rotation: region.rotation ?? 0,
        color: region.color,
        confidence,
        source: "image-region",
        outline: region.outline,
      });
      usedRegions.add(region.id);
    });
  };

  const guidedEyePair = [regions.find((region) => region.id === "guided-eye-left"), regions.find((region) => region.id === "guided-eye-right")]
    .filter((region): region is SemanticRegionCandidate => Boolean(region));
  const eyeResult = guidedEyePair.length === 2
    ? { pair: guidedEyePair.sort((a, b) => a.x - b.x) as [SemanticRegionCandidate, SemanticRegionCandidate], score: 0.8 }
    : findPair(faceCandidates);
  const eyePair = eyeResult.pair;
  const eyeAssignments: Array<{ region: SemanticRegionCandidate; side: SemanticSide; partId: string }> = [];
  if (eyePair) {
    addPairedRegion("eye", eyePair, clamp(0.72 + eyeResult.score * 0.08, 0.68, 0.96));
    eyeAssignments.push(
      { region: eyePair[0], side: "left", partId: "eye-left" },
      { region: eyePair[1], side: "right", partId: "eye-right" },
    );
  } else {
    // Profiles, cyclops characters, and asymmetric creatures can legitimately
    // contain one visible eye. Require it to be in the upper face band so a
    // centered mouth or decoration is not silently relabeled as an eye.
    const singleEye = faceCandidates.filter((region) => relativeY(region) > 0.5)
      .sort((a, b) => {
        const score = (region: SemanticRegionCandidate) => relativeY(region) * 1.8
          - Math.abs(region.x - root.x) / bodyWidth * 0.45
          + Math.min(region.width, region.height) / Math.max(bodyWidth, bodyHeight);
        return score(b) - score(a);
      })[0];
    if (singleEye) {
      const side: SemanticSide = singleEye.x < root.x - bodyWidth * 0.08
        ? "left"
        : singleEye.x > root.x + bodyWidth * 0.08 ? "right" : "center";
      const partId = `eye-${side}`;
      parts.push({
        id: partId,
        kind: "eye",
        side,
        parentId: "body",
        center: { x: singleEye.x, y: singleEye.y, z: 0 },
        size: {
          x: clamp(singleEye.width, bodyWidth * 0.035, bodyWidth * 0.24),
          y: clamp(singleEye.height, bodyHeight * 0.025, bodyHeight * 0.2),
          z: Math.min(singleEye.width, singleEye.height) * 0.18,
        },
        rotation: singleEye.rotation ?? 0,
        color: singleEye.color,
        confidence: 0.67,
        source: "image-region",
        outline: singleEye.outline,
      });
      usedRegions.add(singleEye.id);
      eyeAssignments.push({ region: singleEye, side, partId });
    }
  }

  // A pupil is only accepted when a separate ink component sits inside a
  // detected eye. This keeps the detector faithful to the drawing instead of
  // inventing pupils for dot eyes or closed eyelids.
  if (eyeAssignments.length) {
    eyeAssignments.forEach(({ region: eye, side, partId }) => {
      const pupil = regions.filter((region) => !region.id.startsWith("guided-") && !usedRegions.has(region.id)
        && region.width <= eye.width * 0.72
        && region.height <= eye.height * 0.72
        && Math.abs(region.x - eye.x) <= Math.max(eye.width * 0.42, bodyWidth * 0.018)
        && Math.abs(region.y - eye.y) <= Math.max(eye.height * 0.42, bodyHeight * 0.018))
        .sort((a, b) => {
          const score = (region: SemanticRegionCandidate) => Math.hypot(
            (region.x - eye.x) / Math.max(0.001, eye.width),
            (region.y - eye.y) / Math.max(0.001, eye.height),
          ) + region.width * region.height / Math.max(0.001, eye.width * eye.height) * 0.2;
          return score(a) - score(b);
        })[0];
      if (!pupil) return;
      parts.push({
        id: `pupil-${side}`,
        kind: "pupil",
        side,
        parentId: partId,
        center: { x: pupil.x, y: pupil.y, z: 0 },
        size: {
          x: clamp(pupil.width, bodyWidth * 0.012, eye.width * 0.72),
          y: clamp(pupil.height, bodyHeight * 0.012, eye.height * 0.72),
          z: Math.min(pupil.width, pupil.height) * 0.24,
        },
        rotation: pupil.rotation ?? 0,
        color: pupil.color,
        confidence: 0.9,
        source: "image-region",
        outline: pupil.outline,
      });
      usedRegions.add(pupil.id);
    });
  }

  const eyeY = eyeAssignments.length
    ? eyeAssignments.reduce((sum, assignment) => sum + assignment.region.y, 0) / eyeAssignments.length
    : contourBounds.minY + bodyHeight * 0.64;
  const guidedCheekPair = [regions.find((region) => region.id === "guided-cheek-left"), regions.find((region) => region.id === "guided-cheek-right")]
    .filter((region): region is SemanticRegionCandidate => Boolean(region));
  const cheekResult = guidedCheekPair.length === 2
    ? { pair: guidedCheekPair.sort((a, b) => a.x - b.x) as [SemanticRegionCandidate, SemanticRegionCandidate], score: 0.5 }
    : findPair(faceCandidates.filter((region) => !usedRegions.has(region.id)
      && region.y < eyeY - bodyHeight * 0.055
      && region.y > contourBounds.minY + bodyHeight * 0.25), eyeY - bodyHeight * 0.16);
  if (cheekResult.pair) addPairedRegion("cheek", cheekResult.pair, clamp(0.68 + cheekResult.score * 0.04, 0.6, 0.9));

  const mouth = regions.find((region) => region.id === "guided-mouth") ?? regions.filter((region) => !usedRegions.has(region.id)
    && region.y < eyeY - bodyHeight * 0.035
    && region.y > contourBounds.minY + bodyHeight * 0.18
    && Math.abs(region.x - root.x) < bodyWidth * 0.25)
    .sort((a, b) => {
      const centerScore = (region: SemanticRegionCandidate) => Math.abs(region.x - root.x) / bodyWidth
        + Math.abs(region.y - (eyeY - bodyHeight * 0.19)) / bodyHeight
        - region.width / Math.max(0.01, region.height) * 0.04;
      return centerScore(a) - centerScore(b);
    })[0];
  if (mouth) {
    parts.push({
      id: "mouth",
      kind: "mouth",
      side: "center",
      parentId: "body",
      center: { x: mouth.x, y: mouth.y, z: 0 },
      size: { x: clamp(mouth.width, bodyWidth * 0.05, bodyWidth * 0.3), y: clamp(mouth.height, bodyHeight * 0.02, bodyHeight * 0.12), z: 0.018 },
      rotation: mouth.rotation ?? 0,
      color: mouth.color,
      confidence: 0.82,
      source: "image-region",
      outline: mouth.outline,
    });
    usedRegions.add(mouth.id);
  }
  regions.filter((region) => !usedRegions.has(region.id)
    && region.x > contourBounds.minX + bodyWidth * 0.09
    && region.x < contourBounds.maxX - bodyWidth * 0.09
    && region.y > contourBounds.minY + bodyHeight * 0.09
    && region.y < contourBounds.maxY - bodyHeight * 0.09).slice(0, 6).forEach((region, index) => {
    parts.push({
      id: `marking-${index + 1}`,
      kind: "marking",
      side: region.x < root.x - 0.02 ? "left" : region.x > root.x + 0.02 ? "right" : "center",
      parentId: "body",
      center: { x: region.x, y: region.y, z: 0 },
      size: { x: clamp(region.width, 0.018, bodyWidth * 0.2), y: clamp(region.height, 0.014, bodyHeight * 0.16), z: 0.015 },
      rotation: region.rotation ?? 0,
      color: region.color,
      confidence: 0.62,
      source: "image-region",
      outline: region.outline,
    });
  });

  const branchCandidates = skeleton.filter((point) => {
    if (point === root) return false;
    const distance = Math.hypot(point.x - root.x, point.y - root.y);
    return distance > root.radius * 0.52 && distance + point.radius > root.radius * 1.08;
  });
  const pickBranch = (kind: "ear" | "arm" | "leg", side: "left" | "right") => {
    const sign = side === "left" ? -1 : 1;
    return branchCandidates.filter((point) => {
      const dx = point.x - root.x;
      const dy = point.y - root.y;
      if (dx * sign <= root.radius * 0.08) return false;
      if (kind === "ear") return dy > root.radius * 0.42 && dy > Math.abs(dx) * 0.28;
      if (kind === "arm") return Math.abs(dx) > root.radius * 0.78 && dy > -root.radius * 0.72 && dy < root.radius * 0.55;
      return dy < -root.radius * 0.55 && -dy > Math.abs(dx) * 0.42;
    }).sort((a, b) => {
      const reach = (point: SkeletonPoint) => Math.hypot(point.x - root.x, point.y - root.y) + point.radius * 0.8;
      return reach(b) - reach(a);
    })[0];
  };

  (["left", "right"] as const).forEach((side) => {
    const ear = pickBranch("ear", side);
    if (ear) {
      const earSize = clamp(ear.radius * 2.2, root.radius * 0.22, root.radius * 0.72);
      parts.push({
        id: `ear-${side}`,
        kind: "ear",
        side,
        parentId: "body",
        center: { x: ear.x, y: ear.y, z: 0 },
        anchor: { x: root.x + (ear.x - root.x) * 0.62, y: root.y + (ear.y - root.y) * 0.62, z: 0 },
        size: { x: earSize * 0.76, y: earSize, z: earSize * 0.62 },
        rotation: Math.atan2(-(ear.x - root.x), ear.y - root.y),
        color: bodyColor,
        confidence: 0.76,
        source: "silhouette-branch",
      });
    }
    (["arm", "leg"] as const).forEach((kind) => {
      const endpoint = pickBranch(kind, side);
      if (!endpoint) return;
      const dx = endpoint.x - root.x;
      const dy = endpoint.y - root.y;
      const distance = Math.hypot(dx, dy);
      const unitX = dx / Math.max(0.001, distance);
      const unitY = dy / Math.max(0.001, distance);
      const anchor = { x: root.x + unitX * root.radius * 0.72, y: root.y + unitY * root.radius * 0.72, z: 0 };
      const length = Math.max(root.radius * 0.32, Math.hypot(endpoint.x - anchor.x, endpoint.y - anchor.y));
      const thickness = clamp(endpoint.radius * 1.75, root.radius * 0.13, root.radius * 0.34);
      parts.push({
        id: `${kind}-${side}`,
        kind,
        side,
        parentId: "body",
        center: { x: endpoint.x, y: endpoint.y, z: 0 },
        anchor,
        size: { x: thickness, y: length, z: thickness },
        rotation: Math.atan2(-unitX, unitY),
        color: bodyColor,
        confidence: 0.74,
        source: "silhouette-branch",
      });
      const endKind = kind === "arm" ? "hand" : "foot";
      parts.push({
        id: `${endKind}-${side}`,
        kind: endKind,
        side,
        parentId: `${kind}-${side}`,
        center: { x: endpoint.x, y: endpoint.y, z: 0 },
        size: { x: thickness * 1.28, y: thickness * (endKind === "foot" ? 0.9 : 1.28), z: thickness * 1.08 },
        rotation: 0,
        color: bodyColor,
        confidence: 0.68,
        source: "structural-inference",
      });
    });
  });

  // A stubby arm can merge into a round body and disappear from the medial
  // skeleton. When the drawing has a real paired face, recover only a contour
  // side that shows an inward notch followed by an outward hand bulge. This
  // keeps the fallback evidence-based and avoids inventing arms on faceless
  // shapes or on a smooth opposite side.
  if (eyePair) (["left", "right"] as const).forEach((side) => {
    if (parts.some((part) => part.id === `arm-${side}`)) return;
    const sign = side === "left" ? -1 : 1;
    const sideContour = contour.filter((point) => {
      const dx = (point.x - root.x) * sign;
      const dy = point.y - root.y;
      return dx > root.radius * 0.55 && dy > -root.radius * 1.15 && dy < root.radius * 0.18;
    });
    let notch: { depth: number; endpoint: ContourPoint } | null = null;
    for (let index = 2; index < sideContour.length - 2; index += 1) {
      const current = sideContour[index];
      const before = sideContour.slice(index - 2, index);
      const after = sideContour.slice(index + 1, index + 3);
      const reach = (point: ContourPoint) => Math.abs(point.x - root.x);
      const beforeEdge = before.sort((a, b) => reach(b) - reach(a))[0];
      const afterEdge = after.sort((a, b) => reach(b) - reach(a))[0];
      const depth = Math.min(reach(beforeEdge), reach(afterEdge)) - reach(current);
      if (depth < root.radius * 0.052) continue;
      const endpoint = reach(afterEdge) >= reach(beforeEdge) ? afterEdge : beforeEdge;
      if (!notch || depth > notch.depth) notch = { depth, endpoint };
    }
    if (!notch) return;
    const endpoint = notch.endpoint;
    const dx = endpoint.x - root.x;
    const dy = endpoint.y - root.y;
    const distance = Math.hypot(dx, dy);
    const anchor = {
      x: root.x + dx / Math.max(0.001, distance) * root.radius * 0.72,
      y: root.y + dy / Math.max(0.001, distance) * root.radius * 0.72,
      z: 0,
    };
    const thickness = root.radius * 0.19;
    parts.push({
      id: `arm-${side}`,
      kind: "arm",
      side,
      parentId: "body",
      center: { x: endpoint.x, y: endpoint.y, z: 0 },
      anchor,
      size: { x: thickness, y: Math.max(thickness * 1.5, Math.hypot(endpoint.x - anchor.x, endpoint.y - anchor.y)), z: thickness },
      rotation: Math.atan2(-dx, dy),
      color: bodyColor,
      confidence: 0.58,
      source: "structural-inference",
    });
    parts.push({
      id: `hand-${side}`,
      kind: "hand",
      side,
      parentId: `arm-${side}`,
      center: { x: endpoint.x, y: endpoint.y, z: 0 },
      size: { x: thickness * 1.32, y: thickness * 1.32, z: thickness * 1.08 },
      rotation: 0,
      color: bodyColor,
      confidence: 0.54,
      source: "structural-inference",
    });
  });

  const joints = parts.filter((part) => part.parentId).map((part) => ({
    id: `joint-${part.id}`,
    parentId: part.parentId!,
    childId: part.id,
    x: part.anchor?.x ?? part.center.x,
    y: part.anchor?.y ?? part.center.y,
  }));
  return {
    version: "wallalive-semantic-rig-v2",
    bodyColor,
    lineColor,
    parts,
    joints,
    detectedKinds: [...new Set(parts.map((part) => part.kind))],
  };
}

export function mergeLearnedPartHints(extraction: DrawingExtraction, hints: LearnedPartHint[], latencyMs: number): DrawingExtraction {
  const accepted = hints.filter((hint) => hint.confidence >= (hint.kind === "cheek" || hint.kind === "mouth" ? 0.42 : 0.48));
  if (!accepted.length) return {
    ...extraction,
    learnedRecognition: { model: "wallalive-parts-v2-ensemble", latencyMs, detectedKinds: [] },
  };
  const body = extraction.rig.parts.find((part) => part.kind === "body");
  if (!body) return extraction;
  const toRigPoint = (point: { x: number; y: number }) => ({ x: (point.x - 0.5) * 1.4, y: (0.5 - point.y) * 1.4, z: 0 });
  const toRigHint = (hint: LearnedPartHint) => ({
    ...hint,
    center: toRigPoint(hint.center),
    size: { x: hint.size.x * 1.4, y: hint.size.y * 1.4 },
    endpoints: hint.endpoints?.map(toRigPoint) as [{ x: number; y: number; z: number }, { x: number; y: number; z: number }] | undefined,
  });
  const learned = accepted.map(toRigHint);
  const predictedKinds = new Set(learned.map((hint) => hint.kind));
  const replaceableFaceKinds = new Set<SemanticPartKind>(["eye", "cheek", "mouth"]);
  const parts = extraction.rig.parts.filter((part) => !(
    replaceableFaceKinds.has(part.kind)
      && predictedKinds.has(part.kind as LearnedPartHint["kind"])
  ));
  const regions = extraction.semanticRegions ?? [];
  const usedRegions = new Set<string>();
  const usedIds = new Set(parts.map((part) => part.id));
  const nextId = (kind: SemanticPartKind, side: SemanticSide) => {
    const base = `${kind}-${side}`;
    if (!usedIds.has(base)) {
      usedIds.add(base);
      return base;
    }
    let index = 2;
    while (usedIds.has(`${base}-${index}`)) index += 1;
    const id = `${base}-${index}`;
    usedIds.add(id);
    return id;
  };
  const sideFor = (x: number): SemanticSide => x < body.center.x - body.size.x * 0.065
    ? "left"
    : x > body.center.x + body.size.x * 0.065 ? "right" : "center";
  const nearestRegion = (hint: (typeof learned)[number]) => {
    const candidates = regions.filter((region) => !usedRegions.has(region.id));
    let best: SemanticRegionCandidate | null = null;
    let bestScore = Infinity;
    for (const region of candidates) {
      const distance = Math.hypot(region.x - hint.center.x, region.y - hint.center.y);
      const reach = Math.max(0.055, Math.max(hint.size.x, hint.size.y) * 0.76 + Math.max(region.width, region.height) * 0.34);
      if (distance > reach) continue;
      const sizeError = Math.abs(Math.log(Math.max(0.012, region.width) / Math.max(0.012, hint.size.x)))
        + Math.abs(Math.log(Math.max(0.012, region.height) / Math.max(0.012, hint.size.y)));
      const mouthShapePenalty = hint.kind === "mouth" && region.width < region.height * 0.85 ? 0.9 : 0;
      const score = distance / reach * 1.8 + sizeError * 0.22 + mouthShapePenalty;
      if (score < bestScore) {
        best = region;
        bestScore = score;
      }
    }
    if (best) usedRegions.add(best.id);
    return best;
  };

  for (const hint of learned.filter((candidate) => candidate.kind === "eye" || candidate.kind === "cheek" || candidate.kind === "mouth")) {
    const region = nearestRegion(hint);
    const center = region ? { x: region.x, y: region.y, z: 0 } : hint.center;
    const side = hint.kind === "mouth" ? "center" : sideFor(center.x);
    const width = region?.width ?? hint.size.x;
    const height = region?.height ?? hint.size.y;
    const depthScale = hint.kind === "eye" ? 0.24 : hint.kind === "cheek" ? 0.12 : 0.09;
    parts.push({
      id: nextId(hint.kind, side),
      kind: hint.kind,
      side,
      parentId: "body",
      center,
      size: {
        x: clamp(width, body.size.x * 0.022, body.size.x * (hint.kind === "mouth" ? 0.34 : 0.24)),
        y: clamp(height, body.size.y * 0.016, body.size.y * 0.2),
        z: Math.max(0.012, Math.min(width, height) * depthScale),
      },
      rotation: region?.rotation ?? hint.rotation,
      color: region?.color ?? extraction.rig.lineColor,
      confidence: clamp(hint.confidence * 0.72 + (region ? 0.24 : 0.08), 0, 0.98),
      source: "learned-model",
      outline: region?.outline,
    });
  }

  for (const hint of learned.filter((candidate) => candidate.kind === "ear" || candidate.kind === "arm" || candidate.kind === "leg")) {
    const side = sideFor(hint.center.x);
    const existing = parts.filter((part) => part.kind === hint.kind && part.side === side)
      .sort((a, b) => Math.hypot(a.center.x - hint.center.x, a.center.y - hint.center.y)
        - Math.hypot(b.center.x - hint.center.x, b.center.y - hint.center.y))[0];
    if (existing) {
      existing.confidence = Math.max(existing.confidence, hint.confidence);
      existing.source = "learned-model";
      continue;
    }
    if (hint.kind === "ear") {
      const dx = hint.center.x - body.center.x;
      const dy = hint.center.y - body.center.y;
      parts.push({
        id: nextId("ear", side),
        kind: "ear",
        side,
        parentId: "body",
        center: hint.center,
        anchor: { x: body.center.x + dx * 0.62, y: body.center.y + dy * 0.62, z: 0 },
        size: { x: hint.size.x, y: hint.size.y, z: Math.min(hint.size.x, hint.size.y) * 0.68 },
        rotation: hint.rotation,
        color: extraction.rig.bodyColor,
        confidence: hint.confidence,
        source: "learned-model",
      });
      continue;
    }
    const endpoints = hint.endpoints ?? [body.center, hint.center];
    const ordered = [...endpoints].sort((a, b) => Math.hypot(a.x - body.center.x, a.y - body.center.y)
      - Math.hypot(b.x - body.center.x, b.y - body.center.y));
    const anchor = ordered[0];
    const endpoint = ordered[1];
    const dx = endpoint.x - anchor.x;
    const dy = endpoint.y - anchor.y;
    const length = Math.max(Math.min(hint.size.x, hint.size.y) * 1.5, Math.hypot(dx, dy));
    const thickness = clamp(Math.min(hint.size.x, hint.size.y), body.size.x * 0.035, body.size.x * 0.17);
    parts.push({
      id: nextId(hint.kind, side),
      kind: hint.kind,
      side,
      parentId: "body",
      center: endpoint,
      anchor,
      size: { x: thickness, y: length, z: thickness },
      rotation: Math.atan2(-dx, dy),
      color: extraction.rig.bodyColor,
      confidence: hint.confidence,
      source: "learned-model",
    });
  }

  for (const hint of learned.filter((candidate) => candidate.kind === "hand" || candidate.kind === "foot")) {
    const side = sideFor(hint.center.x);
    const existing = parts.filter((part) => part.kind === hint.kind && part.side === side)
      .sort((a, b) => Math.hypot(a.center.x - hint.center.x, a.center.y - hint.center.y)
        - Math.hypot(b.center.x - hint.center.x, b.center.y - hint.center.y))[0];
    if (existing) {
      existing.confidence = Math.max(existing.confidence, hint.confidence);
      existing.source = "learned-model";
      continue;
    }
    const parentKind: SemanticPartKind = hint.kind === "hand" ? "arm" : "leg";
    const parent = parts.filter((part) => part.kind === parentKind)
      .sort((a, b) => Math.hypot(a.center.x - hint.center.x, a.center.y - hint.center.y)
        - Math.hypot(b.center.x - hint.center.x, b.center.y - hint.center.y))[0];
    if (!parent) continue;
    parts.push({
      id: nextId(hint.kind, side),
      kind: hint.kind,
      side,
      parentId: parent.id,
      center: hint.center,
      size: { x: hint.size.x, y: hint.size.y, z: Math.min(hint.size.x, hint.size.y) * 0.78 },
      rotation: hint.rotation,
      color: extraction.rig.bodyColor,
      confidence: hint.confidence,
      source: "learned-model",
    });
  }

  const joints = parts.filter((part) => part.parentId).map((part) => ({
    id: `joint-${part.id}`,
    parentId: part.parentId!,
    childId: part.id,
    x: part.anchor?.x ?? part.center.x,
    y: part.anchor?.y ?? part.center.y,
  }));
  const detectedKinds = [...new Set(parts.map((part) => part.kind))];
  return {
    ...extraction,
    rig: { ...extraction.rig, parts, joints, detectedKinds },
    learnedRecognition: { model: "wallalive-parts-v2-ensemble", latencyMs, detectedKinds: [...predictedKinds] },
  };
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
  const chromaticInk = new Uint8Array(width * height);
  const scanInsetX = Math.round(width * 0.08);
  const scanInsetTop = Math.round(height * 0.09);
  const scanInsetBottom = Math.round(height * 0.16);
  const focusX = target.x * width;
  const focusY = target.y * height;
  const focusRadiusX = width * 0.27;
  const focusRadiusY = height * 0.33;

  for (let y = scanInsetTop; y < height - scanInsetBottom; y += 1) {
    for (let x = scanInsetX; x < width - scanInsetX; x += 1) {
      if (((x - focusX) / focusRadiusX) ** 2 + ((y - focusY) / focusRadiusY) ** 2 > 1) continue;
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
      const coloredLine = chroma >= Math.max(9, backgroundChroma + 3)
        && colorDistance(pixel, background) > 12
        && lightness < 250;
      if (coloredLine) chromaticInk[y * width + x] = 1;
      if (inkScore(pixel, background) > 54 && (vividInk || darkInk || contrastedStroke)) rawInk[y * width + x] = 1;
    }
  }

  const colorCloseRadius = clamp(Math.round(Math.min(width, height) * 0.014), 5, 9);
  const enclosedTargetRegion = recoverEnclosedTargetRegion(chromaticInk, width, height, target);
  const preconnectedChromaticInk = erode(dilate(chromaticInk, width, height, 2), width, height, 1);
  const isolatedChromaticInk = isolateTargetInk(preconnectedChromaticInk, width, height, target);
  const colorConnectedInk = erode(dilate(isolatedChromaticInk, width, height, colorCloseRadius), width, height, Math.max(2, colorCloseRadius - 3));
  const localizedTargetInk = enclosedTargetRegion ? inkAroundEnclosedRegion(chromaticInk, enclosedTargetRegion, width, height) : null;
  const localizedClosedInk = localizedTargetInk
    ? erode(dilate(localizedTargetInk, width, height, colorCloseRadius), width, height, Math.max(2, colorCloseRadius - 3))
    : null;
  const enclosedComponent = enclosedTargetRegion ? connectedComponents(enclosedTargetRegion, width, height)[0] : null;
  const silhouetteTarget = enclosedComponent ? {
    x: target.x * 0.5 + enclosedComponent.centerX / width * 0.5,
    y: target.y * 0.5 + enclosedComponent.centerY / height * 0.5,
  } : target;
  const recoveredColorSilhouette = localizedClosedInk
    ? recoverTargetSilhouette(localizedClosedInk, width, height, silhouetteTarget, false)
    : recoverTargetSilhouette(colorConnectedInk, width, height, target);
  const colorSolidComponents = connectedComponents(recoveredColorSilhouette, width, height);
  const colorAnchor = chooseDrawing(colorSolidComponents, frame.data, width, height, target);
  const connectedInk = erode(dilate(rawInk, width, height, 2), width, height, 1);
  const fallbackComponents = connectedComponents(connectedInk, width, height);
  const colorInkCount = colorAnchor?.pixels.reduce((count, index) => count + chromaticInk[index], 0) ?? 0;
  const useColorInk = Boolean(colorAnchor && (enclosedTargetRegion || colorInkCount >= Math.max(24, colorAnchor.pixels.length * 0.002)));
  const components = useColorInk ? colorSolidComponents : fallbackComponents;
  const anchor = useColorInk ? colorAnchor : chooseDrawing(fallbackComponents, frame.data, width, height, target);
  if (!anchor) throw new Error("I couldn't find one clear character outline. Tap the drawing, move closer, and capture again.");

  const span = Math.max(anchor.maxX - anchor.minX, anchor.maxY - anchor.minY);
  const anchorArea = Math.max(1, (anchor.maxX - anchor.minX + 1) * (anchor.maxY - anchor.minY + 1));
  const anchorColor = componentColor(anchor, frame.data);
  const selected = useColorInk ? [anchor] : components.filter((component) => {
    if (component === anchor) return true;
    const componentArea = (component.maxX - component.minX + 1) * (component.maxY - component.minY + 1);
    const withinCharacterBounds = component.centerX >= anchor.minX - span * 0.035
      && component.centerX <= anchor.maxX + span * 0.035
      && component.centerY >= anchor.minY - span * 0.035
      && component.centerY <= anchor.maxY + span * 0.035;
    const colorMatches = !useColorInk || hueDistance(componentColor(component, frame.data), anchorColor) < 42;
    return withinCharacterBounds && colorMatches && componentArea <= anchorArea * 0.22 && component.pixels.length >= 8;
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
  const closureRadius = clamp(Math.round(Math.max(cropWidth, cropHeight) * 0.018), 3, 7);
  const sealed = erode(dilate(selectedMask, cropWidth, cropHeight, closureRadius), cropWidth, cropHeight, Math.max(1, closureRadius - 2));
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
  let colorWeight = 0;
  for (let index = 0; index < silhouette.length; index += 1) {
    if (!silhouette[index]) continue;
    const rgba = index * 4;
    transparent.data[rgba] = cropped.data[rgba];
    transparent.data[rgba + 1] = cropped.data[rgba + 1];
    transparent.data[rgba + 2] = cropped.data[rgba + 2];
    transparent.data[rgba + 3] = 255;
    const localX = index % cropWidth;
    const localY = Math.floor(index / cropWidth);
    const sourceIndex = (localY + minY) * width + localX + minX;
    if (useColorInk ? chromaticInk[sourceIndex] : rawInk[sourceIndex]) {
      inkPixels += 1;
      const chroma = Math.max(cropped.data[rgba], cropped.data[rgba + 1], cropped.data[rgba + 2])
        - Math.min(cropped.data[rgba], cropped.data[rgba + 1], cropped.data[rgba + 2]);
      const weight = useColorInk ? Math.max(1, (chroma - backgroundChroma) ** 1.55) : 1;
      colorR += cropped.data[rgba] * weight;
      colorG += cropped.data[rgba + 1] * weight;
      colorB += cropped.data[rgba + 2] * weight;
      colorWeight += weight;
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

  const dominant = colorWeight ? boostInkColor({ r: colorR / colorWeight, g: colorG / colorWeight, b: colorB / colorWeight }) : background;
  const coverage = inkPixels / (width * height);
  const contour = contourFromCanvas(output);
  if (contour.length < 6) {
    const diagnostic = candidateFeatures(anchor, frame.data, width);
    throw new Error(`The selected component was not a closed character (box ${cropWidth}×${cropHeight}, density ${(diagnostic.pixelCount / ((diagnostic.maxX - diagnostic.minX + 1) * (diagnostic.maxY - diagnostic.minY + 1))).toFixed(2)}, chroma ${diagnostic.averageChroma.toFixed(0)}, border ${(diagnostic.edgeFraction ?? 0).toFixed(2)}). Tap inside the character body and capture again.`);
  }
  const skeleton = skeletonFromTexture(output);
  if (!skeleton.length) throw new Error("The drawing body could not be skeletonized. Use one closed, bold character outline.");
  const semantic = analyzeSemanticCanvas(output, dominant, coverage < 0.035);
  const rig = inferSemanticRig(skeleton, contour, semantic.regions, semantic.bodyColor, semantic.lineColor);

  return {
    textureUrl: output.toDataURL("image/png"),
    previewUrl: source.toDataURL("image/jpeg", 0.82),
    contour,
    skeleton,
    rig,
    semanticRegions: semantic.regions,
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
  const bounds = video.getBoundingClientRect();
  const sourceTarget = mapCoverTargetToSource(
    target,
    video.videoWidth,
    video.videoHeight,
    bounds.width || video.clientWidth,
    bounds.height || video.clientHeight,
  );
  return extractDrawingFromSource(video, video.videoWidth, video.videoHeight, sourceTarget);
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

  const limb = (startX: number, startY: number, endX: number, endY: number) => {
    context.beginPath();
    context.moveTo(startX, startY);
    context.lineTo(endX, endY);
    context.strokeStyle = "#18312e";
    context.lineWidth = 48;
    context.stroke();
    context.strokeStyle = "#ff674d";
    context.lineWidth = 29;
    context.stroke();
  };
  limb(124, 247, 55, 194);
  limb(399, 247, 465, 187);
  limb(210, 397, 188, 469);
  limb(315, 400, 337, 469);
  context.strokeStyle = "#18312e";
  context.lineWidth = 15;

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

export async function extractDrawingFromImageUrl(imageUrl: string, target: CaptureTarget = { x: 0.5, y: 0.5 }): Promise<DrawingExtraction> {
  const image = new Image();
  image.decoding = "async";
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("The drawing image could not be loaded."));
    image.src = imageUrl;
  });
  return extractDrawingFromSource(image, image.naturalWidth, image.naturalHeight, target);
}

export async function createAniGenDemoDrawing(): Promise<DrawingExtraction> {
  return extractDrawingFromImageUrl("/anigen-demo-input.png", { x: 0.5, y: 0.47 });
}
