import * as THREE from "three";
import type { ContourPoint, LearnedDepthField } from "./drawing";

export type ArtworkShellResult = {
  geometry: THREE.BufferGeometry;
  frontTriangleCount: number;
  backTriangleCount: number;
  sideTriangleCount: number;
  maximumHalfDepth: number;
  boundaryVertexCount: number;
};

const clamp = (value: number, minimum: number, maximum: number) => Math.min(maximum, Math.max(minimum, value));

function cleanContour(contour: ContourPoint[]) {
  const cleaned: ContourPoint[] = [];
  contour.forEach((point) => {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return;
    const previous = cleaned.at(-1);
    if (!previous || Math.hypot(point.x - previous.x, point.y - previous.y) > 0.002) cleaned.push({ x: point.x, y: point.y });
  });
  if (cleaned.length > 2 && Math.hypot(cleaned[0].x - cleaned.at(-1)!.x, cleaned[0].y - cleaned.at(-1)!.y) < 0.002) cleaned.pop();
  if (cleaned.length < 3) throw new Error("The character outline is too small to build a closed 3D shell.");
  return cleaned;
}

function pointInsideContour(x: number, y: number, contour: ContourPoint[]) {
  let inside = false;
  for (let current = 0, previous = contour.length - 1; current < contour.length; previous = current, current += 1) {
    const a = contour[current];
    const b = contour[previous];
    if ((a.y > y) !== (b.y > y) && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}

function distanceToSegment(x: number, y: number, start: ContourPoint, end: ContourPoint) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  const amount = lengthSquared ? clamp(((x - start.x) * dx + (y - start.y) * dy) / lengthSquared, 0, 1) : 0;
  return Math.hypot(x - (start.x + dx * amount), y - (start.y + dy * amount));
}

function distanceToContour(x: number, y: number, contour: ContourPoint[]) {
  let distance = Infinity;
  for (let index = 0; index < contour.length; index += 1) {
    distance = Math.min(distance, distanceToSegment(x, y, contour[index], contour[(index + 1) % contour.length]));
  }
  return distance;
}

function sampleLearnedDepth(field: Float32Array, depth: LearnedDepthField, x: number, y: number) {
  const modelX = clamp((x / 1.4 + 0.5) * (depth.size - 1), 0, depth.size - 1);
  const modelY = clamp((0.5 - y / 1.4) * (depth.size - 1), 0, depth.size - 1);
  const x0 = Math.floor(modelX);
  const y0 = Math.floor(modelY);
  const x1 = Math.min(depth.size - 1, x0 + 1);
  const y1 = Math.min(depth.size - 1, y0 + 1);
  const amountX = modelX - x0;
  const amountY = modelY - y0;
  const top = field[y0 * depth.size + x0] * (1 - amountX) + field[y0 * depth.size + x1] * amountX;
  const bottom = field[y1 * depth.size + x0] * (1 - amountX) + field[y1 * depth.size + x1] * amountX;
  return (top * (1 - amountY) + bottom * amountY) * depth.depthScale;
}

function signedArea(contour: ContourPoint[]) {
  let area = 0;
  for (let index = 0; index < contour.length; index += 1) {
    const current = contour[index];
    const next = contour[(index + 1) % contour.length];
    area += current.x * next.y - next.x * current.y;
  }
  return area * 0.5;
}

function maximumInteriorDistance(contour: ContourPoint[]) {
  const minX = Math.min(...contour.map((point) => point.x));
  const maxX = Math.max(...contour.map((point) => point.x));
  const minY = Math.min(...contour.map((point) => point.y));
  const maxY = Math.max(...contour.map((point) => point.y));
  let maximum = Math.max((maxX - minX) / 64, (maxY - minY) / 64, 0.01);
  for (let row = 0; row <= 64; row += 1) {
    const y = minY + (maxY - minY) * row / 64;
    for (let column = 0; column <= 64; column += 1) {
      const x = minX + (maxX - minX) * column / 64;
      if (pointInsideContour(x, y, contour)) maximum = Math.max(maximum, distanceToContour(x, y, contour));
    }
  }
  return maximum;
}

type Point2 = { x: number; y: number };

function midpoint(a: Point2, b: Point2): Point2 {
  return { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
}

function subdivideTriangle(a: Point2, b: Point2, c: Point2, levels: number, triangles: Point2[][]) {
  if (levels <= 0) {
    triangles.push([a, b, c]);
    return;
  }
  const ab = midpoint(a, b);
  const bc = midpoint(b, c);
  const ca = midpoint(c, a);
  subdivideTriangle(a, ab, ca, levels - 1, triangles);
  subdivideTriangle(ab, b, bc, levels - 1, triangles);
  subdivideTriangle(ca, bc, c, levels - 1, triangles);
  subdivideTriangle(ab, bc, ca, levels - 1, triangles);
}

export function buildArtworkShellGeometry(
  sourceContour: ContourPoint[],
  depth: LearnedDepthField | null,
  requestedHalfDepth: number,
  inflation = 1,
  subdivisions = 2,
): ArtworkShellResult {
  const contour = cleanContour(sourceContour);
  const triangulationContour = contour.map((point) => new THREE.Vector2(point.x, point.y));
  const faces = THREE.ShapeUtils.triangulateShape(triangulationContour, []);
  if (!faces.length) throw new Error("The character outline could not be triangulated into a stable 3D shell.");
  const outline = triangulationContour.map((point) => ({ x: point.x, y: point.y }));
  const winding = signedArea(outline);
  const maximumDistance = maximumInteriorDistance(outline);
  const maximumHalfDepth = clamp(requestedHalfDepth * inflation, 0.065, 0.18);
  const rimHalfDepth = clamp(maximumHalfDepth * 0.14, 0.012, 0.024);

  const shellDepth = (x: number, y: number, side: "front" | "back") => {
    const normalizedDistance = clamp(distanceToContour(x, y, outline) / maximumDistance, 0, 1);
    const envelope = rimHalfDepth + (maximumHalfDepth - rimHalfDepth) * Math.pow(normalizedDistance, 0.58);
    if (!depth) return envelope;
    const learned = sampleLearnedDepth(side === "front" ? depth.front : depth.back, depth, x, y) * inflation;
    const learnedRatio = clamp(learned / Math.max(0.001, maximumHalfDepth), 0.72, 1.16);
    return clamp(envelope * (0.82 + learnedRatio * 0.18), rimHalfDepth, maximumHalfDepth * 1.04);
  };

  const positions: number[] = [];
  const uvs: number[] = [];
  const vertices = new Map<string, number>();
  const vertexFor = (point: Point2, side: "front" | "back") => {
    const key = `${side}:${Math.round(point.x * 1_000_000)}:${Math.round(point.y * 1_000_000)}`;
    const existing = vertices.get(key);
    if (existing !== undefined) return existing;
    const vertex = positions.length / 3;
    const magnitude = shellDepth(point.x, point.y, side);
    positions.push(point.x, point.y, side === "front" ? magnitude : -magnitude);
    uvs.push(clamp(point.x / 1.4 + 0.5, 0, 1), clamp(point.y / 1.4 + 0.5, 0, 1));
    vertices.set(key, vertex);
    return vertex;
  };

  const frontIndices: number[] = [];
  const backIndices: number[] = [];
  faces.forEach((face) => {
    const base = face.map((index) => outline[index]);
    const triangles: Point2[][] = [];
    subdivideTriangle(base[0], base[1], base[2], clamp(Math.round(subdivisions), 0, 3), triangles);
    triangles.forEach(([a, b, c]) => {
      const cross = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
      const front = cross >= 0 ? [a, b, c] : [a, c, b];
      const frontFace = front.map((point) => vertexFor(point, "front"));
      frontIndices.push(frontFace[0], frontFace[1], frontFace[2]);
      const backFace = front.map((point) => vertexFor(point, "back"));
      backIndices.push(backFace[0], backFace[2], backFace[1]);
    });
  });

  const sideIndices: number[] = [];
  const edgeSegments = 2 ** clamp(Math.round(subdivisions), 0, 3);
  for (let index = 0; index < outline.length; index += 1) {
    const start = outline[index];
    const end = outline[(index + 1) % outline.length];
    for (let segment = 0; segment < edgeSegments; segment += 1) {
      const amountA = segment / edgeSegments;
      const amountB = (segment + 1) / edgeSegments;
      const a = { x: start.x + (end.x - start.x) * amountA, y: start.y + (end.y - start.y) * amountA };
      const b = { x: start.x + (end.x - start.x) * amountB, y: start.y + (end.y - start.y) * amountB };
      const frontA = vertexFor(a, "front");
      const frontB = vertexFor(b, "front");
      const backA = vertexFor(a, "back");
      const backB = vertexFor(b, "back");
      if (winding >= 0) {
        sideIndices.push(frontA, backA, frontB, frontB, backA, backB);
      } else {
        sideIndices.push(frontA, frontB, backA, frontB, backB, backA);
      }
    }
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex([...frontIndices, ...backIndices, ...sideIndices]);
  geometry.clearGroups();
  geometry.addGroup(0, frontIndices.length, 1);
  geometry.addGroup(frontIndices.length, backIndices.length, 2);
  geometry.addGroup(frontIndices.length + backIndices.length, sideIndices.length, 0);
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  return {
    geometry,
    frontTriangleCount: frontIndices.length / 3,
    backTriangleCount: backIndices.length / 3,
    sideTriangleCount: sideIndices.length / 3,
    maximumHalfDepth,
    boundaryVertexCount: outline.length * edgeSegments * 2,
  };
}
