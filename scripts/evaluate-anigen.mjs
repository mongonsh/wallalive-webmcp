#!/usr/bin/env node

import { Client, handle_file } from "@gradio/client";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const SPACE = "VAST-AI/AniGen";
const SPACE_FILE_ROOT = "https://vast-ai-anigen.hf.space/file=";

function parseArgs(argv) {
  const options = {
    benchmark: "eval/varied-drawings/manifest.json",
    outputDir: "/private/tmp/wallalive-anigen-varied",
    report: "eval/anigen-varied-report.json",
    selectedCase: null,
    inspect: null,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (key === "--benchmark") options.benchmark = argv[++index];
    else if (key === "--output-dir") options.outputDir = argv[++index];
    else if (key === "--report") options.report = argv[++index];
    else if (key === "--case") options.selectedCase = argv[++index];
    else if (key === "--inspect") options.inspect = argv[++index];
    else throw new Error(`Unknown argument: ${key}`);
  }
  return options;
}

function collectFiles(value, files = []) {
  if (!value || typeof value !== "object") return files;
  if (Array.isArray(value)) {
    value.forEach((item) => collectFiles(item, files));
    return files;
  }
  if (typeof value.url === "string" || typeof value.path === "string") files.push(value);
  Object.values(value).forEach((item) => collectFiles(item, files));
  return files;
}

function remoteFileUrl(file) {
  const candidate = file?.url ?? file?.path;
  if (!candidate) return null;
  if (/^https?:\/\//i.test(candidate)) return candidate;
  return `${SPACE_FILE_ROOT}${candidate.startsWith("/") ? candidate : `/${candidate}`}`;
}

function parseGlb(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2) {
    throw new Error("Result is not a glTF 2.0 binary file");
  }
  const totalLength = view.getUint32(8, true);
  let cursor = 12;
  let document = null;
  let binary = null;
  while (cursor < totalLength) {
    const length = view.getUint32(cursor, true);
    const type = view.getUint32(cursor + 4, true);
    cursor += 8;
    const chunk = bytes.subarray(cursor, cursor + length);
    cursor += length;
    if (type === 0x4e4f534a) {
      let jsonText = new TextDecoder().decode(chunk);
      while (jsonText.endsWith("\0") || jsonText.endsWith(" ")) jsonText = jsonText.slice(0, -1);
      document = JSON.parse(jsonText);
    } else if (type === 0x004e4942) binary = chunk;
  }
  if (!document || !binary) throw new Error("GLB is missing its JSON or binary chunk");
  return { document, binary };
}

const COMPONENTS = {
  5120: { bytes: 1, read: (view, offset) => view.getInt8(offset), max: 127 },
  5121: { bytes: 1, read: (view, offset) => view.getUint8(offset), max: 255 },
  5122: { bytes: 2, read: (view, offset) => view.getInt16(offset, true), max: 32767 },
  5123: { bytes: 2, read: (view, offset) => view.getUint16(offset, true), max: 65535 },
  5125: { bytes: 4, read: (view, offset) => view.getUint32(offset, true), max: 4294967295 },
  5126: { bytes: 4, read: (view, offset) => view.getFloat32(offset, true), max: 1 },
};
const TYPE_LENGTH = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function accessorReader(document, binary, accessorIndex) {
  const accessor = document.accessors[accessorIndex];
  const bufferView = document.bufferViews[accessor.bufferView];
  const component = COMPONENTS[accessor.componentType];
  const components = TYPE_LENGTH[accessor.type];
  if (!component || !components) throw new Error(`Unsupported accessor ${accessor.componentType}/${accessor.type}`);
  const data = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
  const start = (bufferView.byteOffset ?? 0) + (accessor.byteOffset ?? 0);
  const stride = bufferView.byteStride ?? component.bytes * components;
  const read = (item, componentIndex = 0) => {
    const raw = component.read(data, start + item * stride + componentIndex * component.bytes);
    return accessor.normalized && accessor.componentType !== 5126 ? Math.max(-1, raw / component.max) : raw;
  };
  return { accessor, components, read };
}

function primitiveSurfaceStats(document, binary, primitive) {
  if (primitive.indices == null || (primitive.mode ?? 4) !== 4) return null;
  const indices = accessorReader(document, binary, primitive.indices);
  const edgeCounts = new Map();
  let maximumIndex = 0;
  for (let item = 0; item < indices.accessor.count; item += 1) maximumIndex = Math.max(maximumIndex, indices.read(item));
  const edgeBase = maximumIndex + 1;
  for (let item = 0; item + 2 < indices.accessor.count; item += 3) {
    const triangle = [indices.read(item), indices.read(item + 1), indices.read(item + 2)];
    for (const [left, right] of [[0, 1], [1, 2], [2, 0]]) {
      const minimum = Math.min(triangle[left], triangle[right]);
      const maximum = Math.max(triangle[left], triangle[right]);
      const key = minimum * edgeBase + maximum;
      edgeCounts.set(key, (edgeCounts.get(key) ?? 0) + 1);
    }
  }
  let boundaryEdges = 0;
  let nonManifoldEdges = 0;
  for (const count of edgeCounts.values()) {
    if (count === 1) boundaryEdges += 1;
    if (count > 2) nonManifoldEdges += 1;
  }
  return {
    triangles: Math.floor(indices.accessor.count / 3),
    uniqueEdges: edgeCounts.size,
    boundaryEdges,
    nonManifoldEdges,
  };
}

function inspectGlb(buffer) {
  const { document, binary } = parseGlb(buffer);
  const bounds = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
  const surface = { triangles: 0, uniqueEdges: 0, boundaryEdges: 0, nonManifoldEdges: 0 };
  let vertices = 0;
  let primitives = 0;
  let skinnedPrimitives = 0;
  let coloredPrimitives = 0;
  let validWeightVertices = 0;
  let weightVertices = 0;

  for (const mesh of document.meshes ?? []) {
    for (const primitive of mesh.primitives ?? []) {
      primitives += 1;
      const positionIndex = primitive.attributes?.POSITION;
      if (positionIndex != null) {
        const position = document.accessors[positionIndex];
        vertices += position.count;
        if (position.min && position.max) {
          for (let axis = 0; axis < 3; axis += 1) {
            bounds.min[axis] = Math.min(bounds.min[axis], position.min[axis]);
            bounds.max[axis] = Math.max(bounds.max[axis], position.max[axis]);
          }
        }
      }
      const hasSkin = primitive.attributes?.JOINTS_0 != null && primitive.attributes?.WEIGHTS_0 != null;
      if (hasSkin) {
        skinnedPrimitives += 1;
        const weights = accessorReader(document, binary, primitive.attributes.WEIGHTS_0);
        weightVertices += weights.accessor.count;
        for (let item = 0; item < weights.accessor.count; item += 1) {
          let sum = 0;
          for (let component = 0; component < weights.components; component += 1) sum += weights.read(item, component);
          if (sum > 0.99 && sum < 1.01) validWeightVertices += 1;
        }
      }
      const material = primitive.material == null ? null : document.materials?.[primitive.material];
      if (primitive.attributes?.COLOR_0 != null || material?.pbrMetallicRoughness?.baseColorTexture || material?.pbrMetallicRoughness?.baseColorFactor) {
        coloredPrimitives += 1;
      }
      const stats = primitiveSurfaceStats(document, binary, primitive);
      if (stats) for (const key of Object.keys(surface)) surface[key] += stats[key];
    }
  }

  const spans = bounds.min.map((minimum, axis) => bounds.max[axis] - minimum);
  const maximumSkinJoints = Math.max(0, ...(document.skins ?? []).map((skin) => skin.joints?.length ?? 0));
  const boundaryEdgeRatio = surface.uniqueEdges ? surface.boundaryEdges / surface.uniqueEdges : 1;
  const nonManifoldEdgeRatio = surface.uniqueEdges ? surface.nonManifoldEdges / surface.uniqueEdges : 1;
  const weightNormalizationRatio = weightVertices ? validWeightVertices / weightVertices : 0;
  return {
    bytes: buffer.byteLength,
    meshes: document.meshes?.length ?? 0,
    primitives,
    vertices,
    triangles: surface.triangles,
    bounds: { min: bounds.min, max: bounds.max, spans },
    depthRatio: spans[2] / Math.max(1e-9, Math.max(spans[0], spans[1])),
    skins: document.skins?.length ?? 0,
    bones: maximumSkinJoints,
    skinnedPrimitives,
    weightNormalizationRatio,
    materials: document.materials?.length ?? 0,
    coloredPrimitives,
    boundaryEdges: surface.boundaryEdges,
    boundaryEdgeRatio,
    nonManifoldEdges: surface.nonManifoldEdges,
    nonManifoldEdgeRatio,
    closedSurface: surface.uniqueEdges > 0 && boundaryEdgeRatio <= 0.001 && nonManifoldEdgeRatio <= 0.001,
    trueVolume: spans.every((span) => span > 1e-4) && spans[2] / Math.max(spans[0], spans[1]) >= 0.08,
    rigged: maximumSkinJoints >= 2 && skinnedPrimitives >= 1 && weightNormalizationRatio >= 0.99,
    colorData: coloredPrimitives === primitives && primitives > 0,
  };
}

async function fetchFile(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${url} (${response.status})`);
  return response.arrayBuffer();
}

async function writeReport(reportPath, report) {
  await mkdir(path.dirname(reportPath), { recursive: true });
  await writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);
}

async function generateCase(client, benchmarkDir, outputDir, benchmarkCase) {
  const inputPath = path.resolve(benchmarkDir, benchmarkCase.input);
  console.log(`[${benchmarkCase.id}] preparing ${inputPath}`);
  await client.predict("/start_session", []);
  const prepared = await client.predict("/prepare_input_for_generation", [handle_file(inputPath)]);
  const preparedInput = Array.isArray(prepared.data) ? prepared.data[0] : prepared.data;
  console.log(`[${benchmarkCase.id}] queued on AniGen public GPU`);
  const preview = await client.predict("/generate_preview", [
    preparedInput,
    20260831,
    "ss_flow_solo",
    "slat_flow_auto",
    7.5,
    20,
    3,
    20,
    1,
  ]);
  const files = collectFiles(preview.data);
  const glbs = files.filter((file) => /\.glb(?:$|\?)/i.test(file.url ?? file.path ?? "") || file.mime_type === "model/gltf-binary");
  const meshFile = glbs.find((file) => !/skeleton/i.test(file.url ?? file.path ?? "")) ?? glbs[0];
  const skeletonFile = glbs.find((file) => /skeleton/i.test(file.url ?? file.path ?? "")) ?? glbs[1];
  const meshUrl = remoteFileUrl(meshFile);
  if (!meshUrl) throw new Error("AniGen returned no rigged GLB mesh");
  const meshBuffer = await fetchFile(meshUrl);
  const meshPath = path.join(outputDir, `${benchmarkCase.id}.glb`);
  await writeFile(meshPath, new Uint8Array(meshBuffer));
  const skeletonUrl = remoteFileUrl(skeletonFile);
  if (skeletonUrl && skeletonUrl !== meshUrl) {
    const skeletonBuffer = await fetchFile(skeletonUrl);
    await writeFile(path.join(outputDir, `${benchmarkCase.id}-skeleton.glb`), new Uint8Array(skeletonBuffer));
  }
  const inspection = inspectGlb(meshBuffer);
  console.log(`[${benchmarkCase.id}] ${inspection.vertices} vertices, ${inspection.bones} bones, depth ${(inspection.depthRatio * 100).toFixed(1)}%, closed=${inspection.closedSurface}`);
  return {
    id: benchmarkCase.id,
    input: benchmarkCase.input,
    expected_topology: benchmarkCase.expected_topology,
    expected_visible_parts: benchmarkCase.expected_visible_parts,
    render_palette: benchmarkCase.render_palette,
    output_file: meshPath,
    inspection,
    passed_structural_contract: inspection.closedSurface && inspection.trueVolume && inspection.rigged && inspection.colorData,
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.inspect) {
    const file = await readFile(options.inspect);
    const buffer = file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength);
    const inspection = inspectGlb(buffer);
    console.log(JSON.stringify({ file: options.inspect, inspection }, null, 2));
    if (!(inspection.closedSurface && inspection.trueVolume && inspection.rigged && inspection.colorData)) process.exitCode = 1;
    return;
  }
  const benchmarkPath = path.resolve(options.benchmark);
  const benchmark = JSON.parse(await readFile(benchmarkPath, "utf8"));
  const benchmarkDir = path.dirname(benchmarkPath);
  const cases = benchmark.cases.filter((item) => !options.selectedCase || item.id === options.selectedCase);
  if (!cases.length) throw new Error(`No benchmark case matched ${options.selectedCase}`);
  await mkdir(options.outputDir, { recursive: true });
  const report = {
    benchmark: benchmark.name,
    provider: "AniGen",
    model: "ss_flow_solo + slat_flow_auto",
    generated_at: new Date().toISOString(),
    cases: [],
  };
  const client = await Client.connect(SPACE);
  try {
    for (const benchmarkCase of cases) {
      try {
        report.cases.push(await generateCase(client, benchmarkDir, options.outputDir, benchmarkCase));
      } catch (error) {
        console.error(`[${benchmarkCase.id}] failed:`, error instanceof Error ? error.message : error);
        report.cases.push({ id: benchmarkCase.id, error: error instanceof Error ? error.message : String(error), passed_structural_contract: false });
      }
      await writeReport(options.report, report);
    }
  } finally {
    client.close();
  }
  report.passed = report.cases.length > 0 && report.cases.every((item) => item.passed_structural_contract);
  await writeReport(options.report, report);
  if (!report.passed) process.exitCode = 1;
}

await main();
