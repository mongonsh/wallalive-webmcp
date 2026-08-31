import type { Client } from "@gradio/client";

export const ANIGEN_SPACE = "VAST-AI/AniGen";
export const ANIGEN_PROVIDER = "AniGen" as const;
export const ANIGEN_MODEL = "ss_flow_solo + slat_flow_auto" as const;
export const WALLALIVE_NEURAL_PROVIDER = "WallAlive Sketch Neural Lab" as const;
export const WALLALIVE_NEURAL_MODEL = "sketch-to-render + TripoSR + variable graph skin" as const;

export type NeuralReconstructionPhase =
  | "idle"
  | "consent-required"
  | "connecting"
  | "preparing"
  | "queued"
  | "generating"
  | "downloading"
  | "ready"
  | "error";

export type NeuralProgress = {
  phase: NeuralReconstructionPhase;
  progress: number;
  message: string;
};

export type NeuralAsset = {
  source: "anigen-live" | "wallalive-neural-demo";
  provider: typeof ANIGEN_PROVIDER | typeof WALLALIVE_NEURAL_PROVIDER;
  model: typeof ANIGEN_MODEL | typeof WALLALIVE_NEURAL_MODEL;
  meshUrl: string;
  skeletonUrl?: string;
  processedImageUrl?: string;
  preview: boolean;
  generatedAt: string;
};

type AniGenFile = { url?: string; path?: string; mime_type?: string };

function abortError() {
  return new DOMException("3D generation cancelled", "AbortError");
}

function ensureActive(signal?: AbortSignal) {
  if (signal?.aborted) throw abortError();
}

function collectFiles(value: unknown, files: AniGenFile[] = []): AniGenFile[] {
  if (!value || typeof value !== "object") return files;
  if (Array.isArray(value)) {
    value.forEach((item) => collectFiles(item, files));
    return files;
  }
  const candidate = value as Record<string, unknown>;
  if (typeof candidate.url === "string" || typeof candidate.path === "string") {
    files.push({
      url: typeof candidate.url === "string" ? candidate.url : undefined,
      path: typeof candidate.path === "string" ? candidate.path : undefined,
      mime_type: typeof candidate.mime_type === "string" ? candidate.mime_type : undefined,
    });
  }
  Object.values(candidate).forEach((item) => collectFiles(item, files));
  return files;
}

function fileUrl(file: AniGenFile | undefined) {
  const url = file?.url ?? file?.path;
  if (!url) return null;
  if (/^https?:\/\//i.test(url)) return url;
  return `https://vast-ai-anigen.hf.space/file=${url.startsWith("/") ? url : `/${url}`}`;
}

export function parseAniGenPreview(data: unknown) {
  const files = collectFiles(data);
  const glbs = files.filter((file) => /\.glb(?:$|\?)/i.test(file.url ?? file.path ?? "") || file.mime_type === "model/gltf-binary");
  const mesh = glbs.find((file) => !/skeleton/i.test(file.url ?? file.path ?? "")) ?? glbs[0];
  const skeleton = glbs.find((file) => /skeleton/i.test(file.url ?? file.path ?? "")) ?? glbs[1];
  const meshUrl = fileUrl(mesh);
  if (!meshUrl) throw new Error("AniGen completed without returning a rigged GLB mesh.");
  return { meshUrl, skeletonUrl: fileUrl(skeleton) ?? undefined };
}

async function imageBlob(imageUrl: string, signal?: AbortSignal) {
  ensureActive(signal);
  const response = await fetch(imageUrl, { signal });
  if (!response.ok) throw new Error(`The approved drawing could not be prepared (${response.status}).`);
  const blob = await response.blob();
  return new File([blob], "wallalive-approved-drawing.png", { type: blob.type || "image/png" });
}

async function preserveRemoteFile(url: string, signal?: AbortSignal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`The generated 3D file could not be downloaded (${response.status}).`);
  return URL.createObjectURL(await response.blob());
}

function friendlyError(error: unknown) {
  if (error instanceof DOMException && error.name === "AbortError") return error;
  const message = error instanceof Error ? error.message : String(error);
  if (/quota|GPU task aborted|ZeroGPU|requested vs\./i.test(message)) {
    return new Error("AniGen's public GPU time is temporarily full. The no-wait rigged judge demo still works; live generation can be retried later or pointed at a self-hosted AniGen GPU.");
  }
  if (/failed to fetch|network|load failed/i.test(message)) {
    return new Error("AniGen could not be reached from this browser. Check the connection and try again; the approved drawing remains local.");
  }
  return error instanceof Error ? error : new Error(message);
}

export async function generateAniGenAsset(
  imageUrl: string,
  onProgress?: (progress: NeuralProgress) => void,
  signal?: AbortSignal,
): Promise<NeuralAsset> {
  let client: Client | null = null;
  const update = (phase: NeuralReconstructionPhase, progress: number, message: string) => onProgress?.({ phase, progress, message });
  try {
    ensureActive(signal);
    update("connecting", 0.06, "Connecting to AniGen's rigged 3D model…");
    const gradio = await import("@gradio/client");
    client = await gradio.Client.connect(ANIGEN_SPACE);
    ensureActive(signal);

    const approvedDrawing = await imageBlob(imageUrl, signal);
    await client.predict("/start_session", []);
    ensureActive(signal);
    update("preparing", 0.17, "Preparing only the approved isolated drawing…");
    const prepared = await client.predict<unknown[]>("/prepare_input_for_generation", [gradio.handle_file(approvedDrawing)]);
    ensureActive(signal);

    update("queued", 0.28, "Waiting for the public GPU…");
    update("generating", 0.34, "Generating unseen surfaces, skeleton, and skin weights…");
    const preview = await client.predict<unknown[]>("/generate_preview", [
      Array.isArray(prepared.data) ? prepared.data[0] : prepared.data,
      Math.floor(Math.random() * 2_147_483_647),
      "ss_flow_solo",
      "slat_flow_auto",
      7.5,
      20,
      3,
      20,
      1,
    ]);
    ensureActive(signal);
    const remote = parseAniGenPreview(preview.data);

    update("downloading", 0.91, "Securing the rigged GLB in this browser tab…");
    const meshUrl = await preserveRemoteFile(remote.meshUrl, signal);
    const skeletonUrl = remote.skeletonUrl ? await preserveRemoteFile(remote.skeletonUrl, signal) : undefined;
    update("ready", 1, "Real rigged 3D is ready.");
    return {
      source: "anigen-live",
      provider: ANIGEN_PROVIDER,
      model: ANIGEN_MODEL,
      meshUrl,
      skeletonUrl,
      preview: true,
      generatedAt: new Date().toISOString(),
    };
  } catch (error) {
    throw friendlyError(error);
  } finally {
    client?.close();
  }
}

export function createBundledAniGenAsset(): NeuralAsset {
  return {
    source: "wallalive-neural-demo",
    provider: WALLALIVE_NEURAL_PROVIDER,
    model: WALLALIVE_NEURAL_MODEL,
    meshUrl: "/pip-neural-demo.glb",
    preview: true,
    generatedAt: "2026-08-31T01:41:00.000Z",
  };
}

export function disposeNeuralAsset(asset: NeuralAsset | null) {
  if (!asset) return;
  [asset.meshUrl, asset.skeletonUrl, asset.processedImageUrl].forEach((url) => {
    if (url?.startsWith("blob:")) URL.revokeObjectURL(url);
  });
}
