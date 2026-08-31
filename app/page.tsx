/* eslint-disable @next/next/no-img-element */
"use client";

import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ARStageHandle, CharacterAction } from "./components/ARStage";
import { createAniGenDemoDrawing, createDemoDoodle, POSE_SKELETON_EDGES, type CaptureTarget, type DrawingExtraction, type SemanticPart, type SemanticPartKind, type SemanticSide } from "./lib/drawing";
import { recognizeDrawingFromImageUrl, recognizeDrawingFromVideo, recognizeDrawingParts } from "./lib/learned-parts";
import { createBundledAniGenAsset, disposeNeuralAsset, generateAniGenAsset, type NeuralAsset, type NeuralProgress } from "./lib/anigen";
import type { RiggedAssetInfo } from "./lib/rigged-model";

const ARStage = lazy(() => import("./components/ARStage").then((module) => ({ default: module.ARStage })));

type Actor = "CHILD" | "BROWSER AGENT" | "WALLALIVE";
type AppStep = "ready" | "camera" | "captured" | "alive";
type CameraState = "idle" | "requesting" | "active" | "denied" | "unavailable";
type PanelTab = "agent" | "tools" | "privacy" | "history";

type CharacterState = {
  created: boolean;
  name: string;
  personality: string;
  accent: string;
  inflation: number;
  action: CharacterAction;
  surface: "screen" | "wall" | "floor";
  scale: number;
  storyTitle: string;
};

type Activity = {
  id: string;
  time: string;
  actor: Actor;
  action: string;
  detail: string;
  toolName?: string;
};

type PendingUpload = { url: string; fileName: string };

type WebMCPTool = {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute: (input: Record<string, unknown>, options?: { signal?: AbortSignal }) => Promise<unknown> | unknown;
};

declare global {
  interface Document {
    modelContext?: {
      registerTool: (tool: WebMCPTool, options?: { signal?: AbortSignal }) => Promise<void> | void;
    };
  }
}

const initialCharacter: CharacterState = {
  created: false,
  name: "",
  personality: "curious and kind",
  accent: "#5fc7df",
  inflation: 1,
  action: "idle",
  surface: "screen",
  scale: 1,
  storyTitle: "",
};

const toolNames = [
  ["inspect_wall_scene", "READ"],
  ["reconstruct_rigged_3d_character", "WRITE"],
  ["set_character_personality", "WRITE"],
  ["place_character", "WRITE"],
  ["animate_character", "WRITE"],
  ["recolor_character", "WRITE"],
  ["tell_character_story", "WRITE"],
  ["list_activity", "READ"],
] as const;

const actions: Array<{ action: CharacterAction; label: string; glyph: string }> = [
  { action: "wave", label: "Wave", glyph: "◒" },
  { action: "dance", label: "Dance", glyph: "♪" },
  { action: "hop", label: "Hop", glyph: "↑" },
  { action: "walk", label: "Walk", glyph: "→" },
  { action: "hide", label: "Hide", glyph: "◐" },
  { action: "spin", label: "Spin", glyph: "↻" },
];

const anatomyKinds = ["eye", "cheek", "nose", "mouth", "ear", "arm", "hand", "leg", "foot"] as const satisfies readonly SemanticPartKind[];
const anatomyLabel: Record<(typeof anatomyKinds)[number], string> = {
  eye: "Eye", cheek: "Cheek", nose: "Nose", mouth: "Mouth", ear: "Ear",
  arm: "Arm", hand: "Hand", leg: "Leg", foot: "Foot",
};

const actionProgressive: Record<CharacterAction, string> = {
  idle: "resting",
  wave: "waving",
  dance: "dancing",
  hop: "hopping",
  walk: "walking",
  hide: "hiding",
  spin: "spinning",
};

const stringValue = (value: unknown, fallback = "", max = 180) => typeof value === "string" ? value.trim().slice(0, max) || fallback : fallback;
const numberValue = (value: unknown, fallback: number) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const makeId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
const timeLabel = () => new Date().toLocaleTimeString([], { minute: "2-digit", second: "2-digit" });

function wait(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Story cancelled", "AbortError"));
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(new DOMException("Story cancelled", "AbortError"));
    }, { once: true });
  });
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const stageRef = useRef<ARStageHandle>(null);
  const captureRef = useRef<DrawingExtraction | null>(null);
  const characterRef = useRef<CharacterState>(initialCharacter);
  const activityRef = useRef<Activity[]>([]);
  const neuralAssetRef = useRef<NeuralAsset | null>(null);
  const neuralAbortRef = useRef<AbortController | null>(null);
  const riggedAssetInfoRef = useRef<RiggedAssetInfo | null>(null);
  const externalUploadApprovedRef = useRef(false);
  const rotateGestureRef = useRef<{ pointerId: number; lastX: number; lastY: number; moved: boolean } | null>(null);
  const partDragRef = useRef<{ pointerId: number; partId: string } | null>(null);
  const pendingUploadRef = useRef<PendingUpload | null>(null);

  const [step, setStep] = useState<AppStep>("ready");
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [capture, setCapture] = useState<DrawingExtraction | null>(null);
  const [character, setCharacter] = useState<CharacterState>(initialCharacter);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [panelTab, setPanelTab] = useState<PanelTab>("agent");
  const [webMcpReady, setWebMcpReady] = useState(false);
  const [immersiveAR, setImmersiveAR] = useState(false);
  const [notice, setNotice] = useState("Camera access only begins when you press Start camera.");
  const [agentLine, setAgentLine] = useState("Show me a drawing and I’ll help it find a personality.");
  const [storyCaption, setStoryCaption] = useState("The room is waiting for a new friend.");
  const [demoRunning, setDemoRunning] = useState(false);
  const [captureTarget, setCaptureTarget] = useState<CaptureTarget>({ x: 0.5, y: 0.48 });
  const [neuralAsset, setNeuralAsset] = useState<NeuralAsset | null>(null);
  const [neuralProgress, setNeuralProgress] = useState<NeuralProgress>({ phase: "idle", progress: 0, message: "" });
  const [neuralConsentVisible, setNeuralConsentVisible] = useState(false);
  const [riggedAssetInfo, setRiggedAssetInfo] = useState<RiggedAssetInfo | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [partEditorOpen, setPartEditorOpen] = useState(false);
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  const [pendingPartKind, setPendingPartKind] = useState<(typeof anatomyKinds)[number] | null>(null);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);

  const record = useCallback((actor: Actor, action: string, detail: string, toolName?: string) => {
    const item: Activity = { id: makeId(), time: timeLabel(), actor, action, detail, toolName };
    const next = [item, ...activityRef.current].slice(0, 50);
    activityRef.current = next;
    setActivity(next);
    return item;
  }, []);

  const commitCharacter = useCallback((next: CharacterState, message?: string) => {
    characterRef.current = next;
    setCharacter(next);
    if (message) setNotice(message);
  }, []);

  const commitNeuralAsset = useCallback((next: NeuralAsset | null) => {
    if (neuralAssetRef.current !== next) disposeNeuralAsset(neuralAssetRef.current);
    neuralAssetRef.current = next;
    setNeuralAsset(next);
  }, []);

  const handleRiggedAssetInfo = useCallback((info: RiggedAssetInfo | null) => {
    riggedAssetInfoRef.current = info;
    setRiggedAssetInfo(info);
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("idle");
    setStep(characterRef.current.created ? "alive" : captureRef.current ? "captured" : "ready");
    setNotice("Camera stopped. The approved drawing and character remain only in this tab.");
  }, []);

  useEffect(() => () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    neuralAbortRef.current?.abort();
    disposeNeuralAsset(neuralAssetRef.current);
    if (pendingUploadRef.current) URL.revokeObjectURL(pendingUploadRef.current.url);
  }, []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState("unavailable");
      setNotice("This browser cannot open a camera. The demo doodle still shows the complete experience.");
      return;
    }
    setCameraState("requesting");
    setCaptureTarget({ x: 0.5, y: 0.48 });
    setNotice("Choose Allow to point WallAlive at a drawing.");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraState("active");
      setStep("camera");
      setNotice("Camera is live locally. Tap the character body to target it, then capture.");
      record("CHILD", "Opened the camera", "Video stays on this device and is never exposed as a WebMCP tool.");
    } catch (error) {
      setCameraState("denied");
      setNotice(error instanceof Error && error.name === "NotAllowedError" ? "Camera permission was not granted. Try the demo doodle instead." : "The camera could not start. Try the demo doodle instead.");
    }
  }, [record]);

  const setDrawing = useCallback((next: DrawingExtraction, source: "camera" | "upload" | "demo") => {
    neuralAbortRef.current?.abort();
    neuralAbortRef.current = null;
    commitNeuralAsset(null);
    handleRiggedAssetInfo(null);
    externalUploadApprovedRef.current = false;
    setNeuralConsentVisible(false);
    setNeuralProgress({ phase: "idle", progress: 0, message: "" });
    commitCharacter({ ...initialCharacter, created: true, name: "Pip", accent: next.analysis.secondaryColor });
    captureRef.current = next;
    setCapture(next);
    setSelectedPartId(null);
    setPendingPartKind(null);
    setStep("alive");
    const detected = next.rig.detectedKinds.filter((kind) => kind !== "body").join(", ");
    const learned = next.learnedRecognition;
    setNotice(source !== "demo"
      ? "Safe 3D preview ready. The original artwork stays on the front; uncertain face and limb guesses cannot change its shape."
      : "The safe local 3D preview is ready—or play the no-wait full neural judge demo.");
    setAgentLine(`The selected artwork is preserved on a bounded closed 3D form. Local recognition suggested ${detected || "a character silhouette"}${learned ? ` in ${learned.latencyMs} ms` : ""}, but those unreviewed labels are annotations only—not visible anatomy.`);
    record("WALLALIVE", "Built an identity-safe local 3D preview", `Point-guided cutout · bounded closed relief · original front artwork · ${next.rig.parts.length} unreviewed semantic suggestions · no invented facial geometry · no upload.`);
  }, [commitCharacter, commitNeuralAsset, handleRiggedAssetInfo, record]);

  const recognizeAndSetDrawing = useCallback(async (next: DrawingExtraction, source: "camera" | "upload" | "demo") => {
    setNotice("Preserving the artwork and checking optional rig suggestions locally…");
    try {
      setDrawing(await recognizeDrawingParts(next), source);
    } catch (error) {
      setDrawing(next, source);
      setNotice(`Drawing isolated with exact pixel geometry. Local ML was unavailable: ${error instanceof Error ? error.message : "unknown model error"}`);
    }
  }, [setDrawing]);

  const captureDrawing = useCallback(async () => {
    if (!videoRef.current) return;
    setNotice("Loading precise point-guided segmentation, then preserving the selected artwork locally…");
    try {
      setDrawing(await recognizeDrawingFromVideo(videoRef.current, captureTarget), "camera");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The drawing could not be separated from the wall.");
    }
  }, [captureTarget, setDrawing]);

  const loadDemoDrawing = useCallback(async () => {
    try {
      const demo = createDemoDoodle();
      await recognizeAndSetDrawing(demo, "demo");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The demo drawing could not be created.");
    }
  }, [recognizeAndSetDrawing]);

  const uploadDrawing = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const input = event.currentTarget;
    const file = input.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setNotice("Choose a photo or image file containing one clear character drawing.");
      input.value = "";
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      setNotice("That image is over 12 MB. Choose a smaller photo so recognition stays responsive on mobile.");
      input.value = "";
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    if (pendingUploadRef.current) URL.revokeObjectURL(pendingUploadRef.current.url);
    const next = { url: objectUrl, fileName: file.name };
    pendingUploadRef.current = next;
    setPendingUpload(next);
    setNotice("Tap the character in your photo. The point prompt keeps nearby drawings out.");
    input.value = "";
  }, []);

  const processUploadedDrawing = useCallback(async (target: CaptureTarget) => {
    const pending = pendingUploadRef.current;
    if (!pending) return;
    setPendingUpload(null);
    setNotice("Loading precise point-guided segmentation, then preserving the selected artwork locally…");
    try {
      const drawing = await recognizeDrawingFromImageUrl(pending.url, target);
      setDrawing(drawing, "upload");
      record("CHILD", "Chose a drawing photo", `${pending.fileName} was point-selected, isolated, and recognized locally. The original file was not uploaded.`);
    } catch (error) {
      console.warn("WallAlive local upload recognition was safely rejected", error);
      setNotice(error instanceof Error ? error.message : "The drawing image could not be processed.");
    } finally {
      URL.revokeObjectURL(pending.url);
      pendingUploadRef.current = null;
    }
  }, [record, setDrawing]);

  const cancelPendingUpload = useCallback(() => {
    if (pendingUploadRef.current) URL.revokeObjectURL(pendingUploadRef.current.url);
    pendingUploadRef.current = null;
    setPendingUpload(null);
    setNotice("Photo closed. Choose another image or start the camera.");
  }, []);

  const createCharacter = useCallback((input: Record<string, unknown>, actor: Actor, toolName?: string) => {
    const drawing = captureRef.current;
    if (!drawing) throw new Error("No drawing is approved. The child must capture or choose a drawing first.");
    const neural = neuralAssetRef.current;
    const next: CharacterState = {
      ...characterRef.current,
      created: true,
      name: stringValue(input.name, "Pip", 40),
      personality: stringValue(input.personality, "curious and kind", 120),
      accent: stringValue(input.accent, drawing.analysis.secondaryColor, 20),
      inflation: Math.min(1.35, Math.max(0.7, numberValue(input.inflation, 1))),
      action: "idle",
      storyTitle: "",
    };
    commitCharacter(next, neural ? `${next.name} is now a generated rigged 3D character.` : `${next.name} is now an identity-safe local 3D character.`);
    setStep("alive");
    const graphNodes = drawing.topologyRecognition?.nodes.length ?? drawing.rig.joints.length;
    const graphEdges = drawing.topologyRecognition?.edges.length ?? Math.max(0, graphNodes - 1);
    setAgentLine(neural ? `${next.name} has generated surfaces, colors, bones, and skin weights. The agent can now direct the rig.` : `${next.name} keeps the original front artwork on a bounded closed relief. ${drawing.rig.parts.length} semantic suggestions remain review-only; none can invent or deform its face.`);
    setStoryCaption(`${next.name} lifts away from the wall for the first time.`);
    record(actor, neural ? "Loaded a rigged neural 3D character" : "Loaded the identity-safe local 3D preview", neural
      ? `${next.name} · ${neural.provider} · glTF SkinnedMesh · generated mesh, skeleton, and skin weights.`
      : `${next.name} · closed constrained relief · original front artwork · ${graphNodes} suggested joints · ${graphEdges} suggested branches · semantic suggestions do not deform geometry · no upload.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const requestNeuralConsent = useCallback(() => {
    if (!captureRef.current) return;
    setNeuralConsentVisible(true);
    setNeuralProgress({ phase: "consent-required", progress: 0, message: "Human approval is required before the isolated drawing leaves this tab." });
    setNotice("Review the isolated-image approval. The live camera is never uploaded.");
  }, []);

  const startNeuralReconstruction = useCallback(async () => {
    const drawing = captureRef.current;
    if (!drawing) return;
    neuralAbortRef.current?.abort();
    const controller = new AbortController();
    neuralAbortRef.current = controller;
    setNeuralConsentVisible(false);
    externalUploadApprovedRef.current = true;
    setNotice("AniGen is generating a real mesh, unseen surfaces, skeleton, and skin weights.");
    record("CHILD", "Approved isolated drawing upload", "Only the isolated drawing—not the camera frame—was approved for AniGen.");
    try {
      const asset = await generateAniGenAsset(drawing.textureUrl, (progress) => {
        setNeuralProgress(progress);
        setNotice(progress.message);
      }, controller.signal);
      if (controller.signal.aborted) return;
      commitNeuralAsset(asset);
      createCharacter({ name: "Pip", personality: "curious and kind", accent: drawing.analysis.secondaryColor }, "WALLALIVE");
      record("WALLALIVE", "Generated real rigged 3D", `${asset.provider} returned a colored GLB with generated full geometry and a skinned skeleton.`);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      const message = error instanceof Error ? error.message : "Real 3D generation failed.";
      setNeuralProgress({ phase: "error", progress: 0, message });
      setNotice(message);
      record("WALLALIVE", "Neural 3D generation unavailable", message);
    } finally {
      if (neuralAbortRef.current === controller) neuralAbortRef.current = null;
    }
  }, [commitNeuralAsset, createCharacter, record]);

  const keepPrivatePreview = useCallback(() => {
    setNeuralConsentVisible(false);
    setNeuralProgress({ phase: "idle", progress: 0, message: "" });
    createCharacter({}, "CHILD");
  }, [createCharacter]);

  const setPersonality = useCallback((personality: string, actor: Actor, toolName?: string) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before setting its personality.");
    const next = { ...current, personality: stringValue(personality, current.personality, 120) };
    commitCharacter(next, `${next.name} now feels ${next.personality}.`);
    setAgentLine(`I’ll express “${next.personality}” through movement, without changing the child’s drawing.`);
    record(actor, "Changed the personality", `${next.name} is now ${next.personality}.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const animateCharacter = useCallback((action: CharacterAction, actor: Actor, toolName?: string, caption?: string) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before animating it.");
    const next = { ...current, action };
    commitCharacter(next, `${next.name} is ${actionProgressive[action]}.`);
    setStoryCaption(caption ?? `${next.name} tries a ${action}.`);
    record(actor, `Played ${action}`, caption ?? `${next.name} performs the animation in the live room.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const recolorCharacter = useCallback((accent: string, actor: Actor, toolName?: string) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before changing its 3D edge color.");
    const safeAccent = /^#[0-9a-fA-F]{6}$/.test(accent) ? accent : current.accent;
    const next = { ...current, accent: safeAccent };
    commitCharacter(next, `3D edges changed to ${safeAccent}; original drawing colors preserved.`);
    record(actor, "Changed the 3D accent", `Applied ${safeAccent} only to the generated solid edge.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const placeCharacter = useCallback((x: number, y: number, surface: CharacterState["surface"], scale: number, actor: Actor, toolName?: string) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before placing it.");
    const safeX = Math.min(1, Math.max(0, x));
    const safeY = Math.min(1, Math.max(0, y));
    const safeScale = Math.min(1.55, Math.max(0.55, scale));
    stageRef.current?.placeNormalized(safeX, safeY, safeScale);
    const next = { ...current, surface, scale: safeScale };
    commitCharacter(next, `${next.name} placed on the ${surface}.`);
    record(actor, "Placed the character", `${surface} · x ${safeX.toFixed(2)} · y ${safeY.toFixed(2)} · scale ${safeScale.toFixed(2)}.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const runStory = useCallback(async (title: string, beats: Record<string, unknown>[], actor: Actor, toolName?: string, signal?: AbortSignal) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before telling a story.");
    const storyTitle = stringValue(title, `${current.name}'s first adventure`, 80);
    commitCharacter({ ...current, storyTitle }, `Playing “${storyTitle}”.`);
    record(actor, "Started a mini story", `${storyTitle} · ${beats.length} beats.`, toolName);
    const allowedActions: CharacterAction[] = ["idle", "wave", "dance", "hop", "walk", "hide", "spin"];
    for (const beat of beats.slice(0, 4)) {
      if (signal?.aborted) throw new DOMException("Story cancelled", "AbortError");
      const proposed = stringValue(beat.action, "idle", 20) as CharacterAction;
      const action = allowedActions.includes(proposed) ? proposed : "idle";
      animateCharacter(action, actor, toolName, stringValue(beat.caption, `${characterRef.current.name} ${action}s.`, 120));
      await wait(Math.min(2200, Math.max(650, numberValue(beat.durationMs, 1100))), signal);
    }
    animateCharacter("idle", actor, toolName, `${characterRef.current.name}'s story is ready for another chapter.`);
    window.setTimeout(() => setStoryCaption(""), 1200);
    return { title: storyTitle, beatsPlayed: Math.min(4, beats.length), finalAction: "idle" };
  }, [animateCharacter, commitCharacter, record]);

  const inspectScene = useCallback(() => ({
    drawingApproved: Boolean(captureRef.current),
    drawingAnalysis: captureRef.current?.analysis ?? null,
    reconstruction: captureRef.current ? {
      localIsolation: "MediaPipe MagicTouch point-guided interactive segmentation with compact local-model fallback",
      localPreview: "one continuous bounded closed relief; original artwork is mapped only to the front and unreviewed anatomy cannot deform it",
      method: neuralAssetRef.current ? `${neuralAssetRef.current.provider} full-volume neural mesh + skeleton skinning` : "MagicTouch segmentation + constrained signed-distance relief + Marching Cubes",
      provider: neuralAssetRef.current?.provider ?? "WallAlive local",
      model: neuralAssetRef.current?.model ?? `${captureRef.current.cutoutRecognition?.model ?? "local-alpha-cutout"} + constrained-relief-v2`,
      assetType: neuralAssetRef.current ? "glTF SkinnedMesh" : "Three.js continuous closed SkinnedMesh with identity-preserving front texture",
      topology: neuralAssetRef.current ? "generated full 3D surface including unseen views" : "closed bounded relief; semantic graph is review-only",
      topologyConfidence: captureRef.current.rig.topologyConfidence ?? null,
      backInference: neuralAssetRef.current ? "full neural generative prior" : "bounded hidden-surface relief; plausible, not observed ground truth",
      viewableDegrees: 360,
      contourPoints: captureRef.current.contour.length,
      skeletonPoints: captureRef.current.skeleton.length,
      rigVersion: captureRef.current.rig.version,
      semanticRecognition: captureRef.current.learnedRecognition ?? {
        model: null,
        latencyMs: null,
        detectedKinds: [],
      },
      poseRecognition: captureRef.current.poseRecognition ?? null,
      topologyRecognition: captureRef.current.topologyRecognition ? {
        ...captureRef.current.topologyRecognition,
        contract: "variable graph decoded from learned centerline, endpoints, and junction fields; no fixed human joint count",
      } : null,
      depthRecognition: captureRef.current.depthRecognition ? {
        model: captureRef.current.depthRecognition.model,
        latencyMs: captureRef.current.depthRecognition.latencyMs,
        meanThickness: captureRef.current.depthRecognition.meanThickness,
        meanAsymmetry: captureRef.current.depthRecognition.meanAsymmetry,
        frontBackMirrored: false,
      } : null,
      localPreviewRegions: captureRef.current.rig.parts.map((part) => ({ id: part.id, kind: part.kind, side: part.side, confidence: part.confidence, source: part.source, posePathPoints: part.path?.length ?? 0 })),
      inflation: characterRef.current.inflation,
      neuralModelUsed: Boolean(neuralAssetRef.current),
      generatedAsset: riggedAssetInfoRef.current,
      generationPhase: neuralAssetRef.current ? "neural-ready" : "identity-safe-local-relief-ready",
      neuralUpgrade: neuralAssetRef.current ? "active" : "optional-human-approval-required",
      externalUploadApproved: externalUploadApprovedRef.current,
    } : null,
    character: { ...characterRef.current, textureUrl: undefined },
    cameraFeedExposed: false,
    privacyBoundary: "Camera capture is human-only. WebMCP can request reconstruction but cannot approve or upload; only a visible human action may send the isolated drawing to AniGen.",
    availableAnimations: actions.map((item) => item.action),
    placementModes: immersiveAR ? ["world-hit-test", "camera-overlay"] : ["camera-overlay"],
  }), [immersiveAR]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const controller = new AbortController();
    const base = { type: "object", additionalProperties: false };
    const ok = (payload: Record<string, unknown>) => ({ ok: true, ...payload });
    const fail = (error: unknown) => ({ ok: false, error: error instanceof Error ? error.message : "Tool execution failed." });
    const guard = (signal: AbortSignal) => { if (signal.aborted) throw new DOMException("Tool call cancelled", "AbortError"); };
    const executionSignal = (options?: { signal?: AbortSignal }) => options?.signal ?? controller.signal;
    const tools: WebMCPTool[] = [
      {
        name: "inspect_wall_scene",
        title: "Inspect approved wall drawing",
        description: "Read semantic details about the human-approved drawing, character, AR capability, and privacy boundary. Never returns camera frames or image data.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => { const signal = executionSignal(options); guard(signal); return ok({ scene: inspectScene() }); },
      },
      {
        name: "reconstruct_rigged_3d_character",
        title: "Create the approved rigged 3D character",
        description: "Create a playable identity-safe closed 3D relief from the approved artwork, or request optional neural-full mode. Unreviewed anatomy cannot deform the local geometry. Neural mode can surface human approval but can never approve an upload, open the camera, or receive camera frames.",
        inputSchema: {
          ...base,
          properties: {
            name: { type: "string", minLength: 1, maxLength: 40 },
            personality: { type: "string", minLength: 1, maxLength: 120 },
            accent: { type: "string", pattern: "^#[0-9a-fA-F]{6}$" },
            inflation: { type: "number", minimum: 0.7, maximum: 1.35 },
            reconstructionMode: { type: "string", enum: ["local-private", "neural-full"] },
          },
          required: ["name", "personality", "accent"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            if (!captureRef.current) throw new Error("No drawing is approved. The child must capture or choose a drawing first.");
            const requestedMode = stringValue(input.reconstructionMode, "local-private", 20);
            if (requestedMode === "neural-full" && !neuralAssetRef.current) {
              requestNeuralConsent();
              return ok({ requiresHumanApproval: true, phase: "consent-required", message: "Use the visible approval card to send only the isolated drawing to AniGen." });
            }
            return ok({
              character: createCharacter(input, "BROWSER AGENT", "reconstruct_rigged_3d_character"),
              reconstructionMode: neuralAssetRef.current ? "neural-full" : "local-private",
              localRig: neuralAssetRef.current ? null : {
                topology: captureRef.current.rig.topologyKind ?? null,
                graphNodes: captureRef.current.topologyRecognition?.nodes.length ?? captureRef.current.rig.joints.length,
                graphEdges: captureRef.current.topologyRecognition?.edges.length ?? Math.max(0, captureRef.current.rig.joints.length - 1),
                semanticParts: captureRef.current.rig.parts.length,
                viewableDegrees: 360,
                backInference: "bounded hidden-surface relief; plausible, not observed ground truth",
                semanticStatus: "review-only; unreviewed labels do not create or deform visible geometry",
                depthModel: captureRef.current.depthRecognition?.model ?? null,
                meanDepthAsymmetry: captureRef.current.depthRecognition?.meanAsymmetry ?? null,
              },
              generatedAsset: riggedAssetInfoRef.current,
            });
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "set_character_personality",
        title: "Set character personality",
        description: "Change how the character is described and performed without altering the child's captured artwork.",
        inputSchema: { ...base, properties: { personality: { type: "string", minLength: 1, maxLength: 120 } }, required: ["personality"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); try { guard(signal); return ok({ character: setPersonality(stringValue(input.personality), "BROWSER AGENT", "set_character_personality") }); } catch (error) { return fail(error); } },
      },
      {
        name: "place_character",
        title: "Place character in room",
        description: "Place the created character at a normalized position in the visible AR scene. On WebXR devices the child can tap a real detected surface for final placement.",
        inputSchema: {
          ...base,
          properties: {
            x: { type: "number", minimum: 0, maximum: 1 },
            y: { type: "number", minimum: 0, maximum: 1 },
            surface: { type: "string", enum: ["screen", "wall", "floor"] },
            scale: { type: "number", minimum: 0.55, maximum: 1.55 },
          },
          required: ["x", "y", "surface", "scale"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); try { guard(signal); return ok({ character: placeCharacter(numberValue(input.x, .5), numberValue(input.y, .5), ["wall", "floor"].includes(String(input.surface)) ? input.surface as "wall" | "floor" : "screen", numberValue(input.scale, 1), "BROWSER AGENT", "place_character") }); } catch (error) { return fail(error); } },
      },
      {
        name: "animate_character",
        title: "Animate character",
        description: "Play one safe visible animation on the created character. Does not navigate, capture, upload, or modify the original drawing.",
        inputSchema: { ...base, properties: { action: { type: "string", enum: ["idle", "wave", "dance", "hop", "walk", "hide", "spin"] }, caption: { type: "string", maxLength: 120 } }, required: ["action"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); try { guard(signal); const action = stringValue(input.action, "idle") as CharacterAction; return ok({ character: animateCharacter(action, "BROWSER AGENT", "animate_character", stringValue(input.caption) || undefined) }); } catch (error) { return fail(error); } },
      },
      {
        name: "recolor_character",
        title: "Recolor generated depth",
        description: "Change only the generated 3D solid edge accent. The child's original drawing pixels remain unchanged.",
        inputSchema: { ...base, properties: { accent: { type: "string", pattern: "^#[0-9a-fA-F]{6}$" } }, required: ["accent"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); try { guard(signal); return ok({ character: recolorCharacter(stringValue(input.accent), "BROWSER AGENT", "recolor_character") }); } catch (error) { return fail(error); } },
      },
      {
        name: "tell_character_story",
        title: "Perform mini story",
        description: "Perform a cancellable one-to-four-beat story using safe character animations and short captions in the shared scene.",
        inputSchema: {
          ...base,
          properties: {
            title: { type: "string", minLength: 1, maxLength: 80 },
            beats: {
              type: "array",
              minItems: 1,
              maxItems: 4,
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  action: { type: "string", enum: ["idle", "wave", "dance", "hop", "walk", "hide", "spin"] },
                  caption: { type: "string", minLength: 1, maxLength: 120 },
                  durationMs: { type: "number", minimum: 650, maximum: 2200 },
                },
                required: ["action", "caption"],
              },
            },
          },
          required: ["title", "beats"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); try { guard(signal); const beats = Array.isArray(input.beats) ? input.beats.filter(isRecord) : []; return ok({ story: await runStory(stringValue(input.title), beats, "BROWSER AGENT", "tell_character_story", signal), character: characterRef.current }); } catch (error) { return fail(error); } },
      },
      {
        name: "list_activity",
        title: "List human-agent activity",
        description: "Read recent attributed scene actions. Camera pixels and captured drawing data are intentionally excluded.",
        inputSchema: { ...base, properties: { limit: { type: "number", minimum: 1, maximum: 30 } } },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); guard(signal); const limit = Math.min(30, Math.max(1, numberValue(input.limit, 12))); return ok({ activity: activityRef.current.slice(0, limit), cameraDataIncluded: false }); },
      },
    ];

    Promise.all(tools.map((tool) => Promise.resolve(context.registerTool(tool, { signal: controller.signal })))).then(() => {
      setWebMcpReady(true);
      setNotice(`${tools.length} WebMCP tools are ready. Camera capture remains human-only.`);
    }).catch(() => setWebMcpReady(false));
    return () => controller.abort();
  }, [animateCharacter, createCharacter, inspectScene, placeCharacter, recolorCharacter, requestNeuralConsent, runStory, setPersonality]);

  const runMagicDemo = useCallback(async () => {
    if (demoRunning) return;
    setDemoRunning(true);
    try {
      setAgentLine("Loading the exact drawing and running all six local recognition graphs…");
      const demoInput = await createAniGenDemoDrawing();
      const demo = await recognizeDrawingParts(demoInput).catch(() => demoInput);
      setDrawing(demo, "demo");
      const bundledAsset = createBundledAniGenAsset();
      commitNeuralAsset(bundledAsset);
      externalUploadApprovedRef.current = false;
      setNeuralProgress({ phase: "ready", progress: 1, message: "Verified neural sketch rig loaded." });
      setAgentLine("1 / 4 · This exact drawing became a smoothed, colored 68,326-vertex neural SkinnedMesh with 7 active bones—not a cut-out or extrusion.");
      await wait(450);
      createCharacter({ name: "Pip", personality: "brave on the outside, shy on the inside", accent: "#ce919f", inflation: 1 }, "BROWSER AGENT", "reconstruct_rigged_3d_character");
      setAgentLine("2 / 4 · Sketch-conditioned neural reconstruction produced ears, feet, side arm, a rounded body, unseen surfaces, and semantic skin weights.");
      await wait(850);
      placeCharacter(.68, .53, "wall", 1, "BROWSER AGENT", "place_character");
      setAgentLine("3 / 4 · WebMCP places Pip and directs bones without receiving camera control.");
      await wait(650);
      await runStory("Pip finds their courage", [
        { action: "hide", caption: "Pip hides at the edge of the wall.", durationMs: 800 },
        { action: "hop", caption: "One brave hop into the room.", durationMs: 800 },
        { action: "wave", caption: "A real arm-bone branch waves hello.", durationMs: 1000 },
        { action: "spin", caption: "A full turn reveals generated back geometry.", durationMs: 1400 },
      ], "BROWSER AGENT", "tell_character_story");
      setAgentLine("4 / 4 · The full turn proves real 360° geometry; animation comes from the generated skeleton.");
    } finally {
      setDemoRunning(false);
    }
  }, [commitNeuralAsset, createCharacter, demoRunning, placeCharacter, runStory, setDrawing]);

  const enterAR = useCallback(async () => {
    const result = await stageRef.current?.enterImmersiveAR();
    if (result?.ok) {
      setNotice("Move your phone until a ring appears, then tap a real surface to place the character.");
      record("CHILD", "Entered immersive AR", "Real-world placement uses device hit testing.");
    } else setNotice(result?.error ?? "Immersive AR is unavailable; camera-overlay mode is active.");
  }, [record]);

  const activateStagePoint = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width;
    const y = (event.clientY - bounds.top) / bounds.height;
    if (cameraState === "active") {
      const target = { x: Math.min(0.92, Math.max(0.08, x)), y: Math.min(0.84, Math.max(0.09, y)) };
      setCaptureTarget(target);
      setNotice("Character targeted. Capture will reject paper edges, text, and dense foreground clutter.");
      return;
    }
    if (!characterRef.current.created) return;
    placeCharacter(x, y, "screen", characterRef.current.scale, "CHILD");
  }, [cameraState, placeCharacter]);

  const handleStagePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    rotateGestureRef.current = { pointerId: event.pointerId, lastX: event.clientX, lastY: event.clientY, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const handleStagePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const gesture = rotateGestureRef.current;
    if (!gesture || gesture.pointerId !== event.pointerId || !characterRef.current.created || cameraState === "active") return;
    const dx = event.clientX - gesture.lastX;
    const dy = event.clientY - gesture.lastY;
    if (Math.hypot(dx, dy) > 1.5) gesture.moved = true;
    gesture.lastX = event.clientX;
    gesture.lastY = event.clientY;
    const bounds = event.currentTarget.getBoundingClientRect();
    stageRef.current?.rotateBy((dx / Math.max(1, bounds.width)) * Math.PI * 1.7, (dy / Math.max(1, bounds.height)) * Math.PI * 1.15);
  }, [cameraState]);

  const handleStagePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const gesture = rotateGestureRef.current;
    rotateGestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (gesture?.moved) {
      setNotice("360° model rotated. Drag again to inspect its bounded filled back.");
      return;
    }
    activateStagePoint(event);
  }, [activateStagePoint]);

  const handleARCapability = useCallback((supported: boolean) => {
    setImmersiveAR(supported);
  }, []);

  const handleARPlaced = useCallback((surface: "screen" | "world") => {
    if (surface === "world") commitCharacter({ ...characterRef.current, surface: "wall" });
  }, [commitCharacter]);

  const copyDemoPrompt = useCallback(async () => {
    if (!navigator.clipboard) {
      setNotice("Clipboard access is unavailable. The prompt is shown in the agent panel.");
      return;
    }
    await navigator.clipboard.writeText("Inspect the approved drawing. Turn it into a shy but brave character, place it on the wall, then tell a three-beat story where it hides, hops, and waves.");
    setNotice("Demo prompt copied.");
  }, []);

  const commitRigEdit = useCallback((parts: SemanticPart[], message: string) => {
    const current = captureRef.current;
    if (!current) return;
    const validIds = new Set(parts.map((part) => part.id));
    const normalized = parts.map((part) => part.parentId && !validIds.has(part.parentId) ? { ...part, parentId: "body" } : part);
    const next: DrawingExtraction = {
      ...current,
      rig: {
        ...current.rig,
        parts: normalized,
        joints: normalized.filter((part) => part.parentId && validIds.has(part.parentId)).map((part) => ({
          id: `joint-${part.id}`,
          parentId: part.parentId!,
          childId: part.id,
          x: part.anchor?.x ?? part.center.x,
          y: part.anchor?.y ?? part.center.y,
        })),
        detectedKinds: [...new Set(normalized.map((part) => part.kind))],
      },
    };
    captureRef.current = next;
    setCapture(next);
    setNotice(message);
  }, []);

  const partSide = useCallback((x: number): SemanticSide => {
    const bodyX = captureRef.current?.rig.parts.find((part) => part.kind === "body")?.center.x ?? 0;
    return x < bodyX - 0.04 ? "left" : x > bodyX + 0.04 ? "right" : "center";
  }, []);

  const moveRigPart = useCallback((partId: string, x: number, y: number) => {
    const current = captureRef.current;
    if (!current) return;
    const parts = current.rig.parts.map((part) => {
      if (part.id !== partId) return part;
      const dx = x - part.center.x;
      const dy = y - part.center.y;
      return {
        ...part,
        side: part.kind === "mouth" || part.kind === "nose" ? "center" as const : partSide(x),
        center: { ...part.center, x, y },
        outline: part.outline?.map((point) => ({ x: point.x + dx, y: point.y + dy })),
        path: part.path?.map((point) => ({ ...point, x: point.x + dx, y: point.y + dy })),
      };
    });
    commitRigEdit(parts, "Part position updated. The 3D rig changed with it.");
  }, [commitRigEdit, partSide]);

  const editorPoint = useCallback((svg: SVGSVGElement, clientX: number, clientY: number) => {
    const bounds = svg.getBoundingClientRect();
    const normalizedX = clamp01((clientX - bounds.left) / Math.max(1, bounds.width));
    const normalizedY = clamp01((clientY - bounds.top) / Math.max(1, bounds.height));
    return { x: (normalizedX - 0.5) * 1.4, y: (0.5 - normalizedY) * 1.4 };
  }, []);

  const addRigPart = useCallback((kind: (typeof anatomyKinds)[number], x: number, y: number) => {
    const current = captureRef.current;
    if (!current) return;
    const body = current.rig.parts.find((part) => part.kind === "body");
    if (!body) return;
    const side = kind === "mouth" || kind === "nose" ? "center" : partSide(x);
    const siblings = current.rig.parts.filter((part) => part.kind === kind);
    const dimensions: Record<(typeof anatomyKinds)[number], [number, number, number]> = {
      eye: [0.12, 0.09, 0.035], cheek: [0.1, 0.055, 0.018], nose: [0.06, 0.045, 0.018], mouth: [0.17, 0.06, 0.018], ear: [0.13, 0.18, 0.09],
      arm: [0.075, 0.28, 0.075], hand: [0.1, 0.1, 0.07], leg: [0.085, 0.3, 0.085], foot: [0.13, 0.09, 0.075],
    };
    const [width, height, depth] = dimensions[kind];
    const parentKind = kind === "hand" ? "arm" : kind === "foot" ? "leg" : null;
    const parent = parentKind ? current.rig.parts.filter((part) => part.kind === parentKind)
      .sort((left, right) => Math.hypot(left.center.x - x, left.center.y - y) - Math.hypot(right.center.x - x, right.center.y - y))[0] : null;
    const id = `manual-${kind}-${siblings.length + 1}-${Date.now().toString(36)}`;
    const structural = kind === "arm" || kind === "leg";
    const part: SemanticPart = {
      id,
      kind,
      side,
      parentId: parent?.id ?? "body",
      center: { x, y, z: 0 },
      anchor: structural ? { ...body.center } : undefined,
      size: { x: width, y: height, z: depth },
      rotation: structural ? Math.atan2(-(x - body.center.x), y - body.center.y) : 0,
      color: kind === "eye" || kind === "cheek" || kind === "nose" || kind === "mouth" ? current.rig.lineColor : current.rig.bodyColor,
      confidence: 1,
      source: "structural-inference",
    };
    commitRigEdit([...current.rig.parts, part], `${anatomyLabel[kind]} added exactly where you tapped.`);
    setSelectedPartId(id);
    setPendingPartKind(null);
  }, [commitRigEdit, partSide]);

  const resizeSelectedPart = useCallback((amount: number) => {
    const current = captureRef.current;
    if (!current || !selectedPartId) return;
    const parts = current.rig.parts.map((part) => {
      if (part.id !== selectedPartId) return part;
      const scale = Math.min(1.5, Math.max(0.5, amount));
      return {
        ...part,
        size: { x: part.size.x * scale, y: part.size.y * scale, z: part.size.z * scale },
        outline: part.outline?.map((point) => ({
          x: part.center.x + (point.x - part.center.x) * scale,
          y: part.center.y + (point.y - part.center.y) * scale,
        })),
      };
    });
    commitRigEdit(parts, "Part size updated.");
  }, [commitRigEdit, selectedPartId]);

  const deleteSelectedPart = useCallback(() => {
    const current = captureRef.current;
    if (!current || !selectedPartId) return;
    const selected = current.rig.parts.find((part) => part.id === selectedPartId);
    if (!selected || selected.kind === "body") return;
    commitRigEdit(current.rig.parts.filter((part) => part.id !== selectedPartId), `${selected.kind} removed from the rig.`);
    setSelectedPartId(null);
  }, [commitRigEdit, selectedPartId]);

  const latestAgentActivity = useMemo(() => activity.find((item) => item.actor === "BROWSER AGENT"), [activity]);
  const neuralBusy = ["connecting", "preparing", "queued", "generating", "downloading"].includes(neuralProgress.phase);
  const primaryButton = cameraState === "active"
    ? { label: "CAPTURE DRAWING", action: captureDrawing }
    : capture && !neuralAsset
      ? { label: "UPGRADE NEURAL 3D", action: requestNeuralConsent }
      : { label: "START CAMERA", action: startCamera };
  const stepIndex = step === "ready" ? 0 : step === "camera" ? 1 : character.created ? 3 : 2;

  return (
    <main className="alive-shell">
      <header className="alive-header">
        <a className="alive-brand" href="#play"><span>WALL</span>ALIVE<i>●</i></a>
        <div className="mini-steps" aria-label="Three steps"><span className={stepIndex >= 1 ? "done" : "active"}>1 Scan</span><span className={stepIndex >= 2 ? "done" : ""}>2 Check</span><span className={stepIndex >= 3 ? "done" : ""}>3 Play</span></div>
        <div className="header-actions">
          <div className={`ready-pill ${webMcpReady ? "is-ready" : ""}`}><i /> {webMcpReady ? "8 WEBMCP TOOLS" : "INTERACTIVE DEMO"}</div>
          <button className="inspector-toggle" onClick={() => setInspectorOpen(true)}>WEBMCP</button>
          <button className="judge-demo" onClick={runMagicDemo} disabled={demoRunning}>{demoRunning ? "PLAYING…" : "PLAY JUDGE DEMO"}</button>
        </div>
      </header>

      <section className="alive-layout" id="play">
        <aside className="steps-panel">
          <p className="kicker">THE MAGIC LOOP</p>
          <ol>
            <li className={stepIndex >= 1 ? "done" : "active"}><span>1</span><div><strong>Scan</strong><small>Human opens camera</small></div></li>
            <li className={stepIndex >= 3 ? "done" : stepIndex === 2 ? "active" : ""}><span>2</span><div><strong>Generate</strong><small>Identity-safe closed 3D</small></div></li>
            <li className={stepIndex === 3 ? "active" : ""}><span>3</span><div><strong>Play</strong><small>Agent directs movement</small></div></li>
          </ol>

          {capture ? (
            <div className="drawing-fingerprint">
              <div className="drawing-thumb">
                <img src={capture.textureUrl} alt="Locally isolated drawing" />
                <svg viewBox="0 0 100 100" aria-hidden="true">
                  {capture.poseRecognition?.applicable ? POSE_SKELETON_EDGES.map(([from, to]) => {
                    const start = capture.poseRecognition?.joints.find((joint) => joint.name === from);
                    const end = capture.poseRecognition?.joints.find((joint) => joint.name === to);
                    return start && end ? <line key={`${from}-${to}`} x1={start.x * 100} y1={start.y * 100} x2={end.x * 100} y2={end.y * 100} /> : null;
                  }) : null}
                  {!capture.poseRecognition?.applicable && capture.topologyRecognition?.applicable ? capture.topologyRecognition.edges.map((edge) => {
                    const start = capture.topologyRecognition?.nodes.find((node) => node.id === edge.from);
                    const end = capture.topologyRecognition?.nodes.find((node) => node.id === edge.to);
                    return start && end ? <line key={edge.id} x1={start.x * 100} y1={start.y * 100} x2={end.x * 100} y2={end.y * 100} /> : null;
                  }) : null}
                  {capture.poseRecognition?.applicable
                    ? capture.poseRecognition.joints.map((joint) => <circle key={joint.name} cx={joint.x * 100} cy={joint.y * 100} r="1.7" />)
                    : capture.topologyRecognition?.applicable
                      ? capture.topologyRecognition.nodes.map((node) => <circle key={node.id} cx={node.x * 100} cy={node.y * 100} r={node.role === "root" ? "2.4" : "1.7"} />)
                    : capture.skeleton.map((point, index) => <circle key={index} cx={(point.x / 1.4 + 0.5) * 100} cy={(0.5 - point.y / 1.4) * 100} r={Math.max(1.1, point.radius / 1.4 * 100)} />)}
                </svg>
              </div>
              <div><p className="kicker">{neuralAsset ? "FULL NEURAL RIG + DRAWING PARTS" : "IDENTITY-SAFE LOCAL 3D"}</p><strong>{neuralAsset ? capture.rig.detectedKinds.filter((kind) => kind !== "body").join(" · ") || capture.analysis.shapeHint : "ORIGINAL FRONT · BOUNDED DEPTH"}</strong><span><i style={{ background: capture.rig.bodyColor }} /><i style={{ background: capture.rig.lineColor }} /> {riggedAssetInfo ? `${riggedAssetInfo.bones} BONES · ${riggedAssetInfo.semanticParts} PROJECTED PARTS${riggedAssetInfo.colorTransfer ? " · COLOR MATCHED" : ""} · ${riggedAssetInfo.vertices.toLocaleString()} VERTICES` : neuralAsset ? "RIGGED GLB LOADING" : "CLOSED 360° RELIEF · NO INVENTED FACE"}</span></div>
            </div>
          ) : null}

          <div className="privacy-card">
            <div><b>CAMERA-SAFE BY DESIGN</b><span>◆</span></div>
            <p>Point-guided isolation and the identity-safe 3D preview stay on-device. Only full neural generation sends the isolated drawing after a second visible approval—never live frames.</p>
          </div>
          <input ref={uploadRef} hidden type="file" accept="image/*" onChange={uploadDrawing} />
          <button className="demo-doodle upload-drawing" onClick={() => uploadRef.current?.click()}>UPLOAD A DRAWING PHOTO <span>↥</span></button>
          <button className="demo-doodle" onClick={loadDemoDrawing}>NO CAMERA? TRY A DEMO DOODLE <span>＋</span></button>
        </aside>

        <section className="magic-stage">
          <div className="stage-copy">
            <div><p className="kicker">CAMERA · PAPER · MAGIC</p><h1>Draw it.<br /><em>Wake it.</em></h1></div>
            <div className="stage-ctas">
              {immersiveAR && character.created ? <button className="ar-button" onClick={enterAR}>ENTER REAL AR <span>◎</span></button> : null}
              {cameraState === "active" ? <button className="stop-camera" onClick={stopCamera}>STOP CAMERA</button> : null}
              {cameraState !== "active" && step === "ready" ? <button className="upload-camera" onClick={() => uploadRef.current?.click()}>UPLOAD PHOTO</button> : null}
              <button className="primary-camera" onClick={primaryButton.action} disabled={cameraState === "requesting" || neuralBusy}>{cameraState === "requesting" ? "OPENING…" : neuralBusy ? "GENERATING…" : primaryButton.label}<span>↗</span></button>
            </div>
          </div>
          <div className="notice" role="status" aria-live="polite"><i />{notice}</div>

          {capture ? <div
            className="anatomy-summary"
            data-cutout-model={capture.cutoutRecognition?.model ?? "authored-alpha"}
            data-cutout-confidence={capture.cutoutRecognition?.confidence ?? 1}
            data-cutout-area={capture.cutoutRecognition?.areaPercent ?? capture.analysis.coveragePercent}
          >
            <div>
              <span className="summary-spark">✦</span>
              <p><b>Artwork preserved</b><small>{capture.cutoutRecognition?.model === "mediapipe-magic-touch-v2" ? "point-guided cutout" : "local cutout"}</small></p>
            </div>
            <div className="anatomy-pills"><span>Original face</span><span>Flat details</span><span>Closed back</span></div>
            <button onClick={() => setPartEditorOpen(true)}>REVIEW RIG</button>
          </div> : null}

          <div className={`camera-frame step-${step}`} onPointerDown={handleStagePointerDown} onPointerMove={handleStagePointerMove} onPointerUp={handleStagePointerUp} onPointerCancel={() => { rotateGestureRef.current = null; }}>
            <video ref={videoRef} className={cameraState === "active" ? "camera-video visible" : "camera-video"} autoPlay muted playsInline aria-label="Live local camera preview" />
            {cameraState !== "active" ? <div className="demo-room"><span className="frame-a" /><span className="frame-b" /><span className="shelf" /><span className="plant" /><span className="baseboard" /></div> : null}
            {capture && cameraState !== "active" ? <img className="captured-room" src={capture.previewUrl} alt="Approved drawing preview" /> : null}
            {step === "camera" ? <><div className="capture-guide"><span /><b>TAP CHARACTER · THEN CAPTURE</b></div><div className="capture-target" style={{ left: `${captureTarget.x * 100}%`, top: `${captureTarget.y * 100}%` }}><i /></div></> : null}
            <Suspense fallback={<div className="three-layer" aria-hidden="true" />}>
              <ARStage ref={stageRef} contour={capture?.contour ?? null} skeleton={capture?.skeleton ?? null} textureUrl={capture?.textureUrl ?? null} rig={capture?.rig ?? null} depth={capture?.depthRecognition ?? null} action={character.action} accent={character.accent} inflation={character.inflation} neuralAssetUrl={neuralAsset?.meshUrl ?? null} visible={character.created} onCapability={handleARCapability} onPlaced={handleARPlaced} onNeuralAssetInfo={handleRiggedAssetInfo} />
            </Suspense>
            {neuralConsentVisible ? <div className="neural-consent" role="dialog" aria-modal="true" aria-labelledby="neural-consent-title" onPointerDown={(event) => event.stopPropagation()}>
              <span>FULL NEURAL 3D · HUMAN APPROVAL</span><h2 id="neural-consent-title">Send only this isolated drawing to AniGen?</h2><p>Your identity-safe local relief is already rotatable. AniGen can replace its bounded back with generated unseen geometry, a mesh skeleton, and skin weights. The live camera and room frame are never sent.</p><div><button onClick={startNeuralReconstruction}>GENERATE FULL 3D</button><button onClick={keepPrivatePreview}>KEEP LOCAL PREVIEW</button></div>
            </div> : null}
            {neuralBusy ? <div className="neural-progress" role="status" onPointerDown={(event) => event.stopPropagation()}><span>ANIGEN · RIGGED 3D</span><b>{neuralProgress.message}</b><div><i style={{ width: `${Math.round(neuralProgress.progress * 100)}%` }} /></div><small>{Math.round(neuralProgress.progress * 100)}% · PUBLIC GPU</small></div> : null}
            <div className="camera-hud"><span><i /> {cameraState === "active" ? "LIVE CAMERA · LOCAL" : neuralAsset && character.created ? `FULL NEURAL RIG · ${riggedAssetInfo?.bones ?? "…"} BONES` : character.created ? "IDENTITY-SAFE 3D · LOCAL" : "SAFE DEMO ROOM"}</span><strong>{immersiveAR ? "WEBXR READY" : "CAMERA AR FALLBACK"}</strong></div>
            {character.created && storyCaption ? <div className="story-caption"><span>{character.storyTitle || "LIVE MOMENT"}</span><p>{storyCaption}</p></div> : null}
            {cameraState === "denied" || cameraState === "unavailable" ? <div className="camera-message"><b>CAMERA OPTIONAL</b><p>The demo doodle still proves the complete WebMCP and 3D workflow.</p></div> : null}
          </div>

          <div className="action-tray">
            <div><span>CHARACTER ACTIONS</span><small>{character.created ? `${character.name.toUpperCase()} · ${character.personality.toUpperCase()}` : "WAKE A DRAWING TO PLAY"}</small></div>
            {actions.map((item) => <button key={item.action} disabled={!character.created} className={character.action === item.action ? "active" : ""} onClick={() => animateCharacter(item.action, "CHILD")}><i>{item.glyph}</i>{item.label}</button>)}
          </div>
          <p className="placement-tip">{character.created ? neuralAsset ? "Drag for 360° · Generated back · Actions move the SkinnedMesh bones" : "Drag for 360° · Original artwork on front · Bounded filled back · Recognition cannot deform the face" : "Photograph one clear character—point-guided isolation preserves the original artwork"}</p>
        </section>

        <aside className={`agent-panel ${inspectorOpen ? "is-open" : ""}`} aria-hidden={!inspectorOpen}>
          <button className="inspector-close" onClick={() => setInspectorOpen(false)} aria-label="Close WebMCP inspector">×</button>
          <div className="right-tabs" role="tablist" aria-label="WallAlive inspector">
            {(["agent", "tools", "privacy", "history"] as const).map((tab) => <button key={tab} role="tab" aria-selected={panelTab === tab} className={panelTab === tab ? "active" : ""} onClick={() => setPanelTab(tab)}>{tab}</button>)}
          </div>

          {panelTab === "agent" ? (
            <div className="panel-body">
              <div className="agent-status"><div><i /> BROWSER AGENT</div><span>{webMcpReady ? "CONNECTED" : "DEMO MODE"}</span></div>
              <p className="kicker">SHARED IMAGINATION</p>
              <h2>{agentLine}</h2>
              <p>The agent directs personality, placement, animation, and stories. Camera permission always stays with the child.</p>
              <div className="agent-call"><span>↳</span><div><b>{latestAgentActivity?.toolName ?? "inspect_wall_scene"}</b><small>{latestAgentActivity?.detail ?? "Drawing state visible · Camera private"}</small></div></div>
              <blockquote>“Make it shy. Let it hide, take one brave hop, then wave.”</blockquote>
              <button className="copy-prompt" onClick={copyDemoPrompt}>COPY WINNING DEMO PROMPT <span>⧉</span></button>
            </div>
          ) : null}

          {panelTab === "tools" ? (
            <div className="panel-body">
              <p className="kicker">WEBMCP INSPECTOR</p><h2>Eight tools.<br />Zero camera control.</h2><p>The agent acts on approved state through narrow schemas and shared validation.</p>
              <div className="tools-list">{toolNames.map(([name, mode], index) => <div key={name}><span>{String(index + 1).padStart(2, "0")}</span><code>{name}</code><i>{mode}</i></div>)}</div>
            </div>
          ) : null}

          {panelTab === "privacy" ? (
            <div className="panel-body">
              <p className="kicker">CHILD-SAFE BOUNDARY</p><h2>The camera is not a tool.</h2><p>WallAlive refuses to let an agent open or capture the camera. Neural generation requires a separate visible human approval.</p>
              <ul className="privacy-list"><li><b>Human gesture required</b><span>Start, capture, and isolated-image upload are UI-only.</span></li><li><b>Local extraction</b><span>The room frame is separated in browser memory.</span></li><li><b>Minimal neural input</b><span>Only the approved isolated drawing goes to AniGen.</span></li><li><b>Session-only result</b><span>The generated GLB is held in this browser tab.</span></li></ul>
            </div>
          ) : null}

          {panelTab === "history" ? (
            <div className="panel-body">
              <p className="kicker">VISIBLE PROVENANCE</p><h2>Every action has an author.</h2>
              <div className="history-list">{activity.length ? activity.map((item) => <article key={item.id}><span>{item.time}</span><div><small>{item.actor}</small><b>{item.action}</b><p>{item.detail}</p></div></article>) : <p className="empty-history">The first human or agent action will appear here.</p>}</div>
            </div>
          ) : null}
          <footer className="agent-footer"><span>THE AGENT DIRECTS</span><b>THE CHILD DECIDES</b></footer>
        </aside>
      </section>
      {pendingUpload ? <div className="paper-picker-backdrop" role="dialog" aria-modal="true" aria-labelledby="paper-picker-title">
        <section className="paper-picker">
          <header><div><span>ONE QUICK TAP</span><h2 id="paper-picker-title">Which drawing?</h2></div><button onClick={cancelPendingUpload} aria-label="Close photo">×</button></header>
          <button
            className="paper-picker-image"
            onClick={(event) => {
              const image = event.currentTarget.querySelector("img");
              const bounds = image?.getBoundingClientRect() ?? event.currentTarget.getBoundingClientRect();
              processUploadedDrawing({ x: clamp01((event.clientX - bounds.left) / bounds.width), y: clamp01((event.clientY - bounds.top) / bounds.height) });
            }}
          ><img src={pendingUpload.url} alt="Choose one character from the uploaded sheet" /><span><i /> TAP INSIDE THE CHARACTER</span></button>
          <p>Paper edges, labels, grid lines, and nearby doodles will be treated as background.</p>
        </section>
      </div> : null}
      {partEditorOpen && capture ? <div className="part-editor-backdrop" role="dialog" aria-modal="true" aria-labelledby="part-editor-title">
        <section className="part-editor">
          <header><div><span>ANATOMY CHECK</span><h2 id="part-editor-title">Make it match.</h2></div><button onClick={() => { setPartEditorOpen(false); setPendingPartKind(null); }}>×</button></header>
          <div className="part-editor-workspace">
            <svg
              className={pendingPartKind ? "is-adding" : ""}
              viewBox="0 0 100 100"
              onPointerDown={(event) => {
                if (!pendingPartKind) return;
                const point = editorPoint(event.currentTarget, event.clientX, event.clientY);
                addRigPart(pendingPartKind, point.x, point.y);
              }}
            >
              <image href={capture.textureUrl} x="0" y="0" width="100" height="100" preserveAspectRatio="xMidYMid meet" />
              {capture.rig.parts.filter((part) => anatomyKinds.includes(part.kind as (typeof anatomyKinds)[number])).map((part) => {
                const x = (part.center.x / 1.4 + 0.5) * 100;
                const y = (0.5 - part.center.y / 1.4) * 100;
                const width = Math.max(4, part.size.x / 1.4 * 100);
                const height = Math.max(4, part.size.y / 1.4 * 100);
                const showMaskOutline = part.kind === "eye" || part.kind === "cheek" || part.kind === "nose" || part.kind === "mouth";
                const polygon = showMaskOutline ? part.outline?.map((point) => `${(point.x / 1.4 + 0.5) * 100},${(0.5 - point.y / 1.4) * 100}`).join(" ") : undefined;
                return <g
                  key={part.id}
                  className={`part-marker ${selectedPartId === part.id ? "selected" : ""}`}
                  onPointerDown={(event) => {
                    event.stopPropagation();
                    setSelectedPartId(part.id);
                    setPendingPartKind(null);
                    partDragRef.current = { pointerId: event.pointerId, partId: part.id };
                    event.currentTarget.setPointerCapture(event.pointerId);
                  }}
                  onPointerMove={(event) => {
                    if (partDragRef.current?.pointerId !== event.pointerId || partDragRef.current.partId !== part.id) return;
                    const svg = event.currentTarget.ownerSVGElement;
                    if (!svg) return;
                    const point = editorPoint(svg, event.clientX, event.clientY);
                    moveRigPart(part.id, point.x, point.y);
                  }}
                  onPointerUp={(event) => {
                    partDragRef.current = null;
                    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
                  }}
                >
                  {polygon ? <polygon points={polygon} /> : <ellipse cx={x} cy={y} rx={width / 2} ry={height / 2} />}
                  <circle cx={x} cy={y} r="1.4" />
                  {selectedPartId === part.id ? <text x={x} y={y - Math.max(3, height / 2 + 1.5)}>{part.kind}</text> : null}
                </g>;
              })}
            </svg>
            {pendingPartKind ? <div className="editor-hint">Tap where the {anatomyLabel[pendingPartKind].toLowerCase()} belongs</div> : <div className="editor-hint">Tap a part, then drag it</div>}
          </div>
          <div className="part-editor-tools">
            <p><b>Add missing</b><small>Then tap the drawing</small></p>
            <div>{anatomyKinds.map((kind) => <button key={kind} className={pendingPartKind === kind ? "active" : ""} onClick={() => { setPendingPartKind(kind); setSelectedPartId(null); }}>{anatomyLabel[kind]}</button>)}</div>
            <footer><button disabled={!selectedPartId} onClick={() => resizeSelectedPart(0.86)}>− SIZE</button><button disabled={!selectedPartId} onClick={() => resizeSelectedPart(1.16)}>＋ SIZE</button><button className="remove-part" disabled={!selectedPartId} onClick={deleteSelectedPart}>REMOVE</button><button className="editor-done" onClick={() => { setPartEditorOpen(false); setPendingPartKind(null); }}>LOOKS GOOD</button></footer>
          </div>
        </section>
      </div> : null}
    </main>
  );
}
