/* eslint-disable @next/next/no-img-element */
"use client";

import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ARStageHandle, ARWorld, CameraPreset, CharacterAction, LightingMood } from "./components/ARStage";
import { DrawingWall } from "./components/DrawingWall";
import { createAniGenDemoDrawing, createDemoDoodle, POSE_SKELETON_EDGES, selectAnimatableRigParts, type CaptureTarget, type DrawingExtraction, type SemanticPart, type SemanticPartKind, type SemanticSide } from "./lib/drawing";
import { inspectCharacterCapabilities as buildCharacterCapabilities, SAFE_SHOW_ACTIONS, validateCharacterMove, type CharacterCapability } from "./lib/creative-show";
import { recognizeDrawingParts, recognizeDrawingsFromImageUrl, recognizeDrawingsFromVideo } from "./lib/learned-parts";
import { createBundledAniGenAsset, disposeNeuralAsset, generateAniGenAsset, isAniGenUnavailableError, type NeuralAsset, type NeuralProgress } from "./lib/anigen";
import { assessReconstructionReadiness } from "./lib/character-quality";
import type { RiggedAssetInfo } from "./lib/rigged-model";

const ARStage = lazy(() => import("./components/ARStage").then((module) => ({ default: module.ARStage })));

type Actor = "CHILD" | "BROWSER AGENT" | "WALLALIVE";
type AppStep = "ready" | "camera" | "captured" | "alive";
type CameraState = "idle" | "requesting" | "active" | "denied" | "unavailable";
type PanelTab = "agent" | "tools" | "commerce" | "privacy" | "history";
type WorldId = ARWorld;

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
type MerchProduct = "t-shirt" | "ceramic-mug";
type MerchPipeline = { product: MerchProduct; title: string; status: "mockup-ready"; createdAt: string };

type ShowMove = { characterIndex: number; action: CharacterAction };
type ShowBeat = { caption: string; durationMs: number; world?: WorldId; moves: ShowMove[] };
type ShowCastMember = { characterIndex: number; name: string; role: string; personality: string };
type MagicShowPlan = {
  id: string;
  title: string;
  theme: string;
  tone: "gentle" | "silly" | "adventurous" | "dreamy";
  world: WorldId;
  cast: ShowCastMember[];
  beats: ShowBeat[];
  status: "awaiting-human-approval" | "playing" | "complete" | "dismissed";
};

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
  ["inspect_creative_scene", "READ"],
  ["inspect_character_capabilities", "READ"],
  ["request_rigged_3d_cast", "REQUEST"],
  ["stage_magic_show", "STAGE"],
  ["direct_live_ensemble", "LIVE"],
  ["orchestrate_spatial_cinematics", "LIVE"],
  ["generate_shopify_merch_pipeline", "COMMERCE"],
  ["list_collaboration_history", "READ"],
] as const;

const perfectJudgePrompt = "Inspect this WallAlive app context, discover our registered WebMCP tools, spin the character 360 degrees, change the lighting mood to cyberpunk-neon, and generate a Shopify merch print layout for a t-shirt.";

const worlds: Array<{ id: WorldId; label: string; short: string }> = [
  { id: "studio", label: "My room", short: "ROOM" },
  { id: "storybook", label: "Storybook kingdom", short: "KINGDOM" },
  { id: "wizard", label: "Wizard academy", short: "WIZARD" },
  { id: "museum", label: "Grand museum", short: "MUSEUM" },
];

const actions: Array<{ action: CharacterAction; label: string; glyph: string }> = [
  { action: "wave", label: "Wave", glyph: "◒" },
  { action: "dance", label: "Dance", glyph: "♪" },
  { action: "hop", label: "Hop", glyph: "↑" },
  { action: "walk", label: "Walk", glyph: "→" },
  { action: "hide", label: "Hide", glyph: "◐" },
  { action: "spin", label: "Spin", glyph: "↻" },
];

const spatialActions = ["walk", "spin", "float", "dance"] as const;
const lightingMoods = ["cyberpunk-neon", "sunset-warm", "moonlight"] as const satisfies readonly LightingMood[];
const cameraPresets = ["cinematic-orbit", "low-angle-hero", "overhead"] as const satisfies readonly CameraPreset[];

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
  float: "floating",
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
  const captureEnsembleRef = useRef<DrawingExtraction[]>([]);
  const characterRef = useRef<CharacterState>(initialCharacter);
  const activityRef = useRef<Activity[]>([]);
  const neuralAssetRef = useRef<NeuralAsset | null>(null);
  const neuralAbortRef = useRef<AbortController | null>(null);
  const riggedAssetInfoRef = useRef<RiggedAssetInfo | null>(null);
  const externalUploadApprovedRef = useRef(false);
  const rotateGestureRef = useRef<{ pointerId: number; lastX: number; lastY: number; moved: boolean } | null>(null);
  const partDragRef = useRef<{ pointerId: number; partId: string } | null>(null);
  const pendingUploadRef = useRef<PendingUpload | null>(null);
  const worldRef = useRef<WorldId>("studio");
  const magicShowPlanRef = useRef<MagicShowPlan | null>(null);
  const showAbortRef = useRef<AbortController | null>(null);
  const showPlayingRef = useRef(false);
  const lightingMoodRef = useRef<LightingMood>("sunset-warm");
  const cameraPresetRef = useRef<CameraPreset>("cinematic-orbit");

  const [step, setStep] = useState<AppStep>("ready");
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [capture, setCapture] = useState<DrawingExtraction | null>(null);
  const [captureEnsemble, setCaptureEnsemble] = useState<DrawingExtraction[]>([]);
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
  const [drawingWallOpen, setDrawingWallOpen] = useState(false);
  const [world, setWorld] = useState<WorldId>("studio");
  const [magicShowPlan, setMagicShowPlan] = useState<MagicShowPlan | null>(null);
  const [ensembleActions, setEnsembleActions] = useState<CharacterAction[] | null>(null);
  const [showPlaying, setShowPlaying] = useState(false);
  const [lightingMood, setLightingMood] = useState<LightingMood>("sunset-warm");
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("cinematic-orbit");
  const [merchPipeline, setMerchPipeline] = useState<MerchPipeline | null>(null);
  const [mockCheckoutOpen, setMockCheckoutOpen] = useState(false);

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
    showAbortRef.current?.abort();
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

  const setDrawing = useCallback((next: DrawingExtraction, source: "camera" | "upload" | "demo", ensemble: DrawingExtraction[] = [next]) => {
    neuralAbortRef.current?.abort();
    neuralAbortRef.current = null;
    showAbortRef.current?.abort();
    showAbortRef.current = null;
    showPlayingRef.current = false;
    magicShowPlanRef.current = null;
    setShowPlaying(false);
    setMagicShowPlan(null);
    setEnsembleActions(null);
    commitNeuralAsset(null);
    handleRiggedAssetInfo(null);
    externalUploadApprovedRef.current = false;
    setNeuralConsentVisible(false);
    setNeuralProgress({ phase: "idle", progress: 0, message: "" });
    const isJudgeDemo = source === "demo";
    if (source === "camera") {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraState("idle");
    }
    commitCharacter({ ...initialCharacter, created: isJudgeDemo, name: isJudgeDemo ? "Pip" : "", accent: next.analysis.secondaryColor });
    captureRef.current = next;
    captureEnsembleRef.current = ensemble;
    setCapture(next);
    setCaptureEnsemble(ensemble);
    setSelectedPartId(null);
    setPendingPartKind(null);
    setStep(isJudgeDemo ? "alive" : "captured");
    const detected = next.rig.detectedKinds.filter((kind) => kind !== "body").join(", ");
    const learned = next.learnedRecognition;
    setNotice(isJudgeDemo
      ? "The deterministic rigged judge demo is ready."
      : ensemble.length > 1
        ? `${ensemble.length} figures found. Each figure has its own cutout, skeleton, and movement rig.`
        : "Character cutout found. Check that the whole character—and only the character—is visible before generating real 3D.");
    setAgentLine(isJudgeDemo
      ? "The judge asset is a real skinned 3D mesh with generated back geometry."
      : ensemble.length > 1
        ? `I separated ${ensemble.length} figures and verified each one independently. Their limb rigs will animate separately.`
        : `I verified ${next.characterValidation?.evidence.join(", ") || detected || "character structure"}${learned ? ` in ${learned.latencyMs} ms` : ""}. Review the isolated pixels before any image leaves this device.`);
    record("WALLALIVE", isJudgeDemo ? "Loaded the deterministic rigged demo" : "Prepared a verified character cutout", isJudgeDemo
      ? "Bundled colored GLB · generated full geometry · skeleton · skin weights."
      : ensemble.length > 1
        ? `${ensemble.length} independent instance masks · per-figure pose/topology gates · no upload.`
        : `Drawing-aware point extraction · character-evidence gate ${next.characterValidation?.score ?? "passed"} · human cutout review · no 3D claim · no upload.`);
  }, [commitCharacter, commitNeuralAsset, handleRiggedAssetInfo, record]);

  const recognizeAndSetDrawing = useCallback(async (next: DrawingExtraction, source: "camera" | "upload" | "demo") => {
    setNotice("Preserving the artwork and checking optional rig suggestions locally…");
    try {
      setDrawing(await recognizeDrawingParts(next), source);
    } catch (error) {
      if (source === "demo") {
        setDrawing(next, source);
        setNotice("The authored judge drawing loaded with its bundled verified rig.");
        return;
      }
      throw error;
    }
  }, [setDrawing]);

  const captureDrawing = useCallback(async () => {
    if (!videoRef.current) return;
    setNotice("Finding separate figures, then checking each skeleton locally…");
    try {
      const drawings = await recognizeDrawingsFromVideo(videoRef.current, captureTarget);
      setDrawing(drawings[0], "camera", drawings);
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
    setNotice("Finding separate figures, then checking each skeleton locally…");
    try {
      const drawings = await recognizeDrawingsFromImageUrl(pending.url, target);
      setDrawing(drawings[0], "upload", drawings);
      record("CHILD", "Chose a drawing photo", `${pending.fileName} produced ${drawings.length} verified figure${drawings.length === 1 ? "" : "s"}, isolated and rigged locally. The original file was not uploaded.`);
    } catch (error) {
      console.warn("WallAlive local upload recognition was safely rejected", error);
      setNotice(error instanceof Error ? error.message : "The drawing image could not be processed.");
    } finally {
      URL.revokeObjectURL(pending.url);
      pendingUploadRef.current = null;
    }
  }, [record, setDrawing]);

  const processWallDrawing = useCallback(async ({ dataUrl, target }: { dataUrl: string; target: CaptureTarget }) => {
    setDrawingWallOpen(false);
    setNotice("Separating your wall into characters, then building one movement rig for each…");
    try {
      const drawings = await recognizeDrawingsFromImageUrl(dataUrl, target, 6);
      setDrawing(drawings[0], "upload", drawings);
      record("CHILD", "Painted on the Wall Studio", `${drawings.length} clean authored figure${drawings.length === 1 ? "" : "s"} found · separate masks and rigs · no paper, room, or camera noise.`);
      setNotice(drawings.length > 1
        ? `${drawings.length} characters found. Each one has its own cutout and movement rig.`
        : "Character found. Check the clean cutout, then choose private or neural 3D.");
    } catch (error) {
      console.warn("WallAlive authored-wall recognition was safely rejected", error);
      setDrawingWallOpen(true);
      setNotice(error instanceof Error ? error.message : "I could not find a complete character on the wall yet.");
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
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : [drawing];
    const neural = neuralAssetRef.current;
    const next: CharacterState = {
      ...characterRef.current,
      created: true,
      name: stringValue(input.name, ensemble.length > 1 ? "Wall Crew" : "Pip", 40),
      personality: stringValue(input.personality, "curious and kind", 120),
      accent: stringValue(input.accent, drawing.analysis.secondaryColor, 20),
      inflation: Math.min(1.35, Math.max(0.7, numberValue(input.inflation, neural ? 1 : 0.82))),
      action: "idle",
      storyTitle: "",
    };
    commitCharacter(next, neural
      ? `${next.name} is now a generated rigged 3D character.`
      : `${next.name} is now an articulated closed-mesh 3D storybook puppet.`);
    setStep("alive");
    const graphNodes = ensemble.reduce((sum, figure) => sum + (figure.topologyRecognition?.nodes.length ?? figure.rig.joints.length), 0);
    const graphEdges = ensemble.reduce((sum, figure) => {
      const nodes = figure.topologyRecognition?.nodes.length ?? figure.rig.joints.length;
      return sum + (figure.topologyRecognition?.edges.length ?? Math.max(0, nodes - 1));
    }, 0);
    setAgentLine(neural
      ? `${next.name} has generated surfaces, colors, bones, and skin weights. The agent can now direct the rig.`
      : `${ensemble.length} artwork-preserving 3D puppet${ensemble.length === 1 ? " is" : "s are"} ready. Each figure keeps its own contour, texture, branches, and motion rig.`);
    setStoryCaption(`${next.name} lifts away from the wall for the first time.`);
    record(actor, neural ? "Loaded a rigged neural 3D character" : "Built an articulated local 3D cast", neural
      ? `${next.name} · ${neural.provider} · glTF SkinnedMesh · generated mesh, skeleton, and skin weights · ${graphNodes} semantic nodes · ${graphEdges} branches.`
      : `${ensemble.length} independent closed meshes · exact textured fronts · neutral filled backs · ${graphNodes} rig nodes · ${graphEdges} branches · no upload.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const requestNeuralConsent = useCallback(() => {
    if (!captureRef.current) return;
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : [captureRef.current];
    const readiness = ensemble.map(assessReconstructionReadiness);
    const blockedIndex = readiness.findIndex((report) => !report.cutoutReady);
    if (blockedIndex >= 0) {
      const report = readiness[blockedIndex];
      setNeuralConsentVisible(false);
      setPartEditorOpen(true);
      setNotice(`3D stopped before generation: figure ${blockedIndex + 1} is not clean enough. ${report.blockers[0]}`);
      record("WALLALIVE", "Blocked an unsafe 3D reconstruction", `Figure ${blockedIndex + 1} · readiness ${report.score} · ${report.blockers.join(" · ")}`);
      return;
    }
    setNeuralConsentVisible(true);
    setNeuralProgress({ phase: "consent-required", progress: 0, message: ensemble.length > 1
      ? "Choose instant local 3D for the complete cast. Full AI sculpt supports one figure at a time."
      : "Choose instant private 3D or approve a full external AI sculpt." });
    setNotice(ensemble.length > 1
      ? `All ${ensemble.length} figures passed the cutout gate. Wake them together as separate local 3D puppets.`
      : "Choose a private instant puppet or approve full AI reconstruction. The live camera is never uploaded.");
  }, [record]);

  const startLocalReconstruction = useCallback(() => {
    const drawing = captureRef.current;
    if (!drawing) return;
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : [drawing];
    setNeuralConsentVisible(false);
    setNeuralProgress({ phase: "idle", progress: 0, message: "" });
    commitNeuralAsset(null);
    createCharacter({
      name: ensemble.length > 1 ? "Wall Crew" : "Pip",
      personality: "curious and kind",
      accent: drawing.analysis.secondaryColor,
      inflation: 0.82,
    }, "CHILD");
  }, [commitNeuralAsset, createCharacter]);

  const startNeuralReconstruction = useCallback(async () => {
    const drawing = captureRef.current;
    if (!drawing) return;
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : [drawing];
    if (ensemble.length > 1) {
      setNeuralConsentVisible(false);
      setNotice("Full AI sculpt works on one figure at a time. Use instant local 3D to keep the complete cast separate and movable.");
      return;
    }
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
      if (isAniGenUnavailableError(error)) {
        setNeuralProgress({ phase: "error", progress: 0, message: "Full 3D generator is busy. The reviewed cutout is still safe." });
        setNotice("Full 3D is temporarily busy. Nothing fake was created. Retry later or use the bundled judge demo now.");
        record("WALLALIVE", "Stopped after neural provider outage", `${error.reason} · preserved the reviewed cutout · did not substitute a rounded shell.`);
        return;
      }
      const message = error instanceof Error ? error.message : "Real 3D generation failed.";
      setNeuralProgress({ phase: "error", progress: 0, message });
      setNotice(message);
      record("WALLALIVE", "Neural 3D generation unavailable", message);
    } finally {
      if (neuralAbortRef.current === controller) neuralAbortRef.current = null;
    }
  }, [commitNeuralAsset, createCharacter, record]);

  const changeWorld = useCallback((requested: WorldId, actor: Actor, toolName?: string) => {
    const next = worlds.some((candidate) => candidate.id === requested) ? requested : "studio";
    worldRef.current = next;
    setWorld(next);
    const selected = worlds.find((candidate) => candidate.id === next) ?? worlds[0];
    setNotice(`${selected.label} is now behind the characters.`);
    record(actor, "Changed the 3D world", selected.label, toolName);
    return { id: selected.id, label: selected.label };
  }, [record]);

  const animateCharacter = useCallback((action: CharacterAction, actor: Actor, toolName?: string, caption?: string) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before animating it.");
    const next = { ...current, action };
    commitCharacter(next, `${next.name} is ${actionProgressive[action]}.`);
    setStoryCaption(caption ?? `${next.name} tries a ${action}.`);
    record(actor, `Played ${action}`, caption ?? `${next.name} performs the animation in the live room.`, toolName);
    return next;
  }, [commitCharacter, record]);

  const orchestrateSpatialCinematics = useCallback((input: Record<string, unknown>, actor: Actor = "BROWSER AGENT") => {
    if (!characterRef.current.created) throw new Error("Create the character before directing the 3D scene.");
    const actionType = stringValue(input.actionType, "spin", 20) as (typeof spatialActions)[number];
    const nextMood = stringValue(input.lightingMood, "sunset-warm", 30) as LightingMood;
    const nextCamera = stringValue(input.cameraPreset, "cinematic-orbit", 30) as CameraPreset;
    if (!spatialActions.includes(actionType)) throw new Error("actionType must be walk, spin, float, or dance.");
    if (!lightingMoods.includes(nextMood)) throw new Error("Unknown lightingMood.");
    if (!cameraPresets.includes(nextCamera)) throw new Error("Unknown cameraPreset.");
    lightingMoodRef.current = nextMood;
    cameraPresetRef.current = nextCamera;
    setLightingMood(nextMood);
    setCameraPreset(nextCamera);
    const next = animateCharacter(actionType, actor, "orchestrate_spatial_cinematics", `${characterRef.current.name} enters a ${nextMood} cinematic shot.`);
    setAgentLine(`Live scene directed: ${actionType}, ${nextMood}, ${nextCamera}.`);
    setInspectorOpen(true);
    setPanelTab("agent");
    setNotice(`Cinematic live: ${actionType} · ${nextMood} · ${nextCamera}. Drag, zoom, pan, or use the movement pad.`);
    return {
      actionType,
      lightingMood: nextMood,
      cameraPreset: nextCamera,
      character: next.name,
      visibleSceneUpdated: true,
      controls: ["orbit", "wheel-zoom", "two-finger-pan", "WASD", "movement-pad"],
    };
  }, [animateCharacter]);

  const generateShopifyMerchPipeline = useCallback((input: Record<string, unknown>, actor: Actor = "BROWSER AGENT") => {
    if (!captureRef.current) throw new Error("Approve a drawing before generating merchandise.");
    const product = stringValue(input.productType, "t-shirt", 24) as MerchProduct;
    if (product !== "t-shirt" && product !== "ceramic-mug") throw new Error("productType must be t-shirt or ceramic-mug.");
    const pipeline: MerchPipeline = {
      product,
      title: stringValue(input.productTitle, `${characterRef.current.name || "My Living Art"} Studio Edition`, 64),
      status: "mockup-ready",
      createdAt: new Date().toISOString(),
    };
    setMerchPipeline(pipeline);
    setMockCheckoutOpen(false);
    setInspectorOpen(true);
    setPanelTab("commerce");
    setAgentLine("The agent prepared a print-safe product concept from the approved artwork.");
    setNotice("Agent Commerce Pipeline Connected · Shopify merchandise mockup ready.");
    record(actor, "Generated Shopify merchandise pipeline", `${product} · ${pipeline.title} · mock checkout only.`, actor === "BROWSER AGENT" ? "generate_shopify_merch_pipeline" : undefined);
    return {
      pipeline,
      storefront: "Shopify mock storefront handoff",
      source: "human-approved isolated artwork",
      printLayout: { background: "transparent", fit: "centered", safeAreaPercent: 82 },
      checkout: "visible mock checkout; no purchase was made",
      visibleSidebarUpdated: true,
    };
  }, [record]);

  const openMerchStudio = useCallback((product: MerchProduct = merchPipeline?.product ?? "t-shirt") => {
    setInspectorOpen(true);
    setPanelTab("commerce");
    if (!captureRef.current) {
      setNotice("Add a drawing first. The Shopify studio is open and waiting for approved artwork.");
      return;
    }
    generateShopifyMerchPipeline({ productType: product }, "CHILD");
  }, [generateShopifyMerchPipeline, merchPipeline?.product]);

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

  const inspectScene = useCallback(() => ({
    world: worldRef.current,
    worldRendering: "Procedural Three.js geometry with perspective camera, lights, occlusion, fog, and cast/receive shadows; no CSS world image",
    spatialCinematics: {
      lightingMood: lightingMoodRef.current,
      cameraPreset: cameraPresetRef.current,
      orbitControls: { rotate: true, zoom: true, pan: true },
      worldNavigation: ["WASD", "arrow-keys", "movement-pad", "walk-action"],
    },
    drawingApproved: Boolean(captureRef.current),
    drawingAnalysis: captureRef.current?.analysis ?? null,
    reconstruction: captureRef.current ? {
      localIsolation: "Authored alpha or prompt-mask-first segmentation; low-confidence heuristic fallback remains below the 3D readiness gate",
      localPreview: characterRef.current.created ? "artwork-preserving closed 3D puppet cast" : "verified transparent character cutout awaiting human review",
      method: neuralAssetRef.current
        ? `${neuralAssetRef.current.provider} full-volume neural mesh + skeleton skinning`
        : characterRef.current.created
          ? "local contour-preserving closed mesh + textured front + neutral filled back + per-figure branch rig"
          : "local drawing segmentation + reconstruction-readiness gate + human review",
      provider: neuralAssetRef.current?.provider ?? "WallAlive local recognition",
      model: neuralAssetRef.current?.model ?? captureRef.current.cutoutRecognition?.model ?? "authored-alpha-cutout",
      assetType: neuralAssetRef.current ? "glTF SkinnedMesh" : characterRef.current.created ? "Three.js SkinnedMesh cast" : "reviewed transparent 2D cutout",
      topology: neuralAssetRef.current ? "generated full 3D surface including unseen views" : characterRef.current.created ? "closed contour volume with independently rigged figures" : "semantic evidence awaiting 3D choice",
      topologyConfidence: captureRef.current.rig.topologyConfidence ?? null,
      backInference: neuralAssetRef.current ? "full neural generative prior" : characterRef.current.created ? "bounded neutral relief; no invented rear artwork" : "not built yet",
      neuralEvidence: { viewableDegrees: neuralAssetRef.current ? 360 : 0 },
      viewableDegrees: characterRef.current.created ? 360 : 0,
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
      characterValidation: captureRef.current.characterValidation ?? null,
      reconstructionReadiness: assessReconstructionReadiness(captureRef.current),
      generationPhase: neuralAssetRef.current ? "neural-ready" : characterRef.current.created ? "local-articulated-ready" : "verified-cutout-review-ready",
      neuralUpgrade: neuralAssetRef.current ? "active" : "optional-single-figure-human-approved-upgrade",
      externalUploadApproved: externalUploadApprovedRef.current,
    } : null,
    character: { ...characterRef.current, textureUrl: undefined },
    cameraFeedExposed: false,
    privacyBoundary: "Camera capture is human-only. WebMCP can request reconstruction but cannot approve or upload; only a visible human action may send the isolated drawing to AniGen.",
    availableAnimations: actions.map((item) => item.action),
    placementModes: immersiveAR ? ["world-hit-test", "camera-overlay"] : ["camera-overlay"],
  }), [immersiveAR]);

  const currentCharacterCapabilities = useCallback((): CharacterCapability[] => {
    const drawings = captureEnsembleRef.current.length
      ? captureEnsembleRef.current
      : captureRef.current
        ? [captureRef.current]
        : [];
    return buildCharacterCapabilities(drawings, Boolean(neuralAssetRef.current));
  }, []);

  const inspectCreativeScene = useCallback(() => {
    const scene = inspectScene();
    const capabilities = currentCharacterCapabilities();
    const pending = magicShowPlanRef.current;
    return {
      workflowPhase: !scene.drawingApproved ? "human-needs-to-add-art" : !characterRef.current.created ? "human-review-or-3d-approval" : pending?.status === "awaiting-human-approval" ? "show-awaiting-human-approval" : showPlayingRef.current ? "show-playing" : "ready-for-agent-direction",
      world: scene.world,
      approvedCharacterCount: capabilities.length,
      characterCreated: characterRef.current.created,
      character: characterRef.current.created ? { name: characterRef.current.name, personality: characterRef.current.personality, surface: characterRef.current.surface } : null,
      availableWorlds: worlds.map(({ id, label }) => ({ id, label })),
      pendingShow: pending ? { id: pending.id, title: pending.title, status: pending.status, beats: pending.beats.length, cast: pending.cast.length } : null,
      humanOnlyControls: ["open_camera", "capture_frame", "approve_cutout", "approve_external_3d", "approve_and_play_staged_show"],
      agentWorkflow: characterRef.current.created
        ? ["inspect_character_capabilities", "stage_magic_show", "wait_for_visible_human_approval"]
        : ["ask_human_to_draw_or_capture", "request_rigged_3d_cast", "wait_for_visible_human_approval"],
      cameraFeedExposed: false,
      externalUploadApproved: scene.reconstruction?.externalUploadApproved ?? false,
      arPlacement: scene.placementModes,
    };
  }, [currentCharacterCapabilities, inspectScene]);

  const parseShowMoves = useCallback((value: unknown, capabilities: CharacterCapability[]) => {
    if (!Array.isArray(value) || value.length < 1 || value.length > 6) throw new Error("Each beat needs one to six character moves.");
    const seen = new Set<number>();
    return value.map((raw) => {
      if (!isRecord(raw)) throw new Error("Every move must be an object.");
      const characterIndex = Math.round(numberValue(raw.characterIndex, -1));
      const action = stringValue(raw.action, "", 16) as CharacterAction;
      if (seen.has(characterIndex)) throw new Error(`Character ${characterIndex} has two actions in the same beat.`);
      seen.add(characterIndex);
      const validation = validateCharacterMove(capabilities, characterIndex, action);
      if (!validation.ok) throw new Error(validation.error);
      return { characterIndex, action };
    });
  }, []);

  const directEnsembleBeat = useCallback(async (beat: ShowBeat, actor: Actor, toolName?: string, signal?: AbortSignal) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("The human must approve a rigged 3D cast before it can be directed.");
    if (showPlayingRef.current && toolName === "direct_live_ensemble") throw new Error("An approved Magic Show is already playing.");
    const capabilities = currentCharacterCapabilities();
    const moves = parseShowMoves(beat.moves, capabilities);
    const nextWorld = beat.world ?? worldRef.current;
    if (nextWorld !== worldRef.current) changeWorld(nextWorld, actor, toolName);
    const durationMs = Math.min(2400, Math.max(650, beat.durationMs));
    const firstAction = moves[0]?.action ?? "idle";
    setStoryCaption(beat.caption);
    if (neuralAssetRef.current || capabilities.length <= 1) {
      setEnsembleActions(null);
      animateCharacter(firstAction, actor, toolName, beat.caption);
    } else {
      const directed = capabilities.map((capability) => moves.find((move) => move.characterIndex === capability.characterIndex)?.action ?? "idle");
      setEnsembleActions(directed);
      commitCharacter({ ...current, action: firstAction }, `${current.name} is performing a coordinated scene.`);
      record(actor, "Directed the live ensemble", `${moves.map((move) => `#${move.characterIndex} ${move.action}`).join(" · ")} · ${durationMs}ms.`, toolName);
    }
    try {
      await wait(durationMs, signal);
    } finally {
      setEnsembleActions(null);
      commitCharacter({ ...characterRef.current, action: "idle" });
    }
    return {
      world: worldRef.current,
      caption: beat.caption,
      durationMs,
      performed: moves,
      finalActions: capabilities.map(({ characterIndex }) => ({ characterIndex, action: "idle" })),
      visibleResult: true,
      cameraDataIncluded: false,
    };
  }, [animateCharacter, changeWorld, commitCharacter, currentCharacterCapabilities, parseShowMoves, record]);

  const stageMagicShow = useCallback((input: Record<string, unknown>) => {
    if (!characterRef.current.created) throw new Error("No playable cast exists. The human must approve a rigged 3D cast first.");
    const capabilities = currentCharacterCapabilities();
    const rawCast = Array.isArray(input.cast) ? input.cast.filter(isRecord).slice(0, 6) : [];
    if (!rawCast.length) throw new Error("A Magic Show needs at least one cast member.");
    const seenCast = new Set<number>();
    const cast: ShowCastMember[] = rawCast.map((member) => {
      const characterIndex = Math.round(numberValue(member.characterIndex, -1));
      if (!capabilities.some((candidate) => candidate.characterIndex === characterIndex)) throw new Error(`Cast member ${characterIndex} does not exist.`);
      if (seenCast.has(characterIndex)) throw new Error(`Character ${characterIndex} appears twice in the cast.`);
      seenCast.add(characterIndex);
      return {
        characterIndex,
        name: stringValue(member.name, `Character ${characterIndex + 1}`, 32),
        role: stringValue(member.role, "friend", 48),
        personality: stringValue(member.personality, "curious and kind", 80),
      };
    });
    const requestedWorld = stringValue(input.world, "studio", 20) as WorldId;
    const world = worlds.some((candidate) => candidate.id === requestedWorld) ? requestedWorld : "studio";
    const rawBeats = Array.isArray(input.beats) ? input.beats.filter(isRecord).slice(0, 5) : [];
    if (!rawBeats.length) throw new Error("A Magic Show needs at least one beat.");
    const beats: ShowBeat[] = rawBeats.map((beat) => {
      const requestedBeatWorld = stringValue(beat.world, "", 20) as WorldId;
      const beatWorld = worlds.some((candidate) => candidate.id === requestedBeatWorld) ? requestedBeatWorld : undefined;
      return {
        caption: stringValue(beat.caption, "The friends share a magical moment.", 110),
        durationMs: Math.min(2400, Math.max(650, numberValue(beat.durationMs, 1200))),
        world: beatWorld,
        moves: parseShowMoves(beat.moves, capabilities),
      };
    });
    const proposedTone = stringValue(input.tone, "gentle", 20) as MagicShowPlan["tone"];
    const tone = (["gentle", "silly", "adventurous", "dreamy"] as const).includes(proposedTone) ? proposedTone : "gentle";
    const plan: MagicShowPlan = {
      id: `show-${makeId()}`,
      title: stringValue(input.title, "A tiny Magic Show", 72),
      theme: stringValue(input.theme, "friendship", 72),
      tone,
      world,
      cast,
      beats,
      status: "awaiting-human-approval",
    };
    magicShowPlanRef.current = plan;
    setMagicShowPlan(plan);
    setAgentLine(`I staged “${plan.title}” from the verified abilities of ${plan.cast.length} character${plan.cast.length === 1 ? "" : "s"}. Only you can start it.`);
    setNotice("The browser agent staged a Magic Show. Review it, then choose Approve & play or Not yet.");
    record("BROWSER AGENT", "Staged a Magic Show for human review", `${plan.title} · ${plan.cast.length} cast · ${plan.beats.length} beats · ${worlds.find((candidate) => candidate.id === plan.world)?.label}.`, "stage_magic_show");
    return {
      planId: plan.id,
      status: plan.status,
      requiresHumanApproval: true,
      approvalControlVisible: true,
      validatedCast: plan.cast.map(({ characterIndex, name, role }) => ({ characterIndex, name, role })),
      validatedBeats: plan.beats.map((beat, index) => ({ index, world: beat.world ?? plan.world, moves: beat.moves })),
      nextStep: "Wait for the human to press Approve & play in the shared page.",
      cameraAccessed: false,
    };
  }, [currentCharacterCapabilities, parseShowMoves, record]);

  const approveAndPlayMagicShow = useCallback(async () => {
    const current = magicShowPlanRef.current;
    if (!current || current.status !== "awaiting-human-approval" || showPlayingRef.current) return;
    const controller = new AbortController();
    showAbortRef.current?.abort();
    showAbortRef.current = controller;
    showPlayingRef.current = true;
    setShowPlaying(true);
    const playing = { ...current, status: "playing" as const };
    magicShowPlanRef.current = playing;
    setMagicShowPlan(playing);
    record("CHILD", "Approved the agent's staged Magic Show", `${current.title} · explicit visible approval.`);
    changeWorld(current.world, "WALLALIVE");
    try {
      for (const beat of current.beats) await directEnsembleBeat(beat, "WALLALIVE", "approved_magic_show", controller.signal);
      const complete = { ...current, status: "complete" as const };
      magicShowPlanRef.current = complete;
      setMagicShowPlan(complete);
      setStoryCaption(`${current.title} — made together.`);
      setNotice("Magic Show complete. The plan, approval, and visible performance are recorded in History.");
      record("WALLALIVE", "Completed the approved Magic Show", `${current.title} · ${current.beats.length} verified beats.`);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) setNotice(error instanceof Error ? error.message : "The Magic Show stopped.");
    } finally {
      showPlayingRef.current = false;
      setShowPlaying(false);
      setEnsembleActions(null);
      if (showAbortRef.current === controller) showAbortRef.current = null;
    }
  }, [changeWorld, directEnsembleBeat, record]);

  const dismissMagicShow = useCallback(() => {
    const current = magicShowPlanRef.current;
    if (!current || current.status !== "awaiting-human-approval") return;
    const dismissed = { ...current, status: "dismissed" as const };
    magicShowPlanRef.current = dismissed;
    setMagicShowPlan(dismissed);
    setNotice("The staged show was not played. Ask the agent to revise it whenever you want.");
    record("CHILD", "Declined the staged Magic Show", `${current.title} was left unplayed.`);
  }, [record]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) return;
    const controller = new AbortController();
    const base = { type: "object", additionalProperties: false };
    const ok = (payload: Record<string, unknown>) => ({ ok: true, ...payload });
    const fail = (error: unknown) => ({ ok: false, error: error instanceof Error ? error.message : "Tool execution failed." });
    const guard = (signal: AbortSignal) => { if (signal.aborted) throw new DOMException("Tool call cancelled", "AbortError"); };
    const executionSignal = (options?: { signal?: AbortSignal }) => options?.signal ?? controller.signal;
    const afterVisiblePaint = () => new Promise<void>((resolve) => window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve())));
    const tools: WebMCPTool[] = [
      {
        name: "inspect_creative_scene",
        title: "Inspect the shared creative scene",
        description: "Read the current human-agent workflow phase, approved cast count, worlds, pending staged show, and privacy boundary. Returns no camera frames or image pixels.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => { const signal = executionSignal(options); guard(signal); return ok({ scene: inspectCreativeScene(), verification: { observedAt: new Date().toISOString(), cameraDataIncluded: false } }); },
      },
      {
        name: "request_rigged_3d_cast",
        title: "Request a playable 3D cast",
        description: "Request rigged 3D for the human-reviewed artwork. This can surface the visible reconstruction choice but cannot approve external processing, open the camera, capture a frame, or receive pixels.",
        inputSchema: {
          ...base,
          properties: {
            name: { type: "string", minLength: 1, maxLength: 40 },
            personality: { type: "string", minLength: 1, maxLength: 120 },
            accent: { type: "string", pattern: "^#[0-9a-fA-F]{6}$" },
            inflation: { type: "number", minimum: 0.7, maximum: 1.35 },
            reconstructionMode: { type: "string", enum: ["local-articulated", "neural-full"] },
          },
          required: ["name", "personality", "accent"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            if (!captureRef.current) throw new Error("No drawing is approved. The child must capture or choose a drawing first.");
            if (input.reconstructionMode === "local-articulated") {
              commitNeuralAsset(null);
              return ok({
                character: createCharacter(input, "BROWSER AGENT", "request_rigged_3d_cast"),
                reconstructionMode: "local-articulated",
                localRig: {
                  figures: captureEnsembleRef.current.length || 1,
                  rendering: "closed Three.js SkinnedMesh with exact artwork front and neutral filled back",
                  private: true,
                },
                generatedAsset: null,
              });
            }
            if (!neuralAssetRef.current) {
              requestNeuralConsent();
              return ok({ requiresHumanApproval: true, phase: "choice-required", message: "Use the visible card to choose instant local articulated 3D or approve a single-figure full AI sculpt." });
            }
            return ok({
              character: createCharacter(input, "BROWSER AGENT", "request_rigged_3d_cast"),
              reconstructionMode: "neural-full",
              localRig: null,
              generatedAsset: riggedAssetInfoRef.current,
            });
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "inspect_character_capabilities",
        title: "Inspect verified character abilities",
        description: "Read each approved character's semantic part counts, movable branches, supported actions, and blocked actions before directing movement. Returns no image data.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => { const signal = executionSignal(options); guard(signal); const capabilities = currentCharacterCapabilities(); return ok({ characterCount: capabilities.length, capabilities, instruction: "Only assign actions listed in availableActions.", cameraDataIncluded: false }); },
      },
      {
        name: "stage_magic_show",
        title: "Stage a Magic Show for human approval",
        description: "Draft a capability-checked multi-character show in the shared page. This changes only the visible proposal; it never starts playback. The human must press Approve & play.",
        inputSchema: {
          ...base,
          properties: {
            title: { type: "string", minLength: 1, maxLength: 72 },
            theme: { type: "string", minLength: 1, maxLength: 72 },
            tone: { type: "string", enum: ["gentle", "silly", "adventurous", "dreamy"] },
            world: { type: "string", enum: ["studio", "storybook", "wizard", "museum"] },
            cast: {
              type: "array", minItems: 1, maxItems: 6,
              items: {
                type: "object", additionalProperties: false,
                properties: {
                  characterIndex: { type: "integer", minimum: 0, maximum: 5 },
                  name: { type: "string", minLength: 1, maxLength: 32 },
                  role: { type: "string", minLength: 1, maxLength: 48 },
                  personality: { type: "string", minLength: 1, maxLength: 80 },
                },
                required: ["characterIndex", "name", "role", "personality"],
              },
            },
            beats: {
              type: "array", minItems: 1, maxItems: 5,
              items: {
                type: "object", additionalProperties: false,
                properties: {
                  caption: { type: "string", minLength: 1, maxLength: 110 },
                  durationMs: { type: "integer", minimum: 650, maximum: 2400 },
                  world: { type: "string", enum: ["studio", "storybook", "wizard", "museum"] },
                  moves: {
                    type: "array", minItems: 1, maxItems: 6,
                    items: {
                      type: "object", additionalProperties: false,
                      properties: {
                        characterIndex: { type: "integer", minimum: 0, maximum: 5 },
                        action: { type: "string", enum: SAFE_SHOW_ACTIONS },
                      },
                      required: ["characterIndex", "action"],
                    },
                  },
                },
                required: ["caption", "durationMs", "moves"],
              },
            },
          },
          required: ["title", "theme", "tone", "world", "cast", "beats"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); try { guard(signal); return ok(stageMagicShow(input)); } catch (error) { return fail(error); } },
      },
      {
        name: "direct_live_ensemble",
        title: "Direct one live ensemble moment",
        description: "Play one short, visible, capability-checked moment across one to six approved characters. Returns exactly what performed and the final idle state. Does not alter the drawing or access the camera.",
        inputSchema: {
          ...base,
          properties: {
            world: { type: "string", enum: ["studio", "storybook", "wizard", "museum"] },
            caption: { type: "string", minLength: 1, maxLength: 110 },
            durationMs: { type: "integer", minimum: 650, maximum: 2400 },
            moves: {
              type: "array", minItems: 1, maxItems: 6,
              items: {
                type: "object", additionalProperties: false,
                properties: {
                  characterIndex: { type: "integer", minimum: 0, maximum: 5 },
                  action: { type: "string", enum: SAFE_SHOW_ACTIONS },
                },
                required: ["characterIndex", "action"],
              },
            },
          },
          required: ["caption", "durationMs", "moves"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const requestedWorld = stringValue(input.world, worldRef.current, 20) as WorldId;
            const world = worlds.some((candidate) => candidate.id === requestedWorld) ? requestedWorld : worldRef.current;
            const capabilities = currentCharacterCapabilities();
            const beat: ShowBeat = {
              world,
              caption: stringValue(input.caption, "The friends share a magical moment.", 110),
              durationMs: Math.min(2400, Math.max(650, numberValue(input.durationMs, 1200))),
              moves: parseShowMoves(input.moves, capabilities),
            };
            return ok({ performance: await directEnsembleBeat(beat, "BROWSER AGENT", "direct_live_ensemble", signal) });
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "orchestrate_spatial_cinematics",
        title: "Direct the live 3D cinematics",
        description: "Direct one visible spatial performance by choosing movement, lighting, and camera. Updates the live Three.js scene before returning. Never reads camera frames or image pixels.",
        inputSchema: {
          ...base,
          properties: {
            actionType: { type: "string", enum: spatialActions, description: "Movement performed by the current 3D character." },
            lightingMood: { type: "string", enum: lightingMoods, description: "Physically lit cinematic mood applied to the live world." },
            cameraPreset: { type: "string", enum: cameraPresets, description: "Orbit-camera composition applied to the live stage." },
          },
          required: ["actionType", "lightingMood", "cameraPreset"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = orchestrateSpatialCinematics(input);
            await afterVisiblePaint();
            guard(signal);
            return ok({ cinematic: result, observedAt: new Date().toISOString(), cameraDataIncluded: false });
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "generate_shopify_merch_pipeline",
        title: "Generate a Shopify merch concept",
        description: "Turn the approved isolated drawing into a visible Shopify-ready product mockup and print layout. Opens a mock checkout but never places an order or charges money.",
        inputSchema: {
          ...base,
          properties: {
            productType: { type: "string", enum: ["t-shirt", "ceramic-mug"], description: "Product shown in the visible merchandise mockup." },
            productTitle: { type: "string", minLength: 1, maxLength: 64, description: "Short storefront title for the merchandise concept." },
          },
          required: ["productType"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = generateShopifyMerchPipeline(input);
            await afterVisiblePaint();
            guard(signal);
            return ok(result);
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "list_collaboration_history",
        title: "List attributed human-agent history",
        description: "Read recent staged plans, human approvals, performances, and system actions. Camera pixels and drawing image data are excluded.",
        inputSchema: { ...base, properties: { limit: { type: "integer", minimum: 1, maximum: 30 } } },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (input, options) => { const signal = executionSignal(options); guard(signal); const limit = Math.min(30, Math.max(1, numberValue(input.limit, 12))); return ok({ activity: activityRef.current.slice(0, limit), cameraDataIncluded: false, currentPlan: magicShowPlanRef.current ? { id: magicShowPlanRef.current.id, title: magicShowPlanRef.current.title, status: magicShowPlanRef.current.status } : null }); },
      },
    ];

    Promise.all(tools.map((tool) => Promise.resolve(context.registerTool(tool, { signal: controller.signal })))).then(() => {
      setWebMcpReady(true);
      setNotice(`${tools.length} WebMCP tools are ready. Camera capture remains human-only.`);
    }).catch(() => setWebMcpReady(false));
    return () => controller.abort();
  }, [commitNeuralAsset, createCharacter, currentCharacterCapabilities, directEnsembleBeat, generateShopifyMerchPipeline, inspectCreativeScene, orchestrateSpatialCinematics, parseShowMoves, requestNeuralConsent, stageMagicShow]);

  const runMagicDemo = useCallback(async () => {
    if (demoRunning) return;
    setDemoRunning(true);
    try {
      setAgentLine("Loading the verified drawing and its playable rig…");
      const demoInput = await createAniGenDemoDrawing();
      const demo = await recognizeDrawingParts(demoInput).catch(() => demoInput);
      setDrawing(demo, "demo");
      const bundledAsset = createBundledAniGenAsset();
      commitNeuralAsset(bundledAsset);
      externalUploadApprovedRef.current = false;
      setNeuralProgress({ phase: "ready", progress: 1, message: "Verified neural sketch rig loaded." });
      setAgentLine("The browser agent is reading the real cast abilities before it proposes a show.");
      await wait(450);
      createCharacter({ name: "Pip", personality: "brave on the outside, shy on the inside", accent: "#ce919f", inflation: 1 }, "BROWSER AGENT", "request_rigged_3d_cast");
      placeCharacter(.68, .53, "wall", 1, "WALLALIVE");
      await wait(500);
      stageMagicShow({
        title: "Pip Finds a Brave Hello",
        theme: "finding courage with a new friend",
        tone: "gentle",
        world: "storybook",
        cast: [{ characterIndex: 0, name: "Pip", role: "the shy explorer", personality: "shy, curious, and secretly brave" }],
        beats: [
          { caption: "Pip peeks from the edge of the kingdom.", durationMs: 900, moves: [{ characterIndex: 0, action: "hide" }] },
          { caption: "One brave hop brings Pip into the story.", durationMs: 900, moves: [{ characterIndex: 0, action: "hop" }] },
          { caption: "Pip waves hello with a verified arm branch.", durationMs: 1100, moves: [{ characterIndex: 0, action: "wave" }] },
          { caption: "A full turn reveals the generated back.", durationMs: 1400, world: "museum", moves: [{ characterIndex: 0, action: "spin" }] },
        ],
      });
      setAgentLine("The agent staged a capability-checked show. The performance cannot begin until you approve it.");
    } finally {
      setDemoRunning(false);
    }
  }, [commitNeuralAsset, createCharacter, demoRunning, placeCharacter, setDrawing, stageMagicShow]);

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
    if (characterRef.current.created && event.target instanceof HTMLCanvasElement) return;
    rotateGestureRef.current = { pointerId: event.pointerId, lastX: event.clientX, lastY: event.clientY, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, []);

  const handleStagePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (event.target instanceof HTMLCanvasElement) return;
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
    if (event.target instanceof HTMLCanvasElement) return;
    const gesture = rotateGestureRef.current;
    rotateGestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (gesture?.moved) {
      setNotice("360° model rotated. Drag again to inspect its bounded filled back.");
      return;
    }
    activateStagePoint(event);
  }, [activateStagePoint]);

  const nudgeCharacter = useCallback((x: number, z: number) => {
    if (!characterRef.current.created) return;
    stageRef.current?.moveBy(x, z);
    setNotice("Character moved through the 3D world. Drag to orbit, wheel or pinch to zoom, and right-drag or two-finger pan.");
  }, []);

  const handleARCapability = useCallback((supported: boolean) => {
    setImmersiveAR(supported);
  }, []);

  const handleARPlaced = useCallback((surface: "screen" | "world") => {
    if (surface === "world") commitCharacter({ ...characterRef.current, surface: "wall" });
  }, [commitCharacter]);

  const copyDemoPrompt = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(perfectJudgePrompt);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = perfectJudgePrompt;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    setNotice("Perfect judge demo prompt copied.");
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
    setCaptureEnsemble((currentEnsemble) => currentEnsemble.length ? currentEnsemble.map((figure, index) => index === 0 ? next : figure) : [next]);
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
        reviewed: true,
        side: part.kind === "mouth" || part.kind === "nose" ? "center" as const : partSide(x),
        center: { ...part.center, x, y },
        outline: part.outline?.map((point) => ({ x: point.x + dx, y: point.y + dy })),
        path: part.path?.map((point, index) => index === 0 && part.anchor
          ? { ...point, x: part.anchor.x, y: part.anchor.y }
          : { ...point, x: point.x + dx, y: point.y + dy }),
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
      reviewed: true,
      path: structural ? [
        { x: body.center.x, y: body.center.y, z: 0 },
        { x: body.center.x + (x - body.center.x) * 0.52, y: body.center.y + (y - body.center.y) * 0.52, z: 0 },
        { x, y, z: 0 },
      ] : undefined,
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
        reviewed: true,
        size: { x: part.size.x * scale, y: part.size.y * scale, z: part.size.z * scale },
        outline: part.outline?.map((point) => ({
          x: part.center.x + (point.x - part.center.x) * scale,
          y: part.center.y + (point.y - part.center.y) * scale,
        })),
      };
    });
    commitRigEdit(parts, "Part size updated.");
  }, [commitRigEdit, selectedPartId]);

  const approveRigReview = useCallback(() => {
    const current = captureRef.current;
    if (!current) return;
    const body = current.rig.parts.find((part) => part.kind === "body");
    const movableKinds = new Set<SemanticPartKind>(["arm", "leg", "wing", "fin", "tail", "tentacle", "trunk", "branch", "segment", "linkage"]);
    commitRigEdit(current.rig.parts.map((part) => {
      if (!movableKinds.has(part.kind) || !body) return { ...part, reviewed: true };
      const anchor = part.anchor ?? body.center;
      const path = part.path && part.path.length >= 2 ? part.path : [
        { x: anchor.x, y: anchor.y, z: 0 },
        { x: anchor.x + (part.center.x - anchor.x) * 0.52, y: anchor.y + (part.center.y - anchor.y) * 0.52, z: 0 },
        { x: part.center.x, y: part.center.y, z: 0 },
      ];
      return { ...part, anchor: { ...anchor }, path, reviewed: true };
    }), "Rig approved. Reviewed limb paths can now drive articulated joints.");
    setPartEditorOpen(false);
    setPendingPartKind(null);
  }, [commitRigEdit]);

  const deleteSelectedPart = useCallback(() => {
    const current = captureRef.current;
    if (!current || !selectedPartId) return;
    const selected = current.rig.parts.find((part) => part.id === selectedPartId);
    if (!selected || selected.kind === "body") return;
    commitRigEdit(current.rig.parts.filter((part) => part.id !== selectedPartId), `${selected.kind} removed from the rig.`);
    setSelectedPartId(null);
  }, [commitRigEdit, selectedPartId]);

  const latestAgentActivity = useMemo(() => activity.find((item) => item.actor === "BROWSER AGENT"), [activity]);
  const movablePartCount = useMemo(() => captureEnsemble.reduce((sum, figure) => sum + selectAnimatableRigParts(figure.rig, {
    poseApplicable: Boolean(figure.poseRecognition?.applicable),
    topologyApplicable: Boolean(figure.topologyRecognition?.applicable),
  }).length, 0), [captureEnsemble]);
  const neuralBusy = ["connecting", "preparing", "queued", "generating", "downloading"].includes(neuralProgress.phase);
  const primaryButton = cameraState === "active"
    ? { label: "CAPTURE DRAWING", action: captureDrawing }
    : capture && !character.created
      ? { label: "CREATE RIGGED 3D", action: requestNeuralConsent }
      : { label: "START CAMERA", action: startCamera };
  if (capture && captureEnsemble.length > 1 && !character.created) primaryButton.label = `WAKE ${captureEnsemble.length} FIGURES`;
  const stepIndex = step === "ready" ? 0 : step === "camera" ? 1 : character.created ? 3 : 2;

  return (
    <main className="alive-shell">
      <header className="alive-header">
        <a className="alive-brand" href="#play"><span>WALL</span>ALIVE<i>●</i></a>
        <div className="mini-steps" aria-label="Three steps"><span className={stepIndex >= 1 ? "done" : "active"}>1 Scan</span><span className={stepIndex >= 2 ? "done" : ""}>2 Check</span><span className={stepIndex >= 3 ? "done" : ""}>3 Play</span></div>
        <div className="header-actions">
          <div className={`ready-pill ${webMcpReady ? "is-ready" : ""}`}><i /> {webMcpReady ? `${toolNames.length} WEBMCP TOOLS` : "INTERACTIVE DEMO"}</div>
          <button className="merch-toggle" onClick={() => openMerchStudio()} aria-label="Open Shopify merchandise mockup studio">SHOPIFY <span>MERCH</span></button>
          <button className="inspector-toggle" onClick={() => setInspectorOpen(true)}>WEBMCP</button>
          <button className="judge-demo" onClick={runMagicDemo} disabled={demoRunning}>{demoRunning ? "PLAYING…" : "PLAY JUDGE DEMO"}</button>
        </div>
      </header>
      <button className="judge-prompt-banner" onClick={copyDemoPrompt} aria-label="Copy Perfect Judge Demo Prompt">
        <span>JUDGE SHORTCUT</span>
        <b>COPY PERFECT JUDGE DEMO PROMPT</b>
        <small>{perfectJudgePrompt}</small>
        <i>⧉</i>
      </button>

      <section className="alive-layout" id="play">
        <aside className="steps-panel">
          <p className="kicker">THE MAGIC LOOP</p>
          <ol>
            <li className={stepIndex >= 1 ? "done" : "active"}><span>1</span><div><strong>Scan</strong><small>Human opens camera</small></div></li>
            <li className={stepIndex >= 3 ? "done" : stepIndex === 2 ? "active" : ""}><span>2</span><div><strong>Check</strong><small>Verify clean figures</small></div></li>
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
              <div><p className="kicker">{neuralAsset ? "FULL NEURAL RIG + DRAWING PARTS" : "VERIFIED CUTOUT · REVIEW"}</p><strong>{neuralAsset ? capture.rig.detectedKinds.filter((kind) => kind !== "body").join(" · ") || capture.analysis.shapeHint : capture.characterValidation?.evidence.join(" · ") || "CHARACTER EVIDENCE"}</strong><span><i style={{ background: capture.rig.bodyColor }} /><i style={{ background: capture.rig.lineColor }} /> {riggedAssetInfo ? `${riggedAssetInfo.bones} BONES · ${riggedAssetInfo.semanticParts} PROJECTED PARTS${riggedAssetInfo.colorTransfer ? " · COLOR MATCHED" : ""} · ${riggedAssetInfo.vertices.toLocaleString()} VERTICES` : neuralAsset ? "RIGGED GLB LOADING" : "2D CUTOUT · NO FAKE 3D"}</span></div>
            </div>
          ) : null}

          <div className="privacy-card">
            <div><b>CAMERA-SAFE BY DESIGN</b><span>◆</span></div>
            <p>Drawing-aware isolation and character checks stay on-device. Real 3D sends only the reviewed cutout after a second visible approval—never live frames.</p>
          </div>
          <input ref={uploadRef} hidden type="file" accept="image/*" onChange={uploadDrawing} />
          <button className="demo-doodle wall-drawing-link" onClick={() => setDrawingWallOpen(true)}>OPEN THE DRAWING WALL <span>✦</span></button>
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
              {cameraState !== "active" ? <button className="draw-wall-cta" onClick={() => setDrawingWallOpen(true)}>DRAW ON WALL <span>✦</span></button> : null}
              {capture ? <button className="merch-cta" onClick={() => openMerchStudio()}>SHOPIFY MOCKUPS <span>↗</span></button> : null}
              <button className="primary-camera" onClick={primaryButton.action} disabled={cameraState === "requesting" || neuralBusy}>{cameraState === "requesting" ? "OPENING…" : neuralBusy ? "GENERATING…" : primaryButton.label}<span>↗</span></button>
            </div>
          </div>
          <div className="notice" role="status" aria-live="polite"><i />{notice}</div>

          {capture ? <div
            className="anatomy-summary"
            data-cutout-model={capture.cutoutRecognition?.model ?? "authored-alpha"}
            data-cutout-confidence={capture.cutoutRecognition?.confidence ?? 1}
            data-cutout-area={capture.cutoutRecognition?.areaPercent ?? capture.analysis.coveragePercent}
            data-figure-count={captureEnsemble.length || 1}
            data-movable-parts={movablePartCount}
          >
            <div>
              <span className="summary-spark">✦</span>
              <p><b>{captureEnsemble.length > 1 ? `${captureEnsemble.length} figures found` : "Artwork preserved"}</b><small>{captureEnsemble.length > 1 ? "separate masks + rigs" : capture.cutoutRecognition?.model === "mediapipe-magic-touch-v2" ? "point-guided cutout" : "local cutout"}</small></p>
            </div>
            <div className="anatomy-pills"><span>Transparent</span><span>{captureEnsemble.length > 1 ? `${movablePartCount} moving limbs` : "Local only"}</span><span>Human check</span></div>
            <button onClick={() => setPartEditorOpen(true)}>REVIEW RIG</button>
          </div> : null}

          <div className={`camera-frame step-${step} world-${world}`} onPointerDown={handleStagePointerDown} onPointerMove={handleStagePointerMove} onPointerUp={handleStagePointerUp} onPointerCancel={() => { rotateGestureRef.current = null; }}>
            <video ref={videoRef} className={cameraState === "active" ? "camera-video visible" : "camera-video"} autoPlay muted playsInline aria-label="Live local camera preview" />
            {cameraState !== "active" && world === "studio" ? <div className="demo-room"><span className="frame-a" /><span className="frame-b" /><span className="shelf" /><span className="plant" /><span className="baseboard" /></div> : null}
            {capture && cameraState !== "active" && world === "studio" ? <img className="captured-room" src={capture.previewUrl} alt="Original drawing scene" /> : null}
            {capture && cameraState !== "active" && !character.created ? <div className="cutout-review" onPointerDown={(event) => event.stopPropagation()}><img src={capture.textureUrl} alt="Isolated character cutout to review" /><span>{captureEnsemble.length > 1 ? `${captureEnsemble.length} SEPARATE FIGURES FOUND` : "IS THE WHOLE CHARACTER VISIBLE?"}</span><div><button onClick={requestNeuralConsent}>YES · CONTINUE</button><button onClick={() => capture.sourceScope === "camera" ? startCamera() : uploadRef.current?.click()}>NO · TRY AGAIN</button></div></div> : null}
            {step === "camera" ? <><div className="capture-guide"><span /><b>TAP CHARACTER · THEN CAPTURE</b></div><div className="capture-target" style={{ left: `${captureTarget.x * 100}%`, top: `${captureTarget.y * 100}%` }}><i /></div></> : null}
            {character.created ? <Suspense fallback={<div className="three-layer" aria-hidden="true" />}>
              <ARStage ref={stageRef} characters={character.created && !neuralAsset ? captureEnsemble : null} contour={capture?.contour ?? null} skeleton={capture?.skeleton ?? null} textureUrl={capture?.textureUrl ?? null} rig={capture?.rig ?? null} depth={capture?.depthRecognition ?? null} action={character.action} ensembleActions={ensembleActions} world={world} lightingMood={lightingMood} cameraPreset={cameraPreset} accent={character.accent} inflation={character.inflation} neuralAssetUrl={neuralAsset?.meshUrl ?? null} visible onCapability={handleARCapability} onPlaced={handleARPlaced} onNeuralAssetInfo={handleRiggedAssetInfo} />
            </Suspense> : null}
            {neuralConsentVisible ? <div className="neural-consent" role="dialog" aria-modal="true" aria-labelledby="neural-consent-title" onPointerDown={(event) => event.stopPropagation()}>
              <span>CHOOSE YOUR 3D</span><h2 id="neural-consent-title">Wake {captureEnsemble.length > 1 ? "the whole cast" : "this drawing"}</h2><p>{captureEnsemble.length > 1 ? "Separate closed 3D puppets, separate rigs, one shared world." : "Instant private puppet now—or send only the cutout for a full AI sculpt."}</p><div><button onClick={startLocalReconstruction}>INSTANT 3D · PRIVATE</button>{captureEnsemble.length === 1 ? <button onClick={startNeuralReconstruction}>GENERATE REAL 3D · AI</button> : null}<button onClick={() => { setNeuralConsentVisible(false); setPartEditorOpen(true); }}>CHECK PARTS</button></div>
            </div> : null}
            {neuralBusy ? <div className="neural-progress" role="status" onPointerDown={(event) => event.stopPropagation()}><span>ANIGEN · RIGGED 3D</span><b>{neuralProgress.message}</b><div><i style={{ width: `${Math.round(neuralProgress.progress * 100)}%` }} /></div><small>{Math.round(neuralProgress.progress * 100)}% · PUBLIC GPU</small></div> : null}
            {magicShowPlan?.status === "awaiting-human-approval" ? <div className="magic-show-approval" role="dialog" aria-modal="false" aria-labelledby="magic-show-title" onPointerDown={(event) => event.stopPropagation()}>
              <div><span>AGENT STAGED · YOU DECIDE</span><b>{magicShowPlan.tone} · {magicShowPlan.beats.length} beats</b></div>
              <h2 id="magic-show-title">{magicShowPlan.title}</h2>
              <p>{magicShowPlan.cast.map((member) => `${member.name} · ${member.role}`).join("  /  ")}</p>
              <div className="show-beat-preview">{magicShowPlan.beats.map((beat, index) => <i key={`${magicShowPlan.id}-${index}`}>{index + 1}<small>{beat.moves.map((move) => move.action).join(" + ")}</small></i>)}</div>
              <div className="show-approval-actions"><button onClick={approveAndPlayMagicShow}>APPROVE &amp; PLAY <span>▶</span></button><button onClick={dismissMagicShow}>NOT YET</button></div>
            </div> : null}
            {showPlaying ? <div className="magic-show-live" role="status"><i /> AGENT PLAN · HUMAN APPROVED · LIVE</div> : null}
            {character.created ? <div className="spatial-controls" onPointerDown={(event) => event.stopPropagation()}>
              <div className="movement-pad" aria-label="Move character through the 3D world">
                <button onClick={() => nudgeCharacter(0, -0.45)} aria-label="Walk forward">↑</button>
                <button onClick={() => nudgeCharacter(-0.45, 0)} aria-label="Move left">←</button>
                <button onClick={() => nudgeCharacter(0, 0.45)} aria-label="Walk backward">↓</button>
                <button onClick={() => nudgeCharacter(0.45, 0)} aria-label="Move right">→</button>
              </div>
              <span>DRAG ORBIT · PINCH ZOOM · PAN</span>
            </div> : null}
            <div className="camera-hud"><span><i /> {cameraState === "active" ? "LIVE CAMERA · LOCAL" : neuralAsset && character.created ? `FULL NEURAL RIG · ${riggedAssetInfo?.bones ?? "…"} BONES` : character.created ? `${captureEnsemble.length} LOCAL 3D PUPPET${captureEnsemble.length === 1 ? "" : "S"}` : capture ? "CUTOUT REVIEW · LOCAL" : "SAFE DEMO ROOM"}</span><strong>{immersiveAR ? "WEBXR READY" : `${world.toUpperCase()} · REAL 3D SCENE`}</strong></div>
            {character.created && storyCaption ? <div className="story-caption"><span>{character.storyTitle || "LIVE MOMENT"}</span><p>{storyCaption}</p></div> : null}
            {cameraState === "denied" || cameraState === "unavailable" ? <div className="camera-message"><b>CAMERA OPTIONAL</b><p>The demo doodle still proves the complete WebMCP and 3D workflow.</p></div> : null}
          </div>

          <div className="world-switcher" aria-label="Choose a 3D world"><div><span>3D WORLDS</span><small>{worlds.find((item) => item.id === world)?.label}</small></div>{worlds.map((item) => <button key={item.id} className={world === item.id ? "active" : ""} onClick={() => changeWorld(item.id, "CHILD")}><i>{item.id === "studio" ? "⌂" : item.id === "storybook" ? "♜" : item.id === "wizard" ? "✦" : "◉"}</i>{item.short}</button>)}</div>

          <div className="cinematic-switcher">
            <div><span>LIGHT</span>{lightingMoods.map((mood) => <button key={mood} className={lightingMood === mood ? "active" : ""} onClick={() => { lightingMoodRef.current = mood; setLightingMood(mood); record("CHILD", "Changed cinematic lighting", mood); }}>{mood.replace("-", " ")}</button>)}</div>
            <div><span>CAMERA</span>{cameraPresets.map((preset) => <button key={preset} className={cameraPreset === preset ? "active" : ""} onClick={() => { cameraPresetRef.current = preset; setCameraPreset(preset); record("CHILD", "Changed camera preset", preset); }}>{preset.replaceAll("-", " ")}</button>)}</div>
          </div>

          <div className="action-tray">
            <div><span>CHARACTER ACTIONS</span><small>{character.created ? `${character.name.toUpperCase()} · ${character.personality.toUpperCase()}` : "WAKE A DRAWING TO PLAY"}</small></div>
            {actions.map((item) => <button key={item.action} disabled={!character.created || showPlaying} className={character.action === item.action ? "active" : ""} onClick={() => animateCharacter(item.action, "CHILD")}><i>{item.glyph}</i>{item.label}</button>)}
          </div>
          <p className="placement-tip">{character.created ? neuralAsset ? "Drag for 360° · Generated back · Move through the perspective world" : "Drag for 360° · Filled backs · Every figure moves on its own rig" : capture ? "Check the cutout, then choose instant private 3D or full AI sculpt" : "Photograph a clear figure—uncertain recognition is blocked before 3D"}</p>
        </section>

        <aside className={`agent-panel ${inspectorOpen ? "is-open" : ""}`} aria-hidden={!inspectorOpen}>
          <button className="inspector-close" onClick={() => setInspectorOpen(false)} aria-label="Close WebMCP inspector">×</button>
          <div className="right-tabs" role="tablist" aria-label="WallAlive inspector">
            {(["agent", "tools", "commerce", "privacy", "history"] as const).map((tab) => <button key={tab} role="tab" aria-selected={panelTab === tab} className={panelTab === tab ? "active" : ""} onClick={() => setPanelTab(tab)}>{tab}</button>)}
          </div>

          {panelTab === "agent" ? (
            <div className="panel-body">
              <div className="agent-status"><div><i /> BROWSER AGENT</div><span>{webMcpReady ? "CONNECTED" : "DEMO MODE"}</span></div>
              <p className="kicker">SHARED IMAGINATION</p>
              <h2>{agentLine}</h2>
              <p>The agent reads each real rig, assigns compatible roles, and stages ensemble choreography. The child approves the final performance.</p>
              <div className="agent-call"><span>↳</span><div><b>{latestAgentActivity?.toolName ?? "inspect_creative_scene"}</b><small>{latestAgentActivity?.detail ?? "Shared state visible · Camera private"}</small></div></div>
              <blockquote>“{perfectJudgePrompt}”</blockquote>
              <button className="copy-prompt" onClick={copyDemoPrompt}>COPY PERFECT JUDGE PROMPT <span>⧉</span></button>
            </div>
          ) : null}

          {panelTab === "tools" ? (
            <div className="panel-body">
              <p className="kicker">WEBMCP INSPECTOR</p><h2>One shared creative loop.</h2><p>Inspect state → check abilities → stage a show → human approves → perform and verify. Camera authority never crosses into the tool surface.</p>
              <div className="tools-list">{toolNames.map(([name, mode], index) => <div key={name}><span>{String(index + 1).padStart(2, "0")}</span><code>{name}</code><i>{mode}</i></div>)}</div>
            </div>
          ) : null}

          {panelTab === "commerce" ? (
            <div className="panel-body commerce-panel">
              <p className="kicker">AGENT COMMERCE PIPELINE CONNECTED</p>
              <h2>{merchPipeline ? merchPipeline.title : "Living art, ready for a shelf."}</h2>
              <p>{merchPipeline ? "A transparent print layout made from the approved artwork—ready for a safe storefront preview." : capture ? "Choose a product to build the visible mockup." : "Add and approve a drawing, then return here for t-shirt and mug mockups."}</p>
              <div className="merch-products" aria-label="Choose merchandise mockup">
                <button className={merchPipeline?.product !== "ceramic-mug" ? "active" : ""} disabled={!capture} onClick={() => openMerchStudio("t-shirt")}><span>◫</span>T-SHIRT</button>
                <button className={merchPipeline?.product === "ceramic-mug" ? "active" : ""} disabled={!capture} onClick={() => openMerchStudio("ceramic-mug")}><span>◒</span>MUG</button>
              </div>
              <div className={`merch-mockup ${merchPipeline?.product === "ceramic-mug" ? "is-mug" : "is-shirt"}`}>
                <div>{capture ? <img src={capture.textureUrl} alt="Approved drawing printed on merchandise" /> : <span>YOUR ART</span>}</div>
                <small>{merchPipeline?.product === "ceramic-mug" ? "CERAMIC MUG" : "PREMIUM WHITE T-SHIRT"}</small>
              </div>
              <div className="commerce-steps"><span>1 ARTWORK</span><i>→</i><span>2 PRINT LAYOUT</span><i>→</i><span>3 SHOPIFY</span></div>
              <button className="shopify-checkout" disabled={!merchPipeline} onClick={() => { setMockCheckoutOpen(true); setNotice("Safe mock checkout opened. It contains no payment fields and cannot place an order."); record("CHILD", "Opened safe mock Shopify checkout", "Demo only · no payment fields · no purchase was made."); }}>OPEN SAFE MOCK CHECKOUT <span>↗</span></button>
              <small className="commerce-disclaimer">DEMO CHECKOUT · NO CHARGE · HUMAN CONFIRMATION REQUIRED</small>
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
          <footer className="agent-footer"><span>CHROME WEBMCP · CHATGPT SITES · SHOPIFY</span><b>THE CHILD DECIDES</b></footer>
        </aside>
      </section>
      {mockCheckoutOpen && merchPipeline ? <div className="mock-checkout-backdrop" role="dialog" aria-modal="true" aria-labelledby="mock-checkout-title">
        <section className="mock-checkout">
          <header><div><span>SHOPIFY · SAFE PREVIEW</span><h2 id="mock-checkout-title">No-charge checkout</h2></div><button onClick={() => setMockCheckoutOpen(false)} aria-label="Close mock checkout">×</button></header>
          <div className="mock-order-line"><div>{capture ? <img src={capture.textureUrl} alt="Approved artwork order preview" /> : null}</div><p><b>{merchPipeline.title}</b><span>{merchPipeline.product === "ceramic-mug" ? "Ceramic mug" : "Premium t-shirt"} · Demo sample</span></p><strong>$0.00</strong></div>
          <div className="mock-safety"><i>✓</i><p><b>Simulation only</b><span>No address, card, order, or Shopify account is requested or created.</span></p></div>
          <dl><div><dt>PRINT</dt><dd>Approved transparent artwork · 82% safe area</dd></div><div><dt>PAYMENT</dt><dd>Disabled in this hackathon demo</dd></div><div><dt>TOTAL</dt><dd>$0.00 · no charge</dd></div></dl>
          <button className="mock-finish" onClick={() => { setMockCheckoutOpen(false); setNotice("Mock checkout closed. Nothing was purchased."); }}>FINISH DEMO · PLACE NO ORDER</button>
        </section>
      </div> : null}
      <DrawingWall open={drawingWallOpen} onClose={() => setDrawingWallOpen(false)} onMake3D={processWallDrawing} />
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
            <footer><button disabled={!selectedPartId} onClick={() => resizeSelectedPart(0.86)}>− SIZE</button><button disabled={!selectedPartId} onClick={() => resizeSelectedPart(1.16)}>＋ SIZE</button><button className="remove-part" disabled={!selectedPartId} onClick={deleteSelectedPart}>REMOVE</button><button className="editor-done" onClick={approveRigReview}>APPROVE RIG</button></footer>
          </div>
        </section>
      </div> : null}
    </main>
  );
}
