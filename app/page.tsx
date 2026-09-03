/* eslint-disable @next/next/no-img-element */
"use client";

import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import type { CSSProperties } from "react";
import type { ARStageHandle, ARWorld, CameraPreset, CharacterAction, LightingMood, ModelPaintBrush, ModelPaintInspection, ModelPaintTool, WorldObjectInteraction } from "./components/ARStage";
import { DrawingWall } from "./components/DrawingWall";
import { SharedRoomPanel } from "./components/SharedRoomPanel";
import { appendCaptureTarget, createAniGenDemoDrawing, createDemoDoodle, POSE_SKELETON_EDGES, selectAnimatableRigParts, type CaptureTarget, type DrawingExtraction, type SemanticPart, type SemanticPartKind, type SemanticSide } from "./lib/drawing";
import { inspectCharacterCapabilities as buildCharacterCapabilities, SAFE_SHOW_ACTIONS, validateCharacterMove, type CharacterCapability } from "./lib/creative-show";
import { recognizeDrawingParts, recognizeDrawingsAtImageTargets, recognizeDrawingsAtVideoTargets } from "./lib/learned-parts";
import { createBundledAniGenAsset, disposeNeuralAsset, generateAniGenAsset, isAniGenUnavailableError, type NeuralAsset, type NeuralProgress } from "./lib/anigen";
import { assessReconstructionReadiness } from "./lib/character-quality";
import { getAccessibleWorldInteraction } from "./lib/world-interactions";
import { buildLearningProgress, type LearningReflection, type ReflectionRevision } from "./lib/learning-progress";
import type { RiggedAssetInfo } from "./lib/rigged-model";
import {
  CREATOR_PRODUCT_IDS,
  buildCreatorHandoff,
  buildShopifyProductsCsv,
  buildShopifyStoreBlueprint,
  recommendCreatorProducts,
  stageCreatorDrop as buildCreatorDrop,
  type ArtworkCommerceProfile,
  type CreatorAudience,
  type CreatorDrop,
  type CreatorGoal,
  type CreatorProductId,
  type CreatorRecommendation,
  type CreatorVibe,
} from "./lib/creator-commerce";
import { useSharedRoom } from "./lib/use-shared-room";
import { registerAndVerifyWebMCP, type RegisterableWebMCPTool, type WebMCPModelContext, type WebMCPRuntimeCheck } from "./lib/webmcp-runtime";

const ARStage = lazy(() => import("./components/ARStage").then((module) => ({ default: module.ARStage })));

type Actor = "CHILD" | "BROWSER AGENT" | "WALLALIVE";
type AppStep = "ready" | "camera" | "captured" | "alive";
type CameraState = "idle" | "requesting" | "active" | "denied" | "unavailable";
type PanelTab = "agent" | "learning" | "tools" | "commerce" | "privacy" | "history";
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

type PendingUpload = { url: string; fileName: string; origin: "photo" | "wall" };

type ShowMove = { characterIndex: number; action: CharacterAction };
type ShowBeat = { caption: string; durationMs: number; world?: WorldId; moves: ShowMove[] };
type ShowCastMember = { characterIndex: number; name: string; role: string; personality: string };
type MagicShowPlan = {
  id: string;
  title: string;
  theme: string;
  learningGoal: string;
  tone: "gentle" | "silly" | "adventurous" | "dreamy";
  world: WorldId;
  cast: ShowCastMember[];
  beats: ShowBeat[];
  stagedBy: Actor;
  status: "awaiting-human-approval" | "playing" | "complete" | "dismissed";
};
type WorldActivity = { title: string; prompt: string; objectIds: string[]; reward: string };
type PaintAdventure = {
  id: string;
  title: string;
  prompt: string;
  tool: ModelPaintTool;
  palette: string[];
  steps: string[];
  status: "awaiting-child" | "painting" | "dismissed";
};

type WebMCPTool = RegisterableWebMCPTool;
type WebMCPStatus = "unsupported" | "registering" | "registered" | "verified" | "error";

declare global {
  interface Document {
    modelContext?: WebMCPModelContext;
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
  ["inspect_learning_progress", "READ"],
  ["stage_next_learning_challenge", "STAGE"],
  ["inspect_character_capabilities", "READ"],
  ["inspect_reconstruction_readiness", "READ"],
  ["request_character_repair", "REQUEST"],
  ["request_rigged_3d_cast", "REQUEST"],
  ["stage_magic_show", "STAGE"],
  ["direct_live_ensemble", "LIVE"],
  ["orchestrate_spatial_cinematics", "LIVE"],
  ["recommend_creator_products", "ADVISE"],
  ["stage_shopify_import_kit", "STAGE"],
  ["inspect_shopify_import_kit", "READ"],
  ["inspect_shared_room", "READ"],
  ["prepare_room_invite", "STAGE"],
  ["interact_story_world", "LIVE"],
  ["inspect_3d_paint_studio", "READ"],
  ["stage_3d_paint_adventure", "STAGE"],
  ["list_collaboration_history", "READ"],
] as const;

const suggestedJudgePrompt = "Help my child and me turn this drawing into a tiny painted story. Inspect the creative scene, 3D paint studio, and shared room. Propose a joyful palette and stage a short paint adventure, but let my child paint the model by touch. Then stage a three-beat show using only verified actions and wait for us to approve playback. Never access camera frames or image pixels, grade the child, publish, or buy anything.";

const worlds: Array<{ id: WorldId; label: string; short: string }> = [
  { id: "studio", label: "My room", short: "ROOM" },
  { id: "storybook", label: "Storybook kingdom", short: "KINGDOM" },
  { id: "wizard", label: "Wizard academy", short: "WIZARD" },
  { id: "museum", label: "Grand museum", short: "MUSEUM" },
];

const worldActivities: Record<WorldId, WorldActivity> = {
  studio: { title: "Make a tiny movie", prompt: "Touch the projector and maker table. Give every character one job in the scene.", objectIds: ["studio-projector", "studio-maker-table"], reward: "Your cast made its first mini movie together." },
  storybook: { title: "Firefly hide & seek", prompt: "Find three glowing fireflies, then unlock the castle gate as a team.", objectIds: ["storybook-firefly-0", "storybook-firefly-1", "storybook-firefly-2", "storybook-gate"], reward: "The lantern trail is complete—the castle opens." },
  wizard: { title: "The cooperation spell", prompt: "Read the spell book, gather three crystals, then enter the living portal.", objectIds: ["wizard-spell-book", "wizard-crystal-a", "wizard-crystal-b", "wizard-crystal-c", "wizard-portal"], reward: "The spell worked because every character had a role." },
  museum: { title: "Curate a living gallery", prompt: "Touch each artwork and the motion sculpture. Tell what you notice together.", objectIds: ["museum-art-0", "museum-art-1", "museum-art-2", "museum-sculpture"], reward: "Your cast curated a new exhibition story." },
};

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
const modelPaintTools: Array<{ id: ModelPaintTool; label: string; glyph: string }> = [
  { id: "brush", label: "Brush", glyph: "●" },
  { id: "spray", label: "Spray", glyph: "⁙" },
  { id: "oil", label: "Oil", glyph: "≈" },
  { id: "spill", label: "Splash", glyph: "✦" },
];
const modelPaintPalette = ["#ff5d73", "#ffb21c", "#ffe55c", "#43c59e", "#42a5f5", "#7557d9", "#f8f5ed", "#263645"];
const emptyPaintInspection: ModelPaintInspection = { strokeCount: 0, paintedSurfaceCount: 0, colors: [], tools: [] };

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
  const activeFigureIndexRef = useRef(0);
  const worldRef = useRef<WorldId>("studio");
  const magicShowPlanRef = useRef<MagicShowPlan | null>(null);
  const completedShowBeatsRef = useRef(0);
  const learningReflectionRef = useRef<LearningReflection | null>(null);
  const showAbortRef = useRef<AbortController | null>(null);
  const showPlayingRef = useRef(false);
  const lightingMoodRef = useRef<LightingMood>("sunset-warm");
  const cameraPresetRef = useRef<CameraPreset>("cinematic-orbit");
  const worldInteractionActorRef = useRef<{ actor: Actor; toolName?: string }>({ actor: "CHILD" });
  const paintGestureRef = useRef<{ pointerId: number; painted: boolean } | null>(null);
  const paintAdventureRef = useRef<PaintAdventure | null>(null);
  const paintBrushRef = useRef<ModelPaintBrush>({ tool: "brush", color: "#42a5f5", size: 0.42 });
  const paintInspectionRef = useRef<ModelPaintInspection>(emptyPaintInspection);

  const [step, setStep] = useState<AppStep>("ready");
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [capture, setCapture] = useState<DrawingExtraction | null>(null);
  const [captureEnsemble, setCaptureEnsemble] = useState<DrawingExtraction[]>([]);
  const [character, setCharacter] = useState<CharacterState>(initialCharacter);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [panelTab, setPanelTab] = useState<PanelTab>("agent");
  const [webMcpStatus, setWebMcpStatus] = useState<WebMCPStatus>("registering");
  const [webMcpCheck, setWebMcpCheck] = useState<WebMCPRuntimeCheck | null>(null);
  const [immersiveAR, setImmersiveAR] = useState(false);
  const [rendererAvailable, setRendererAvailable] = useState<boolean | null>(null);
  const [notice, setNotice] = useState("Camera stays on this device.");
  const [agentLine, setAgentLine] = useState("Create together.");
  const [storyCaption, setStoryCaption] = useState("Ready for a new friend.");
  const [demoRunning, setDemoRunning] = useState(false);
  const [cameraTargets, setCameraTargets] = useState<CaptureTarget[]>([]);
  const [neuralAsset, setNeuralAsset] = useState<NeuralAsset | null>(null);
  const [neuralProgress, setNeuralProgress] = useState<NeuralProgress>({ phase: "idle", progress: 0, message: "" });
  const [neuralConsentVisible, setNeuralConsentVisible] = useState(false);
  const [riggedAssetInfo, setRiggedAssetInfo] = useState<RiggedAssetInfo | null>(null);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [partEditorOpen, setPartEditorOpen] = useState(false);
  const [selectedPartId, setSelectedPartId] = useState<string | null>(null);
  const [pendingPartKind, setPendingPartKind] = useState<(typeof anatomyKinds)[number] | null>(null);
  const [pendingUpload, setPendingUpload] = useState<PendingUpload | null>(null);
  const [pendingUploadTargets, setPendingUploadTargets] = useState<CaptureTarget[]>([]);
  const [activeFigureIndex, setActiveFigureIndex] = useState(0);
  const [drawingWallOpen, setDrawingWallOpen] = useState(false);
  const [sharedRoomOpen, setSharedRoomOpen] = useState(false);
  const [dismissedInvite, setDismissedInvite] = useState("");
  const [world, setWorld] = useState<WorldId>("studio");
  const [magicShowPlan, setMagicShowPlan] = useState<MagicShowPlan | null>(null);
  const [completedShowBeats, setCompletedShowBeats] = useState(0);
  const [learningReflection, setLearningReflection] = useState<LearningReflection | null>(null);
  const [reflectionRetell, setReflectionRetell] = useState("");
  const [reflectionNextChange, setReflectionNextChange] = useState<ReflectionRevision>("new-ending");
  const [ensembleActions, setEnsembleActions] = useState<CharacterAction[] | null>(null);
  const [showPlaying, setShowPlaying] = useState(false);
  const [lightingMood, setLightingMood] = useState<LightingMood>("sunset-warm");
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>("cinematic-orbit");
  const [creatorAudience, setCreatorAudience] = useState<CreatorAudience>("family");
  const [creatorGoal, setCreatorGoal] = useState<CreatorGoal>("keepsake");
  const [creatorVibe, setCreatorVibe] = useState<CreatorVibe>("sunny");
  const [creatorRecommendations, setCreatorRecommendations] = useState<CreatorRecommendation[]>([]);
  const [creatorDrop, setCreatorDrop] = useState<CreatorDrop | null>(null);
  const [adultExportApproved, setAdultExportApproved] = useState(false);
  const [worldInteractions, setWorldInteractions] = useState<Record<WorldId, string[]>>({ studio: [], storybook: [], wizard: [], museum: [] });
  const [lastWorldMoment, setLastWorldMoment] = useState<WorldObjectInteraction | null>(null);
  const [paintStudioOpen, setPaintStudioOpen] = useState(false);
  const [paintBrush, setPaintBrush] = useState<ModelPaintBrush>({ tool: "brush", color: "#42a5f5", size: 0.42 });
  const [paintInspection, setPaintInspection] = useState<ModelPaintInspection>(emptyPaintInspection);
  const [paintAdventure, setPaintAdventure] = useState<PaintAdventure | null>(null);
  const sharedRoom = useSharedRoom();
  const prepareRoomInvite = sharedRoom.prepareInvite;
  const search = useSyncExternalStore(
    () => () => undefined,
    () => window.location.search,
    () => "",
  );
  const inviteParams = useMemo(() => new URLSearchParams(search), [search]);
  const invitedRoom = inviteParams.get("room")?.trim().toUpperCase().slice(0, 16) ?? "";
  const invitedUsername = inviteParams.get("invite")?.trim().slice(0, 24) ?? "";
  const sharedRoomVisible = sharedRoomOpen || Boolean(invitedRoom && dismissedInvite !== search);
  const sharedRoomStateRef = useRef({ session: sharedRoom.session, participants: sharedRoom.participants, operations: sharedRoom.operations, status: sharedRoom.status });

  useEffect(() => {
    sharedRoomStateRef.current = { session: sharedRoom.session, participants: sharedRoom.participants, operations: sharedRoom.operations, status: sharedRoom.status };
  }, [sharedRoom.operations, sharedRoom.participants, sharedRoom.session, sharedRoom.status]);

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

  const commitPaintAdventure = useCallback((next: PaintAdventure | null) => {
    paintAdventureRef.current = next;
    setPaintAdventure(next);
  }, []);

  const commitPaintInspection = useCallback((next: ModelPaintInspection) => {
    paintInspectionRef.current = next;
    setPaintInspection(next);
  }, []);

  const choosePaintBrush = useCallback((next: ModelPaintBrush) => {
    paintBrushRef.current = next;
    setPaintBrush(next);
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
    if (pendingUploadRef.current?.url.startsWith("blob:")) URL.revokeObjectURL(pendingUploadRef.current.url);
  }, []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState("unavailable");
      setNotice("This browser cannot open a camera. The demo doodle still shows the complete experience.");
      return;
    }
    setCameraState("requesting");
    setCameraTargets([]);
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
      setNotice("Camera is live locally. Tap each character body, then capture only those selections.");
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
    completedShowBeatsRef.current = 0;
    learningReflectionRef.current = null;
    setShowPlaying(false);
    setMagicShowPlan(null);
    setCompletedShowBeats(0);
    setLearningReflection(null);
    setReflectionRetell("");
    setReflectionNextChange("new-ending");
    setEnsembleActions(null);
    setPaintStudioOpen(false);
    paintGestureRef.current = null;
    commitPaintAdventure(null);
    commitPaintInspection(emptyPaintInspection);
    commitNeuralAsset(null);
    handleRiggedAssetInfo(null);
    externalUploadApprovedRef.current = false;
    setNeuralConsentVisible(false);
    setRendererAvailable(null);
    setNeuralProgress({ phase: "idle", progress: 0, message: "" });
    const isJudgeDemo = source === "demo";
    if (source === "camera") {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setCameraState("idle");
    }
    commitCharacter({ ...initialCharacter, created: isJudgeDemo, name: isJudgeDemo ? "Sunny" : "", accent: next.analysis.secondaryColor });
    captureRef.current = next;
    captureEnsembleRef.current = ensemble;
    activeFigureIndexRef.current = 0;
    setCapture(next);
    setCaptureEnsemble(ensemble);
    setActiveFigureIndex(0);
    setSelectedPartId(null);
    setPendingPartKind(null);
    setStep(isJudgeDemo ? "alive" : "captured");
    const detected = next.rig.detectedKinds.filter((kind) => kind !== "body").join(", ");
    const learned = next.learnedRecognition;
    setNotice(isJudgeDemo
      ? "The deterministic rigged judge demo is ready."
      : ensemble.length > 1
        ? `${ensemble.length} selected figures isolated. Review each rig; movement unlocks only after verified branches pass.`
        : "Character cutout found. Check that the whole character—and only the character—is visible before generating real 3D.");
    setAgentLine(isJudgeDemo
      ? "The judge asset is a real skinned 3D mesh with generated back geometry."
      : ensemble.length > 1
        ? `I isolated ${ensemble.length} exact selections independently. Movement stays locked for any figure that still needs rig review.`
        : `I verified ${next.characterValidation?.evidence.join(", ") || detected || "character structure"}${learned ? ` in ${learned.latencyMs} ms` : ""}. Review the isolated pixels before any image leaves this device.`);
    record("WALLALIVE", isJudgeDemo ? "Loaded the deterministic rigged demo" : "Prepared a verified character cutout", isJudgeDemo
      ? "Bundled colored GLB · generated full geometry · skeleton · skin weights."
      : ensemble.length > 1
        ? `${ensemble.length} independent instance masks · per-figure pose/topology gates · no upload.`
        : `Drawing-aware point extraction · character-evidence gate ${next.characterValidation?.score ?? "passed"} · human cutout review · no 3D claim · no upload.`);
  }, [commitCharacter, commitNeuralAsset, commitPaintAdventure, commitPaintInspection, handleRiggedAssetInfo, record]);

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
    if (!cameraTargets.length) {
      setNotice("Tap at least one character before capturing.");
      return;
    }
    setNotice(`Checking ${cameraTargets.length} exact camera selection${cameraTargets.length === 1 ? "" : "s"} locally…`);
    try {
      const drawings = await recognizeDrawingsAtVideoTargets(videoRef.current, cameraTargets);
      setDrawing(drawings[0], "camera", drawings);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The drawing could not be separated from the wall.");
    }
  }, [cameraTargets, setDrawing]);

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
    if (pendingUploadRef.current?.url.startsWith("blob:")) URL.revokeObjectURL(pendingUploadRef.current.url);
    const next: PendingUpload = { url: objectUrl, fileName: file.name, origin: "photo" };
    pendingUploadRef.current = next;
    setPendingUpload(next);
    setPendingUploadTargets([]);
    setNotice("Tap every character you want, then scan only those selections.");
    input.value = "";
  }, []);

  const processUploadedDrawing = useCallback(async () => {
    const pending = pendingUploadRef.current;
    if (!pending) return;
    const targets = pendingUploadTargets;
    if (!targets.length) {
      setNotice("Tap at least one character before scanning.");
      return;
    }
    setPendingUpload(null);
    setNotice(`Checking ${targets.length} exact selection${targets.length === 1 ? "" : "s"} locally…`);
    try {
      const drawings = await recognizeDrawingsAtImageTargets(pending.url, targets);
      setDrawing(drawings[0], "upload", drawings);
      const rejectedCount = targets.filter((target) => !drawings.some((drawing) => {
        const accepted = drawing.sourceTarget;
        return accepted && Math.hypot(accepted.x - target.x, accepted.y - target.y) < 0.08;
      })).length;
      record("CHILD", pending.origin === "wall" ? "Selected the cast from the Wall Studio" : "Selected the cast in a drawing photo", `${pending.fileName} · ${targets.length} explicit target${targets.length === 1 ? "" : "s"} · ${drawings.length} accepted cutout${drawings.length === 1 ? "" : "s"} · ${rejectedCount} rejected safely · no automatic background cast · original pixels stayed in this tab.`);
      setNotice(rejectedCount
        ? `${drawings.length}/${targets.length} selections passed. ${rejectedCount} uncertain selection${rejectedCount === 1 ? " was" : "s were"} left out instead of becoming a broken 3D figure.`
        : `${drawings.length} selected figure${drawings.length === 1 ? "" : "s"} passed the cutout gate. Review each rig before movement.`);
    } catch (error) {
      console.warn("WallAlive local upload recognition was safely rejected", error);
      setNotice(error instanceof Error ? error.message : "The drawing image could not be processed.");
    } finally {
      if (pending.url.startsWith("blob:")) URL.revokeObjectURL(pending.url);
      pendingUploadRef.current = null;
      setPendingUploadTargets([]);
    }
  }, [pendingUploadTargets, record, setDrawing]);

  const processWallDrawing = useCallback(async ({ dataUrl }: { dataUrl: string; target: CaptureTarget }) => {
    setDrawingWallOpen(false);
    const next: PendingUpload = { url: dataUrl, fileName: "Wall Studio artwork", origin: "wall" };
    pendingUploadRef.current = next;
    setPendingUpload(next);
    setPendingUploadTargets([]);
    setNotice("Tap every Wall Studio character you want in the cast. Only those exact selections will be scanned.");
  }, []);

  const cancelPendingUpload = useCallback(() => {
    if (pendingUploadRef.current?.url.startsWith("blob:")) URL.revokeObjectURL(pendingUploadRef.current.url);
    pendingUploadRef.current = null;
    setPendingUpload(null);
    setPendingUploadTargets([]);
    setNotice("Photo closed. Choose another image or start the camera.");
  }, []);

  const createCharacter = useCallback((input: Record<string, unknown>, actor: Actor, toolName?: string) => {
    const drawing = captureRef.current;
    if (!drawing) throw new Error("No drawing is approved. The child must capture or choose a drawing first.");
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : [drawing];
    const neural = neuralAssetRef.current;
    const readiness = ensemble.map(assessReconstructionReadiness);
    const motionReadyCount = readiness.filter((report) => report.motionReady).length;
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
      : motionReadyCount === ensemble.length
        ? `${next.name} is now a reviewed, articulated spatial puppet.`
        : `${next.name} is now a static spatial puppet. ${ensemble.length - motionReadyCount} figure${ensemble.length - motionReadyCount === 1 ? " needs" : "s need"} a rig check before limb movement.`);
    setStep("alive");
    const graphNodes = ensemble.reduce((sum, figure) => sum + (figure.topologyRecognition?.nodes.length ?? figure.rig.joints.length), 0);
    const graphEdges = ensemble.reduce((sum, figure) => {
      const nodes = figure.topologyRecognition?.nodes.length ?? figure.rig.joints.length;
      return sum + (figure.topologyRecognition?.edges.length ?? Math.max(0, nodes - 1));
    }, 0);
    setAgentLine(neural
      ? `${next.name} has generated surfaces, colors, bones, and skin weights. The agent can now direct the rig.`
      : `${ensemble.length} artwork-preserving spatial puppet${ensemble.length === 1 ? "" : "s"} ready · ${motionReadyCount}/${ensemble.length} verified for limb movement.`);
    setStoryCaption(`${next.name} lifts away from the wall for the first time.`);
    record(actor, neural ? "Loaded a rigged neural 3D character" : "Built an articulated local 3D cast", neural
      ? `${next.name} · ${neural.provider} · glTF SkinnedMesh · generated mesh, skeleton, and skin weights · ${graphNodes} semantic nodes · ${graphEdges} branches.`
      : `${ensemble.length} closed-relief spatial preview${ensemble.length === 1 ? "" : "s"} · exact artwork fronts · ${motionReadyCount}/${ensemble.length} motion-ready · ${graphNodes} rig nodes · ${graphEdges} candidate branches · no upload.`, toolName);
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
    const motionBlocked = readiness.filter((report) => !report.motionReady).length;
    setNeuralConsentVisible(true);
    setNeuralProgress({ phase: "consent-required", progress: 0, message: ensemble.length > 1
      ? `Choose instant local 3D for the complete cast. ${motionBlocked ? `${motionBlocked} figure${motionBlocked === 1 ? " needs" : "s need"} review before limb movement.` : "Every figure has verified movement."}`
      : motionBlocked ? "The cutout can become a static spatial puppet. Review the skeleton to unlock verified limb movement, or approve a full external AI sculpt." : "Choose instant private 3D or approve a full external AI sculpt." });
    setNotice(ensemble.length > 1
      ? `All ${ensemble.length} figures passed the cutout gate · ${ensemble.length - motionBlocked}/${ensemble.length} passed the motion gate.`
      : motionBlocked ? "The cutout is clean, but the skeleton still needs your check before limb movement." : "Choose a private instant puppet or approve full AI reconstruction. The live camera is never uploaded.");
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

  const handleWorldInteraction = useCallback((interaction: WorldObjectInteraction, actor: Actor = "CHILD", toolName?: string) => {
    setWorldInteractions((current) => ({
      ...current,
      [interaction.world]: current[interaction.world].includes(interaction.id)
        ? current[interaction.world]
        : [...current[interaction.world], interaction.id],
    }));
    setLastWorldMoment(interaction);
    setStoryCaption(interaction.story);
    setNotice(`${interaction.label}: ${interaction.story}`);
    record(actor, `${interaction.verb} ${interaction.label}`, interaction.story, toolName);
    return { objectId: interaction.id, label: interaction.label, verb: interaction.verb, story: interaction.story, visibleWorldUpdated: true };
  }, [record]);

  const handleStageWorldInteraction = useCallback((interaction: WorldObjectInteraction) => {
    const source = worldInteractionActorRef.current;
    handleWorldInteraction(interaction, source.actor, source.toolName);
  }, [handleWorldInteraction]);

  const interactWithWorldObject = useCallback((objectId: string, actor: Actor = "BROWSER AGENT") => {
    const currentWorld = worldRef.current;
    const activity = worldActivities[currentWorld];
    if (!activity.objectIds.includes(objectId)) throw new Error(`That object is not part of ${worldRef.current}. Inspect the scene for current object ids.`);
    const toolName = actor === "BROWSER AGENT" ? "interact_story_world" : undefined;
    worldInteractionActorRef.current = { actor, toolName };
    const activated = stageRef.current?.interactWorldObject(objectId) ?? false;
    worldInteractionActorRef.current = { actor: "CHILD" };
    if (!activated) {
      const fallbackInteraction = getAccessibleWorldInteraction(currentWorld, objectId);
      if (!fallbackInteraction) throw new Error("That story object is not ready yet.");
      handleWorldInteraction(fallbackInteraction, actor, toolName);
    }
    return { objectId, world: currentWorld, activated: true, renderedIn3D: activated, accessibleFallback: !activated, visibleWorldUpdated: true };
  }, [handleWorldInteraction]);

  const animateCharacter = useCallback((action: CharacterAction, actor: Actor, toolName?: string, caption?: string) => {
    const current = characterRef.current;
    if (!current.created) throw new Error("Create the character before animating it.");
    const figures = captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [];
    const unsupported = buildCharacterCapabilities(figures, Boolean(neuralAssetRef.current)).filter((capability) => !capability.availableActions.includes(action));
    if (unsupported.length) throw new Error(`${action} is locked for figure ${unsupported.map((capability) => capability.characterIndex + 1).join(", ")} because the needed skeleton branch was not verified.`);
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

  const getArtworkCommerceProfile = useCallback((): ArtworkCommerceProfile => {
    const current = captureRef.current;
    if (!current) throw new Error("Approve a drawing before building a creator collection.");
    const figures = captureEnsembleRef.current.length ? captureEnsembleRef.current : [current];
    const movableParts = figures.reduce((sum, figure) => sum + selectAnimatableRigParts(figure.rig, {
      poseApplicable: Boolean(figure.poseRecognition?.applicable),
      topologyApplicable: Boolean(figure.topologyRecognition?.applicable),
    }).length, 0);
    return {
      characterName: characterRef.current.name || "My Drawing",
      figureCount: figures.length,
      aspectRatio: current.analysis.aspectRatio,
      coveragePercent: current.analysis.coveragePercent,
      edgeEnergy: current.analysis.edgeEnergy,
      dominantColor: current.analysis.dominantColor,
      secondaryColor: current.analysis.secondaryColor,
      movableParts,
      semanticParts: [...new Set(figures.flatMap((figure) => figure.rig.detectedKinds))],
      hasRigged3D: Boolean(neuralAssetRef.current),
      activeWorld: worldRef.current,
      storyTitle: characterRef.current.storyTitle,
      contributors: sharedRoomStateRef.current.participants.map((participant) => participant.username),
    };
  }, []);

  const recommendProductsForArtwork = useCallback((input: Record<string, unknown> = {}, actor: Actor = "BROWSER AGENT") => {
    const audience = stringValue(input.audience, creatorAudience, 20) as CreatorAudience;
    const goal = stringValue(input.goal, creatorGoal, 20) as CreatorGoal;
    if (!["family", "classroom", "community"].includes(audience)) throw new Error("Unknown audience.");
    if (!["keepsake", "fundraiser", "portfolio"].includes(goal)) throw new Error("Unknown creator goal.");
    const profile = getArtworkCommerceProfile();
    const recommendations = recommendCreatorProducts(profile, goal, audience);
    setCreatorAudience(audience);
    setCreatorGoal(goal);
    setCreatorRecommendations(recommendations);
    setCreatorDrop(null);
    setAdultExportApproved(false);
    setInspectorOpen(true);
    setPanelTab("commerce");
    setAgentLine(`I matched ${profile.characterName} to products that preserve its silhouette and detail.`);
    setNotice("Creator Shop is ready. Review the product reasoning, then ask the agent to stage a drop.");
    record(actor, "Recommended a creator collection", `${recommendations.slice(0, 3).map((item) => item.label).join(" · ")} · based on approved artwork evidence.`, actor === "BROWSER AGENT" ? "recommend_creator_products" : undefined);
    return {
      artwork: { ...profile, dominantColor: profile.dominantColor, secondaryColor: profile.secondaryColor },
      recommendations,
      visibleCreatorShopUpdated: true,
      nextAgentSteps: ["Ask the human which products and story feel right", "Stage a Shopify import kit for review"],
      cameraDataIncluded: false,
      imagePixelsIncluded: false,
    };
  }, [creatorAudience, creatorGoal, getArtworkCommerceProfile, record]);

  const stageShopifyImportKit = useCallback((input: Record<string, unknown> = {}, actor: Actor = "BROWSER AGENT") => {
    const audience = stringValue(input.audience, creatorAudience, 20) as CreatorAudience;
    const goal = stringValue(input.goal, creatorGoal, 20) as CreatorGoal;
    const vibe = stringValue(input.vibe, creatorVibe, 20) as CreatorVibe;
    const rawProducts = Array.isArray(input.productIds) ? input.productIds : [];
    const productIds = rawProducts.filter((item): item is CreatorProductId => typeof item === "string" && CREATOR_PRODUCT_IDS.includes(item as CreatorProductId));
    const drop = buildCreatorDrop({
      profile: getArtworkCommerceProfile(),
      dropName: stringValue(input.dropName, "", 72),
      story: stringValue(input.story, "", 240),
      audience,
      goal,
      vibe,
      productIds,
    });
    setCreatorAudience(audience);
    setCreatorGoal(goal);
    setCreatorVibe(vibe);
    setCreatorRecommendations(recommendCreatorProducts(getArtworkCommerceProfile(), goal, audience));
    setCreatorDrop(drop);
    setAdultExportApproved(false);
    setInspectorOpen(true);
    setPanelTab("commerce");
    setAgentLine(`The Creator Drop “${drop.name}” is staged—not published. An adult decides whether to export it.`);
    setNotice("Import kit staged for review. This app is not connected to a Shopify store.");
    record(actor, "Staged a Shopify import kit", `${drop.name} · ${drop.products.length} draft products · adult approval required · store not connected.`, actor === "BROWSER AGENT" ? "stage_shopify_import_kit" : undefined);
    return {
      drop,
      visibleCreatorShopUpdated: true,
      requiresHumanApproval: true,
      humanOnlyNextAction: "Approve and download the Shopify import kit",
      publishesToShopify: false,
      createsOrders: false,
      imagePixelsIncluded: false,
    };
  }, [creatorAudience, creatorGoal, creatorVibe, getArtworkCommerceProfile, record]);

  const inspectCreatorDrop = useCallback(() => ({
    drop: creatorDrop,
    recommendations: creatorRecommendations,
    approval: { required: true, granted: adultExportApproved },
    handoff: creatorDrop ? "Draft CSV + storefront blueprint + adult checklist" : null,
    shopifyConnection: "not-connected",
    optionalNextStep: "An adult may import these files into a real Shopify store. That separate storefront—not WallAlive—would expose Shopify's native WebMCP tools.",
    publishesToShopify: false,
    createsOrders: false,
    imagePixelsIncluded: false,
  }), [adultExportApproved, creatorDrop, creatorRecommendations]);

  const openCreatorStudio = useCallback(() => {
    setInspectorOpen(true);
    setPanelTab("commerce");
    if (!captureRef.current) {
      setNotice("Start with a drawing. The Creator Shop will unlock after the artwork is approved.");
      return;
    }
    if (!creatorRecommendations.length) recommendProductsForArtwork({}, "CHILD");
  }, [creatorRecommendations.length, recommendProductsForArtwork]);

  const downloadTextFile = useCallback((fileName: string, contents: string, type: string) => {
    const url = URL.createObjectURL(new Blob([contents], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = fileName;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, []);

  const approveCreatorExport = useCallback(() => {
    if (!creatorDrop) return;
    setAdultExportApproved(true);
    setCreatorDrop({ ...creatorDrop, status: "approved-for-export" });
    setNotice("Adult approval recorded. The offline Shopify import files are ready; no store is connected.");
    record("CHILD", "Approved the Creator Drop export", `${creatorDrop.name} may be downloaded as draft files; ${creatorDrop.contributors.length} contributor permission checks recorded; nothing was published.`);
  }, [creatorDrop, record]);

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
    worldRendering: "Original high-resolution procedural PBR Three.js set with perspective, textured surfaces, lights, occlusion, fog, shadows, and raycast interactions; no flat backdrop or third-party franchise art",
    interactiveStory: {
      title: worldActivities[worldRef.current].title,
      prompt: worldActivities[worldRef.current].prompt,
      objectIds: worldActivities[worldRef.current].objectIds,
      completedObjectIds: worldInteractions[worldRef.current],
    },
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
      localPreview: characterRef.current.created
        ? assessReconstructionReadiness(captureRef.current).motionReady
          ? "instant artwork-preserving spatial puppet with verified movement branches; not a full neural sculpt"
          : "instant artwork-preserving static spatial puppet; limb movement remains locked pending human rig review"
        : "verified transparent character cutout awaiting human review",
      method: neuralAssetRef.current
        ? `${neuralAssetRef.current.provider} full-volume neural mesh + skeleton skinning`
        : characterRef.current.created
          ? "local contour-preserving relief mesh + full-resolution artwork front + neutral filled back + only confidence-gated branch bones"
          : "local drawing segmentation + reconstruction-readiness gate + human review",
      provider: neuralAssetRef.current?.provider ?? "WallAlive local recognition",
      model: neuralAssetRef.current?.model ?? captureRef.current.cutoutRecognition?.model ?? "authored-alpha-cutout",
      assetType: neuralAssetRef.current ? "glTF SkinnedMesh" : characterRef.current.created ? "Three.js closed-relief spatial preview" : "reviewed transparent 2D cutout",
      topology: neuralAssetRef.current ? "generated full 3D surface including unseen views" : characterRef.current.created ? "closed relief volume; a preview, not unseen-view reconstruction" : "semantic evidence awaiting 3D choice",
      topologyConfidence: captureRef.current.rig.topologyConfidence ?? null,
      backInference: neuralAssetRef.current ? "full neural generative prior" : characterRef.current.created ? "bounded neutral relief; no invented rear artwork" : "not built yet",
      neuralEvidence: { fullSculptDegrees: neuralAssetRef.current ? 360 : 0 },
      orbitableDegrees: characterRef.current.created ? 360 : 0,
      fullSculptDegrees: neuralAssetRef.current ? 360 : 0,
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
      generationPhase: neuralAssetRef.current ? "neural-ready" : characterRef.current.created ? assessReconstructionReadiness(captureRef.current).motionReady ? "local-motion-ready" : "local-static-needs-rig-review" : "verified-cutout-review-ready",
      neuralUpgrade: neuralAssetRef.current ? "active" : "optional-single-figure-human-approved-upgrade",
      externalUploadApproved: externalUploadApprovedRef.current,
    } : null,
    character: { ...characterRef.current, textureUrl: undefined },
    cameraFeedExposed: false,
    privacyBoundary: "Camera capture is human-only. WebMCP can request reconstruction but cannot approve or upload; only a visible human action may send the isolated drawing to AniGen.",
    availableAnimations: actions.map((item) => item.action),
    placementModes: immersiveAR ? ["world-hit-test", "camera-overlay"] : ["camera-overlay"],
  }), [immersiveAR, worldInteractions]);

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
      sharedRoom: sharedRoomStateRef.current.session ? {
        roomId: sharedRoomStateRef.current.session.roomId,
        participantCount: sharedRoomStateRef.current.participants.length,
        vectorOperationCount: sharedRoomStateRef.current.operations.length,
      } : null,
      pendingShow: pending ? { id: pending.id, title: pending.title, learningGoal: pending.learningGoal, status: pending.status, beats: pending.beats.length, cast: pending.cast.length, stagedBy: pending.stagedBy } : null,
      humanOnlyControls: ["open_camera", "capture_frame", "approve_cutout", "approve_external_3d", "approve_and_play_staged_show"],
      agentWorkflow: characterRef.current.created
        ? ["inspect_reconstruction_readiness", "inspect_learning_progress", "stage_next_learning_challenge", "wait_for_visible_human_approval"]
        : ["ask_human_to_draw_or_capture", "inspect_reconstruction_readiness", "request_character_repair_if_needed", "request_rigged_3d_cast", "wait_for_visible_human_approval"],
      cameraFeedExposed: false,
      externalUploadApproved: scene.reconstruction?.externalUploadApproved ?? false,
      arPlacement: scene.placementModes,
    };
  }, [currentCharacterCapabilities, inspectScene]);

  const inspectLearningProgress = useCallback(() => {
    const plan = magicShowPlanRef.current;
    const room = sharedRoomStateRef.current;
    return buildLearningProgress({
      story: plan ? {
        title: plan.title,
        learningGoal: plan.learningGoal,
        plannedBeats: plan.beats.length,
        completedBeats: completedShowBeatsRef.current,
        status: plan.status,
      } : null,
      reflection: learningReflectionRef.current,
      humanTurns: activityRef.current.filter((item) => item.actor === "CHILD").length,
      agentTurns: activityRef.current.filter((item) => item.actor === "BROWSER AGENT").length,
      participantCount: room.session ? Math.max(1, room.participants.length) : 1,
      sharedVectorOperations: room.operations.length,
      worldInteractions: Object.fromEntries(worlds.map(({ id }) => [id, worldInteractions[id].length])),
      worldTotals: Object.fromEntries(worlds.map(({ id }) => [id, worldActivities[id].objectIds.length])),
    });
  }, [worldInteractions]);

  const inspectReconstructionReadiness = useCallback(() => {
    const drawings = captureEnsembleRef.current.length
      ? captureEnsembleRef.current
      : captureRef.current
        ? [captureRef.current]
        : [];
    const capabilities = currentCharacterCapabilities();
    const figures = drawings.map((drawing, index) => {
      const readiness = assessReconstructionReadiness(drawing);
      const capability = capabilities[index];
      return {
        characterIndex: index,
        selectedTarget: drawing.sourceTarget ?? null,
        cutoutReady: readiness.cutoutReady,
        motionReady: readiness.motionReady,
        readinessScore: readiness.score,
        blockers: readiness.blockers,
        warnings: readiness.warnings,
        detectedParts: capability?.semanticParts ?? {},
        movableParts: capability?.movableParts ?? [],
        availableActions: capability?.availableActions ?? ["idle", "hop", "hide", "spin"],
        recommendedHumanAction: !readiness.cutoutReady
          ? "Reselect this figure from a cleaner image."
          : !readiness.motionReady
            ? "Open the visible anatomy check and correct or approve movement branches."
            : "Ready for capability-checked movement.",
      };
    });
    return {
      figureCount: figures.length,
      allCutoutsReady: figures.length > 0 && figures.every((figure) => figure.cutoutReady),
      allMotionReady: figures.length > 0 && figures.every((figure) => figure.motionReady),
      figures,
      policy: "A clean cutout may become a static spatial puppet. Playable movement is claimed only for verified or human-reviewed branches.",
      cameraDataIncluded: false,
      artworkPixelsIncluded: false,
    };
  }, [currentCharacterCapabilities]);

  const saveLearningReflection = useCallback(() => {
    const retell = reflectionRetell.trim().slice(0, 360);
    if (retell.length < 3) {
      setNotice("Tell one thing that happened before saving your Story Passport.");
      return;
    }
    const reflection: LearningReflection = {
      retell,
      nextChange: reflectionNextChange,
      savedAt: new Date().toISOString(),
    };
    learningReflectionRef.current = reflection;
    setLearningReflection(reflection);
    setNotice("Story Passport saved privately in this tab. The agent can now suggest one evidence-based revision.");
    record("CHILD", "Reflected on the completed story", `Retell recorded · next revision: ${reflectionNextChange.replaceAll("-", " ")}.`);
  }, [record, reflectionNextChange, reflectionRetell]);

  const downloadLearningEvidence = useCallback(() => {
    const progress = inspectLearningProgress();
    const story = progress.story;
    const evidence = progress.observedEvidence;
    const lines = [
      "WALLALIVE STORY PASSPORT",
      "Observational evidence — not a grade or measured learning gain",
      "",
      `Story: ${story?.title ?? "Not staged"}`,
      `Learning goal: ${story?.learningGoal ?? "Not set"}`,
      `Sequence performed: ${story?.completedBeats ?? 0}/${story?.plannedBeats ?? 0} beats`,
      `Interactive world moments: ${evidence.completedWorldInteractions}`,
      `Completed worlds: ${evidence.completedWorlds.join(", ") || "None yet"}`,
      `Creators present: ${evidence.participantCount}`,
      `Shared drawing operations: ${evidence.sharedVectorOperations}`,
      "",
      `Learner retell: ${progress.reflection?.retell ?? "Not recorded"}`,
      `Learner's next revision: ${progress.reflection?.nextChange.replaceAll("-", " ") ?? "Not chosen"}`,
      "",
      `Suggested next scaffold: ${progress.suggestedNextScaffold}`,
      "Camera frames and artwork pixels are not included.",
    ];
    downloadTextFile("wallalive-story-passport.txt", lines.join("\n"), "text/plain;charset=utf-8");
    record("CHILD", "Downloaded a private Story Passport", "Structured story evidence only · no camera frames or artwork pixels.");
  }, [downloadTextFile, inspectLearningProgress, record]);

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

  const stageMagicShow = useCallback((input: Record<string, unknown>, actor: Actor = "BROWSER AGENT", toolName = "stage_magic_show") => {
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
      learningGoal: stringValue(input.learningGoal, "story sequencing and confident expression", 96),
      tone,
      world,
      cast,
      beats,
      stagedBy: actor,
      status: "awaiting-human-approval",
    };
    magicShowPlanRef.current = plan;
    completedShowBeatsRef.current = 0;
    learningReflectionRef.current = null;
    setMagicShowPlan(plan);
    setCompletedShowBeats(0);
    setLearningReflection(null);
    setReflectionRetell("");
    setAgentLine(actor === "BROWSER AGENT"
      ? `I staged “${plan.title}” from the verified abilities of ${plan.cast.length} character${plan.cast.length === 1 ? "" : "s"}. Only you can start it.`
      : `The guided demo staged “${plan.title}.” A real browser agent uses the same validated planning function through WebMCP.`);
    setNotice(`${actor === "BROWSER AGENT" ? "The browser agent" : "WallAlive's guided demo"} staged a learning story. Review it, then choose Approve & play or Not yet.`);
    record(actor, "Staged a learning story for human review", `${plan.title} · goal: ${plan.learningGoal} · ${plan.cast.length} cast · ${plan.beats.length} beats · ${worlds.find((candidate) => candidate.id === plan.world)?.label}.`, toolName);
    return {
      planId: plan.id,
      status: plan.status,
      learningGoal: plan.learningGoal,
      stagedBy: plan.stagedBy,
      requiresHumanApproval: true,
      approvalControlVisible: true,
      validatedCast: plan.cast.map(({ characterIndex, name, role }) => ({ characterIndex, name, role })),
      validatedBeats: plan.beats.map((beat, index) => ({ index, world: beat.world ?? plan.world, moves: beat.moves })),
      nextStep: "Wait for the human to press Approve & play in the shared page.",
      cameraAccessed: false,
    };
  }, [currentCharacterCapabilities, parseShowMoves, record]);

  const stageNextLearningChallenge = useCallback((input: Record<string, unknown>) => {
    const progress = inspectLearningProgress();
    const capabilities = currentCharacterCapabilities();
    if (!capabilities.length || !characterRef.current.created) throw new Error("A human-approved spatial cast is required before an adaptive challenge can be staged.");
    const requestedFocus = stringValue(input.focus, "", 24);
    const focus = (["sequencing", "collaboration", "expression", "observation"] as const).includes(requestedFocus as "sequencing")
      ? requestedFocus as "sequencing" | "collaboration" | "expression" | "observation"
      : progress.observedEvidence.participantCount > 1
        ? "collaboration"
        : progress.phase === "performed-needs-reflection"
          ? "expression"
          : "sequencing";
    const worldCompletion = worlds.map(({ id }) => ({
      id,
      ratio: worldActivities[id].objectIds.length ? worldInteractions[id].length / worldActivities[id].objectIds.length : 0,
    })).sort((left, right) => left.ratio - right.ratio);
    const nextWorld = worldCompletion[0]?.id ?? "studio";
    const revision = progress.reflection?.nextChange;
    const learningGoal = focus === "collaboration"
      ? "take turns, assign roles, and combine ideas into one three-part story"
      : focus === "expression"
        ? "name a feeling, show it through action, and retell what changed"
        : focus === "observation"
          ? "notice three interactive details and use them in a beginning, middle, and ending"
          : "sequence a clear beginning, middle, and ending, then retell the order";
    const actionPriority: CharacterAction[][] = [
      ["wave", "hop", "spin", "hide"],
      ["walk", "dance", "spin", "hop"],
      ["dance", "wave", "hide", "spin"],
    ];
    const movesForBeat = (beatIndex: number) => capabilities.map((capability, castIndex) => ({
      characterIndex: capability.characterIndex,
      action: actionPriority[(beatIndex + castIndex) % actionPriority.length].find((action) => capability.availableActions.includes(action)) ?? "spin",
    }));
    const cast = capabilities.map((capability, index) => ({
      characterIndex: capability.characterIndex,
      name: capabilities.length === 1 ? characterRef.current.name || "Pip" : `Friend ${index + 1}`,
      role: focus === "collaboration" ? ["idea starter", "helper", "story finisher"][index % 3] : ["explorer", "problem solver", "storyteller"][index % 3],
      personality: characterRef.current.personality || "curious and kind",
    }));
    const revisionPhrase = revision ? ` using the learner's ${revision.replaceAll("-", " ")} choice` : "";
    const staged = stageMagicShow({
      title: revision ? "Our Story, Reimagined" : "The Three-Moment Quest",
      theme: `${focus}${revisionPhrase}`,
      learningGoal,
      tone: focus === "expression" ? "gentle" : "adventurous",
      world: nextWorld,
      cast,
      beats: [
        { caption: "First, every friend notices the challenge and chooses a role.", durationMs: 1050, world: nextWorld, moves: movesForBeat(0) },
        { caption: "Next, the cast combines different moves to solve it together.", durationMs: 1250, moves: movesForBeat(1) },
        { caption: "Finally, they show what changed and get ready to tell the story back.", durationMs: 1150, moves: movesForBeat(2) },
      ],
    }, "BROWSER AGENT", "stage_next_learning_challenge");
    return {
      ...staged,
      adaptiveBasis: {
        previousPhase: progress.phase,
        learnerRevisionChoice: revision ?? null,
        participantCount: progress.observedEvidence.participantCount,
        leastExploredWorld: nextWorld,
        focus,
        interpretationBoundary: progress.interpretationBoundary,
      },
    };
  }, [currentCharacterCapabilities, inspectLearningProgress, stageMagicShow, worldInteractions]);

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
      for (const [index, beat] of current.beats.entries()) {
        await directEnsembleBeat(beat, "WALLALIVE", "approved_magic_show", controller.signal);
        completedShowBeatsRef.current = index + 1;
        setCompletedShowBeats(index + 1);
      }
      const complete = { ...current, status: "complete" as const };
      magicShowPlanRef.current = complete;
      setMagicShowPlan(complete);
      setStoryCaption(`${current.title} — made together.`);
      setNotice("Magic Show complete. Retell what happened in your Story Passport.");
      record("WALLALIVE", "Completed the approved Magic Show", `${current.title} · ${current.beats.length} verified beats.`);
      setInspectorOpen(true);
      setPanelTab("learning");
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

  const inspect3DPaintStudio = useCallback(() => ({
    characterReady: characterRef.current.created,
    interaction: "Touch or drag directly on the live 3D model. Paint follows the hit surface through UV textures or spatial vertex color, while the model remains orbitable outside paint mode.",
    availableTools: modelPaintTools.map(({ id, label }) => ({ id, label })),
    availablePalette: modelPaintPalette,
    currentBrush: paintBrushRef.current,
    paint: stageRef.current?.inspectPaint() ?? paintInspectionRef.current,
    stagedAdventure: paintAdventureRef.current,
    humanControl: {
      agentMayStagePalette: true,
      agentMayPaintPixels: false,
      childAppliesEveryStroke: true,
      undoAndResetVisible: true,
    },
    cameraDataIncluded: false,
    artworkPixelsIncluded: false,
  }), []);

  const stage3DPaintAdventure = useCallback((input: Record<string, unknown>) => {
    if (!characterRef.current.created) throw new Error("No 3D character is ready. Ask the family to approve a drawing and create its character first.");
    const requestedTool = stringValue(input.tool, "brush", 12) as ModelPaintTool;
    const tool = modelPaintTools.some((candidate) => candidate.id === requestedTool) ? requestedTool : "brush";
    const requestedPalette = Array.isArray(input.palette)
      ? input.palette.filter((color): color is string => typeof color === "string" && /^#[0-9a-f]{6}$/i.test(color)).slice(0, 5).map((color) => color.toLowerCase())
      : [];
    if (!requestedPalette.length) throw new Error("The palette needs one to five colors in #RRGGBB format.");
    const steps = Array.isArray(input.steps)
      ? input.steps.map((step) => stringValue(step, "", 80)).filter(Boolean).slice(0, 4)
      : [];
    if (!steps.length) throw new Error("Add one to four short, child-friendly painting steps.");
    const adventure: PaintAdventure = {
      id: makeId(),
      title: stringValue(input.title, "Color magic", 48),
      prompt: stringValue(input.prompt, "What happens when these colors meet?", 100),
      tool,
      palette: requestedPalette,
      steps,
      status: "awaiting-child",
    };
    commitPaintAdventure(adventure);
    choosePaintBrush({ ...paintBrushRef.current, tool, color: requestedPalette[0] });
    setPaintStudioOpen(true);
    setAgentLine(`I staged “${adventure.title}.” Your child makes every mark.`);
    setNotice("A paint idea is ready. Tap Start painting—or Not now. The agent changed no pixels.");
    record("BROWSER AGENT", "Staged a 3D paint adventure", `${adventure.title} · ${tool} · ${requestedPalette.length} colors · child applies every stroke.`, "stage_3d_paint_adventure");
    return {
      adventure,
      visibleReviewOpen: true,
      requiresChildAction: true,
      pixelsChanged: false,
      nextStep: "The child chooses Start painting, then touches the live 3D character.",
    };
  }, [choosePaintBrush, commitPaintAdventure, record]);

  const beginPaintAdventure = useCallback(() => {
    const current = paintAdventureRef.current;
    if (current) commitPaintAdventure({ ...current, status: "painting" });
    setPaintStudioOpen(true);
    setNotice("Paint the character itself. Drag for brush, spray, or oil; tap once for a splash.");
    record("CHILD", "Started the staged paint adventure", current?.title ?? "Free paint");
  }, [commitPaintAdventure, record]);

  const dismissPaintAdventure = useCallback(() => {
    const current = paintAdventureRef.current;
    if (current) commitPaintAdventure({ ...current, status: "dismissed" });
    setNotice("Paint idea dismissed. The model was not changed.");
    record("CHILD", "Dismissed the paint adventure", current?.title ?? "Paint suggestion");
  }, [commitPaintAdventure, record]);

  const undoModelPaint = useCallback(() => {
    const next = stageRef.current?.undoPaint() ?? emptyPaintInspection;
    commitPaintInspection(next);
    setNotice(next.strokeCount ? `Undid one stroke. ${next.strokeCount} left.` : "Paint returned to the original drawing.");
  }, [commitPaintInspection]);

  const resetModelPaint = useCallback(() => {
    const next = stageRef.current?.resetPaint() ?? emptyPaintInspection;
    commitPaintInspection(next);
    setNotice("All added paint cleared. The original drawing is safe.");
    record("CHILD", "Cleared added 3D paint", "Original artwork texture restored; no source pixels changed.");
  }, [commitPaintInspection, record]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) {
      window.queueMicrotask(() => {
        setWebMcpStatus("unsupported");
        setWebMcpCheck(null);
      });
      return;
    }
    const controller = new AbortController();
    const base = { type: "object", additionalProperties: false };
    const ok = (payload: Record<string, unknown>) => ({ ok: true, ...payload });
    const fail = (error: unknown) => {
      if (error instanceof DOMException && error.name === "AbortError") throw error;
      return { ok: false, error: error instanceof Error ? error.message : "Tool execution failed." };
    };
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
        name: "inspect_learning_progress",
        title: "Inspect the learner's Story Passport",
        description: "Read the visible plan, completed beats, collaborative turns, world activity, reflection, and suggested next scaffold. This is observational evidence, not a grade; it returns no camera frames or artwork pixels.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => {
          const signal = executionSignal(options);
          guard(signal);
          return ok({ progress: inspectLearningProgress(), observedAt: new Date().toISOString() });
        },
      },
      {
        name: "stage_next_learning_challenge",
        title: "Stage the learner's next challenge",
        description: "Use the private Story Passport, participant count, least-explored world, learner revision choice, and verified character actions to stage one visible next challenge. It is observational—not grading—and the human must approve playback.",
        inputSchema: { ...base, properties: { focus: { type: "string", enum: ["sequencing", "collaboration", "expression", "observation"], description: "Optional learning focus. If omitted, WallAlive chooses from visible Story Passport evidence." } } },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = stageNextLearningChallenge(input);
            await afterVisiblePaint();
            guard(signal);
            return ok(result);
          } catch (error) { return fail(error); }
        },
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
              if (characterRef.current.created && !neuralAssetRef.current) return ok({ alreadyCreated: true, reconstructionMode: "local-articulated", readiness: inspectReconstructionReadiness() });
              requestNeuralConsent();
              await afterVisiblePaint();
              guard(signal);
              return ok({
                requiresHumanApproval: true,
                phase: "choice-required",
                requestedMode: "local-articulated",
                readiness: inspectReconstructionReadiness(),
                message: "The visible reconstruction card is open. The human must choose the private spatial puppet or review a blocked figure; the agent cannot create or approve it.",
              });
            }
            if (!neuralAssetRef.current) {
              requestNeuralConsent();
              await afterVisiblePaint();
              guard(signal);
              return ok({ requiresHumanApproval: true, phase: "choice-required", message: "Use the visible card to choose instant local articulated 3D or approve a single-figure full AI sculpt." });
            }
            const result = ok({
              character: createCharacter(input, "BROWSER AGENT", "request_rigged_3d_cast"),
              reconstructionMode: "neural-full",
              localRig: null,
              generatedAsset: riggedAssetInfoRef.current,
            });
            await afterVisiblePaint();
            guard(signal);
            return result;
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
        name: "inspect_reconstruction_readiness",
        title: "Inspect every selected figure before 3D",
        description: "Read per-figure cutout and motion readiness, blockers, warnings, verified movable parts, and the next human repair action. Returns structured evidence only—never camera frames or artwork pixels.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => { const signal = executionSignal(options); guard(signal); return ok({ readiness: inspectReconstructionReadiness(), observedAt: new Date().toISOString() }); },
      },
      {
        name: "request_character_repair",
        title: "Request a visible human rig repair",
        description: "Open the anatomy check for one selected figure and explain its current blockers. The agent cannot move joints, alter the drawing, approve the rig, or access pixels; the human performs and approves every correction.",
        inputSchema: {
          ...base,
          properties: {
            characterIndex: { type: "integer", minimum: 0, maximum: 5, description: "Figure index returned by inspect_reconstruction_readiness." },
            focus: { type: "string", enum: ["cutout", "face", "arms", "legs", "movement"], description: "Area the agent wants the human to verify." },
          },
          required: ["characterIndex", "focus"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const drawings = captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [];
            const characterIndex = Math.round(numberValue(input.characterIndex, -1));
            const drawing = drawings[characterIndex];
            if (!drawing) throw new Error(`Figure ${characterIndex} does not exist. Inspect reconstruction readiness first.`);
            const readiness = assessReconstructionReadiness(drawing);
            activeFigureIndexRef.current = characterIndex;
            setActiveFigureIndex(characterIndex);
            setSelectedPartId(null);
            setPendingPartKind(null);
            setPartEditorOpen(true);
            const focus = stringValue(input.focus, "movement", 16);
            const guidance = !readiness.cutoutReady
              ? `The cutout has a blocker: ${readiness.blockers[0]} Close this check and reselect the figure if the boundary is wrong.`
              : `Check ${focus}; drag incorrect parts, add missing parts, remove false parts, then use Approve rig.`;
            setAgentLine(`I found uncertainty in figure ${characterIndex + 1}. You stay in control of the correction.`);
            setNotice(guidance);
            record("BROWSER AGENT", "Requested a human anatomy check", `Figure ${characterIndex + 1} · ${focus} · agent made no pixel or rig changes.`, "request_character_repair");
            await afterVisiblePaint();
            guard(signal);
            return ok({ characterIndex, focus, readinessBeforeRepair: readiness, visibleRepairOpen: true, requiresHumanApproval: true, agentChangedRig: false, guidance });
          } catch (error) { return fail(error); }
        },
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
            learningGoal: { type: "string", minLength: 1, maxLength: 96 },
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
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = stageMagicShow(input);
            await afterVisiblePaint();
            guard(signal);
            return ok(result);
          } catch (error) { return fail(error); }
        },
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
        name: "recommend_creator_products",
        title: "Recommend products for this artwork",
        description: "Rank product formats using the approved artwork's figure count, silhouette, detail, colors, and goal. Opens visible recommendations but never publishes, buys, or returns image pixels.",
        inputSchema: {
          ...base,
          properties: {
            audience: { type: "string", enum: ["family", "classroom", "community"], description: "Who the reviewed collection is for." },
            goal: { type: "string", enum: ["keepsake", "fundraiser", "portfolio"], description: "What the creator wants the collection to achieve." },
          },
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = recommendProductsForArtwork(input);
            await afterVisiblePaint();
            guard(signal);
            return ok(result);
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "stage_shopify_import_kit",
        title: "Stage a Shopify import kit",
        description: "Create a visible offline import kit with product copy, print placement, palette, pricing, and storefront sections. WallAlive is not connected to a Shopify store; it cannot publish, manage a cart, or place orders.",
        inputSchema: {
          ...base,
          properties: {
            dropName: { type: "string", minLength: 1, maxLength: 72, description: "Collection name shown in the review panel." },
            story: { type: "string", minLength: 1, maxLength: 240, description: "Short creator story for the collection page." },
            audience: { type: "string", enum: ["family", "classroom", "community"], description: "Who the reviewed collection is for." },
            goal: { type: "string", enum: ["keepsake", "fundraiser", "portfolio"], description: "Collection purpose used for recommendations." },
            vibe: { type: "string", enum: ["sunny", "storybook", "bold", "museum"], description: "Visual direction for the storefront blueprint." },
            productIds: { type: "array", minItems: 1, maxItems: 4, items: { type: "string", enum: CREATOR_PRODUCT_IDS }, description: "Draft products to include." },
          },
          required: ["dropName", "story", "audience", "goal", "vibe"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = stageShopifyImportKit(input);
            await afterVisiblePaint();
            guard(signal);
            return ok(result);
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "inspect_shopify_import_kit",
        title: "Inspect the staged Shopify import kit",
        description: "Read the visible offline draft, recommendation evidence, approval status, and explicit not-connected state. Returns no camera frames, artwork pixels, payment data, or order data.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => { const signal = executionSignal(options); guard(signal); return ok(inspectCreatorDrop()); },
      },
      {
        name: "inspect_shared_room",
        title: "Inspect the collaborative room",
        description: "Read guest creators, room code, vector-operation count, and the current interactive quest. Returns handles and structured state only—never artwork pixels, camera frames, or private tokens.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => {
          const signal = executionSignal(options);
          guard(signal);
          const room = sharedRoomStateRef.current;
          return ok({
            active: Boolean(room.session),
            roomId: room.session?.roomId ?? null,
            creators: room.participants.map((person) => ({ username: person.username, accent: person.accent })),
            sharedVectorOperations: room.operations.length,
            sync: room.status,
            quest: { world: worldRef.current, ...worldActivities[worldRef.current], completedObjectIds: worldInteractions[worldRef.current] },
            pixelsIncluded: false,
            sessionTokenIncluded: false,
          });
        },
      },
      {
        name: "prepare_room_invite",
        title: "Prepare a creator room invite",
        description: "Prepare a visible invite link for a guest username in the current drawing room. It does not contact the person or send a message; the human chooses how to share the link.",
        inputSchema: { ...base, properties: { username: { type: "string", minLength: 2, maxLength: 24, description: "Guest creator handle to place on the invite." } }, required: ["username"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            if (!sharedRoomStateRef.current.session) {
              setSharedRoomOpen(true);
              await afterVisiblePaint();
              guard(signal);
              return ok({ requiresHumanAction: true, action: "Create or join a room in the visible collaboration panel.", messageSent: false });
            }
            const result = await prepareRoomInvite(stringValue(input.username, "", 24));
            setSharedRoomOpen(true);
            setAgentLine(`Invite prepared for @${result.friend}. You decide where to share it.`);
            record("BROWSER AGENT", "Prepared a room invite", `@${result.friend} · no message sent`, "prepare_room_invite");
            await afterVisiblePaint();
            guard(signal);
            return ok({ username: result.friend, inviteUrl: result.inviteUrl, visibleReviewOpen: true, messageSent: false, pixelsIncluded: false });
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "interact_story_world",
        title: "Touch an object in the story world",
        description: "Activate one listed 3D story object in the visible world and return its narrative effect. Use inspect_shared_room first. It cannot invent hidden objects or bypass character capability checks.",
        inputSchema: { ...base, properties: { objectId: { type: "string", minLength: 1, maxLength: 64, description: "Exact object id from the current quest." } }, required: ["objectId"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = interactWithWorldObject(stringValue(input.objectId, "", 64));
            await afterVisiblePaint();
            guard(signal);
            return ok(result);
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "inspect_3d_paint_studio",
        title: "Inspect the child's 3D paint studio",
        description: "Read the current paint tool, available child-safe palette, stroke evidence, and any staged paint adventure. Returns structured state only, without camera frames or artwork pixels.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, options) => {
          const signal = executionSignal(options);
          guard(signal);
          return ok({ studio: inspect3DPaintStudio(), observedAt: new Date().toISOString() });
        },
      },
      {
        name: "stage_3d_paint_adventure",
        title: "Stage a 3D paint adventure for the child",
        description: "Stage one visible palette, texture tool, creative prompt, and one-to-four short steps beside the live 3D model. The tool changes no pixels; the child decides whether to start and applies every stroke by touch.",
        inputSchema: {
          ...base,
          properties: {
            title: { type: "string", minLength: 1, maxLength: 48, description: "Short playful adventure title." },
            prompt: { type: "string", minLength: 1, maxLength: 100, description: "Open-ended creative question for the child." },
            tool: { type: "string", enum: ["brush", "spray", "oil", "spill"], description: "Starting texture; the child can change it." },
            palette: { type: "array", minItems: 1, maxItems: 5, items: { type: "string", pattern: "^#[0-9a-fA-F]{6}$" }, description: "One to five proposed colors." },
            steps: { type: "array", minItems: 1, maxItems: 4, items: { type: "string", minLength: 1, maxLength: 80 }, description: "Short optional ideas, not commands." },
          },
          required: ["title", "prompt", "tool", "palette", "steps"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, options) => {
          const signal = executionSignal(options);
          try {
            guard(signal);
            const result = stage3DPaintAdventure(input);
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

    registerAndVerifyWebMCP(context, tools, controller.signal).then((result) => {
      if (controller.signal.aborted) return;
      setWebMcpCheck(result);
      setWebMcpStatus(result.status);
    }).catch((error) => {
      if (controller.signal.aborted || (error instanceof DOMException && error.name === "AbortError")) return;
      console.error("WallAlive WebMCP runtime check failed", error);
      setWebMcpCheck(null);
      setWebMcpStatus("error");
    });
    return () => controller.abort();
  }, [commitNeuralAsset, createCharacter, currentCharacterCapabilities, directEnsembleBeat, inspect3DPaintStudio, inspectCreativeScene, inspectCreatorDrop, inspectLearningProgress, inspectReconstructionReadiness, interactWithWorldObject, orchestrateSpatialCinematics, parseShowMoves, prepareRoomInvite, recommendProductsForArtwork, record, requestNeuralConsent, stage3DPaintAdventure, stageMagicShow, stageNextLearningChallenge, stageShopifyImportKit, worldInteractions]);

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
      setAgentLine(`The guided demo is loading a verified rig. A real agent can inspect, direct, and stage a Creator Drop through ${toolNames.length} WebMCP tools.`);
      await wait(450);
      createCharacter({ name: "Sunny", personality: "brave on the outside, silly at heart", accent: "#f6c958", inflation: 1 }, "WALLALIVE", "judge_demo");
      placeCharacter(.68, .53, "wall", 1, "WALLALIVE");
      await wait(500);
      stageMagicShow({
        title: "Sunny Finds a Brave Hello",
        theme: "finding courage with a new friend",
        learningGoal: "sequence a beginning, middle, and ending; name an emotion; retell the story",
        tone: "gentle",
        world: "storybook",
        cast: [{ characterIndex: 0, name: "Sunny", role: "the playful explorer", personality: "silly, curious, and secretly brave" }],
        beats: [
          { caption: "Sunny peeks from the edge of the kingdom.", durationMs: 900, moves: [{ characterIndex: 0, action: "hide" }] },
          { caption: "One brave hop brings Sunny into the story.", durationMs: 900, moves: [{ characterIndex: 0, action: "hop" }] },
          { caption: "Sunny waves hello with a verified arm branch.", durationMs: 1100, moves: [{ characterIndex: 0, action: "wave" }] },
          { caption: "A full turn reveals the generated back.", durationMs: 1400, world: "museum", moves: [{ characterIndex: 0, action: "spin" }] },
        ],
      }, "WALLALIVE", "judge_demo");
      setAgentLine("The guided example is ready. A real browser agent can stage a new version; either way, only you can approve playback.");
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
      setCameraTargets((current) => {
        const next = appendCaptureTarget(current, target, 6);
        setNotice(next.length === current.length
          ? "That character is already selected. Tap a different character or capture now."
          : `${next.length} character${next.length === 1 ? "" : "s"} selected. Tap more, or capture only this cast.`);
        return next;
      });
      return;
    }
    if (!characterRef.current.created) return;
    placeCharacter(x, y, "screen", characterRef.current.scale, "CHILD");
  }, [cameraState, placeCharacter]);

  const handleStagePointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (paintStudioOpen && characterRef.current.created) {
      if (!(event.target instanceof HTMLCanvasElement)) return;
      event.preventDefault();
      const bounds = event.currentTarget.getBoundingClientRect();
      stageRef.current?.beginPaintStroke(paintBrushRef.current);
      const result = stageRef.current?.paintAtNormalized(
        (event.clientX - bounds.left) / Math.max(1, bounds.width),
        (event.clientY - bounds.top) / Math.max(1, bounds.height),
        event.pressure || 0.5,
      );
      paintGestureRef.current = { pointerId: event.pointerId, painted: Boolean(result?.painted) };
      event.currentTarget.setPointerCapture(event.pointerId);
      if (!result?.painted) setNotice("Touch the 3D character to paint it. Empty world space stays clean.");
      return;
    }
    if (characterRef.current.created && event.target instanceof HTMLCanvasElement) return;
    rotateGestureRef.current = { pointerId: event.pointerId, lastX: event.clientX, lastY: event.clientY, moved: false };
    event.currentTarget.setPointerCapture(event.pointerId);
  }, [paintStudioOpen]);

  const handleStagePointerMove = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const paintGesture = paintGestureRef.current;
    if (paintStudioOpen && paintGesture?.pointerId === event.pointerId) {
      event.preventDefault();
      const bounds = event.currentTarget.getBoundingClientRect();
      const result = stageRef.current?.paintAtNormalized(
        (event.clientX - bounds.left) / Math.max(1, bounds.width),
        (event.clientY - bounds.top) / Math.max(1, bounds.height),
        event.pressure || 0.5,
      );
      if (result?.painted) paintGesture.painted = true;
      return;
    }
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
  }, [cameraState, paintStudioOpen]);

  const handleStagePointerUp = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    const paintGesture = paintGestureRef.current;
    if (paintGesture?.pointerId === event.pointerId) {
      paintGestureRef.current = null;
      const next = stageRef.current?.endPaintStroke() ?? emptyPaintInspection;
      commitPaintInspection(next);
      if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
      if (paintGesture.painted) {
        setNotice(`${paintBrushRef.current.tool === "spill" ? "Splash" : paintBrushRef.current.tool} added in 3D. Undo is always available.`);
        record("CHILD", "Painted directly on the 3D character", `${paintBrushRef.current.tool} · ${paintBrushRef.current.color} · ${next.strokeCount} strokes`);
      }
      return;
    }
    if (event.target instanceof HTMLCanvasElement) return;
    const gesture = rotateGestureRef.current;
    rotateGestureRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    if (gesture?.moved) {
      setNotice(neuralAssetRef.current ? "Full 3D sculpt rotated." : "Instant spatial preview rotated. Full unseen-view sculpt is a separate neural quality tier.");
      return;
    }
    activateStagePoint(event);
  }, [activateStagePoint, commitPaintInspection, record]);

  const handleStagePointerCancel = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    rotateGestureRef.current = null;
    if (paintGestureRef.current?.pointerId === event.pointerId) {
      paintGestureRef.current = null;
      commitPaintInspection(stageRef.current?.endPaintStroke() ?? emptyPaintInspection);
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }, [commitPaintInspection]);

  const nudgeCharacter = useCallback((x: number, z: number) => {
    if (!characterRef.current.created) return;
    stageRef.current?.moveBy(x, z);
    setNotice("Character moved through the 3D world. Drag to orbit, wheel or pinch to zoom, and right-drag or two-finger pan.");
  }, []);

  const handleARCapability = useCallback((supported: boolean) => {
    setImmersiveAR(supported);
  }, []);

  const handleRendererCapability = useCallback((supported: boolean) => {
    setRendererAvailable(supported);
  }, []);

  const handleARPlaced = useCallback((surface: "screen" | "world") => {
    if (surface === "world") commitCharacter({ ...characterRef.current, surface: "wall" });
  }, [commitCharacter]);

  const copyDemoPrompt = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(suggestedJudgePrompt);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = suggestedJudgePrompt;
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      textarea.remove();
    }
    setNotice("Family prompt copied. Paste it into ChatGPT while this page is open.");
  }, []);

  const selectRigFigure = useCallback((index: number) => {
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [];
    const safeIndex = Math.min(Math.max(0, index), Math.max(0, ensemble.length - 1));
    activeFigureIndexRef.current = safeIndex;
    setActiveFigureIndex(safeIndex);
    setSelectedPartId(null);
    setPendingPartKind(null);
  }, []);

  const openRigEditor = useCallback((index = 0) => {
    selectRigFigure(index);
    setPartEditorOpen(true);
    setNotice(`Reviewing figure ${index + 1}. Move, resize, add, or remove parts before approving movement.`);
  }, [selectRigFigure]);

  const commitRigEdit = useCallback((parts: SemanticPart[], message: string) => {
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [];
    const index = Math.min(activeFigureIndexRef.current, Math.max(0, ensemble.length - 1));
    const current = ensemble[index];
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
    const nextEnsemble = ensemble.map((figure, figureIndex) => figureIndex === index ? next : figure);
    captureEnsembleRef.current = nextEnsemble;
    setCaptureEnsemble(nextEnsemble);
    if (index === 0) {
      captureRef.current = next;
      setCapture(next);
    }
    setNotice(message);
  }, []);

  const partSide = useCallback((x: number): SemanticSide => {
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [];
    const current = ensemble[activeFigureIndexRef.current];
    const bodyX = current?.rig.parts.find((part) => part.kind === "body")?.center.x ?? 0;
    return x < bodyX - 0.04 ? "left" : x > bodyX + 0.04 ? "right" : "center";
  }, []);

  const moveRigPart = useCallback((partId: string, x: number, y: number) => {
    const current = (captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [])[activeFigureIndexRef.current];
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
    const current = (captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [])[activeFigureIndexRef.current];
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
    const current = (captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [])[activeFigureIndexRef.current];
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
    const current = (captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [])[activeFigureIndexRef.current];
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
    }), `Figure ${activeFigureIndexRef.current + 1} approved. Reviewed limb paths can now drive articulated joints.`);
    setPartEditorOpen(false);
    setPendingPartKind(null);
  }, [commitRigEdit]);

  const deleteSelectedPart = useCallback(() => {
    const current = (captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [])[activeFigureIndexRef.current];
    if (!current || !selectedPartId) return;
    const selected = current.rig.parts.find((part) => part.id === selectedPartId);
    if (!selected || selected.kind === "body") return;
    commitRigEdit(current.rig.parts.filter((part) => part.id !== selectedPartId), `${selected.kind} removed from the rig.`);
    setSelectedPartId(null);
  }, [commitRigEdit, selectedPartId]);

  const removeFigure = useCallback((index: number) => {
    const ensemble = captureEnsembleRef.current.length ? captureEnsembleRef.current : captureRef.current ? [captureRef.current] : [];
    if (ensemble.length <= 1) {
      setNotice("Keep one figure in the cast. Choose Try again if this cutout is wrong.");
      return;
    }
    const next = ensemble.filter((_, figureIndex) => figureIndex !== index);
    captureEnsembleRef.current = next;
    captureRef.current = next[0];
    setCaptureEnsemble(next);
    setCapture(next[0]);
    selectRigFigure(Math.min(index, next.length - 1));
    setNotice(`Removed background selection ${index + 1}. ${next.length} figure${next.length === 1 ? "" : "s"} remain.`);
    record("CHILD", "Removed a mistaken cast selection", `Figure ${index + 1} removed before 3D; ${next.length} figure${next.length === 1 ? "" : "s"} remain.`);
  }, [record, selectRigFigure]);

  const latestAgentActivity = useMemo(() => activity.find((item) => item.actor === "BROWSER AGENT"), [activity]);
  const activeRigDrawing = captureEnsemble[activeFigureIndex] ?? capture;
  const ensembleReadiness = useMemo(() => captureEnsemble.map(assessReconstructionReadiness), [captureEnsemble]);
  const movablePartCount = useMemo(() => captureEnsemble.reduce((sum, figure) => sum + selectAnimatableRigParts(figure.rig, {
    poseApplicable: Boolean(figure.poseRecognition?.applicable),
    topologyApplicable: Boolean(figure.topologyRecognition?.applicable),
  }).length, 0), [captureEnsemble]);
  const sharedSupportedActions = useMemo(() => {
    const capabilities = buildCharacterCapabilities(captureEnsemble, Boolean(neuralAsset));
    return new Set(SAFE_SHOW_ACTIONS.filter((action) => capabilities.length > 0 && capabilities.every((capability) => capability.availableActions.includes(action))));
  }, [captureEnsemble, neuralAsset]);
  const learningProgress = useMemo(() => buildLearningProgress({
    story: magicShowPlan ? {
      title: magicShowPlan.title,
      learningGoal: magicShowPlan.learningGoal,
      plannedBeats: magicShowPlan.beats.length,
      completedBeats: completedShowBeats,
      status: magicShowPlan.status,
    } : null,
    reflection: learningReflection,
    humanTurns: activity.filter((item) => item.actor === "CHILD").length,
    agentTurns: activity.filter((item) => item.actor === "BROWSER AGENT").length,
    participantCount: sharedRoom.session ? Math.max(1, sharedRoom.participants.length) : 1,
    sharedVectorOperations: sharedRoom.operations.length,
    worldInteractions: Object.fromEntries(worlds.map(({ id }) => [id, worldInteractions[id].length])),
    worldTotals: Object.fromEntries(worlds.map(({ id }) => [id, worldActivities[id].objectIds.length])),
  }), [activity, completedShowBeats, learningReflection, magicShowPlan, sharedRoom.operations.length, sharedRoom.participants.length, sharedRoom.session, worldInteractions]);
  const neuralBusy = ["connecting", "preparing", "queued", "generating", "downloading"].includes(neuralProgress.phase);
  const primaryButton = cameraState === "active"
    ? { label: "CAPTURE DRAWING", action: captureDrawing }
    : capture && !character.created
      ? { label: "CHOOSE 3D QUALITY", action: requestNeuralConsent }
      : { label: "START CAMERA", action: startCamera };
  if (capture && captureEnsemble.length > 1 && !character.created) primaryButton.label = `WAKE ${captureEnsemble.length} FIGURES`;
  const stepIndex = step === "ready" ? 0 : step === "camera" ? 1 : character.created ? 3 : 2;
  const showLegacyStart: boolean = false;
  const webMcpReady = webMcpStatus === "registered" || webMcpStatus === "verified";
  const webMcpLabel = webMcpStatus === "verified" ? `WEBMCP ✓ ${webMcpCheck?.registeredCount ?? toolNames.length}`
    : webMcpStatus === "registered" ? "WEBMCP LIVE"
      : webMcpStatus === "registering" ? "WEBMCP…"
        : webMcpStatus === "error" ? "WEBMCP ERROR"
          : "WEBMCP OFF";

  return (
    <main className="alive-shell" data-webmcp-status={webMcpStatus} data-webmcp-tool-count={webMcpCheck?.registeredCount ?? 0} data-webmcp-probe={webMcpCheck?.verifiedTool ?? "none"}>
      <header className="alive-header">
        <a className="alive-brand" href="#play"><span>WALL</span>ALIVE<i>●</i></a>
        <div className="mini-steps" aria-label="Creative journey"><span className={stepIndex >= 1 ? "done" : "active"}>Make</span><i>→</i><span className={stepIndex >= 2 ? "done" : ""}>Wake</span><i>→</i><span className={stepIndex >= 3 ? "done" : ""}>Share</span></div>
        <div className="header-actions">
          <div className={`ready-pill ${webMcpReady ? "is-ready" : ""}`} title={webMcpStatus === "unsupported" ? "Open in ChatGPT or Chrome with WebMCP enabled" : webMcpLabel}><i /> {webMcpLabel}</div>
          <button className={`room-toggle ${sharedRoom.session ? "is-live" : ""}`} onClick={() => setSharedRoomOpen(true)} aria-label="Open collaborative drawing room">{sharedRoom.session ? `${sharedRoom.participants.length} LIVE` : "DRAW TOGETHER"}</button>
          <button className="merch-toggle" onClick={openCreatorStudio} aria-label="Open Creator Shop">SHOP</button>
          <button className="inspector-toggle" onClick={() => { setInspectorOpen(true); setPanelTab("agent"); }}>AGENT</button>
          <button className="agent-prompt" onClick={copyDemoPrompt} aria-label="Copy the agent demo prompt">PROMPT</button>
          <button className="judge-demo" onClick={runMagicDemo} disabled={demoRunning}>{demoRunning ? "WAKING…" : "SEE THE MAGIC"}</button>
        </div>
      </header>
      <input ref={uploadRef} hidden type="file" accept="image/*" onChange={uploadDrawing} />

      <section className="alive-layout" id="play">
        {showLegacyStart ? (<aside className="steps-panel" hidden aria-hidden="true">
          <p className="kicker">START HERE</p>
          <h2>Make a new friend.</h2>
          <p className="start-copy">Draw in the studio, choose a picture, or point the camera at artwork on a wall.</p>
          <div className="start-choices">
            <button onClick={() => setDrawingWallOpen(true)}><i>✦</i><span><b>Draw something</b><small>Full paint studio</small></span><em>→</em></button>
            <button onClick={() => setSharedRoomOpen(true)}><i>∞</i><span><b>Draw together</b><small>{sharedRoom.session ? `${sharedRoom.participants.length} creator${sharedRoom.participants.length === 1 ? "" : "s"} live` : "Invite a friend"}</small></span><em>→</em></button>
            <button onClick={() => uploadRef.current?.click()}><i>▧</i><span><b>Use a picture</b><small>Paper or digital art</small></span><em>→</em></button>
            <button onClick={cameraState === "active" ? captureDrawing : startCamera}><i>◉</i><span><b>{cameraState === "active" ? "Capture now" : "Find wall art"}</b><small>Camera stays private</small></span><em>→</em></button>
          </div>
          <button className="demo-doodle" onClick={loadDemoDrawing}>Try the friendly demo <span>＋</span></button>
          <ol className="journey-trail">
            <li className={stepIndex >= 1 ? "done" : "active"}><span>1</span><div><strong>Make</strong><small>Choose an artwork</small></div></li>
            <li className={stepIndex >= 3 ? "done" : stepIndex === 2 ? "active" : ""}><span>2</span><div><strong>Wake</strong><small>Check and animate</small></div></li>
            <li className={stepIndex === 3 ? "active" : ""}><span>3</span><div><strong>Share</strong><small>Story or Creator Drop</small></div></li>
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
              <div><p className="kicker">{neuralAsset ? "FULL NEURAL RIG + DRAWING PARTS" : "VERIFIED CUTOUT · REVIEW"}</p><strong>{neuralAsset ? capture.rig.detectedKinds.filter((kind) => kind !== "body").join(" · ") || capture.analysis.shapeHint : capture.characterValidation?.evidence.join(" · ") || "CHARACTER EVIDENCE"}</strong><span><i style={{ background: capture.rig.bodyColor }} /><i style={{ background: capture.rig.lineColor }} /> {riggedAssetInfo ? `${riggedAssetInfo.bones} BONES · ${riggedAssetInfo.semanticParts} PROJECTED PARTS${riggedAssetInfo.colorTransfer ? " · COLOR MATCHED" : ""} · ${riggedAssetInfo.vertices.toLocaleString()} VERTICES` : neuralAsset ? rendererAvailable === false ? "3D ASSET VERIFIED · PREVIEW NEEDS WEBGL" : "RIGGED GLB LOADING" : "2D CUTOUT · NO FAKE 3D"}</span></div>
            </div>
          ) : null}

          <div className="privacy-card">
            <div><b>Private by default</b><span>✓</span></div>
            <p>The agent never sees camera frames. You approve any AI sculpt or store export.</p>
          </div>
        </aside>) : null}

        <section className="magic-stage">
          <div className="stage-copy">
            <div><p className="kicker">DRAW · WAKE · PLAY</p><h1>Draw it.<br /><em>Bring it to life.</em></h1><p className="stage-purpose">You draw. The agent guides. You approve. Draw, upload, or scan. Then play together in 3D.</p><div className="human-agent-loop"><span><b>YOU</b> draw</span><i>+</i><span><b>CHATGPT</b> guides</span><i>→</i><span><b>YOU</b> approve &amp; play</span></div></div>
            <div className="stage-ctas">
              {immersiveAR && character.created ? <button className="ar-button" onClick={enterAR}>ENTER REAL AR <span>◎</span></button> : null}
              {cameraState === "active" ? <button className="stop-camera" onClick={stopCamera}>STOP CAMERA</button> : null}
              {cameraState !== "active" && step === "ready" ? <button className="upload-camera" onClick={() => uploadRef.current?.click()}>UPLOAD</button> : null}
              {cameraState !== "active" && !capture ? <button className="draw-wall-cta" onClick={() => setDrawingWallOpen(true)}>DRAW <span>✦</span></button> : null}
              {cameraState !== "active" && !capture ? <button className="share-wall-cta" onClick={() => setSharedRoomOpen(true)}>TOGETHER <span>∞</span></button> : null}
              <button className="webmcp-coach" onClick={() => { void copyDemoPrompt(); setInspectorOpen(true); setPanelTab("agent"); }}>ASK CHATGPT <span>✦</span></button>
              {capture ? <button className="merch-cta" onClick={openCreatorStudio}>MAKE PRODUCTS <span>↗</span></button> : null}
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
              <p><b>{captureEnsemble.length > 1 ? `${captureEnsemble.length} figures selected` : "Artwork preserved"}</b><small>{captureEnsemble.length > 1 ? "separate cutouts · rig checks" : capture.cutoutRecognition?.model === "mediapipe-magic-touch-v2" ? "point-guided cutout" : "local cutout"}</small></p>
            </div>
            <div className="anatomy-pills"><span>Transparent</span><span>{captureEnsemble.length > 1 ? `${movablePartCount} moving limbs` : "Local only"}</span><span>Human check</span></div>
            <button onClick={() => openRigEditor(0)}>REVIEW RIG</button>
          </div> : null}

          {captureEnsemble.length > 1 ? <div className="cast-check" aria-label="Selected character readiness">
            <div><span>YOUR CAST</span><small>Check every figure before movement</small></div>
            <div className="cast-check-list">{captureEnsemble.map((figure, index) => {
              const readiness = ensembleReadiness[index];
              const status = !readiness?.cutoutReady ? "RESELECT" : readiness.motionReady ? "READY" : "FIX RIG";
              return <article key={`${figure.textureUrl.slice(-24)}-${index}`} className={readiness?.motionReady ? "is-ready" : "needs-review"}>
                <button className="cast-figure" onClick={() => openRigEditor(index)} aria-label={`Review figure ${index + 1}, ${status.toLowerCase()}`}>
                  <img src={figure.textureUrl} alt="" /><b>{index + 1}</b><span>{status}</span>
                </button>
                <button className="cast-remove" onClick={() => removeFigure(index)} aria-label={`Remove figure ${index + 1}`}>×</button>
              </article>;
            })}</div>
          </div> : null}

          <div className={`camera-frame step-${step} world-${world}${paintStudioOpen ? " paint-mode" : ""}`} onPointerDown={handleStagePointerDown} onPointerMove={handleStagePointerMove} onPointerUp={handleStagePointerUp} onPointerCancel={handleStagePointerCancel}>
            <video ref={videoRef} className={cameraState === "active" ? "camera-video visible" : "camera-video"} autoPlay muted playsInline aria-label="Live local camera preview" />
            {cameraState !== "active" && world === "studio" ? <div className="demo-room"><span className="frame-a" /><span className="frame-b" /><span className="shelf" /><span className="plant" /><span className="baseboard" /></div> : null}
            {capture && cameraState !== "active" && world === "studio" ? <img className="captured-room" src={capture.previewUrl} alt="Original drawing scene" /> : null}
            {capture && cameraState !== "active" && !character.created ? <div className="cutout-review" onPointerDown={(event) => event.stopPropagation()}><img src={capture.textureUrl} alt="Isolated character cutout to review" /><span>{captureEnsemble.length > 1 ? `${captureEnsemble.length} SEPARATE FIGURES FOUND` : "IS THE WHOLE CHARACTER VISIBLE?"}</span><div><button onClick={requestNeuralConsent}>YES · CONTINUE</button><button onClick={() => capture.sourceScope === "camera" ? startCamera() : uploadRef.current?.click()}>NO · TRY AGAIN</button></div></div> : null}
            {step === "camera" ? <><div className="capture-guide"><span /><b>TAP EACH CHARACTER · THEN CAPTURE</b></div>{cameraTargets.map((target, index) => <div key={`${target.x}-${target.y}-${index}`} className="capture-target camera-cast-target" style={{ left: `${target.x * 100}%`, top: `${target.y * 100}%` }}><i>{index + 1}</i></div>)}</> : null}
            {character.created ? <Suspense fallback={<div className="three-layer" aria-hidden="true" />}>
              <ARStage ref={stageRef} characters={character.created && !neuralAsset ? captureEnsemble : null} contour={capture?.contour ?? null} skeleton={capture?.skeleton ?? null} textureUrl={capture?.textureUrl ?? null} rig={capture?.rig ?? null} depth={capture?.depthRecognition ?? null} action={character.action} ensembleActions={ensembleActions} world={world} lightingMood={lightingMood} cameraPreset={cameraPreset} accent={character.accent} inflation={character.inflation} neuralAssetUrl={neuralAsset?.meshUrl ?? null} paintEnabled={paintStudioOpen} visible onCapability={handleARCapability} onRendererCapability={handleRendererCapability} onPlaced={handleARPlaced} onNeuralAssetInfo={handleRiggedAssetInfo} onWorldInteraction={handleStageWorldInteraction} />
            </Suspense> : null}
            {character.created && !paintStudioOpen ? <button className="paint-studio-launch" onClick={() => { setPaintStudioOpen(true); setNotice("Touch the 3D character to paint. Choose a texture and color first."); }} onPointerDown={(event) => event.stopPropagation()}><i>🎨</i> PAINT 3D</button> : null}
            {character.created && paintStudioOpen ? <section className="model-paint-studio" aria-label="Paint directly on the 3D character" onPointerDown={(event) => event.stopPropagation()}>
              <header><div><span>PAINT THE 3D</span><b>{paintInspection.strokeCount ? `${paintInspection.strokeCount} stroke${paintInspection.strokeCount === 1 ? "" : "s"}` : "Touch the character"}</b></div><button onClick={() => { setPaintStudioOpen(false); setNotice("Paint saved on this 3D character. Drag the scene to orbit again."); }} aria-label="Finish painting">DONE</button></header>
              {paintAdventure?.status === "awaiting-child" ? <div className="paint-adventure" role="dialog" aria-label="Agent paint idea">
                <span>CHATGPT IDEA · YOU DECIDE</span><h3>{paintAdventure.title}</h3><p>{paintAdventure.prompt}</p>
                <div>{paintAdventure.steps.map((paintStep, index) => <i key={`${paintAdventure.id}-${index}`}><b>{index + 1}</b>{paintStep}</i>)}</div>
                <footer><button onClick={beginPaintAdventure}>START PAINTING</button><button onClick={dismissPaintAdventure}>NOT NOW</button></footer>
              </div> : null}
              <div className="paint-tools" role="group" aria-label="Paint texture">{modelPaintTools.map((item) => <button key={item.id} className={paintBrush.tool === item.id ? "active" : ""} onClick={() => choosePaintBrush({ ...paintBrushRef.current, tool: item.id })}><i>{item.glyph}</i>{item.label}</button>)}</div>
              <div className="paint-colors" role="group" aria-label="Paint color">{[...new Set([...(paintAdventure?.palette ?? []), ...modelPaintPalette])].slice(0, 10).map((color) => <button key={color} className={paintBrush.color === color ? "active" : ""} style={{ "--paint-color": color } as CSSProperties} onClick={() => choosePaintBrush({ ...paintBrushRef.current, color })} aria-label={`Choose ${color}`} />)}</div>
              <label className="paint-size"><span>SIZE</span><input type="range" min="0.08" max="1" step="0.02" value={paintBrush.size} onChange={(event) => choosePaintBrush({ ...paintBrushRef.current, size: Number(event.target.value) })} /><i style={{ "--brush-size": `${10 + paintBrush.size * 20}px`, "--paint-color": paintBrush.color } as CSSProperties} /></label>
              <footer className="paint-history"><button onClick={undoModelPaint} disabled={!paintInspection.strokeCount}>↶ UNDO</button><button onClick={resetModelPaint} disabled={!paintInspection.strokeCount}>CLEAR</button><small>Original stays safe</small></footer>
            </section> : null}
            {neuralConsentVisible ? <div className="neural-consent" role="dialog" aria-modal="true" aria-labelledby="neural-consent-title" onPointerDown={(event) => event.stopPropagation()}>
              <span>CHOOSE YOUR QUALITY</span><h2 id="neural-consent-title">Wake {captureEnsemble.length > 1 ? "the whole cast" : "this drawing"}</h2><p>{ensembleReadiness.some((report) => !report.motionReady) ? "A clean static spatial puppet is available now. Check every uncertain rig before expecting arms or legs to move." : captureEnsemble.length > 1 ? "Every selected figure passed the motion gate. Full sculpt currently handles one figure at a time." : "Use an instant private spatial puppet—or approve a full AI sculpt with generated unseen views."}</p><div><button onClick={startLocalReconstruction}>{ensembleReadiness.every((report) => report.motionReady) ? "PLAYABLE PUPPET · PRIVATE" : "STATIC PUPPET · PRIVATE"}</button>{captureEnsemble.length === 1 ? <button onClick={startNeuralReconstruction}>FULL 3D SCULPT · AI</button> : null}<button onClick={() => { setNeuralConsentVisible(false); openRigEditor(ensembleReadiness.findIndex((report) => !report.motionReady) < 0 ? 0 : ensembleReadiness.findIndex((report) => !report.motionReady)); }}>CHECK SKELETON</button></div>
            </div> : null}
            {neuralBusy ? <div className="neural-progress" role="status" onPointerDown={(event) => event.stopPropagation()}><span>ANIGEN · RIGGED 3D</span><b>{neuralProgress.message}</b><div><i style={{ width: `${Math.round(neuralProgress.progress * 100)}%` }} /></div><small>{Math.round(neuralProgress.progress * 100)}% · PUBLIC GPU</small></div> : null}
            {magicShowPlan?.status === "awaiting-human-approval" ? <div className="magic-show-approval" role="dialog" aria-modal="false" aria-labelledby="magic-show-title" onPointerDown={(event) => event.stopPropagation()}>
              <div><span>{magicShowPlan.stagedBy === "BROWSER AGENT" ? "AGENT STAGED" : "GUIDED DEMO"} · YOU DECIDE</span><b>{magicShowPlan.tone} · {magicShowPlan.beats.length} beats</b></div>
              <h2 id="magic-show-title">{magicShowPlan.title}</h2>
              <p>{magicShowPlan.cast.map((member) => `${member.name} · ${member.role}`).join("  /  ")}</p>
              <p className="show-learning-goal"><b>LEARNING GOAL</b>{magicShowPlan.learningGoal}</p>
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
            <div className="camera-hud"><span><i /> {cameraState === "active" ? "LIVE CAMERA · LOCAL" : neuralAsset && character.created ? rendererAvailable === false ? "FULL NEURAL ASSET · PREVIEW PAUSED" : `FULL NEURAL RIG · ${riggedAssetInfo?.bones ?? "…"} BONES` : character.created ? `${captureEnsemble.length} INSTANT SPATIAL PUPPET${captureEnsemble.length === 1 ? "" : "S"}` : capture ? "CUTOUT REVIEW · LOCAL" : "SAFE DEMO ROOM"}</span><strong>{immersiveAR ? "WEBXR READY" : rendererAvailable === false ? "ACCESSIBLE STORY MODE" : `${world.toUpperCase()} · INTERACTIVE 3D`}</strong></div>
            {character.created && storyCaption ? <div className="story-caption"><span>{character.storyTitle || "LIVE MOMENT"}</span><p>{storyCaption}</p></div> : null}
            {cameraState === "denied" || cameraState === "unavailable" ? <div className="camera-message"><b>CAMERA OPTIONAL</b><p>The demo doodle still proves the complete WebMCP and 3D workflow.</p></div> : null}
          </div>

          {character.created ? <div className="world-switcher" aria-label="Choose a 3D world"><div><span>CHOOSE A WORLD</span><small>{worlds.find((item) => item.id === world)?.label}</small></div>{worlds.map((item) => <button key={item.id} className={world === item.id ? "active" : ""} onClick={() => changeWorld(item.id, "CHILD")}><i>{item.id === "studio" ? "⌂" : item.id === "storybook" ? "♜" : item.id === "wizard" ? "✦" : "◉"}</i>{item.short}</button>)}</div> : null}

          {character.created ? <section className={`world-quest ${worldInteractions[world].length >= worldActivities[world].objectIds.length ? "is-complete" : ""}`} aria-label={`${worldActivities[world].title} interactive activity`}>
            <div className="quest-copy"><span>PLAY THIS WORLD</span><h3>{worldActivities[world].title}</h3><p>{worldInteractions[world].length >= worldActivities[world].objectIds.length ? worldActivities[world].reward : worldActivities[world].prompt}</p></div>
            <div className="quest-objects">{worldActivities[world].objectIds.map((id, index) => <button key={id} className={worldInteractions[world].includes(id) ? "found" : ""} onClick={() => interactWithWorldObject(id, "CHILD")}><i>{worldInteractions[world].includes(id) ? "✓" : index + 1}</i><span>{id.split("-").slice(1).join(" ").replace(/[0-9]/g, "").trim()}</span></button>)}</div>
            <div className="quest-progress"><i style={{ width: `${worldInteractions[world].length / worldActivities[world].objectIds.length * 100}%` }} /><span>{worldInteractions[world].length}/{worldActivities[world].objectIds.length}</span></div>
            {lastWorldMoment?.world === world ? <small className="quest-moment">{lastWorldMoment.story}</small> : null}
          </section> : null}

          {character.created ? <div className="cinematic-switcher">
            <div><span>LIGHT</span>{lightingMoods.map((mood) => <button key={mood} className={lightingMood === mood ? "active" : ""} onClick={() => { lightingMoodRef.current = mood; setLightingMood(mood); record("CHILD", "Changed cinematic lighting", mood); }}>{mood.replace("-", " ")}</button>)}</div>
            <div><span>CAMERA</span>{cameraPresets.map((preset) => <button key={preset} className={cameraPreset === preset ? "active" : ""} onClick={() => { cameraPresetRef.current = preset; setCameraPreset(preset); record("CHILD", "Changed camera preset", preset); }}>{preset.replaceAll("-", " ")}</button>)}</div>
          </div> : null}

          {character.created ? <div className="action-tray">
            <div><span>CHARACTER ACTIONS</span><small>{character.created ? `${character.name.toUpperCase()} · ${character.personality.toUpperCase()}` : "WAKE A DRAWING TO PLAY"}</small></div>
            {actions.map((item) => <button key={item.action} disabled={!character.created || showPlaying || !sharedSupportedActions.has(item.action)} className={character.action === item.action ? "active" : ""} title={sharedSupportedActions.has(item.action) ? `${item.label} with the verified rig` : `${item.label} needs a verified skeleton branch`} onClick={() => animateCharacter(item.action, "CHILD")}><i>{item.glyph}</i>{item.label}</button>)}
          </div> : null}
          <p className="placement-tip">{character.created ? neuralAsset ? "Full sculpt · generated unseen views · verified rig" : "Instant puppet · artwork preserved · only verified branches move" : capture ? "Check the cutout and skeleton, then choose instant puppet or full AI sculpt" : "Photograph a clear figure—uncertain recognition is blocked before 3D"}</p>
          <div className="learning-loop" aria-label="WallAlive learning loop"><b>LEARNING LOOP</b><span>Imagine</span><i>→</i><span>Sequence</span><i>→</i><span>Perform</span><i>→</i><span>Reflect</span></div>
          {character.created ? <button className={`story-passport-peek phase-${learningProgress.phase}`} onClick={() => { setInspectorOpen(true); setPanelTab("learning"); }}>
            <span>STORY PASSPORT</span>
            <b>{learningReflection ? "Reflection saved" : completedShowBeats ? "Tell it back" : magicShowPlan ? "Story in progress" : "Make thinking visible"}</b>
            <i>{magicShowPlan ? `${completedShowBeats}/${magicShowPlan.beats.length} beats` : "OPEN"} →</i>
          </button> : null}
        </section>

        <aside className={`agent-panel ${inspectorOpen ? "is-open" : ""}`} aria-hidden={!inspectorOpen}>
          <button className="inspector-close" onClick={() => setInspectorOpen(false)} aria-label="Close WebMCP inspector">×</button>
          <div className="right-tabs" role="tablist" aria-label="WallAlive inspector">
            {(["agent", "learning", "tools", "commerce", "privacy", "history"] as const).map((tab) => <button key={tab} role="tab" aria-selected={panelTab === tab} className={panelTab === tab ? "active" : ""} onClick={() => setPanelTab(tab)}>{tab}</button>)}
          </div>

          {panelTab === "agent" ? (
            <div className="panel-body">
              <div className="agent-status"><div><i /> BROWSER AGENT</div><span>{webMcpLabel}</span></div>
              <p className="kicker">SHARED IMAGINATION</p>
              <h2>{agentLine}</h2>
              <p>The agent reads each real rig, assigns compatible roles, and stages ensemble choreography. The child approves the final performance.</p>
              <div className="agent-call"><span>↳</span><div><b>{latestAgentActivity?.toolName ?? "inspect_creative_scene"}</b><small>{latestAgentActivity?.detail ?? "Shared state visible · Camera private"}</small></div></div>
              <blockquote>“{suggestedJudgePrompt}”</blockquote>
              <button className="copy-prompt" onClick={copyDemoPrompt}>COPY FAMILY PROMPT <span>⧉</span></button>
            </div>
          ) : null}

          {panelTab === "learning" ? (
            <div className="panel-body learning-panel">
              <p className="kicker">STORY PASSPORT · PRIVATE LEARNING EVIDENCE</p>
              <h2>{learningProgress.story?.title ?? "Plan it. Play it. Tell it back."}</h2>
              <p>{learningProgress.story?.learningGoal ?? "Complete a short story or world quest, then capture the learner's own retell and next creative choice."}</p>
              <div className="learning-metrics" aria-label="Observed story progress">
                <div><b>{learningProgress.story ? `${learningProgress.story.completedBeats}/${learningProgress.story.plannedBeats}` : "0"}</b><span>beats performed</span></div>
                <div><b>{learningProgress.observedEvidence.completedWorldInteractions}</b><span>world moments</span></div>
                <div><b>{learningProgress.observedEvidence.participantCount}</b><span>creator{learningProgress.observedEvidence.participantCount === 1 ? "" : "s"}</span></div>
              </div>
              {learningProgress.phase === "performed-needs-reflection" || learningProgress.phase === "reflected" ? <section className="reflection-routine">
                <label htmlFor="story-retell"><span>WHAT HAPPENED?</span><small>Try “First… then… finally…”</small></label>
                <textarea id="story-retell" value={reflectionRetell} maxLength={360} onChange={(event) => setReflectionRetell(event.target.value)} placeholder="First my character… Then… Finally…" />
                <fieldset><legend>What should change next?</legend>{([
                  ["new-ending", "New ending"],
                  ["new-feeling", "New feeling"],
                  ["add-friend", "Add a friend"],
                ] as const).map(([value, label]) => <button type="button" key={value} className={reflectionNextChange === value ? "active" : ""} onClick={() => setReflectionNextChange(value)}>{label}</button>)}</fieldset>
                <button className="save-reflection" onClick={saveLearningReflection}>{learningReflection ? "UPDATE REFLECTION" : "SAVE MY REFLECTION"} <span>✦</span></button>
              </section> : <div className="learning-next"><i>→</i><span><b>Next</b>{learningProgress.suggestedNextScaffold}</span></div>}
              {learningReflection ? <button className="download-passport" onClick={downloadLearningEvidence}>DOWNLOAD TEACHER / PARENT NOTE <span>↓</span></button> : null}
              <small className="evidence-boundary">Evidence, not a grade. Saved in this tab; no camera frames or artwork pixels.</small>
            </div>
          ) : null}

          {panelTab === "tools" ? (
            <div className="panel-body">
              <p className="kicker">WEBMCP INSPECTOR</p><h2>Agent sees uncertainty. You fix it.</h2><p>Inspect each figure → request visible repair → adapt the next challenge → human approves → perform and reflect. Camera pixels, correction, grading, publishing, and purchase stay outside the tool surface.</p>
              <div className="tools-list">{toolNames.map(([name, mode], index) => <div key={name}><span>{String(index + 1).padStart(2, "0")}</span><code>{name}</code><i>{mode}</i></div>)}</div>
            </div>
          ) : null}

          {panelTab === "commerce" ? (
            <div className="panel-body commerce-panel">
              <p className="kicker">CREATOR PRODUCTS · OFFLINE IMPORT KIT</p>
              <h2>{creatorDrop?.name ?? "Turn imagination into a tiny collection."}</h2>
              <p>{capture ? "The agent studies the approved artwork, explains which products fit, and designs a draft storefront. An adult owns the final export." : "Approve a drawing first. Then the Creator Shop can recommend products without exposing camera frames or image pixels to the agent."}</p>

              {capture ? <>
                <div className="creator-intent" aria-label="Creator collection direction">
                  <label><span>For</span><select value={creatorAudience} onChange={(event) => { setCreatorAudience(event.target.value as CreatorAudience); setCreatorDrop(null); }}><option value="family">Family</option><option value="classroom">Classroom</option><option value="community">Community</option></select></label>
                  <label><span>Goal</span><select value={creatorGoal} onChange={(event) => { setCreatorGoal(event.target.value as CreatorGoal); setCreatorDrop(null); }}><option value="keepsake">Keepsake</option><option value="fundraiser">Fundraiser</option><option value="portfolio">Portfolio</option></select></label>
                  <label><span>Vibe</span><select value={creatorVibe} onChange={(event) => { setCreatorVibe(event.target.value as CreatorVibe); setCreatorDrop(null); }}><option value="sunny">Sunny</option><option value="storybook">Storybook</option><option value="bold">Bold</option><option value="museum">Museum</option></select></label>
                </div>

                {!creatorRecommendations.length ? <button className="creator-agent-action" onClick={() => recommendProductsForArtwork({ audience: creatorAudience, goal: creatorGoal }, "CHILD")}><span>✦</span><b>Ask the agent what fits</b><i>→</i></button> : null}

                {creatorRecommendations.length ? <div className="recommendation-shelf" aria-label="Agent product recommendations">
                  {creatorRecommendations.slice(0, 4).map((product, index) => <article key={product.id} className={index === 0 ? "top-pick" : ""}>
                    <div><i>{product.glyph}</i><span>{index === 0 ? "BEST FIT" : `${product.score}% FIT`}</span></div>
                    <h3>{product.label}</h3>
                    <p>{product.reason}</p>
                    <small>${product.price} draft price · {product.placement}</small>
                  </article>)}
                </div> : null}

                {creatorRecommendations.length && !creatorDrop ? <button className="creator-agent-action is-primary" onClick={() => stageShopifyImportKit({
                  audience: creatorAudience,
                  goal: creatorGoal,
                  vibe: creatorVibe,
                  dropName: `${character.name || "My Drawing"}’s Little World`,
                  story: `${character.name || "A new friend"} began as a drawing and learned how to move, tell a story, and step into a tiny collection.`,
                  productIds: creatorRecommendations.slice(0, 3).map((product) => product.id),
                }, "CHILD")}><span>✦</span><b>Build my Creator Drop</b><i>→</i></button> : null}

                {creatorDrop ? <section className={`creator-store-preview vibe-${creatorDrop.vibe}`} aria-label="Offline Shopify import preview">
                  <div className="store-browser"><i /><i /><i /><span>SHOPIFY IMPORT PREVIEW · NOT CONNECTED</span></div>
                  <div className="store-hero" style={{ "--store-accent": creatorDrop.palette.accent, "--store-highlight": creatorDrop.palette.highlight } as CSSProperties}>
                    <div><small>{creatorDrop.storefront.announcement}</small><h3>{creatorDrop.name}</h3><p>{creatorDrop.story}</p><span className="storefront-cta" aria-hidden="true">{creatorDrop.threeDExperience.enabled ? "EXPLORE THE 3D STORY" : "MEET THE COLLECTION"}</span></div>
                    <figure>{capture ? <img src={capture.textureUrl} alt="Approved artwork in the staged collection" /> : null}<i>✦</i>{creatorDrop.threeDExperience.enabled ? <b>360° 3D · {creatorDrop.threeDExperience.activeWorld}</b> : null}</figure>
                  </div>
                  <div className="store-products">{creatorDrop.products.map((product) => <div key={product.id}><i>{product.glyph}</i><span><b>{product.label}</b><small>${product.price}.00</small></span></div>)}</div>
                  {creatorDrop.contributors.length ? <div className="store-contributors"><span>CREATED TOGETHER</span>{creatorDrop.contributors.map((contributor) => <i key={contributor.username}>@{contributor.username}</i>)}</div> : null}
                </section> : null}

                {creatorDrop ? <div className="adult-export-boundary">
                  <div><i>✓</i><span><b>Adult + creator permissions first</b><small>{creatorDrop.contributors.length ? `Confirm permission for ${creatorDrop.contributors.length} credited creators. ` : ""}Nothing is published, purchased, or sent to Shopify.</small></span></div>
                  {!adultExportApproved ? <button onClick={approveCreatorExport}>ADULT · CONFIRM PERMISSIONS</button> : <div className="export-files">
                    <button onClick={() => downloadTextFile("wallalive-products.csv", buildShopifyProductsCsv(creatorDrop), "text/csv;charset=utf-8")}>Products CSV <span>↓</span></button>
                    <button onClick={() => downloadTextFile("wallalive-store-blueprint.json", buildShopifyStoreBlueprint(creatorDrop), "application/json")}>Store blueprint <span>↓</span></button>
                    <button onClick={() => downloadTextFile("wallalive-adult-handoff.md", buildCreatorHandoff(creatorDrop), "text/markdown;charset=utf-8")}>Adult checklist <span>↓</span></button>
                    {capture ? <a href={capture.textureUrl} download="wallalive-print-art.png">Print artwork <span>↓</span></a> : null}
                    {neuralAsset?.meshUrl ? <a href={neuralAsset.meshUrl} download="wallalive-rigged-character.glb">Rigged 3D model <span>↓</span></a> : null}
                  </div>}
                </div> : null}

                <div className="shopify-handoff"><span>WALLALIVE</span><i>→</i><span>IMPORT FILES</span><i>→</i><span>ADULT REVIEW</span><i>→</i><b>STORE NOT CONNECTED</b></div>
              </> : <button className="creator-agent-action" onClick={() => setInspectorOpen(false)}><span>✦</span><b>Make a drawing first</b><i>→</i></button>}
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
          <footer className="agent-footer"><span>OPENAI SITES · CLOUDFLARE D1 · CHROME WEBMCP</span><b>HUMANS APPROVE</b></footer>
        </aside>
      </section>
      <DrawingWall
        open={drawingWallOpen}
        onClose={() => setDrawingWallOpen(false)}
        onMake3D={processWallDrawing}
        sharedSession={sharedRoom.session}
        sharedParticipants={sharedRoom.participants}
        sharedOperations={sharedRoom.operations}
        onSharedOperation={sharedRoom.appendOperation}
      />
          <SharedRoomPanel
            key={`${invitedRoom}:${invitedUsername}`}
            open={sharedRoomVisible}
        session={sharedRoom.session}
        participants={sharedRoom.participants}
        status={sharedRoom.status}
        message={sharedRoom.message}
        invitedRoom={invitedRoom}
        invitedUsername={invitedUsername}
            onClose={() => { setSharedRoomOpen(false); setDismissedInvite(search); }}
        onCreate={async (username) => { const result = await sharedRoom.createRoom(username); record("CHILD", "Opened a shared drawing room", `Room ${result.roomId} is ready for invited creators.`); }}
        onJoin={async (roomId, username) => { const result = await sharedRoom.joinRoom(roomId, username); record("CHILD", "Joined a shared drawing room", `@${result.username} joined room ${result.roomId}.`); }}
        onInvite={async (username) => { const result = await sharedRoom.prepareInvite(username); record("CHILD", "Prepared a private room invite", `Invite prepared for @${result.friend}; WallAlive did not send a message.`); return result; }}
        onLeave={() => { sharedRoom.leaveRoom(); record("CHILD", "Left the shared room", "The local artwork stayed in this browser tab."); }}
        onOpenWall={() => { setSharedRoomOpen(false); setDismissedInvite(search); setDrawingWallOpen(true); }}
      />
      {pendingUpload ? <div className="paper-picker-backdrop" role="dialog" aria-modal="true" aria-labelledby="paper-picker-title">
        <section className="paper-picker">
          <header><div><span>SELECT THE CAST · UP TO 6</span><h2 id="paper-picker-title">Tap every character.</h2></div><button onClick={cancelPendingUpload} aria-label="Close photo">×</button></header>
          <div className="paper-picker-image"><button
            className="paper-picker-canvas"
            aria-label="Tap the center of each character to select it"
            onClick={(event) => {
              const bounds = event.currentTarget.getBoundingClientRect();
              const target = { x: clamp01((event.clientX - bounds.left) / bounds.width), y: clamp01((event.clientY - bounds.top) / bounds.height) };
              setPendingUploadTargets((current) => appendCaptureTarget(current, target, 6));
            }}
          ><img src={pendingUpload.url} alt="Uploaded artwork with selected character markers" />{pendingUploadTargets.map((target, index) => <i key={`${target.x}-${target.y}`} className="paper-target" style={{ left: `${target.x * 100}%`, top: `${target.y * 100}%` }}>{index + 1}</i>)}</button></div>
          <footer className="paper-picker-actions"><button disabled={!pendingUploadTargets.length} onClick={processUploadedDrawing}>SCAN {pendingUploadTargets.length || ""} CHARACTER{pendingUploadTargets.length === 1 ? "" : "S"}</button><button disabled={!pendingUploadTargets.length} onClick={() => setPendingUploadTargets((current) => current.slice(0, -1))}>UNDO</button><button disabled={!pendingUploadTargets.length} onClick={() => setPendingUploadTargets([])}>CLEAR</button></footer>
          <p>Only numbered selections become characters. Labels, paper edges, and nearby decoration are ignored.</p>
        </section>
      </div> : null}
      {partEditorOpen && activeRigDrawing ? <div className="part-editor-backdrop" role="dialog" aria-modal="true" aria-labelledby="part-editor-title">
        <section className="part-editor">
          <header><div><span>ANATOMY CHECK · FIGURE {activeFigureIndex + 1}/{captureEnsemble.length || 1}</span><h2 id="part-editor-title">Make it match.</h2></div><button onClick={() => { setPartEditorOpen(false); setPendingPartKind(null); }}>×</button></header>
          {captureEnsemble.length > 1 ? <nav className="part-editor-cast" aria-label="Choose figure to repair">{captureEnsemble.map((figure, index) => <button key={index} className={activeFigureIndex === index ? "active" : ""} onClick={() => selectRigFigure(index)}><img src={figure.textureUrl} alt="" /><span>FIGURE {index + 1}</span><small>{ensembleReadiness[index]?.motionReady ? "READY" : "CHECK"}</small></button>)}</nav> : null}
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
              <image href={activeRigDrawing.textureUrl} x="0" y="0" width="100" height="100" preserveAspectRatio="xMidYMid meet" />
              {activeRigDrawing.rig.parts.filter((part) => anatomyKinds.includes(part.kind as (typeof anatomyKinds)[number])).map((part) => {
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
