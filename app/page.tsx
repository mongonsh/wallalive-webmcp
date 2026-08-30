/* eslint-disable @next/next/no-img-element */
"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ARStage, type ARStageHandle, type BodyShape, type CharacterAction } from "./components/ARStage";
import { createDemoDoodle, extractDrawingFromVideo, type DrawingExtraction } from "./lib/drawing";

type Actor = "CHILD" | "BROWSER AGENT" | "WALLALIVE";
type AppStep = "ready" | "camera" | "captured" | "alive";
type CameraState = "idle" | "requesting" | "active" | "denied" | "unavailable";
type PanelTab = "agent" | "tools" | "privacy" | "history";

type CharacterState = {
  created: boolean;
  name: string;
  personality: string;
  bodyShape: BodyShape;
  eyeStyle: "curious" | "sleepy" | "sparkly";
  accent: string;
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

type WebMCPTool = {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute: (input: Record<string, unknown>, options: { signal: AbortSignal }) => Promise<unknown> | unknown;
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
  bodyShape: "round",
  eyeStyle: "curious",
  accent: "#5fc7df",
  action: "idle",
  surface: "screen",
  scale: 1,
  storyTitle: "",
};

const toolNames = [
  ["inspect_wall_scene", "READ"],
  ["create_character_from_drawing", "WRITE"],
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
  const streamRef = useRef<MediaStream | null>(null);
  const stageRef = useRef<ARStageHandle>(null);
  const captureRef = useRef<DrawingExtraction | null>(null);
  const characterRef = useRef<CharacterState>(initialCharacter);
  const activityRef = useRef<Activity[]>([]);

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

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("idle");
    setStep(characterRef.current.created ? "alive" : captureRef.current ? "captured" : "ready");
    setNotice("Camera stopped. The approved drawing and character remain only in this tab.");
  }, []);

  useEffect(() => () => streamRef.current?.getTracks().forEach((track) => track.stop()), []);

  const startCamera = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraState("unavailable");
      setNotice("This browser cannot open a camera. The demo doodle still shows the complete experience.");
      return;
    }
    setCameraState("requesting");
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
      setNotice("Camera is live locally. Center one drawing inside the dotted frame.");
      record("CHILD", "Opened the camera", "Video stays on this device and is never exposed as a WebMCP tool.");
    } catch (error) {
      setCameraState("denied");
      setNotice(error instanceof Error && error.name === "NotAllowedError" ? "Camera permission was not granted. Try the demo doodle instead." : "The camera could not start. Try the demo doodle instead.");
    }
  }, [record]);

  const setDrawing = useCallback((next: DrawingExtraction, source: "camera" | "demo") => {
    captureRef.current = next;
    setCapture(next);
    setStep("captured");
    setNotice(source === "camera" ? "Drawing isolated locally. Press Wake it up to add depth and personality." : "Demo drawing ready. Press Wake it up—or run the full magic demo.");
    setAgentLine(`I found a ${next.analysis.shapeHint}, ${next.analysis.edgeEnergy} drawing. Its strongest color is ${next.analysis.dominantColor}.`);
    record("WALLALIVE", "Isolated a drawing", `${next.analysis.coveragePercent}% foreground · ${next.analysis.shapeHint} silhouette · no upload.`);
  }, [record]);

  const captureDrawing = useCallback(() => {
    if (!videoRef.current) return;
    try {
      setDrawing(extractDrawingFromVideo(videoRef.current), "camera");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The drawing could not be separated from the wall.");
    }
  }, [setDrawing]);

  const loadDemoDrawing = useCallback(() => {
    try {
      const demo = createDemoDoodle();
      setDrawing(demo, "demo");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The demo drawing could not be created.");
    }
  }, [setDrawing]);

  const createCharacter = useCallback((input: Record<string, unknown>, actor: Actor, toolName?: string) => {
    const drawing = captureRef.current;
    if (!drawing) throw new Error("No drawing is approved. The child must capture or choose a drawing first.");
    const requestedShape = stringValue(input.bodyShape, drawing.analysis.shapeHint, 20);
    const bodyShape: BodyShape = ["round", "tall", "wide", "spiky"].includes(requestedShape) ? requestedShape as BodyShape : drawing.analysis.shapeHint;
    const eyeStyleInput = stringValue(input.eyeStyle, "curious", 20);
    const eyeStyle = ["curious", "sleepy", "sparkly"].includes(eyeStyleInput) ? eyeStyleInput as CharacterState["eyeStyle"] : "curious";
    const next: CharacterState = {
      ...characterRef.current,
      created: true,
      name: stringValue(input.name, "Pip", 40),
      personality: stringValue(input.personality, "curious and kind", 120),
      bodyShape,
      eyeStyle,
      accent: stringValue(input.accent, drawing.analysis.secondaryColor, 20),
      action: "idle",
      storyTitle: "",
    };
    commitCharacter(next, `${next.name} is alive in 3D.`);
    setStep("alive");
    setAgentLine(`${next.name} feels ${next.personality}. I’ll protect the original colors while we play.`);
    setStoryCaption(`${next.name} blinks for the very first time.`);
    record(actor, "Woke the drawing", `${next.name} · ${next.bodyShape} body · ${next.personality}.`, toolName);
    return next;
  }, [commitCharacter, record]);

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
    record(actor, "Changed the 3D accent", `Applied ${safeAccent} only to generated depth and limbs.`, toolName);
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
    return { title: storyTitle, beatsPlayed: Math.min(4, beats.length), finalAction: "idle" };
  }, [animateCharacter, commitCharacter, record]);

  const inspectScene = useCallback(() => ({
    drawingApproved: Boolean(captureRef.current),
    drawingAnalysis: captureRef.current?.analysis ?? null,
    character: { ...characterRef.current, textureUrl: undefined },
    cameraFeedExposed: false,
    privacyBoundary: "Camera capture is human-only. WebMCP tools receive semantic drawing analysis, never live frames or image data.",
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
    const tools: WebMCPTool[] = [
      {
        name: "inspect_wall_scene",
        title: "Inspect approved wall drawing",
        description: "Read semantic details about the human-approved drawing, character, AR capability, and privacy boundary. Never returns camera frames or image data.",
        inputSchema: { ...base, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, { signal }) => { guard(signal); return ok({ scene: inspectScene() }); },
      },
      {
        name: "create_character_from_drawing",
        title: "Wake approved drawing",
        description: "Turn the current human-approved drawing into a local 2.5D character. This cannot open the camera or capture a new image.",
        inputSchema: {
          ...base,
          properties: {
            name: { type: "string", minLength: 1, maxLength: 40 },
            personality: { type: "string", minLength: 1, maxLength: 120 },
            bodyShape: { type: "string", enum: ["round", "tall", "wide", "spiky"] },
            eyeStyle: { type: "string", enum: ["curious", "sleepy", "sparkly"] },
            accent: { type: "string", pattern: "^#[0-9a-fA-F]{6}$" },
          },
          required: ["name", "personality", "bodyShape", "eyeStyle", "accent"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => { try { guard(signal); return ok({ character: createCharacter(input, "BROWSER AGENT", "create_character_from_drawing") }); } catch (error) { return fail(error); } },
      },
      {
        name: "set_character_personality",
        title: "Set character personality",
        description: "Change how the character is described and performed without altering the child's captured artwork.",
        inputSchema: { ...base, properties: { personality: { type: "string", minLength: 1, maxLength: 120 } }, required: ["personality"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => { try { guard(signal); return ok({ character: setPersonality(stringValue(input.personality), "BROWSER AGENT", "set_character_personality") }); } catch (error) { return fail(error); } },
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
        execute: async (input, { signal }) => { try { guard(signal); return ok({ character: placeCharacter(numberValue(input.x, .5), numberValue(input.y, .5), ["wall", "floor"].includes(String(input.surface)) ? input.surface as "wall" | "floor" : "screen", numberValue(input.scale, 1), "BROWSER AGENT", "place_character") }); } catch (error) { return fail(error); } },
      },
      {
        name: "animate_character",
        title: "Animate character",
        description: "Play one safe visible animation on the created character. Does not navigate, capture, upload, or modify the original drawing.",
        inputSchema: { ...base, properties: { action: { type: "string", enum: ["idle", "wave", "dance", "hop", "walk", "hide", "spin"] }, caption: { type: "string", maxLength: 120 } }, required: ["action"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => { try { guard(signal); const action = stringValue(input.action, "idle") as CharacterAction; return ok({ character: animateCharacter(action, "BROWSER AGENT", "animate_character", stringValue(input.caption) || undefined) }); } catch (error) { return fail(error); } },
      },
      {
        name: "recolor_character",
        title: "Recolor generated depth",
        description: "Change only the generated 3D edge and limb accent. The child's original drawing pixels remain unchanged.",
        inputSchema: { ...base, properties: { accent: { type: "string", pattern: "^#[0-9a-fA-F]{6}$" } }, required: ["accent"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => { try { guard(signal); return ok({ character: recolorCharacter(stringValue(input.accent), "BROWSER AGENT", "recolor_character") }); } catch (error) { return fail(error); } },
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
        execute: async (input, { signal }) => { try { guard(signal); const beats = Array.isArray(input.beats) ? input.beats.filter(isRecord) : []; return ok({ story: await runStory(stringValue(input.title), beats, "BROWSER AGENT", "tell_character_story", signal), character: characterRef.current }); } catch (error) { return fail(error); } },
      },
      {
        name: "list_activity",
        title: "List human-agent activity",
        description: "Read recent attributed scene actions. Camera pixels and captured drawing data are intentionally excluded.",
        inputSchema: { ...base, properties: { limit: { type: "number", minimum: 1, maximum: 30 } } },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (input, { signal }) => { guard(signal); const limit = Math.min(30, Math.max(1, numberValue(input.limit, 12))); return ok({ activity: activityRef.current.slice(0, limit), cameraDataIncluded: false }); },
      },
    ];

    Promise.all(tools.map((tool) => Promise.resolve(context.registerTool(tool, { signal: controller.signal })))).then(() => {
      setWebMcpReady(true);
      setNotice(`${tools.length} WebMCP tools are ready. Camera capture remains human-only.`);
    }).catch(() => setWebMcpReady(false));
    return () => controller.abort();
  }, [animateCharacter, createCharacter, inspectScene, placeCharacter, recolorCharacter, runStory, setPersonality]);

  const runMagicDemo = useCallback(async () => {
    if (demoRunning) return;
    setDemoRunning(true);
    try {
      const demo = createDemoDoodle();
      setDrawing(demo, "demo");
      setAgentLine("1 / 4 · I can see the approved shape and colors—but not a live camera feed.");
      await wait(550);
      createCharacter({ name: "Pip", personality: "brave on the outside, shy on the inside", bodyShape: "round", eyeStyle: "curious", accent: "#5fc7df" }, "BROWSER AGENT", "create_character_from_drawing");
      setAgentLine("2 / 4 · Pip keeps every mark from the child’s drawing and gains real depth.");
      await wait(700);
      placeCharacter(.68, .53, "wall", 1, "BROWSER AGENT", "place_character");
      setAgentLine("3 / 4 · The agent places Pip without controlling the camera.");
      await wait(650);
      await runStory("Pip finds their courage", [
        { action: "hide", caption: "Pip hides at the edge of the wall.", durationMs: 800 },
        { action: "hop", caption: "One brave hop into the room.", durationMs: 800 },
        { action: "wave", caption: "Pip peeks out and waves hello.", durationMs: 1000 },
      ], "BROWSER AGENT", "tell_character_story");
      setAgentLine("4 / 4 · A drawing became a character, and the child stayed in control.");
    } finally {
      setDemoRunning(false);
    }
  }, [createCharacter, demoRunning, placeCharacter, runStory, setDrawing]);

  const enterAR = useCallback(async () => {
    const result = await stageRef.current?.enterImmersiveAR();
    if (result?.ok) {
      setNotice("Move your phone until a ring appears, then tap a real surface to place the character.");
      record("CHILD", "Entered immersive AR", "Real-world placement uses device hit testing.");
    } else setNotice(result?.error ?? "Immersive AR is unavailable; camera-overlay mode is active.");
  }, [record]);

  const handleStageClick = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
    if (!characterRef.current.created) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width;
    const y = (event.clientY - bounds.top) / bounds.height;
    placeCharacter(x, y, "screen", characterRef.current.scale, "CHILD");
  }, [placeCharacter]);

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

  const latestAgentActivity = useMemo(() => activity.find((item) => item.actor === "BROWSER AGENT"), [activity]);
  const primaryButton = cameraState === "active" ? { label: "CAPTURE DRAWING", action: captureDrawing } : step === "captured" ? { label: "WAKE IT UP", action: () => createCharacter({}, "CHILD") } : { label: "START CAMERA", action: startCamera };
  const stepIndex = step === "ready" ? 0 : step === "camera" ? 1 : step === "captured" ? 2 : 3;

  return (
    <main className="alive-shell">
      <header className="alive-header">
        <a className="alive-brand" href="#play"><span>WALL</span>ALIVE<i>●</i></a>
        <p>YOUR DRAWING. YOUR ROOM. YOUR STORY.</p>
        <div className="header-actions">
          <div className={`ready-pill ${webMcpReady ? "is-ready" : ""}`}><i /> {webMcpReady ? "8 WEBMCP TOOLS" : "INTERACTIVE DEMO"}</div>
          <button className="judge-demo" onClick={runMagicDemo} disabled={demoRunning}>{demoRunning ? "PLAYING…" : "PLAY JUDGE DEMO"}</button>
        </div>
      </header>

      <section className="alive-layout" id="play">
        <aside className="steps-panel">
          <p className="kicker">THE MAGIC LOOP</p>
          <ol>
            <li className={stepIndex >= 1 ? "done" : "active"}><span>1</span><div><strong>Scan</strong><small>Human opens camera</small></div></li>
            <li className={stepIndex >= 3 ? "done" : stepIndex === 2 ? "active" : ""}><span>2</span><div><strong>Wake</strong><small>Drawing gains depth</small></div></li>
            <li className={stepIndex === 3 ? "active" : ""}><span>3</span><div><strong>Play</strong><small>Agent directs movement</small></div></li>
          </ol>

          {capture ? (
            <div className="drawing-fingerprint">
              <div className="drawing-thumb"><img src={capture.textureUrl} alt="Locally isolated drawing" /></div>
              <div><p className="kicker">DRAWING DNA</p><strong>{capture.analysis.shapeHint} · {capture.analysis.edgeEnergy}</strong><span><i style={{ background: capture.analysis.dominantColor }} /><i style={{ background: capture.analysis.secondaryColor }} /> {capture.analysis.coveragePercent}% INK</span></div>
            </div>
          ) : null}

          <div className="privacy-card">
            <div><b>CAMERA-SAFE BY DESIGN</b><span>◆</span></div>
            <p>Only the child can start or capture. The agent receives shapes and colors—not the live camera or image pixels.</p>
          </div>
          <button className="demo-doodle" onClick={loadDemoDrawing}>NO CAMERA? TRY A DEMO DOODLE <span>＋</span></button>
        </aside>

        <section className="magic-stage">
          <div className="stage-copy">
            <div><p className="kicker">LIVE CAMERA PLAYGROUND</p><h1>What if their drawing<br /><em>jumped off the wall?</em></h1></div>
            <div className="stage-ctas">
              {immersiveAR && character.created ? <button className="ar-button" onClick={enterAR}>ENTER REAL AR <span>◎</span></button> : null}
              {cameraState === "active" ? <button className="stop-camera" onClick={stopCamera}>STOP CAMERA</button> : null}
              <button className="primary-camera" onClick={primaryButton.action} disabled={cameraState === "requesting"}>{cameraState === "requesting" ? "OPENING…" : primaryButton.label}<span>↗</span></button>
            </div>
          </div>

          <div className={`camera-frame step-${step}`} onPointerDown={handleStageClick}>
            <video ref={videoRef} className={cameraState === "active" ? "camera-video visible" : "camera-video"} autoPlay muted playsInline aria-label="Live local camera preview" />
            {cameraState !== "active" ? <div className="demo-room"><span className="frame-a" /><span className="frame-b" /><span className="shelf" /><span className="plant" /><span className="baseboard" /></div> : null}
            {capture && cameraState !== "active" ? <img className="captured-room" src={capture.previewUrl} alt="Approved drawing preview" /> : null}
            {step === "camera" ? <div className="capture-guide"><span /><b>KEEP ONE DRAWING INSIDE</b></div> : null}
            <ARStage ref={stageRef} textureUrl={capture?.textureUrl ?? null} action={character.action} accent={character.accent} bodyShape={character.bodyShape} visible={character.created} onCapability={handleARCapability} onPlaced={handleARPlaced} />
            <div className="camera-hud"><span><i /> {cameraState === "active" ? "LIVE CAMERA · LOCAL" : character.created ? "3D CHARACTER LIVE" : "SAFE DEMO ROOM"}</span><strong>{immersiveAR ? "WEBXR READY" : "CAMERA AR FALLBACK"}</strong></div>
            {character.created ? <div className="story-caption"><span>{character.storyTitle || "LIVE MOMENT"}</span><p>{storyCaption}</p></div> : null}
            {cameraState === "denied" || cameraState === "unavailable" ? <div className="camera-message"><b>CAMERA OPTIONAL</b><p>The demo doodle still proves the complete WebMCP and 3D workflow.</p></div> : null}
          </div>

          <div className="action-tray">
            <div><span>CHARACTER ACTIONS</span><small>{character.created ? `${character.name.toUpperCase()} · ${character.personality.toUpperCase()}` : "WAKE A DRAWING TO PLAY"}</small></div>
            {actions.map((item) => <button key={item.action} disabled={!character.created} className={character.action === item.action ? "active" : ""} onClick={() => animateCharacter(item.action, "CHILD")}><i>{item.glyph}</i>{item.label}</button>)}
          </div>
          <p className="placement-tip">{character.created ? "Tap anywhere in the room to move the character · Try the action buttons or ask your browser agent" : "Use a bold, colorful drawing on a plain wall or sheet of paper for the cleanest capture"}</p>
        </section>

        <aside className="agent-panel">
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
              <p className="kicker">CHILD-SAFE BOUNDARY</p><h2>The camera is not a tool.</h2><p>WallAlive intentionally refuses to let an agent open, capture, or upload the camera.</p>
              <ul className="privacy-list"><li><b>Human gesture required</b><span>Start and capture are UI-only.</span></li><li><b>Local extraction</b><span>Pixels are processed in browser memory.</span></li><li><b>Semantic agent view</b><span>Only color, shape, and approved state.</span></li><li><b>Session-only art</b><span>No image is saved after the tab closes.</span></li></ul>
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
      <div className="notice" role="status" aria-live="polite"><i />{notice}</div>
    </main>
  );
}
