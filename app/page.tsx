"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

type Visual = "establish" | "arrival" | "insert" | "close" | "reverse" | "exit";
type Actor = "DIRECTOR" | "BROWSER AGENT" | "CUTROOM";

type Shot = {
  id: string;
  number: number;
  title: string;
  shotType: string;
  duration: number;
  description: string;
  visual: Visual;
  umbrellaVisible: boolean;
  screenDirection: "left" | "right";
  wardrobe: string;
  locked: boolean;
  createdBy: Actor;
};

type Branch = {
  id: string;
  name: string;
  note: string;
  shots: Shot[];
};

type Constraint = {
  id: string;
  label: string;
  rule: string;
  locked: boolean;
};

type Activity = {
  id: string;
  timestamp: string;
  actor: Actor;
  action: string;
  detail: string;
  toolName?: string;
};

type StudioState = {
  sceneTitle: string;
  branches: Branch[];
  activeBranchId: string;
  selectedShotId: string;
  constraints: Constraint[];
  activity: Activity[];
  version: number;
};

type ContinuityIssue = {
  shotId: string;
  shotNumber: number;
  rule: string;
  detail: string;
};

type ContinuityReport = {
  branchId: string;
  checkedShots: number;
  issues: ContinuityIssue[];
};

type WebMCPTool = {
  name: string;
  title: string;
  description: string;
  inputSchema: Record<string, unknown>;
  annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
  execute: (
    input: Record<string, unknown>,
    options: { signal: AbortSignal },
  ) => Promise<string> | string;
};

declare global {
  interface Document {
    modelContext?: {
      registerTool: (
        tool: WebMCPTool,
        options?: { signal?: AbortSignal },
      ) => Promise<void> | void;
    };
  }
}

const STORAGE_KEY = "cutroom-studio-v1";

const initialShots: Shot[] = [
  {
    id: "shot-01",
    number: 1,
    title: "Last wash",
    shotType: "WIDE",
    duration: 4,
    description: "Mara waits alone as the final washer slows to a stop.",
    visual: "establish",
    umbrellaVisible: true,
    screenDirection: "right",
    wardrobe: "navy coat",
    locked: true,
    createdBy: "DIRECTOR",
  },
  {
    id: "shot-02",
    number: 2,
    title: "A second shadow",
    shotType: "MEDIUM",
    duration: 3,
    description: "A stranger crosses the glass before the bell can ring.",
    visual: "arrival",
    umbrellaVisible: true,
    screenDirection: "right",
    wardrobe: "navy coat",
    locked: false,
    createdBy: "BROWSER AGENT",
  },
  {
    id: "shot-03",
    number: 3,
    title: "The handle turns",
    shotType: "INSERT",
    duration: 2,
    description: "Mara grips the red handle. The dryer sound drops away.",
    visual: "insert",
    umbrellaVisible: true,
    screenDirection: "right",
    wardrobe: "navy coat",
    locked: false,
    createdBy: "BROWSER AGENT",
  },
];

const seedState: StudioState = {
  sceneTitle: "The Red Umbrella",
  branches: [
    {
      id: "cut-a",
      name: "Cut A",
      note: "Suspense through withheld information",
      shots: initialShots,
    },
  ],
  activeBranchId: "cut-a",
  selectedShotId: "shot-01",
  constraints: [
    {
      id: "red-umbrella",
      label: "VISUAL ANCHOR",
      rule: "The red umbrella stays visible in every shot.",
      locked: true,
    },
  ],
  activity: [
    {
      id: "activity-seed",
      timestamp: "00:00",
      actor: "DIRECTOR",
      action: "Locked the anchor",
      detail: "Shot 01 and the red umbrella are protected.",
    },
    {
      id: "activity-inspect",
      timestamp: "00:04",
      actor: "BROWSER AGENT",
      action: "Inspected the board",
      detail: "Read 3 shots, 1 branch, and 1 creative lock.",
      toolName: "inspect_storyboard",
    },
  ],
  version: 1,
};

const expansionDrafts: Omit<Shot, "id" | "number" | "locked" | "createdBy">[] = [
  {
    title: "Eyes in the chrome",
    shotType: "CLOSE",
    duration: 3,
    description: "The stranger appears only in the washer door reflection.",
    visual: "close",
    umbrellaVisible: true,
    screenDirection: "right",
    wardrobe: "navy coat",
  },
  {
    title: "No one at the door",
    shotType: "REVERSE",
    duration: 4,
    description: "Mara turns. The entrance is empty, but the bell still moves.",
    visual: "reverse",
    umbrellaVisible: true,
    screenDirection: "right",
    wardrobe: "navy coat",
  },
  {
    title: "Leave it spinning",
    shotType: "WIDE",
    duration: 5,
    description: "She exits with the umbrella. One machine starts by itself.",
    visual: "exit",
    umbrellaVisible: true,
    screenDirection: "right",
    wardrobe: "navy coat",
  },
];

const toolNames = [
  "inspect_storyboard",
  "create_shot",
  "update_shot",
  "lock_creative_decision",
  "expand_sequence",
  "create_alternate_cut",
  "check_continuity",
  "select_cut",
];

const cloneSeed = (): StudioState => JSON.parse(JSON.stringify(seedState)) as StudioState;
const nowLabel = () => new Date().toLocaleTimeString([], { minute: "2-digit", second: "2-digit" });
const makeId = (prefix: string) => `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const stringValue = (value: unknown, fallback = "") => typeof value === "string" ? value.trim().slice(0, 280) : fallback;
const numberValue = (value: unknown, fallback: number) => typeof value === "number" && Number.isFinite(value) ? value : fallback;
const isVisual = (value: unknown): value is Visual => ["establish", "arrival", "insert", "close", "reverse", "exit"].includes(String(value));

function activeBranchOf(state: StudioState) {
  return state.branches.find((branch) => branch.id === state.activeBranchId) ?? state.branches[0];
}

function addActivity(
  state: StudioState,
  actor: Actor,
  action: string,
  detail: string,
  toolName?: string,
) {
  const activity: Activity = {
    id: makeId("activity"),
    timestamp: nowLabel(),
    actor,
    action,
    detail,
    toolName,
  };
  return { ...state, activity: [activity, ...state.activity].slice(0, 40), version: state.version + 1 };
}

function inspectState(state: StudioState) {
  const branch = activeBranchOf(state);
  return {
    scene: state.sceneTitle,
    version: state.version,
    activeBranch: { id: branch.id, name: branch.name, note: branch.note },
    branches: state.branches.map(({ id, name, shots }) => ({ id, name, shotCount: shots.length })),
    constraints: state.constraints,
    selectedShotId: state.selectedShotId,
    shots: branch.shots,
    guardrail: "Locked shots cannot be changed by agent tools. Create an alternate cut to explore safely.",
  };
}

function checkBranch(branch: Branch, umbrellaLock: boolean): ContinuityReport {
  const issues: ContinuityIssue[] = [];
  const baselineDirection = branch.shots[0]?.screenDirection;
  const baselineWardrobe = branch.shots[0]?.wardrobe;

  branch.shots.forEach((shot) => {
    if (umbrellaLock && !shot.umbrellaVisible) {
      issues.push({ shotId: shot.id, shotNumber: shot.number, rule: "red_umbrella", detail: "Locked visual anchor is missing." });
    }
    if (baselineDirection && shot.screenDirection !== baselineDirection) {
      issues.push({ shotId: shot.id, shotNumber: shot.number, rule: "screen_direction", detail: `Expected ${baselineDirection}; found ${shot.screenDirection}.` });
    }
    if (baselineWardrobe && shot.wardrobe !== baselineWardrobe) {
      issues.push({ shotId: shot.id, shotNumber: shot.number, rule: "wardrobe", detail: `Expected ${baselineWardrobe}; found ${shot.wardrobe}.` });
    }
  });

  return { branchId: branch.id, checkedShots: branch.shots.length, issues };
}

function ShotArtwork({ shot }: { shot: Shot }) {
  return (
    <div className="frame-art" data-visual={shot.visual} aria-label={`Paper-cut storyboard frame: ${shot.title}`}>
      <span className="window-glow" />
      <span className="washer washer-one" />
      <span className="washer washer-two" />
      <span className="door-shape" />
      <span className="figure"><i /></span>
      {shot.visual === "arrival" || shot.visual === "reverse" ? <span className="shadow-figure" /> : null}
      {shot.umbrellaVisible ? <span className="umbrella" /> : null}
      <span className="paper-grain" />
      <span className="frame-index">{String(shot.number).padStart(2, "0")}</span>
      {shot.locked ? <span className="frame-lock">◆ LOCKED</span> : null}
      <span className="frame-author">{shot.createdBy === "DIRECTOR" ? "HUMAN" : "AGENT"}</span>
    </div>
  );
}

function ShotCard({ shot, selected, onSelect }: { shot: Shot; selected: boolean; onSelect: () => void }) {
  return (
    <button className={`shot-card ${selected ? "selected" : ""}`} onClick={onSelect} aria-pressed={selected}>
      <div className="shot-meta">
        <span>SHOT {String(shot.number).padStart(2, "0")}</span>
        <span>{shot.locked ? "ANCHOR LOCKED" : shot.createdBy === "BROWSER AGENT" ? "AGENT DRAFT" : "DIRECTOR"}</span>
      </div>
      <ShotArtwork shot={shot} />
      <div className="shot-copy">
        <span>{shot.shotType} · {shot.duration} SEC</span>
        <h3>{shot.title}</h3>
        <p>{shot.description}</p>
      </div>
    </button>
  );
}

export default function Home() {
  const [studio, setStudio] = useState<StudioState>(cloneSeed);
  const studioRef = useRef(studio);
  const [panelTab, setPanelTab] = useState<"relay" | "tools" | "history">("relay");
  const [webMcpStatus, setWebMcpStatus] = useState<"checking" | "ready" | "fallback">("fallback");
  const [continuityReport, setContinuityReport] = useState<ContinuityReport | null>(null);
  const [notice, setNotice] = useState("Board restored. Creative locks are active.");
  const [demoRunning, setDemoRunning] = useState(false);
  const [demoCue, setDemoCue] = useState("Ask your browser agent to inspect this board.");

  const commit = useCallback((next: StudioState, message?: string) => {
    studioRef.current = next;
    setStudio(next);
    if (typeof window !== "undefined") localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    if (message) setNotice(message);
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return;
    try {
      const parsed = JSON.parse(saved) as StudioState;
      if (parsed.branches?.length && parsed.constraints?.length) {
        const timer = window.setTimeout(() => {
          studioRef.current = parsed;
          setStudio(parsed);
          setNotice(`Restored CutRoom version ${parsed.version}.`);
        }, 0);
        return () => window.clearTimeout(timer);
      }
    } catch {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const logRead = useCallback((toolName: string, action: string, detail: string) => {
    const next = addActivity(studioRef.current, "BROWSER AGENT", action, detail, toolName);
    commit(next, `${toolName} completed.`);
    return next;
  }, [commit]);

  const appendShots = useCallback((drafts: Record<string, unknown>[], actor: Actor, toolName?: string) => {
    const current = studioRef.current;
    const branch = activeBranchOf(current);
    const room = Math.max(0, 6 - branch.shots.length);
    const accepted = drafts.slice(0, room).map((draft, index): Shot => ({
      id: makeId("shot"),
      number: branch.shots.length + index + 1,
      title: stringValue(draft.title, `Untitled beat ${branch.shots.length + index + 1}`),
      shotType: stringValue(draft.shotType, "MEDIUM").toUpperCase(),
      duration: Math.min(12, Math.max(1, numberValue(draft.duration, 3))),
      description: stringValue(draft.description, "A new beat in the sequence."),
      visual: isVisual(draft.visual) ? draft.visual : "close",
      umbrellaVisible: typeof draft.umbrellaVisible === "boolean" ? draft.umbrellaVisible : true,
      screenDirection: draft.screenDirection === "left" ? "left" : "right",
      wardrobe: stringValue(draft.wardrobe, "navy coat"),
      locked: false,
      createdBy: actor,
    }));

    if (!accepted.length) return { state: current, added: [] as Shot[] };
    const branches = current.branches.map((item) => item.id === branch.id ? { ...item, shots: [...item.shots, ...accepted] } : item);
    let next = { ...current, branches, selectedShotId: accepted[0].id };
    next = addActivity(next, actor, "Expanded the sequence", `Added ${accepted.length} coverage beat${accepted.length === 1 ? "" : "s"} to ${branch.name}.`, toolName);
    commit(next, `${accepted.length} new shots added without changing the locked anchor.`);
    return { state: next, added: accepted };
  }, [commit]);

  const createShot = useCallback((input: Record<string, unknown>, actor: Actor, toolName?: string) => {
    const result = appendShots([input], actor, toolName);
    const shot = result.added[0];
    if (!shot) throw new Error("The active cut already has six shots. Update an unlocked shot or create an alternate cut.");
    return shot;
  }, [appendShots]);

  const patchShot = useCallback((shotId: string, patch: Record<string, unknown>, actor: Actor, toolName?: string, recordActivity = true) => {
    const current = studioRef.current;
    const branch = activeBranchOf(current);
    const existing = branch.shots.find((shot) => shot.id === shotId);
    if (!existing) throw new Error(`Shot ${shotId} was not found in ${branch.name}.`);
    if (existing.locked) throw new Error(`Shot ${shotId} is director-locked. Create an alternate cut instead.`);

    const updated: Shot = {
      ...existing,
      title: patch.title === undefined ? existing.title : stringValue(patch.title, existing.title),
      shotType: patch.shotType === undefined ? existing.shotType : stringValue(patch.shotType, existing.shotType).toUpperCase(),
      duration: patch.duration === undefined ? existing.duration : Math.min(12, Math.max(1, numberValue(patch.duration, existing.duration))),
      description: patch.description === undefined ? existing.description : stringValue(patch.description, existing.description),
      visual: patch.visual === undefined ? existing.visual : isVisual(patch.visual) ? patch.visual : existing.visual,
      umbrellaVisible: typeof patch.umbrellaVisible === "boolean" ? patch.umbrellaVisible : existing.umbrellaVisible,
      screenDirection: patch.screenDirection === "left" || patch.screenDirection === "right" ? patch.screenDirection : existing.screenDirection,
      wardrobe: patch.wardrobe === undefined ? existing.wardrobe : stringValue(patch.wardrobe, existing.wardrobe),
    };
    const branches = current.branches.map((item) => item.id === branch.id ? { ...item, shots: item.shots.map((shot) => shot.id === shotId ? updated : shot) } : item);
    let next = { ...current, branches };
    if (recordActivity) next = addActivity(next, actor, "Updated a shot", `${String(updated.number).padStart(2, "0")} · ${updated.title}`, toolName);
    commit(next, recordActivity ? `Shot ${String(updated.number).padStart(2, "0")} updated.` : undefined);
    return updated;
  }, [commit]);

  const lockShot = useCallback((shotId: string, actor: Actor, toolName?: string) => {
    const current = studioRef.current;
    const branch = activeBranchOf(current);
    const target = branch.shots.find((shot) => shot.id === shotId);
    if (!target) throw new Error(`Shot ${shotId} was not found.`);
    const branches = current.branches.map((item) => item.id === branch.id ? { ...item, shots: item.shots.map((shot) => shot.id === shotId ? { ...shot, locked: true } : shot) } : item);
    let next = { ...current, branches };
    next = addActivity(next, actor, "Locked a creative decision", `Shot ${String(target.number).padStart(2, "0")} is protected from agent edits.`, toolName);
    commit(next, `Shot ${String(target.number).padStart(2, "0")} locked.`);
    return target;
  }, [commit]);

  const toggleHumanLock = useCallback((shotId: string) => {
    const current = studioRef.current;
    const branch = activeBranchOf(current);
    const target = branch.shots.find((shot) => shot.id === shotId);
    if (!target) return;
    const branches = current.branches.map((item) => item.id === branch.id ? { ...item, shots: item.shots.map((shot) => shot.id === shotId ? { ...shot, locked: !shot.locked } : shot) } : item);
    let next = { ...current, branches };
    next = addActivity(next, "DIRECTOR", target.locked ? "Released a lock" : "Locked a creative decision", `Shot ${String(target.number).padStart(2, "0")} is ${target.locked ? "editable" : "protected"}.`);
    commit(next, `Shot ${String(target.number).padStart(2, "0")} ${target.locked ? "unlocked" : "locked"}.`);
  }, [commit]);

  const createAlternate = useCallback((name: string, changes: Record<string, unknown>[], actor: Actor, toolName?: string) => {
    const current = studioRef.current;
    const source = activeBranchOf(current);
    const branchId = makeId("cut");
    const changedIds: string[] = [];
    const shots = source.shots.map((shot) => {
      const change = changes.find((item) => stringValue(item.shotId) === shot.id);
      if (!change || shot.locked) return { ...shot };
      changedIds.push(shot.id);
      return {
        ...shot,
        title: change.title === undefined ? shot.title : stringValue(change.title, shot.title),
        description: change.description === undefined ? shot.description : stringValue(change.description, shot.description),
        shotType: change.shotType === undefined ? shot.shotType : stringValue(change.shotType, shot.shotType).toUpperCase(),
        visual: change.visual === undefined ? shot.visual : isVisual(change.visual) ? change.visual : shot.visual,
        umbrellaVisible: typeof change.umbrellaVisible === "boolean" ? change.umbrellaVisible : shot.umbrellaVisible,
        screenDirection: change.screenDirection === "left" || change.screenDirection === "right" ? change.screenDirection : shot.screenDirection,
        wardrobe: change.wardrobe === undefined ? shot.wardrobe : stringValue(change.wardrobe, shot.wardrobe),
        createdBy: actor,
      };
    });
    const branch: Branch = {
      id: branchId,
      name: stringValue(name, `Cut ${String.fromCharCode(65 + current.branches.length)}`),
      note: "Alternate rhythm; locked decisions inherited",
      shots,
    };
    let next = { ...current, branches: [...current.branches, branch], activeBranchId: branchId, selectedShotId: changedIds[0] ?? shots[0]?.id ?? "" };
    next = addActivity(next, actor, "Created an alternate cut", `${branch.name} inherits ${shots.filter((shot) => shot.locked).length} locks and changes ${changedIds.length} shots.`, toolName);
    commit(next, `${branch.name} created. The original cut is untouched.`);
    return branch;
  }, [commit]);

  const selectBranch = useCallback((branchId: string, actor: Actor, toolName?: string) => {
    const current = studioRef.current;
    const branch = current.branches.find((item) => item.id === branchId);
    if (!branch) throw new Error(`Cut ${branchId} was not found.`);
    let next = { ...current, activeBranchId: branchId, selectedShotId: branch.shots[0]?.id ?? "" };
    next = addActivity(next, actor, "Switched the working cut", `${branch.name} is now active.`, toolName);
    commit(next, `${branch.name} selected.`);
    return branch;
  }, [commit]);

  const runContinuity = useCallback((branchId?: string, actor: Actor = "DIRECTOR", toolName?: string) => {
    const current = studioRef.current;
    const branch = current.branches.find((item) => item.id === branchId) ?? activeBranchOf(current);
    const report = checkBranch(branch, current.constraints.some((constraint) => constraint.id === "red-umbrella" && constraint.locked));
    const next = addActivity(current, actor, "Checked continuity", report.issues.length ? `Found ${report.issues.length} issue${report.issues.length === 1 ? "" : "s"} across ${report.checkedShots} shots.` : `All ${report.checkedShots} shots honor the active locks.`, toolName);
    commit(next, report.issues.length ? `${report.issues.length} continuity issue found.` : "Continuity is clean.");
    setContinuityReport(report);
    setPanelTab("relay");
    return report;
  }, [commit]);

  useEffect(() => {
    const context = document.modelContext;
    if (!context?.registerTool) {
      return;
    }

    const controller = new AbortController();
    const ok = (payload: Record<string, unknown>) => JSON.stringify({ ok: true, ...payload });
    const fail = (error: unknown) => JSON.stringify({ ok: false, error: error instanceof Error ? error.message : "Tool execution failed." });
    const guard = (signal: AbortSignal) => { if (signal.aborted) throw new DOMException("Tool call cancelled", "AbortError"); };
    const objectArray = (value: unknown) => Array.isArray(value) ? value.filter(isRecord) : [];
    const baseSchema = { type: "object", additionalProperties: false };

    const tools: WebMCPTool[] = [
      {
        name: "inspect_storyboard",
        title: "Inspect storyboard",
        description: "Read the active CutRoom scene, shots, branches, selected shot, and director-locked constraints before proposing edits.",
        inputSchema: { ...baseSchema, properties: {} },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (_input, { signal }) => {
          guard(signal);
          const snapshot = inspectState(studioRef.current);
          return ok({ storyboard: snapshot });
        },
      },
      {
        name: "create_shot",
        title: "Create shot",
        description: "Append one structured storyboard shot to the active cut. Use only after inspect_storyboard; locked shots remain unchanged.",
        inputSchema: {
          ...baseSchema,
          properties: {
            title: { type: "string", minLength: 1, maxLength: 80 },
            shotType: { type: "string", enum: ["WIDE", "MEDIUM", "CLOSE", "INSERT", "REVERSE"] },
            duration: { type: "number", minimum: 1, maximum: 12 },
            description: { type: "string", minLength: 1, maxLength: 280 },
            visual: { type: "string", enum: ["establish", "arrival", "insert", "close", "reverse", "exit"] },
            umbrellaVisible: { type: "boolean" },
            screenDirection: { type: "string", enum: ["left", "right"] },
            wardrobe: { type: "string", maxLength: 80 },
          },
          required: ["title", "shotType", "duration", "description", "visual", "umbrellaVisible", "screenDirection", "wardrobe"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try { guard(signal); const shot = createShot(input, "BROWSER AGENT", "create_shot"); return ok({ created: shot, version: studioRef.current.version }); } catch (error) { return fail(error); }
        },
      },
      {
        name: "update_shot",
        title: "Update unlocked shot",
        description: "Revise an existing unlocked shot in the active cut. Director-locked shots are rejected with a guardrail message.",
        inputSchema: {
          ...baseSchema,
          properties: {
            shotId: { type: "string", minLength: 1, maxLength: 80 },
            title: { type: "string", minLength: 1, maxLength: 80 },
            shotType: { type: "string", enum: ["WIDE", "MEDIUM", "CLOSE", "INSERT", "REVERSE"] },
            duration: { type: "number", minimum: 1, maximum: 12 },
            description: { type: "string", minLength: 1, maxLength: 280 },
            visual: { type: "string", enum: ["establish", "arrival", "insert", "close", "reverse", "exit"] },
            umbrellaVisible: { type: "boolean" },
            screenDirection: { type: "string", enum: ["left", "right"] },
            wardrobe: { type: "string", maxLength: 80 },
          },
          required: ["shotId"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try { guard(signal); const shotId = stringValue(input.shotId); const updated = patchShot(shotId, input, "BROWSER AGENT", "update_shot"); return ok({ updated, version: studioRef.current.version }); } catch (error) { return fail(error); }
        },
      },
      {
        name: "lock_creative_decision",
        title: "Lock creative decision",
        description: "Protect a shot from future agent edits. Locks are intentionally one-way for agents; only the director can release one in the UI.",
        inputSchema: { ...baseSchema, properties: { shotId: { type: "string", minLength: 1, maxLength: 80 } }, required: ["shotId"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try { guard(signal); const shot = lockShot(stringValue(input.shotId), "BROWSER AGENT", "lock_creative_decision"); return ok({ lockedShotId: shot.id, version: studioRef.current.version }); } catch (error) { return fail(error); }
        },
      },
      {
        name: "expand_sequence",
        title: "Expand sequence",
        description: "Append one to six agent-proposed coverage beats to the active cut in one atomic action while preserving every director lock.",
        inputSchema: {
          ...baseSchema,
          properties: {
            shots: {
              type: "array",
              minItems: 1,
              maxItems: 6,
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  title: { type: "string", minLength: 1, maxLength: 80 },
                  shotType: { type: "string", enum: ["WIDE", "MEDIUM", "CLOSE", "INSERT", "REVERSE"] },
                  duration: { type: "number", minimum: 1, maximum: 12 },
                  description: { type: "string", minLength: 1, maxLength: 280 },
                  visual: { type: "string", enum: ["establish", "arrival", "insert", "close", "reverse", "exit"] },
                  umbrellaVisible: { type: "boolean" },
                  screenDirection: { type: "string", enum: ["left", "right"] },
                  wardrobe: { type: "string", maxLength: 80 },
                },
                required: ["title", "shotType", "duration", "description", "visual", "umbrellaVisible", "screenDirection", "wardrobe"],
              },
            },
          },
          required: ["shots"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try { guard(signal); const result = appendShots(objectArray(input.shots), "BROWSER AGENT", "expand_sequence"); return ok({ added: result.added, activeCut: activeBranchOf(result.state).id, version: result.state.version }); } catch (error) { return fail(error); }
        },
      },
      {
        name: "create_alternate_cut",
        title: "Create alternate cut",
        description: "Branch the active cut, inherit all locked shots, and apply proposed changes only to unlocked shots so the original stays recoverable.",
        inputSchema: {
          ...baseSchema,
          properties: {
            name: { type: "string", minLength: 1, maxLength: 60 },
            changes: {
              type: "array",
              maxItems: 6,
              items: {
                type: "object",
                additionalProperties: false,
                properties: {
                  shotId: { type: "string", minLength: 1, maxLength: 80 },
                  title: { type: "string", maxLength: 80 },
                  description: { type: "string", maxLength: 280 },
                  shotType: { type: "string", enum: ["WIDE", "MEDIUM", "CLOSE", "INSERT", "REVERSE"] },
                  visual: { type: "string", enum: ["establish", "arrival", "insert", "close", "reverse", "exit"] },
                  umbrellaVisible: { type: "boolean" },
                  screenDirection: { type: "string", enum: ["left", "right"] },
                  wardrobe: { type: "string", maxLength: 80 },
                },
                required: ["shotId"],
              },
            },
          },
          required: ["name", "changes"],
        },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try { guard(signal); const branch = createAlternate(stringValue(input.name, "Alternate Cut"), objectArray(input.changes), "BROWSER AGENT", "create_alternate_cut"); return ok({ createdBranch: { id: branch.id, name: branch.name, shotCount: branch.shots.length }, inheritedLocks: branch.shots.filter((shot) => shot.locked).map((shot) => shot.id), version: studioRef.current.version }); } catch (error) { return fail(error); }
        },
      },
      {
        name: "check_continuity",
        title: "Check continuity",
        description: "Audit a cut for the locked red umbrella, screen direction, and wardrobe continuity. Returns exact shot IDs and repairable issues.",
        inputSchema: { ...baseSchema, properties: { branchId: { type: "string", maxLength: 80 } } },
        annotations: { readOnlyHint: true, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try {
            guard(signal);
            const current = studioRef.current;
            const branch = current.branches.find((item) => item.id === stringValue(input.branchId)) ?? activeBranchOf(current);
            const report = checkBranch(branch, current.constraints.some((constraint) => constraint.id === "red-umbrella" && constraint.locked));
            return ok({ report, version: current.version });
          } catch (error) { return fail(error); }
        },
      },
      {
        name: "select_cut",
        title: "Select working cut",
        description: "Switch the visible working board to a known branch ID returned by inspect_storyboard without deleting or merging any cut.",
        inputSchema: { ...baseSchema, properties: { branchId: { type: "string", minLength: 1, maxLength: 80 } }, required: ["branchId"] },
        annotations: { readOnlyHint: false, untrustedContentHint: true },
        execute: async (input, { signal }) => {
          try { guard(signal); const branch = selectBranch(stringValue(input.branchId), "BROWSER AGENT", "select_cut"); return ok({ activeBranch: { id: branch.id, name: branch.name }, version: studioRef.current.version }); } catch (error) { return fail(error); }
        },
      },
    ];

    Promise.all(tools.map((tool) => Promise.resolve(context.registerTool(tool, { signal: controller.signal })))).then(() => {
      setWebMcpStatus("ready");
      setNotice(`${tools.length} WebMCP tools registered for your browser agent.`);
    }).catch(() => setWebMcpStatus("fallback"));

    return () => controller.abort();
  }, [appendShots, createAlternate, createShot, lockShot, patchShot, selectBranch]);

  const branch = activeBranchOf(studio);
  const selectedShot = branch.shots.find((shot) => shot.id === studio.selectedShotId) ?? branch.shots[0];
  const lockedCount = branch.shots.filter((shot) => shot.locked).length;
  const totalDuration = branch.shots.reduce((sum, shot) => sum + shot.duration, 0);

  const lastAgentActivity = useMemo(
    () => studio.activity.find((item) => item.actor === "BROWSER AGENT") ?? studio.activity[0],
    [studio.activity],
  );

  const expandFromUI = () => {
    appendShots(expansionDrafts.map((shot) => ({ ...shot })), "BROWSER AGENT", "expand_sequence");
    setPanelTab("relay");
    setDemoCue("Coverage complete. Lock a shot or branch the ending.");
  };

  const branchFromUI = () => {
    const changeTarget = branch.shots.find((shot) => !shot.locked);
    createAlternate("Cut B · Stranger First", changeTarget ? [{ shotId: changeTarget.id, title: "The stranger waits", description: "Reveal the stranger before Mara notices the moving bell.", visual: "reverse", screenDirection: "left" }] : [], "BROWSER AGENT", "create_alternate_cut");
    setPanelTab("relay");
    setDemoCue("Alternate cut created. Run continuity to catch the direction change.");
  };

  const resetBoard = useCallback(() => {
    const next = cloneSeed();
    commit(next, "Demo board reset to the director's seed.");
    setContinuityReport(null);
    setPanelTab("relay");
    setDemoCue("Ask your browser agent to inspect this board.");
  }, [commit]);

  const runJudgeDemo = async () => {
    if (demoRunning) return;
    setDemoRunning(true);
    resetBoard();
    setDemoCue("1 / 4 · Agent inspects the director's seed and locked anchor.");
    await new Promise((resolve) => setTimeout(resolve, 700));
    logRead("inspect_storyboard", "Inspected the board", "Read 3 shots and the locked red umbrella constraint.");
    setDemoCue("2 / 4 · Agent expands to six shots without touching the lock.");
    await new Promise((resolve) => setTimeout(resolve, 700));
    appendShots(expansionDrafts.map((shot) => ({ ...shot })), "BROWSER AGENT", "expand_sequence");
    setDemoCue("3 / 4 · Director locks the insert; agent branches safely.");
    await new Promise((resolve) => setTimeout(resolve, 700));
    const currentBranch = activeBranchOf(studioRef.current);
    const insert = currentBranch.shots.find((shot) => shot.visual === "insert");
    if (insert) lockShot(insert.id, "DIRECTOR");
    const changeTarget = activeBranchOf(studioRef.current).shots.find((shot) => !shot.locked);
    createAlternate("Cut B · Stranger First", changeTarget ? [{ shotId: changeTarget.id, title: "The stranger waits", description: "Reveal the stranger before Mara notices the moving bell.", visual: "reverse", screenDirection: "left" }] : [], "BROWSER AGENT", "create_alternate_cut");
    setDemoCue("4 / 4 · Continuity finds the one risky direction change.");
    await new Promise((resolve) => setTimeout(resolve, 700));
    runContinuity(undefined, "BROWSER AGENT", "check_continuity");
    setDemoCue("Demo complete · The human stayed in control of every irreversible decision.");
    setDemoRunning(false);
  };

  return (
    <main className="studio-shell">
      <header className="topbar">
        <a className="brand-block" href="#board" aria-label="CutRoom home"><span className="brand-cut">CUT</span><span>ROOM</span><em>α</em></a>
        <div className="scene-identity"><span>SCENE 12 · INT. LAUNDROMAT · NIGHT</span><strong>{studio.sceneTitle}</strong></div>
        <div className="header-actions">
          <div className={`connection-pill ${webMcpStatus}`}><i /> {webMcpStatus === "ready" ? "8 SITE TOOLS READY" : webMcpStatus === "checking" ? "CHECKING TOOLS" : "INTERACTIVE DEMO"}</div>
          <button className="demo-button" onClick={runJudgeDemo} disabled={demoRunning}>{demoRunning ? "PLAYING…" : "PLAY JUDGE DEMO"}</button>
        </div>
      </header>

      <section className="workspace">
        <aside className="left-rail">
          <p className="eyebrow">DIRECTOR&apos;S LOOP</p>
          <ol className="loop-list">
            <li className="complete"><span>1</span><div><strong>Seed</strong><small>Set the intention</small></div></li>
            <li className={branch.shots.length >= 6 ? "complete" : "active"}><span>2</span><div><strong>Expand</strong><small>Find the coverage</small></div></li>
            <li className={lockedCount > 1 ? "complete" : ""}><span>3</span><div><strong>Lock</strong><small>Protect the truth</small></div></li>
            <li className={studio.branches.length > 1 ? "complete" : ""}><span>4</span><div><strong>Branch</strong><small>Explore without loss</small></div></li>
          </ol>

          <div className="branch-switcher" aria-label="Storyboard cuts">
            <p className="eyebrow">WORKING CUT</p>
            {studio.branches.map((item) => (
              <button key={item.id} className={item.id === branch.id ? "active" : ""} onClick={() => selectBranch(item.id, "DIRECTOR")}>
                <span>{item.name}</span><small>{item.shots.length} SHOTS</small>
              </button>
            ))}
          </div>

          <button className="constraint-card" onClick={() => setNotice("Only the director can release this creative lock.")}>
            <div><span className="lock-dot">◆</span><b>CREATIVE LOCK</b></div>
            <p>{studio.constraints[0].rule}</p>
            <small>LOCKED BY DIRECTOR</small>
          </button>

          <div className="primary-stack">
            <button className="expand-button" onClick={expandFromUI} disabled={branch.shots.length >= 6}>EXPAND TO 6 SHOTS <span>↗</span></button>
            <button className="branch-button" onClick={branchFromUI} disabled={studio.branches.length >= 3}>CREATE ALTERNATE CUT <span>＋</span></button>
          </div>
        </aside>

        <section className="storyboard" id="board">
          <div className="board-heading">
            <div><p className="eyebrow">{branch.name.toUpperCase()} · WORKING BOARD · V{studio.version}</p><h1>Direct the intention.<br />Let the agent find the coverage.</h1><p className="branch-note">{branch.note}</p></div>
            <div className="board-stats"><span>{lockedCount} / {branch.shots.length}</span><small>SHOTS LOCKED</small></div>
          </div>

          <div className="shot-grid">
            {branch.shots.map((shot) => <ShotCard key={shot.id} shot={shot} selected={shot.id === selectedShot?.id} onSelect={() => commit({ ...studioRef.current, selectedShotId: shot.id })} />)}
            {branch.shots.length < 6 ? (
              <button className="empty-shot" onClick={expandFromUI}><span>＋</span><b>ASK AGENT FOR COVERAGE</b><small>{6 - branch.shots.length} SHOTS OPEN</small></button>
            ) : null}
          </div>

          <div className="timeline-rule"><span>00:00</span><i /><b>{totalDuration} SEC CUT</b><i /><span>00:{String(totalDuration).padStart(2, "0")}</span></div>

          {selectedShot ? (
            <section className="shot-editor" aria-label="Selected shot editor">
              <div className="editor-label"><span>EDITING</span><b>{String(selectedShot.number).padStart(2, "0")}</b><small>{selectedShot.locked ? "DIRECTOR LOCKED" : "LIVE & EDITABLE"}</small></div>
              <label><span>TITLE</span><input value={selectedShot.title} disabled={selectedShot.locked} onChange={(event) => patchShot(selectedShot.id, { title: event.target.value }, "DIRECTOR", undefined, false)} onBlur={() => setNotice(`Shot ${String(selectedShot.number).padStart(2, "0")} saved locally.`)} /></label>
              <label className="description-field"><span>INTENTION</span><textarea value={selectedShot.description} disabled={selectedShot.locked} onChange={(event) => patchShot(selectedShot.id, { description: event.target.value }, "DIRECTOR", undefined, false)} onBlur={() => setNotice(`Shot ${String(selectedShot.number).padStart(2, "0")} saved locally.`)} /></label>
              <label><span>SIZE</span><select value={selectedShot.shotType} disabled={selectedShot.locked} onChange={(event) => patchShot(selectedShot.id, { shotType: event.target.value }, "DIRECTOR")}><option>WIDE</option><option>MEDIUM</option><option>CLOSE</option><option>INSERT</option><option>REVERSE</option></select></label>
              <button className={`lock-button ${selectedShot.locked ? "locked" : ""}`} onClick={() => toggleHumanLock(selectedShot.id)}>{selectedShot.locked ? "RELEASE LOCK" : "LOCK THIS SHOT"}</button>
            </section>
          ) : null}
        </section>

        <aside className="right-rail">
          <div className="right-tabs" role="tablist" aria-label="CutRoom inspector">
            {(["relay", "tools", "history"] as const).map((tab) => <button key={tab} role="tab" aria-selected={panelTab === tab} className={panelTab === tab ? "active" : ""} onClick={() => setPanelTab(tab)}>{tab}</button>)}
          </div>

          {panelTab === "relay" ? (
            <div className="panel-content">
              <div className="relay-heading"><div><i /> LIVE RELAY</div><span>V{studio.version}</span></div>
              <div className="relay-copy"><p className="eyebrow">BROWSER AGENT</p><h2>{demoCue}</h2><p>CutRoom gives the visiting agent structured actions, while locks and provenance keep the director in control.</p></div>
              {lastAgentActivity ? <div className="tool-call"><div><span>↳</span><b>{lastAgentActivity.toolName ?? "shared_state"}</b></div><p>{lastAgentActivity.detail}</p></div> : null}
              {continuityReport ? (
                <div className={`continuity-card ${continuityReport.issues.length ? "has-issues" : "clean"}`}>
                  <span>CONTINUITY REPORT</span>
                  <strong>{continuityReport.issues.length ? `${continuityReport.issues.length} ISSUE FOUND` : "CLEAN CUT"}</strong>
                  {continuityReport.issues.slice(0, 2).map((issue) => <p key={`${issue.shotId}-${issue.rule}`}>Shot {String(issue.shotNumber).padStart(2, "0")} · {issue.detail}</p>)}
                </div>
              ) : <div className="agent-note"><span>AGENT NOTE</span><p>Every shot keeps the red umbrella visible. Run a check after branching.</p></div>}
              <button className="check-button" onClick={() => runContinuity()}>RUN CONTINUITY CHECK <span>✓</span></button>
              <p className="privacy-note">NO MODEL API · NO ACCOUNT · BOARD STAYS IN THIS BROWSER</p>
            </div>
          ) : null}

          {panelTab === "tools" ? (
            <div className="panel-content">
              <div className="panel-title"><p className="eyebrow">WEBMCP INSPECTOR</p><h2>8 narrow tools.<br />One shared canvas.</h2><p>Your browser agent supplies the intelligence. CutRoom supplies safe, verifiable actions.</p></div>
              <div className="tools-list">{toolNames.map((name, index) => <div key={name}><span>{String(index + 1).padStart(2, "0")}</span><code>{name}</code><i>{[0, 6].includes(index) ? "READ" : "WRITE"}</i></div>)}</div>
            </div>
          ) : null}

          {panelTab === "history" ? (
            <div className="panel-content">
              <div className="panel-title"><p className="eyebrow">PROVENANCE</p><h2>Nothing happens in the dark.</h2><p>Human and agent actions share one visible, attributable history.</p></div>
              <div className="history-list">{studio.activity.map((item) => <article key={item.id}><span>{item.timestamp}</span><div><small>{item.actor}</small><b>{item.action}</b><p>{item.detail}</p></div></article>)}</div>
              <button className="reset-button" onClick={resetBoard}>RESET DEMO BOARD</button>
            </div>
          ) : null}

          <footer className="relay-footer"><span>YOUR AGENT, YOUR MODEL</span><i>WEBMCP</i></footer>
        </aside>
      </section>
      <div className="notice" role="status" aria-live="polite"><i />{notice}</div>
    </main>
  );
}
