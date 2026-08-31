import type { CharacterAction } from "../components/ARStage";
import { selectAnimatableRigParts, type DrawingExtraction, type SemanticPartKind } from "./drawing.ts";

export const SAFE_SHOW_ACTIONS: readonly CharacterAction[] = ["idle", "wave", "dance", "hop", "walk", "hide", "spin"];

export type CharacterCapability = {
  characterIndex: number;
  semanticParts: Partial<Record<SemanticPartKind, number>>;
  movableParts: Array<{ id: string; kind: SemanticPartKind; side: string; confidence: number }>;
  availableActions: CharacterAction[];
  blockedActions: Array<{ action: CharacterAction; reason: string }>;
  rigBasis: "learned-pose" | "learned-topology" | "safe-root-only" | "full-neural";
};

const countParts = (drawing: DrawingExtraction) => drawing.rig.parts.reduce<Partial<Record<SemanticPartKind, number>>>((counts, part) => {
  counts[part.kind] = (counts[part.kind] ?? 0) + 1;
  return counts;
}, {});

export function inspectCharacterCapabilities(drawings: DrawingExtraction[], fullNeural = false): CharacterCapability[] {
  return drawings.map((drawing, characterIndex) => {
    const movable = selectAnimatableRigParts(drawing.rig, {
      poseApplicable: Boolean(drawing.poseRecognition?.applicable),
      topologyApplicable: Boolean(drawing.topologyRecognition?.applicable),
    });
    const kinds = new Set(movable.map((part) => part.kind));
    const available = new Set<CharacterAction>(["idle", "hop", "hide", "spin"]);
    if (fullNeural || ["arm", "hand", "wing", "tentacle", "tail"].some((kind) => kinds.has(kind as SemanticPartKind))) available.add("wave");
    if (fullNeural || movable.length > 0) available.add("dance");
    if (fullNeural || ["leg", "foot", "tentacle"].some((kind) => kinds.has(kind as SemanticPartKind))) available.add("walk");
    const availableActions = SAFE_SHOW_ACTIONS.filter((action) => available.has(action));
    const blockedActions = SAFE_SHOW_ACTIONS.filter((action) => !available.has(action)).map((action) => ({
      action,
      reason: action === "wave" ? "No verified waving branch" : action === "walk" ? "No verified leg or walking branch" : "The verified rig cannot perform this action",
    }));
    return {
      characterIndex,
      semanticParts: countParts(drawing),
      movableParts: movable.map((part) => ({ id: part.id, kind: part.kind, side: part.side, confidence: Number(part.confidence.toFixed(3)) })),
      availableActions,
      blockedActions,
      rigBasis: fullNeural
        ? "full-neural"
        : drawing.poseRecognition?.applicable
          ? "learned-pose"
          : drawing.topologyRecognition?.applicable
            ? "learned-topology"
            : "safe-root-only",
    };
  });
}

export function validateCharacterMove(capabilities: CharacterCapability[], characterIndex: number, action: CharacterAction) {
  const capability = capabilities.find((candidate) => candidate.characterIndex === characterIndex);
  if (!capability) return { ok: false as const, error: `Character ${characterIndex} does not exist in the approved cast.` };
  if (!SAFE_SHOW_ACTIONS.includes(action)) return { ok: false as const, error: `“${action}” is not a child-safe WallAlive action.` };
  if (!capability.availableActions.includes(action)) {
    const blocked = capability.blockedActions.find((candidate) => candidate.action === action);
    return { ok: false as const, error: `Character ${characterIndex} cannot ${action}: ${blocked?.reason ?? "the verified rig does not support it"}.` };
  }
  return { ok: true as const, capability };
}
