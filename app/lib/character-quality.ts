import type { ContourPoint, DrawingExtraction, SemanticPartKind } from "./drawing.ts";

export type CharacterValidation = {
  accepted: boolean;
  score: number;
  rectangularity: number;
  axisAlignedEdgeFraction: number;
  evidence: string[];
  reason: string;
};

export type ReconstructionReadiness = {
  cutoutReady: boolean;
  motionReady: boolean;
  score: number;
  blockers: string[];
  warnings: string[];
  evidence: string[];
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

function contourMetrics(contour: ContourPoint[]) {
  if (contour.length < 3) return { rectangularity: 1, axisAlignedEdgeFraction: 1 };
  let twiceArea = 0;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let alignedLength = 0;
  let perimeter = 0;
  for (let index = 0; index < contour.length; index += 1) {
    const point = contour[index];
    const next = contour[(index + 1) % contour.length];
    minX = Math.min(minX, point.x);
    minY = Math.min(minY, point.y);
    maxX = Math.max(maxX, point.x);
    maxY = Math.max(maxY, point.y);
    twiceArea += point.x * next.y - next.x * point.y;
    const dx = next.x - point.x;
    const dy = next.y - point.y;
    const length = Math.hypot(dx, dy);
    perimeter += length;
    if (length && (Math.abs(dx) / length < 0.12 || Math.abs(dy) / length < 0.12)) alignedLength += length;
  }
  const boxArea = Math.max(1e-6, (maxX - minX) * (maxY - minY));
  return {
    rectangularity: clamp01(Math.abs(twiceArea) / 2 / boxArea),
    axisAlignedEdgeFraction: clamp01(alignedLength / Math.max(1e-6, perimeter)),
  };
}

const strongPartConfidence: Partial<Record<SemanticPartKind, number>> = {
  eye: 0.58,
  mouth: 0.52,
  nose: 0.58,
  ear: 0.62,
  arm: 0.62,
  hand: 0.58,
  leg: 0.62,
  foot: 0.58,
  wing: 0.62,
  fin: 0.62,
  tail: 0.62,
  tentacle: 0.62,
  trunk: 0.62,
  branch: 0.62,
  linkage: 0.62,
};

/**
 * Refuses to turn a coherent paper/screen patch into a character. General
 * segmentation confidence is deliberately not character evidence: the gate
 * requires learned facial, pose, or topology anchors after isolation.
 */
export function assessCharacterExtraction(extraction: DrawingExtraction): CharacterValidation {
  const shape = contourMetrics(extraction.contour);
  const learnedParts = extraction.rig.parts.filter((part) => (
    ["learned-model", "learned-pose", "learned-topology"].includes(part.source)
    && part.confidence >= (strongPartConfidence[part.kind] ?? 0.64)
  ));
  const count = (kind: SemanticPartKind) => learnedParts.filter((part) => part.kind === kind).length;
  const face = count("eye") >= 2 || (count("eye") >= 1 && count("mouth") >= 1);
  const articulatedKinds = new Set(learnedParts
    .filter((part) => ["arm", "hand", "leg", "foot", "wing", "fin", "tail", "tentacle", "trunk", "branch", "linkage"].includes(part.kind))
    .map((part) => part.kind));
  const articulated = articulatedKinds.size >= 2 && learnedParts.filter((part) => articulatedKinds.has(part.kind)).length >= 2;
  const confidentPoseJoints = extraction.poseRecognition?.joints.filter((joint) => joint.confidence >= 0.46).length ?? 0;
  const pose = Boolean(extraction.poseRecognition?.applicable && confidentPoseJoints >= 6);
  const topology = extraction.topologyRecognition;
  const graph = Boolean(
    topology?.applicable
    && topology.kindConfidence >= 0.62
    && topology.fieldConfidence >= 0.32
    && topology.nodes.filter((node) => node.confidence >= 0.42).length >= 3
    && topology.edges.filter((edge) => edge.confidence >= 0.38).length >= 2,
  );
  const evidence = [face ? "face" : "", articulated ? "articulated-parts" : "", pose ? "pose" : "", graph ? "topology" : ""].filter(Boolean);
  const slabLike = shape.rectangularity >= 0.72 && shape.axisAlignedEdgeFraction >= 0.48;
  const weakEvidence = evidence.length === 0 || (!face && !pose && evidence.length < 2);
  const suspiciousCoverage = (extraction.cutoutRecognition?.areaPercent ?? extraction.analysis.coveragePercent) >= 58;
  const accepted = evidence.length > 0 && !(slabLike && weakEvidence) && !(suspiciousCoverage && !face && !pose);
  const evidenceScore = Math.min(0.72, evidence.length * 0.2 + (face ? 0.14 : 0) + (pose ? 0.1 : 0));
  const shapePenalty = slabLike ? 0.34 : shape.rectangularity >= 0.82 ? 0.2 : 0;
  const score = clamp01(0.18 + evidenceScore - shapePenalty - (suspiciousCoverage && !face ? 0.14 : 0));
  const reason = accepted
    ? `Verified character evidence: ${evidence.join(", ")}.`
    : slabLike
      ? "The selected region looks like paper, a screen, or a rectangular camera patch—not one complete character."
      : "I could not verify a face, articulated parts, pose, or character topology inside the selected region.";
  return {
    accepted,
    score: Number(score.toFixed(3)),
    rectangularity: Number(shape.rectangularity.toFixed(3)),
    axisAlignedEdgeFraction: Number(shape.axisAlignedEdgeFraction.toFixed(3)),
    evidence,
    reason,
  };
}

export function requireCharacterExtraction(extraction: DrawingExtraction): DrawingExtraction {
  const characterValidation = assessCharacterExtraction(extraction);
  if (!characterValidation.accepted) {
    throw new Error(`${characterValidation.reason} Tap the middle of one character and keep the full head and limbs inside the guide.`);
  }
  return { ...extraction, characterValidation };
}

/**
 * A segmentation mask and a playable rig are different claims. This second
 * gate prevents a barely accepted crop from becoming a 3D character, and it
 * prevents unverified joints from being presented as movable anatomy.
 */
export function assessReconstructionReadiness(extraction: DrawingExtraction): ReconstructionReadiness {
  const validation = extraction.characterValidation ?? assessCharacterExtraction(extraction);
  const cutoutConfidence = extraction.cutoutRecognition?.confidence ?? 1;
  const areaPercent = extraction.cutoutRecognition?.areaPercent ?? extraction.analysis.coveragePercent;
  const confidentPoseJoints = extraction.poseRecognition?.joints.filter((joint) => joint.confidence >= 0.46).length ?? 0;
  const poseReady = Boolean(extraction.poseRecognition?.applicable && confidentPoseJoints >= 6);
  const topology = extraction.topologyRecognition;
  const topologyReady = Boolean(
    topology?.applicable
    && topology.kindConfidence >= 0.62
    && topology.fieldConfidence >= 0.32
    && topology.nodes.filter((node) => node.confidence >= 0.42).length >= 3
    && topology.edges.filter((edge) => edge.confidence >= 0.38).length >= 2,
  );
  const learnedBranches = extraction.rig.parts.filter((part) => (
    ["arm", "hand", "leg", "foot", "wing", "fin", "tail", "tentacle", "trunk", "branch", "linkage"].includes(part.kind)
    && ["learned-model", "learned-pose", "learned-topology"].includes(part.source)
    && part.confidence >= 0.62
  )).length;

  const blockers: string[] = [];
  const warnings: string[] = [];
  if (!validation.accepted || validation.score < 0.44) blockers.push(validation.reason);
  if (cutoutConfidence < 0.62) blockers.push("The character boundary is too uncertain. Tap the character again or use a cleaner photo.");
  if (areaPercent < 1.2) blockers.push("The selected character is too small to reconstruct safely.");
  if (areaPercent > 56) blockers.push("The selection still contains too much paper or background.");
  if (extraction.contour.length < 12) blockers.push("The outline is too simple to preserve the drawing safely.");
  if (!poseReady && !topologyReady) warnings.push("The automatic skeleton is not reliable yet. Review the joints before asking it to move.");
  if (!poseReady && !topologyReady && learnedBranches < 2) warnings.push("Arms and legs were not verified. Movement stays locked instead of inventing anatomy.");

  const cutoutReady = blockers.length === 0;
  const motionReady = cutoutReady && (poseReady || topologyReady) && learnedBranches >= 1;
  const score = clamp01(
    validation.score * 0.46
    + cutoutConfidence * 0.28
    + Math.min(1, extraction.contour.length / 48) * 0.12
    + (poseReady ? 0.08 : 0)
    + (topologyReady ? 0.06 : 0),
  );
  return {
    cutoutReady,
    motionReady,
    score: Number(score.toFixed(3)),
    blockers,
    warnings,
    evidence: validation.evidence,
  };
}
