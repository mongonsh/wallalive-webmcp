const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

export function averageLogitConfidence(logits: Float32Array, indices: number[], channelOffset = 0) {
  if (!indices.length) return 0;
  return indices.reduce((total, index) => total + sigmoid(logits[channelOffset + index]), 0) / indices.length;
}

type SpatialHint = {
  kind: string;
  center: { x: number; y: number };
  size: { x: number; y: number };
  rotation: number;
};

export function sameSemanticInstance(candidate: SpatialHint, hint: SpatialHint) {
  if (candidate.kind !== hint.kind) return false;
  const separation = Math.hypot(candidate.center.x - hint.center.x, candidate.center.y - hint.center.y);
  const extent = Math.max(candidate.size.x, candidate.size.y, hint.size.x, hint.size.y) * 0.75;
  const minimumReach = hint.kind === "cheek" ? 0.085 : 0.065;
  const reach = Math.min(0.14, Math.max(minimumReach, extent));
  if (separation >= reach) return false;
  if (hint.kind !== "arm" && hint.kind !== "leg") return true;
  const rawAngle = Math.abs(candidate.rotation - hint.rotation) % Math.PI;
  return Math.min(rawAngle, Math.PI - rawAngle) < 0.52;
}
