const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

export type InkColorSample = { r: number; g: number; b: number };

const clampByte = (value: number) => Math.min(255, Math.max(0, value));

function enhanceInkColor(color: InkColorSample) {
  const average = (color.r + color.g + color.b) / 3;
  return {
    r: clampByte(average + (color.r - average) * 2.35),
    g: clampByte(average + (color.g - average) * 2.35),
    b: clampByte(average + (color.b - average) * 2.35),
  };
}

/**
 * Finds the dominant drawn hue without letting a small, highly saturated
 * background pattern (for example blue graph paper) overpower a longer ink
 * stroke. Neutral drawings fall back to their darkest sampled pixels.
 */
export function selectDominantInkColor(samples: InkColorSample[]) {
  if (!samples.length) return null;
  const binCount = 24;
  const votes = new Float64Array(binCount);
  const red = new Float64Array(binCount);
  const green = new Float64Array(binCount);
  const blue = new Float64Array(binCount);
  let totalVotes = 0;

  for (const sample of samples) {
    const maximum = Math.max(sample.r, sample.g, sample.b);
    const minimum = Math.min(sample.r, sample.g, sample.b);
    const chroma = maximum - minimum;
    const darkness = 255 - (sample.r + sample.g + sample.b) / 3;
    if (chroma < 4 || darkness < 8) continue;
    let hue = maximum === sample.r
      ? ((sample.g - sample.b) / chroma + 6) % 6
      : maximum === sample.g ? (sample.b - sample.r) / chroma + 2 : (sample.r - sample.g) / chroma + 4;
    hue *= 60;
    const bin = Math.floor(hue / (360 / binCount)) % binCount;
    // Linear chroma rewards a real colored stroke without allowing a handful
    // of vivid grid pixels to dominate thousands of softer ink pixels.
    const vote = chroma * Math.sqrt(Math.max(1, darkness));
    votes[bin] += vote;
    red[bin] += sample.r * vote;
    green[bin] += sample.g * vote;
    blue[bin] += sample.b * vote;
    totalVotes += vote;
  }

  if (totalVotes > 0) {
    let winner = 0;
    let winningVote = -Infinity;
    for (let index = 0; index < binCount; index += 1) {
      const neighborhood = votes[index]
        + votes[(index - 1 + binCount) % binCount] * 0.72
        + votes[(index + 1) % binCount] * 0.72;
      if (neighborhood > winningVote) {
        winner = index;
        winningVote = neighborhood;
      }
    }
    const selected = [(winner - 1 + binCount) % binCount, winner, (winner + 1) % binCount];
    const selectedVote = selected.reduce((total, index) => total + votes[index], 0);
    if (selectedVote >= totalVotes * 0.14) {
      return enhanceInkColor({
        r: selected.reduce((total, index) => total + red[index], 0) / selectedVote,
        g: selected.reduce((total, index) => total + green[index], 0) / selectedVote,
        b: selected.reduce((total, index) => total + blue[index], 0) / selectedVote,
      });
    }
  }

  const darkest = [...samples]
    .sort((a, b) => (a.r + a.g + a.b) - (b.r + b.g + b.b))
    .slice(0, Math.max(1, Math.ceil(samples.length * 0.3)));
  return enhanceInkColor({
    r: darkest.reduce((total, sample) => total + sample.r, 0) / darkest.length,
    g: darkest.reduce((total, sample) => total + sample.g, 0) / darkest.length,
    b: darkest.reduce((total, sample) => total + sample.b, 0) / darkest.length,
  });
}

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
