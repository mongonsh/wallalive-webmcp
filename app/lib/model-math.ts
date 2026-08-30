const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));

export function averageLogitConfidence(logits: Float32Array, indices: number[], channelOffset = 0) {
  if (!indices.length) return 0;
  return indices.reduce((total, index) => total + sigmoid(logits[channelOffset + index]), 0) / indices.length;
}
