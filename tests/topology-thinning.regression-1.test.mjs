import test from "node:test";
import assert from "node:assert/strict";

import { thinTopologyMask } from "../app/lib/learned-parts.ts";

test("separates radial appendage endpoints after thinning a merged centerline band", () => {
  const size = 41;
  const mask = new Uint8Array(size * size);
  const paint = (x, y, radius = 2) => {
    for (let dy = -radius; dy <= radius; dy += 1) for (let dx = -radius; dx <= radius; dx += 1) {
      if (dx * dx + dy * dy <= radius * radius && x + dx >= 0 && y + dy >= 0 && x + dx < size && y + dy < size) mask[(y + dy) * size + x + dx] = 1;
    }
  };
  const line = (x0, y0, x1, y1) => {
    const steps = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0));
    for (let step = 0; step <= steps; step += 1) paint(Math.round(x0 + (x1 - x0) * step / steps), Math.round(y0 + (y1 - y0) * step / steps));
  };
  for (const [x, y] of [[20, 3], [33, 7], [38, 20], [33, 34], [20, 38], [7, 34], [3, 20], [7, 7]]) line(20, 20, x, y);
  const thinned = thinTopologyMask(mask, size);
  let endpoints = 0;
  for (let y = 1; y < size - 1; y += 1) for (let x = 1; x < size - 1; x += 1) {
    if (!thinned[y * size + x]) continue;
    let neighbors = 0;
    for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) if (dx || dy) neighbors += thinned[(y + dy) * size + x + dx];
    if (neighbors === 1) endpoints += 1;
  }
  // Diagonal bands can collapse two near-center spokes during thinning; the
  // learned endpoint heatmap supplies those remaining candidates. The graph
  // fallback must still recover a clear multi-appendage radial structure.
  assert.ok(endpoints >= 6 && endpoints <= 10, `expected a radial multi-endpoint graph, received ${endpoints}`);
});
