import test from "node:test";
import assert from "node:assert/strict";

import { extractionSearchWindow, hasMeaningfulSelectedAlpha, recoverTargetSilhouette } from "../app/lib/drawing.ts";

// Regression: uploaded photos reused the tight live-camera focus ellipse and
// cut the user's top ears and lower feet out before semantic ML could run.
test("selected images scan the full centered character while camera capture stays clutter-bounded", () => {
  const camera = extractionSearchWindow(480, 580, "camera");
  const selected = extractionSearchWindow(480, 580, "selected-image");

  assert.ok(selected.scanInsetTop < camera.scanInsetTop);
  assert.ok(selected.scanInsetBottom < camera.scanInsetBottom);
  assert.ok(selected.focusRadiusX > camera.focusRadiusX * 1.6);
  assert.ok(selected.focusRadiusY > camera.focusRadiusY * 1.4);
  assert.ok(selected.scanInsetTop / 580 <= 0.03);
  assert.ok(selected.scanInsetBottom / 580 <= 0.04);
  const nearCorner = ((480 * 0.47) / selected.focusRadiusX) ** 2 + ((580 * 0.47) / selected.focusRadiusY) ** 2;
  assert.ok(nearCorner < 1, "the selected-image search ellipse must include diagonal appendages");
});

test("large uploaded-photo masks do not overflow the JavaScript argument stack", () => {
  const width = 480;
  const height = 480;
  const mask = new Uint8Array(width * height);
  for (let y = 36; y < height - 36; y += 1) for (let x = 34; x < width - 34; x += 1) mask[y * width + x] = 1;
  const recovered = recoverTargetSilhouette(mask, width, height, { x: 0.5, y: 0.5 });
  assert.equal(recovered.length, mask.length);
  assert.equal(recovered[Math.floor(height / 2) * width + Math.floor(width / 2)], 1);
});

test("selected transparent artwork uses its authored alpha instead of splitting colored fills at dark outlines", () => {
  const data = new Uint8ClampedArray(100 * 4);
  for (let pixel = 0; pixel < 38; pixel += 1) data[pixel * 4 + 3] = 255;
  assert.equal(hasMeaningfulSelectedAlpha(data, "selected-image"), true);
  assert.equal(hasMeaningfulSelectedAlpha(data, "camera"), false);
  data.fill(255, 3);
  assert.equal(hasMeaningfulSelectedAlpha(data, "selected-image"), false);
});
