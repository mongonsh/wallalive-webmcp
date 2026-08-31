import test from "node:test";
import assert from "node:assert/strict";

import { extractionSearchWindow } from "../app/lib/drawing.ts";

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
});
