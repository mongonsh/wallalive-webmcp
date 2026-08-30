import test from "node:test";
import assert from "node:assert/strict";
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";

// Regression: ISSUE-006 — Three.js inflated the initial page chunk above 700 kB
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("keeps the initial page chunk below the build warning threshold", async () => {
  const chunkDirectory = new URL("../dist/client/_next/static/chunks/", import.meta.url);
  const files = (await readdir(chunkDirectory)).filter((file) => file.startsWith("page-") && file.endsWith(".js"));
  assert.ok(files.length > 0, "the production build must emit a page chunk");
  const sizes = await Promise.all(files.map(async (file) => (await stat(join(chunkDirectory.pathname, file))).size));

  assert.ok(Math.max(...sizes) < 500_000, `initial page chunk is still ${Math.max(...sizes).toLocaleString()} bytes`);
});
