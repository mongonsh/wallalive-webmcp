import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

// Regression: ISSUE-004 — npm test skipped every focused ML regression file
// Found by /qa on 2026-08-31
// Report: .gstack/qa-reports/qa-report-localhost-2026-08-31.md
test("the standard test command includes every test module", async () => {
  const packageJson = JSON.parse(await readFile(new URL("../package.json", import.meta.url), "utf8"));

  assert.match(packageJson.scripts.test, /node --test tests\/\*\.test\.mjs/);
});
