import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// Regression: ISSUE-002 — a decorative mock-store CTA was exposed as a dead button
// Found by /qa on 2026-09-03
// Report: .gstack/qa-reports/qa-report-wallalive-webmcp-2026-09-03.md
test("storefront mockup never advertises a non-functional action", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(page, /<span className="storefront-cta" aria-hidden="true">/);
  assert.doesNotMatch(page, /<button tabIndex=\{-1\}>\{creatorDrop\.threeDExperience/);
  assert.match(css, /\.storefront-cta\s*\{/);
  assert.doesNotMatch(css, /\.store-hero button/);
});
