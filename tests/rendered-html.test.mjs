import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the CutRoom product shell", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>CutRoom — Direct the intention<\/title>/i);
  assert.match(html, /Direct the intention/);
  assert.match(html, /The Red Umbrella/);
  assert.match(html, /PLAY JUDGE DEMO/);
  assert.match(html, /CREATIVE LOCK/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project/i);
});

test("registers a strict, guardrailed WebMCP collaboration surface", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const expectedTools = [
    "inspect_storyboard",
    "create_shot",
    "update_shot",
    "lock_creative_decision",
    "expand_sequence",
    "create_alternate_cut",
    "check_continuity",
    "select_cut",
  ];

  for (const tool of expectedTools) {
    assert.match(page, new RegExp(`name: ["']${tool}["']`));
  }

  assert.match(page, /document\.modelContext/);
  assert.match(page, /registerTool\(tool, \{ signal: controller\.signal \}\)/);
  assert.match(page, /additionalProperties: false/);
  assert.match(page, /readOnlyHint/);
  assert.match(page, /director-locked\. Create an alternate cut instead/);
  assert.match(page, /localStorage\.setItem/);
  assert.match(page, /checkBranch\(/);
});
