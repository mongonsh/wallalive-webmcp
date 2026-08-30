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

test("server-renders the WallAlive product shell and security headers", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  assert.equal(response.headers.get("permissions-policy"), "tools=(self), camera=(self), xr-spatial-tracking=(self)");
  assert.equal(response.headers.get("x-content-type-options"), "nosniff");

  const html = await response.text();
  assert.match(html, /<title>WallAlive — Draw it\. Wake it\. Play\.<\/title>/i);
  assert.match(html, /What if their drawing/);
  assert.match(html, /jumped off the wall/);
  assert.match(html, /START CAMERA/);
  assert.match(html, /PLAY JUDGE DEMO/);
  assert.match(html, /CAMERA-SAFE BY DESIGN/);
  assert.doesNotMatch(html, /codex-preview|react-loading-skeleton|Starter Project|CutRoom/i);
});

test("registers eight strict WebMCP tools without camera authority", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  const expectedTools = [
    "inspect_wall_scene",
    "create_character_from_drawing",
    "set_character_personality",
    "place_character",
    "animate_character",
    "recolor_character",
    "tell_character_story",
    "list_activity",
  ];

  for (const tool of expectedTools) assert.match(page, new RegExp(`name: ["']${tool}["']`));

  const registeredNames = [...page.matchAll(/name: ["']([^"']+)["']/g)].map((match) => match[1]);
  assert.equal(registeredNames.filter((name) => expectedTools.includes(name)).length, 8);
  assert.equal(registeredNames.some((name) => /camera|capture|upload/.test(name)), false);
  assert.match(page, /document\.modelContext/);
  assert.match(page, /registerTool\(tool, \{ signal: controller\.signal \}\)/);
  assert.match(page, /additionalProperties: false/);
  assert.match(page, /readOnlyHint/);
  assert.match(page, /Camera capture is human-only/);
  assert.match(page, /cameraFeedExposed: false/);
});

test("implements local drawing extraction and real WebXR hit testing", async () => {
  const drawing = await readFile(new URL("../app/lib/drawing.ts", import.meta.url), "utf8");
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");

  assert.match(drawing, /getImageData/);
  assert.match(drawing, /toDataURL/);
  assert.doesNotMatch(drawing, /fetch\(|XMLHttpRequest|WebSocket/);
  assert.match(stage, /isSessionSupported\("immersive-ar"\)/);
  assert.match(stage, /requestSession\("immersive-ar"/);
  assert.match(stage, /requiredFeatures: \["hit-test"\]/);
  assert.match(stage, /getHitTestResults/);
});
