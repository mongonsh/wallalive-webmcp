import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { normalizeUsername, sharedRoomInviteUrl, validateSharedOperation } from "../app/lib/collaboration.ts";

test("shared drawing operations are compact, normalized, and identity-attributed", () => {
  const operation = validateSharedOperation({
    id: "op-1",
    participantId: "person-1",
    author: "mika<script>",
    kind: "gesture",
    tool: "brush",
    color: "#ff674d",
    size: 18,
    points: [{ x: -2, y: .5, pressure: 9 }, { x: .8, y: 4, pressure: .4 }],
    seed: 42,
    createdAt: "2026-09-02T00:00:00.000Z",
  });
  assert.ok(operation);
  assert.equal(operation.author, "mikascript");
  assert.deepEqual(operation.points, [{ x: 0, y: .5, pressure: 1 }, { x: .8, y: 1, pressure: .4 }]);
});

test("room invite URLs carry only room code and sanitized guest handle", () => {
  const url = new URL(sharedRoomInviteUrl("https://wallalive.example/play", "MOON7", "sora & team"));
  assert.equal(url.searchParams.get("room"), "MOON7");
  assert.equal(url.searchParams.get("invite"), normalizeUsername("sora & team"));
  assert.equal(url.searchParams.has("token"), false);
});

test("the D1 room endpoint stores vector operations, not artwork pixels", async () => {
  const source = await readFile(new URL("../app/api/rooms/route.ts", import.meta.url), "utf8");
  assert.match(source, /shared_drawing_ops/);
  assert.match(source, /validateSharedOperation/);
  assert.doesNotMatch(source, /dataUrl|base64|image_blob/);
  assert.match(source, /token_hash/);
});

test("the drawing wall replays remote operations and prevents divergent shared undo", async () => {
  const source = await readFile(new URL("../app/components/DrawingWall.tsx", import.meta.url), "utf8");
  assert.match(source, /renderSharedOperation/);
  assert.match(source, /onSharedOperation/);
  assert.match(source, /Undo is disabled in a live shared room/);
});

test("the mobile drawing wall keeps touch coordinates aligned and preserves unfinished work", async () => {
  const source = await readFile(new URL("../app/components/DrawingWall.tsx", import.meta.url), "utf8");
  const css = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const canvasRule = css.match(/\.wall-canvas-wrap canvas \{[\s\S]*?\n\}/)?.[0] ?? "";

  assert.match(source, /initializedWallRef/);
  assert.match(source, /onPointerCancel=\{pointerUp\}/);
  assert.match(source, /className="wall-guide"/);
  assert.match(canvasRule, /width:\s*auto/);
  assert.match(canvasRule, /height:\s*auto/);
  assert.match(canvasRule, /aspect-ratio:\s*1400\s*\/\s*850/);
  assert.doesNotMatch(canvasRule, /object-fit:\s*contain/);
});

test("an invitation wins over a stale room session and explains the three-step handoff", async () => {
  const panel = await readFile(new URL("../app/components/SharedRoomPanel.tsx", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(panel, /const joiningInvitation = Boolean\(invitedRoom/);
  assert.match(panel, /!session \|\| joiningInvitation/);
  assert.match(panel, /JOIN THIS ROOM/);
  assert.match(panel, /SHARE INVITE/);
  assert.match(panel, /1 · Share link/);
  assert.match(page, /onOpenWall=\{\(\) => \{ setSharedRoomOpen\(false\); setDismissedInvite\(search\); setDrawingWallOpen\(true\); \}\}/);
});

test("the first screen states the human-agent value in one sentence", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /You draw\. The agent guides\. You approve\./);
  assert.match(page, /ASK CHATGPT/);
  assert.match(page, /COPY FAMILY PROMPT/);
});
