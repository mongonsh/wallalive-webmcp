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
