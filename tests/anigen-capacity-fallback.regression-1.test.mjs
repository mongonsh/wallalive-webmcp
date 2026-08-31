import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const anigen = await readFile(new URL("../app/lib/anigen.ts", import.meta.url), "utf8");
const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

test("classifies shared GPU capacity and network failures as recoverable provider outages", () => {
  assert.match(anigen, /export class AniGenUnavailableError extends Error/);
  assert.match(anigen, /reason: "capacity" \| "network"/);
  assert.match(anigen, /new AniGenUnavailableError\("capacity"/);
  assert.match(anigen, /new AniGenUnavailableError\("network"/);
  assert.match(anigen, /space metadata could not be loaded/);
  assert.match(anigen, /export function isAniGenUnavailableError/);
});

test("continues with an honest private 3D fallback instead of stopping on public GPU quota", () => {
  assert.match(page, /if \(isAniGenUnavailableError\(error\)\)/);
  assert.match(page, /localFallbackRef\.current = true/);
  assert.match(page, /setLocalFallbackActive\(true\)/);
  assert.match(page, /createCharacter\(\{ name: "Pip"/);
  assert.match(page, /Private on-device 3D is ready/);
  assert.match(page, /ON-DEVICE 3D · PRIVATE/);
  assert.doesNotMatch(page, /public GPU time is temporarily full/);
});
