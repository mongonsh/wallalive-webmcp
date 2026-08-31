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

test("stops safely instead of substituting a rounded shell when public GPU capacity is unavailable", () => {
  assert.match(page, /if \(isAniGenUnavailableError\(error\)\)/);
  assert.match(page, /Nothing fake was created/);
  assert.match(page, /did not substitute a rounded shell/);
  assert.doesNotMatch(page, /localFallbackRef/);
  assert.doesNotMatch(page, /setLocalFallbackActive/);
  assert.doesNotMatch(page, /public GPU time is temporarily full/);
});
