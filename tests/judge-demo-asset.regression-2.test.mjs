import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("judge demo uses the earlier high-detail AniGen character and input artwork", async () => {
  const [anigen, drawing, page] = await Promise.all([
    readFile(new URL("../app/lib/anigen.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/lib/drawing.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
  ]);

  const bundledAsset = anigen.slice(anigen.indexOf("export function createBundledAniGenAsset"));
  assert.match(bundledAsset, /source: "anigen-demo"/);
  assert.match(bundledAsset, /meshUrl: "\/anigen-demo\.glb"/);
  assert.doesNotMatch(bundledAsset, /pip-neural-demo\.glb/);
  assert.match(drawing, /extractDrawingFromImageUrl\("\/anigen-demo-input\.png"/);
  assert.match(page, /createCharacter\(\{ name: "Sunny"/);
  assert.match(page, /Sunny Finds a Brave Hello/);
});
