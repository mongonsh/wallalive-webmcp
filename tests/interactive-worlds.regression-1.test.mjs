import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

test("every story world exposes real raycastable objects", async () => {
  const stage = await readFile(new URL("../app/components/ARStage.tsx", import.meta.url), "utf8");
  for (const id of ["studio-projector", "storybook-gate", "wizard-spell-book", "wizard-portal", "museum-sculpture"]) {
    assert.match(stage, new RegExp(id));
  }
  assert.match(stage, /THREE\.Raycaster/);
  assert.match(stage, /wallaliveInteraction/);
  assert.match(stage, /onWorldInteraction/);
});

test("WebMCP can inspect a quest and touch only listed scene objects", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");
  assert.match(page, /name: "inspect_shared_room"/);
  assert.match(page, /name: "interact_story_world"/);
  assert.match(page, /activity\.objectIds\.includes\(objectId\)/);
  assert.match(page, /pixelsIncluded: false/);
});
