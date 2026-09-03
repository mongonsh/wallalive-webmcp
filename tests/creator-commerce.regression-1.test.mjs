import assert from "node:assert/strict";
import test from "node:test";

import {
  buildCreatorHandoff,
  buildShopifyProductsCsv,
  buildShopifyStoreBlueprint,
  recommendCreatorProducts,
  stageCreatorDrop,
} from "../app/lib/creator-commerce.ts";

const profile = {
  characterName: "Pip",
  figureCount: 3,
  aspectRatio: 1.08,
  coveragePercent: 61,
  edgeEnergy: "scribbly",
  dominantColor: "#ff765f",
  secondaryColor: "#62c8dc",
  movableParts: 7,
  semanticParts: ["body", "eye", "mouth", "arm", "leg", "ear"],
  hasRigged3D: true,
  activeWorld: "storybook",
  storyTitle: "Pip Finds a Brave Hello",
  contributors: ["mika", "sora"],
};

test("product recommendations use artwork evidence and creator intent", () => {
  const recommendations = recommendCreatorProducts(profile, "fundraiser", "classroom");
  assert.equal(recommendations.length, 8);
  assert.equal(recommendations[0].id, "story-zine");
  assert.match(recommendations[0].reason, /3 characters/);
  assert.ok(recommendations.every((item) => item.score >= 1 && item.score <= 99));
});

test("creator drop remains a draft until a visible adult approval", () => {
  const drop = stageCreatorDrop({
    profile,
    dropName: "Pip and Friends",
    story: "Three classroom drawings learn to cooperate.",
    audience: "classroom",
    goal: "fundraiser",
    vibe: "storybook",
    productIds: ["sticker-sheet", "art-print", "tote-bag"],
    now: "2026-09-02T00:00:00.000Z",
  });
  assert.equal(drop.status, "awaiting-adult-approval");
  assert.equal(drop.safety.publishesToShopify, false);
  assert.equal(drop.safety.createsOrders, false);
  assert.equal(drop.safety.includesImagePixelsInToolResult, false);
  assert.equal(drop.safety.contributorPermissionsRequired, true);
  assert.deepEqual(drop.contributors.map((item) => item.username), ["mika", "sora"]);
  assert.equal(drop.threeDExperience.heroMode, "interactive-model");
  assert.deepEqual(new Set(drop.products.map((item) => item.id)), new Set(["sticker-sheet", "art-print", "tote-bag"]));
});

test("Shopify handoff exports importable draft data and explicit safety guidance", () => {
  const drop = stageCreatorDrop({ profile, dropName: "Pip's World", audience: "family", goal: "keepsake" });
  const csv = buildShopifyProductsCsv(drop);
  const blueprint = JSON.parse(buildShopifyStoreBlueprint(drop));
  const handoff = buildCreatorHandoff(drop);

  assert.match(csv, /^"Handle","Title","Body \(HTML\)"/);
  assert.match(csv, /"draft"/);
  assert.equal(blueprint.importMode, "draft-only");
  assert.equal(blueprint.storefrontMcpHandoff.wallaliveNeverDoes.includes("place order"), true);
  assert.equal(blueprint.storefrontMcpHandoff.nativeTools.includes("search_catalog"), true);
  assert.match(handoff, /@mika/);
  assert.match(handoff, /Adult review checklist/);
  assert.match(handoff, /not connected to Shopify/);
});
