export const CREATOR_PRODUCT_IDS = ["art-print", "sticker-sheet", "t-shirt", "tote-bag", "ceramic-mug"] as const;

export type CreatorProductId = (typeof CREATOR_PRODUCT_IDS)[number];
export type CreatorAudience = "family" | "classroom" | "community";
export type CreatorGoal = "keepsake" | "fundraiser" | "portfolio";
export type CreatorVibe = "sunny" | "storybook" | "bold" | "museum";

export type ArtworkCommerceProfile = {
  characterName: string;
  figureCount: number;
  aspectRatio: number;
  coveragePercent: number;
  edgeEnergy: "soft" | "scribbly" | "bold";
  dominantColor: string;
  secondaryColor: string;
  movableParts: number;
  semanticParts: string[];
  hasRigged3D: boolean;
  activeWorld: string;
  storyTitle: string;
};

export type CreatorRecommendation = {
  id: CreatorProductId;
  label: string;
  glyph: string;
  score: number;
  price: number;
  reason: string;
  placement: string;
};

export type CreatorDrop = {
  id: string;
  name: string;
  story: string;
  audience: CreatorAudience;
  goal: CreatorGoal;
  vibe: CreatorVibe;
  status: "awaiting-adult-approval" | "approved-for-export";
  createdAt: string;
  source: "human-approved-artwork";
  palette: { ink: string; paper: string; accent: string; highlight: string };
  products: CreatorRecommendation[];
  storefront: {
    announcement: string;
    heroHeading: string;
    heroCopy: string;
    sections: Array<{ type: string; heading: string; purpose: string }>;
  };
  threeDExperience: {
    enabled: boolean;
    heroMode: "interactive-model" | "artwork-poster";
    activeWorld: string;
    motionPrompt: string;
    shopifyMediaNote: string;
  };
  safety: {
    requiresAdultApproval: true;
    publishesToShopify: false;
    createsOrders: false;
    includesImagePixelsInToolResult: false;
  };
};

const PRODUCT_META: Record<CreatorProductId, Omit<CreatorRecommendation, "score" | "reason">> = {
  "art-print": { id: "art-print", label: "Art print", glyph: "▧", price: 18, placement: "Centered, 88% of printable area" },
  "sticker-sheet": { id: "sticker-sheet", label: "Sticker sheet", glyph: "✦", price: 9, placement: "Individual figures, 4 mm safe cut line" },
  "t-shirt": { id: "t-shirt", label: "T-shirt", glyph: "♧", price: 28, placement: "Front chest, 28 cm maximum width" },
  "tote-bag": { id: "tote-bag", label: "Tote bag", glyph: "▱", price: 24, placement: "Centered front, 24 cm maximum width" },
  "ceramic-mug": { id: "ceramic-mug", label: "Mug", glyph: "◒", price: 19, placement: "Two-sided wrap, face kept away from handle" },
};

const roundScore = (score: number) => Math.max(1, Math.min(99, Math.round(score)));

export function recommendCreatorProducts(
  profile: ArtworkCommerceProfile,
  goal: CreatorGoal,
  audience: CreatorAudience,
): CreatorRecommendation[] {
  const ensemble = profile.figureCount > 1;
  const detailed = profile.semanticParts.length >= 5 || profile.edgeEnergy === "scribbly";
  const wide = profile.aspectRatio > 1.18;
  const compact = profile.coveragePercent < 42;

  const score: Record<CreatorProductId, number> = {
    "art-print": 79 + (detailed ? 12 : 0) + (goal === "portfolio" ? 8 : 0),
    "sticker-sheet": 72 + (ensemble ? 18 : 0) + (audience === "classroom" ? 8 : 0),
    "t-shirt": 70 + (!wide ? 10 : -5) + (profile.movableParts >= 2 ? 4 : 0),
    "tote-bag": 66 + (wide ? 9 : 0) + (goal === "fundraiser" ? 10 : 0),
    "ceramic-mug": 62 + (!ensemble && compact ? 14 : 0) + (audience === "family" ? 5 : 0),
  };

  const reason: Record<CreatorProductId, string> = {
    "art-print": detailed
      ? "Best match for preserving the drawing’s small marks and original color story."
      : "Gives the full silhouette room to breathe without cropping the artwork.",
    "sticker-sheet": ensemble
      ? `Keeps all ${profile.figureCount} characters separate, so every figure becomes its own sticker.`
      : "Turns expressive parts into a playful, low-cost mini collection.",
    "t-shirt": wide
      ? "Works as a smaller chest emblem so the wide silhouette stays readable."
      : "The compact silhouette reads clearly from a distance on a front print.",
    "tote-bag": goal === "fundraiser"
      ? "A practical, giftable item that suits a school or community fundraiser."
      : "The generous flat print area preserves the character’s outer shape.",
    "ceramic-mug": compact
      ? "The centered character fits the curved print zone without losing its face."
      : "A two-sided layout can pair the character with its name or short story.",
  };

  return CREATOR_PRODUCT_IDS
    .map((id) => ({ ...PRODUCT_META[id], score: roundScore(score[id]), reason: reason[id] }))
    .sort((a, b) => b.score - a.score);
}

export function stageCreatorDrop(input: {
  profile: ArtworkCommerceProfile;
  dropName?: string;
  story?: string;
  audience?: CreatorAudience;
  goal?: CreatorGoal;
  vibe?: CreatorVibe;
  productIds?: CreatorProductId[];
  now?: string;
}): CreatorDrop {
  const audience = input.audience ?? "family";
  const goal = input.goal ?? "keepsake";
  const vibe = input.vibe ?? "sunny";
  const recommendations = recommendCreatorProducts(input.profile, goal, audience);
  const selected = input.productIds?.length
    ? recommendations.filter((item) => input.productIds?.includes(item.id)).slice(0, 4)
    : recommendations.slice(0, 3);
  const name = (input.dropName?.trim() || `${input.profile.characterName || "My Drawing"}’s Little World`).slice(0, 72);
  const story = (input.story?.trim() || `${input.profile.characterName || "A new friend"} began as a drawing and learned how to move, tell a story, and step into a tiny collection.`).slice(0, 240);

  return {
    id: `drop-${Date.now().toString(36)}`,
    name,
    story,
    audience,
    goal,
    vibe,
    status: "awaiting-adult-approval",
    createdAt: input.now ?? new Date().toISOString(),
    source: "human-approved-artwork",
    palette: {
      ink: "#173b3a",
      paper: "#fff8ea",
      accent: input.profile.dominantColor,
      highlight: input.profile.secondaryColor,
    },
    products: selected,
    storefront: {
      announcement: audience === "classroom" ? "Made from imagination in our classroom" : "A tiny collection from a very big imagination",
      heroHeading: name,
      heroCopy: story,
      sections: [
        { type: "image-banner", heading: name, purpose: "Introduce the artwork and collection story." },
        { type: "custom-liquid", heading: "Meet the character in 3D", purpose: "Embed the adult-approved GLB as an interactive model with a poster fallback." },
        { type: "featured-collection", heading: "Meet the collection", purpose: "Show the recommended products in one calm shelf." },
        { type: "rich-text", heading: "From drawing to 3D", purpose: "Explain the creative process without exposing private images." },
        { type: "multicolumn", heading: "Made with care", purpose: "State adult approval, print checks, and creator credit." },
      ],
    },
    threeDExperience: {
      enabled: input.profile.hasRigged3D,
      heroMode: input.profile.hasRigged3D ? "interactive-model" : "artwork-poster",
      activeWorld: input.profile.activeWorld,
      motionPrompt: input.profile.storyTitle || `${input.profile.characterName} waves hello`,
      shopifyMediaNote: input.profile.hasRigged3D
        ? "Upload the exported GLB as Shopify product media after adult review; preserve the artwork poster as the accessible fallback."
        : "Use the approved artwork poster until an adult-approved GLB is available.",
    },
    safety: {
      requiresAdultApproval: true,
      publishesToShopify: false,
      createsOrders: false,
      includesImagePixelsInToolResult: false,
    },
  };
}

const csvCell = (value: string | number | boolean) => `"${String(value).replaceAll('"', '""')}"`;

export function buildShopifyProductsCsv(drop: CreatorDrop) {
  const header = [
    "Handle", "Title", "Body (HTML)", "Vendor", "Product Category", "Type", "Tags", "Published",
    "Option1 Name", "Option1 Value", "Variant SKU", "Variant Price", "Variant Requires Shipping", "Variant Taxable", "Status",
  ];
  const rows = drop.products.map((product, index) => {
    const handle = `${drop.name}-${product.id}`.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64);
    return [
      handle,
      `${drop.name} — ${product.label}`,
      `<p>${drop.story}</p><p><strong>Print placement:</strong> ${product.placement}</p>`,
      "WallAlive Creator Studio",
      "Arts & Entertainment > Artwork",
      product.label,
      `wallalive,creator-drop,${drop.audience},${drop.goal}`,
      false,
      "Title",
      "Default Title",
      `WA-${drop.id.slice(-6).toUpperCase()}-${index + 1}`,
      product.price.toFixed(2),
      true,
      true,
      "draft",
    ];
  });
  return [header, ...rows].map((row) => row.map(csvCell).join(",")).join("\n");
}

export function buildShopifyStoreBlueprint(drop: CreatorDrop) {
  return JSON.stringify({
    wallaliveVersion: 1,
    importMode: "draft-only",
    collection: {
      title: drop.name,
      description: drop.story,
      handle: drop.name.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 64),
    },
    theme: {
      palette: drop.palette,
      template: {
        sections: Object.fromEntries(drop.storefront.sections.map((section, index) => [`wallalive_${index + 1}`, { type: section.type, settings: { heading: section.heading, text: section.purpose } }])),
        order: drop.storefront.sections.map((_section, index) => `wallalive_${index + 1}`),
      },
    },
    threeDExperience: drop.threeDExperience,
    products: drop.products,
    storefrontMcpHandoff: {
      discovery: "Use the Shopify storefront's native WebMCP tools or Storefront MCP after these drafts are imported and reviewed.",
      allowedNextActions: ["search catalog", "recommend live products", "manage cart after shopper confirmation"],
      wallaliveNeverDoes: ["publish products", "charge payment", "place order"],
    },
    safety: drop.safety,
  }, null, 2);
}

export function buildCreatorHandoff(drop: CreatorDrop) {
  return [
    `# ${drop.name}`,
    "",
    drop.story,
    "",
    "## Adult review checklist",
    "",
    "- Confirm creator name/credit and permission to share the artwork.",
    "- Check every print placement and product title in Shopify draft mode.",
    "- Add print-provider images and fulfillment details inside Shopify.",
    "- Preview the storefront on mobile before publishing.",
    "- Keep products unpublished until an adult intentionally approves them.",
    "",
    "## Agent handoff",
    "",
    "After import and human review, use Shopify's native WebMCP storefront tools for live catalog search and cart management. WallAlive does not publish, purchase, or charge.",
  ].join("\n");
}
