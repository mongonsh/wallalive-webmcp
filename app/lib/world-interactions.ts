import type { ARWorld, WorldObjectInteraction } from "../components/ARStage";

const interactionCopy: Record<string, Omit<WorldObjectInteraction, "id" | "world">> = {
  "studio-projector": { label: "Story projector", verb: "play", story: "The projector turns the cast's drawings into a tiny shadow-movie scene." },
  "studio-maker-table": { label: "Maker table", verb: "build", story: "The maker table opens a prop-building challenge for the whole cast." },
  "storybook-firefly-0": { label: "Firefly 1", verb: "find", story: "A hidden firefly joins the cast's lantern trail." },
  "storybook-firefly-1": { label: "Firefly 2", verb: "find", story: "A hidden firefly joins the cast's lantern trail." },
  "storybook-firefly-2": { label: "Firefly 3", verb: "find", story: "A hidden firefly joins the cast's lantern trail." },
  "storybook-gate": { label: "Castle gate", verb: "unlock", story: "The gate opens only when two characters cooperate." },
  "wizard-spell-book": { label: "Spell book", verb: "read", story: "The spell book gives every character a role based on its verified movements." },
  "wizard-crystal-a": { label: "Tide crystal", verb: "cast", story: "The tide crystal teaches the cast to match color, sound, and motion." },
  "wizard-crystal-b": { label: "Moon crystal", verb: "cast", story: "The moon crystal asks one character to float while another spins." },
  "wizard-crystal-c": { label: "Sun crystal", verb: "cast", story: "The sun crystal completes the cooperative spell sequence." },
  "wizard-portal": { label: "Living portal", verb: "enter", story: "The portal reveals the next chapter after the spell ingredients are found." },
  "museum-art-0": { label: "River of Shapes", verb: "curate", story: "This artwork asks the cast to compare color, shape, and feeling." },
  "museum-art-1": { label: "The Brave Mark", verb: "curate", story: "This artwork asks the cast to compare color, shape, and feeling." },
  "museum-art-2": { label: "Golden Echo", verb: "curate", story: "This artwork asks the cast to compare color, shape, and feeling." },
  "museum-sculpture": { label: "Motion sculpture", verb: "inspect", story: "The sculpture turns as the cast tells a new interpretation together." },
};

export function getAccessibleWorldInteraction(world: ARWorld, objectId: string): WorldObjectInteraction | null {
  const copy = interactionCopy[objectId];
  if (!copy || !objectId.startsWith(`${world}-`)) return null;
  return { id: objectId, world, ...copy };
}

export const accessibleWorldObjectIds = Object.freeze(Object.keys(interactionCopy));
