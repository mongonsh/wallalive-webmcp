# CutRoom demo script (2:20 target)

## 0:00–0:18 — The problem

**Voiceover:** “Storyboard tools can generate pictures. But directing is deciding what must not change. CutRoom turns a browser agent into a creative collaborator that can explore without erasing the director’s intent.”

Show the initial three-shot board. Point to Shot 01 and the red-umbrella creative lock.

## 0:18–0:40 — WebMCP is the collaboration layer

Open the **Tools** panel.

**Voiceover:** “CutRoom exposes eight narrow WebMCP tools. There is no embedded model or API key. The visiting browser agent reads and acts on the exact live board through structured schemas.”

Prompt the browser agent:

> Inspect this storyboard and tell me what the director has locked.

Show `inspect_storyboard` in the Live Relay and the activity history.

## 0:40–1:08 — Expand without losing intent

Prompt:

> Expand this to six shots. Keep the red umbrella visible, maintain screen direction and wardrobe, and build suspense without revealing the stranger directly.

Show three new paper-cut frames appear. Select one agent-created shot and edit its intention manually.

**Voiceover:** “Human edits and agent calls use the same state and validation. Every result is immediately visible and locally persisted.”

## 1:08–1:30 — The guardrail moment

Prompt:

> Change Shot 01 into a close-up and remove the umbrella.

Show `update_shot` reject the request because Shot 01 is director-locked.

**Voiceover:** “The lock is enforced in the write path, not merely drawn in the interface. An agent cannot quietly override it.”

## 1:30–1:57 — Branch safely

Prompt:

> Create an alternate cut called “Stranger First” that reveals the stranger earlier. Preserve locked shots.

Switch between Cut A and Cut B to prove the original is untouched. Point to inherited lock badges.

## 1:57–2:15 — Catch the mistake

Prompt:

> Check continuity in this alternate cut.

Show the intentional screen-direction issue returned with its exact shot number.

**Voiceover:** “The agent can now repair a precise issue instead of regenerating the scene.”

## 2:15–2:20 — Close

**Voiceover:** “CutRoom: direct the intention, let the agent find the coverage.”

End on the full board with the WebMCP inspector visible.

## Recording notes

- Record at 1440×900 or 1920×1080, browser zoom 90–100%.
- Hide bookmarks and notifications.
- Keep the final edit under three minutes.
- Use hard cuts; no long intro animation.
- Make the lock rejection, branch switch, and continuity result large enough to read.
