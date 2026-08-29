# CutRoom

**Direct the intention. Let the agent find the coverage.**

CutRoom is a human-agent storyboard studio built for the WebMCP Challenge. A filmmaker sets the creative anchor, then any compatible browser agent can inspect the board, propose coverage, lock decisions, branch an alternate cut, and audit continuity inside the same visible interface.

The product does not embed a model or require an API key. The visiting browser agent supplies the intelligence; CutRoom supplies the shared state, narrow tools, creative guardrails, and provenance.

## Why this is a WebMCP-native product

This is not a chat box beside a storyboard generator. WebMCP is the collaboration layer:

- The agent reads the exact live board instead of relying on screenshots or DOM guessing.
- Tool calls and human actions update the same application state.
- Director locks are enforced inside every write path, including agent tools.
- Alternate cuts preserve the original rather than overwriting creative work.
- Every action is visible, attributable, locally persisted, and easy to verify.
- A continuity tool returns exact shot IDs and repairable issues.

## WebMCP tool surface

| Tool | Mode | Purpose |
| --- | --- | --- |
| `inspect_storyboard` | Read | Returns the scene, active cut, shots, branches, selection, locks, and version. |
| `create_shot` | Write | Appends one structured shot to the active cut. |
| `update_shot` | Write | Revises an unlocked shot; rejects director-locked shots. |
| `lock_creative_decision` | Write | Protects a shot from future agent edits. Agent locks are one-way. |
| `expand_sequence` | Write | Adds up to six proposed coverage beats atomically. |
| `create_alternate_cut` | Write | Clones the active cut, inherits locks, and changes only unlocked shots. |
| `check_continuity` | Read | Audits umbrella visibility, screen direction, and wardrobe. |
| `select_cut` | Write | Switches the visible working branch without deleting or merging. |

All tools use strict JSON schemas, narrow inputs, cancellation signals, annotations, shared validation, and result payloads that include enough state to verify the outcome.

## Fast demo

Open the app and press **Play Judge Demo**. In under five seconds it demonstrates the complete collaboration loop:

1. Inspect the director's seed and creative lock.
2. Expand three beats into a six-shot sequence.
3. Lock the insert and create an alternate cut without changing Cut A.
4. Run continuity and identify the intentional screen-direction risk by shot number.

In a WebMCP-enabled browser, a visiting agent can perform the same flow using the registered tools. A strong first prompt is:

> Inspect this storyboard. Expand it to six shots while preserving every director lock. Lock the strongest insert, create an alternate cut that reveals the stranger earlier, then check continuity and summarize every change.

## Local development

Requires Node.js `>=22.13.0`.

```bash
npm ci
npm run dev
```

Validation:

```bash
npm run lint
npm test
```

## Architecture

- React 19 + TypeScript, built with vinext for Cloudflare/Sites
- `document.modelContext.registerTool()` imperative WebMCP integration
- One canonical mutation layer shared by the UI and tool executors
- Local-first persistence through `localStorage`
- Deterministic paper-cut storyboard art rendered with HTML and CSS
- No account, backend, model dependency, uploaded screenplay, or API key

## Design decisions

CutRoom deliberately targets a narrow but real moment in creative work: a director has an intention worth protecting and needs coverage options quickly. The product's novelty is the collaboration contract—not generic image generation. Human locks, safe branching, continuity checks, provenance, and reversible exploration make the browser agent feel like a respectful creative collaborator.

## License

MIT — see [LICENSE](./LICENSE).
