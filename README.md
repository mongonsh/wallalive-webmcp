# WallAlive

**Draw it. Wake it. Play.**

**Live demo:** https://wallalive-webmcp.mungunshagai-tb.chatgpt.site

**Public source:** https://github.com/mongonsh/wallalive-webmcp

WallAlive turns a child’s drawing into a layered 3D character that can live and perform in the real room. A child explicitly opens the camera and approves a drawing; then a compatible browser agent can name it, shape its personality, place it, animate it, recolor its generated depth, and direct a mini story through WebMCP.

The camera is intentionally **not** a WebMCP tool. Drawing extraction happens in browser memory, no image is uploaded, and the agent receives semantic shape/color analysis only after human approval.

## Why this needs WebMCP

WallAlive is a shared imagination surface, not a chat box attached to an AR demo:

- Human and agent actions update the same visible character and scene.
- The browser agent composes personality, placement, motion, and story through narrow tools.
- A human-only camera boundary is enforced by capability design, not prompt instructions.
- Every action is visibly attributed to `CHILD`, `BROWSER AGENT`, or `WALLALIVE`.
- Tool calls return verification-rich state so the agent can observe what actually happened.
- The app has no embedded model, API key, account, or server-side image pipeline.

## The magic loop

1. **Scan:** the child presses **Start camera** and centers a bold drawing.
2. **Wake:** WallAlive separates the drawing locally and creates a seven-layer 2.5D character with real 3D eyes, limbs, lighting, and shadow.
3. **Play:** a browser agent places the character and directs movements or a four-beat story.
4. **Enter AR:** supported Android/WebXR devices use surface hit testing; every other modern browser gets the camera-overlay experience.

Press **Play Judge Demo** for a deterministic, camera-free version of the complete loop.

## WebMCP tool surface

| Tool | Mode | Purpose |
| --- | --- | --- |
| `inspect_wall_scene` | Read | Returns approved drawing semantics, character state, AR capability, and the privacy boundary. |
| `create_character_from_drawing` | Write | Wakes the human-approved drawing with a name, personality, body shape, eyes, and generated accent. |
| `set_character_personality` | Write | Changes performance intent without altering the child’s original pixels. |
| `place_character` | Write | Places and scales the character at a normalized position in the visible scene. |
| `animate_character` | Write | Plays one of seven safe visible actions. |
| `recolor_character` | Write | Recolors only generated depth and limbs; original drawing colors stay untouched. |
| `tell_character_story` | Write | Performs a cancellable one-to-four-beat story with animation and captions. |
| `list_activity` | Read | Returns recent attributed actions without camera or image data. |

All tools have strict JSON schemas, `additionalProperties: false`, cancellation signals, read/write annotations, bounded inputs, shared validation, and explicit error results. There is deliberately no `open_camera`, `capture_image`, or `upload_drawing` tool.

## Demo prompt

> Inspect the approved drawing. Turn it into a shy but brave character, place it on the wall, then tell a three-beat story where it hides, hops, and waves.

## Browser support

| Capability | Support |
| --- | --- |
| Camera capture + 3D overlay | Modern mobile and desktop browsers over HTTPS |
| Immersive room placement | WebXR `immersive-ar` + `hit-test`, typically Chrome on compatible Android devices |
| iPhone/iPad and non-WebXR browsers | Full camera-overlay fallback; immersive hit testing is not claimed |
| No-camera judging | Built-in demo doodle and one-click judge sequence |

WallAlive currently makes a convincing layered 2.5D character from the captured silhouette; it does not claim photogrammetric mesh reconstruction. High-contrast art on a plain background produces the cleanest extraction.

## Architecture

```text
human camera gesture
        │ local Canvas pixel extraction
        ▼
approved texture + shape/color analysis
        │                     │
        │                     └── WebMCP reads semantic state only
        ▼
Three.js layered character ◄──── WebMCP personality / place / animate / story
        │
        ├── WebXR hit-test placement when supported
        └── transparent camera overlay everywhere else
```

- React 19 + TypeScript, built with vinext for Cloudflare/Sites
- Three.js transparent WebGL character layer
- WebXR `immersive-ar` session with real-world hit testing
- `document.modelContext.registerTool()` imperative WebMCP integration
- One canonical action layer shared by UI controls and tool executors
- Session-only Canvas data URLs; no network path in drawing extraction
- Permissions Policy for WebMCP tools, camera, and WebXR; plus referrer, MIME sniffing, and frame protection headers

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
npm audit --omit=dev
```

## License

MIT — see [LICENSE](./LICENSE).
