# WallAlive

**Draw it. Wake it. Play.**

**Live demo:** https://wallalive-webmcp.mungunshagai-tb.chatgpt.site

**Public source:** https://github.com/mongonsh/wallalive-webmcp

WallAlive turns a child’s drawing into a closed, rounded 3D character that can live and perform in the real room. A child explicitly opens the camera and approves a drawing; then a compatible browser agent can reconstruct its volume, name it, shape its personality, place it, animate it, recolor its generated surface, and direct a mini story through WebMCP.

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

1. **Scan:** the child presses **Start camera**, taps the character, and captures. Local adaptive ink detection preserves vivid or dark strokes while rejecting smooth lighting changes.
2. **Sculpt:** WallAlive scores line-art candidates against the tap, rejects dense foreground clutter and rectangular paper borders, closes small gaps, and flood-fills the chosen silhouette. A distance transform extracts medial-axis nodes with a local radius at each node. Those disks become overlapping 3D spheres in a 64³ implicit field; Marching Cubes polygonizes their union as a rounded front, sides, and back.
3. **Play:** a browser agent places the character and directs movements or a four-beat story.
4. **Enter AR:** supported Android/WebXR devices use surface hit testing; every other modern browser gets the camera-overlay experience.

Press **Play Judge Demo** for a deterministic, camera-free version of the complete loop.

## WebMCP tool surface

| Tool | Mode | Purpose |
| --- | --- | --- |
| `inspect_wall_scene` | Read | Returns approved drawing semantics, character state, AR capability, and the privacy boundary. |
| `reconstruct_volumetric_character` | Write | Converts the approved silhouette’s medial skeleton and local radii into a closed 64³ sphere-union volume with a name, personality, accent, and bounded inflation strength. |
| `set_character_personality` | Write | Changes performance intent without altering the child’s original pixels. |
| `place_character` | Write | Places and scales the character at a normalized position in the visible scene. |
| `animate_character` | Write | Plays one of seven safe visible actions. |
| `recolor_character` | Write | Recolors only the generated solid edge; original drawing colors stay untouched. |
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

WallAlive constructs a genuine closed polygonal 3D surface from the captured silhouette; it does not claim photogrammetry, neural 360° inference, or anatomy that was not drawn. This is a browser-native implementation of the silhouette-inflation family introduced by Teddy and later used for playful single-view modeling in Monster Mash. High-contrast closed line art produces the cleanest extraction. See [RESEARCH.md](./docs/RESEARCH.md) for the algorithm decision and neural upgrade boundary.

## Architecture

```text
human camera gesture + tap target
        │ adaptive ink mask + target-aware candidate scoring
        ▼
clutter/border rejection + flood-filled silhouette
        │
        ▼
medial-axis ridge nodes + local distance radii
        │                     │
        │                     └── WebMCP reads semantic state only
        ▼
64³ implicit sphere union + Marching Cubes ◄── WebMCP reconstruct / place / animate / story
        │
        ├── curved original-art front + generated rounded back
        ├── WebXR hit-test placement when supported
        └── transparent camera overlay everywhere else
```

- React 19 + TypeScript, built with vinext for Cloudflare/Sites
- Three.js Marching Cubes over a medial-skeleton sphere union, plus a CPU-curved original-art surface with recomputed normals
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
