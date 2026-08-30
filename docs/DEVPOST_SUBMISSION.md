# WallAlive — Devpost submission packet

Deadline: **September 3, 2026 at 1:00 PM PDT** / **September 4, 2026 at 5:00 AM JST**

## Project name

WallAlive

## Tagline

Draw it. Wake it. Play.

## One-line pitch

A child-safe AR playground that turns one approved drawing into a rigged 3D character a browser agent can place, animate, and direct through WebMCP.

## Links

- Live app: https://wallalive-webmcp.mungunshagai-tb.chatgpt.site
- Public repository: https://github.com/mongonsh/wallalive-webmcp
- Public YouTube demo: `PASTE_PUBLIC_YOUTUBE_URL_HERE`
- Prepared video: `outputs/WallAlive-WebMCP-demo.mp4`

## Inspiration

Children already treat drawings as living characters. A few lines can have a name, a voice, and an entire world behind them. WallAlive asks: **what if a child’s drawing could step off the wall into the room—and a browser agent could become its director?**

## What it does

1. The child explicitly opens the camera, taps one character, and captures it.
2. Local Canvas processing separates the drawing from smooth lighting, page borders, text, and foreground clutter.
3. A second visible approval explains that only the isolated drawing—not the live camera or room frame—will go to the 3D model.
4. AniGen jointly predicts a full mesh, unseen surfaces, skeleton, and skinning weights from that one image.
5. WallAlive loads the GLB as a Three.js `SkinnedMesh`. Seven actions drive the generated bone branches; drag/spin exposes true 360° geometry and the generated back.
6. Eight WebMCP tools let the agent inspect exact model state, create a personality, place and scale the rig, animate it, recolor an accent, and perform a cancellable mini story.
7. WebXR hit testing places the same character on a real surface when supported; every other browser gets the camera-overlay experience.

The one-click judge demo loads a verified AniGen fixture immediately. It contains one colored `SkinnedMesh`, 20 bones, and 159,930 vertices, so judging never depends on shared public GPU quota.

## Why WallAlive is a strong fit for WebMCP

WebMCP is the collaboration layer, not a chat box attached to AR. The child authors the input and keeps sensor authority. The agent composes personality, placement, motion, and narrative through structured actions inside the same visible world.

The safety boundary is structural: **the camera is not a tool**. No registered tool can open the camera, capture a frame, approve an external upload, or retrieve pixels. `reconstruct_rigged_3d_character` can surface the visible approval UI, then stops for the human. This boundary is enforced by capability design, not a prompt.

Without WebMCP, an agent would infer changing 3D state through screenshots and brittle clicks. With WebMCP, it can inspect provider/model/mesh/bone metadata, invoke bounded actions, verify results, and stay synchronized with the child.

## WebMCP implementation

WallAlive registers eight imperative tools with `document.modelContext.registerTool()`:

- `inspect_wall_scene`
- `reconstruct_rigged_3d_character`
- `set_character_personality`
- `place_character`
- `animate_character`
- `recolor_character`
- `tell_character_story`
- `list_activity`

Each tool uses strict JSON Schema, `additionalProperties: false`, bounded values, cancellation signals, read/write annotations, and shared validation. UI controls and tool executors call the same canonical action layer. Inspection returns the neural provider, model, asset type, mesh/bone/vertex counts, generation phase, approval state, and `cameraFeedExposed: false`.

## How we built it

- React 19 + TypeScript, vinext, Cloudflare Workers, and ChatGPT Sites
- Local target-aware drawing isolation with connected components, clutter/border rejection, morphology, and flood fill
- AniGen `ss_flow_solo` + `slat_flow_auto` through a lazy `@gradio/client` connection
- Three.js `GLTFLoader`, real `SkinnedMesh` assets, generated-bone actions, lighting, shadow, and 360° interaction
- WebXR `immersive-ar` with `hit-test`
- Imperative WebMCP registration through `document.modelContext`
- Human-only camera and isolated-image approval boundary
- Automated tests that parse the shipped GLB and verify its mesh, skeleton, bone branches, skin indices, and skin weights

## Challenges

The first version proved that a thick silhouette is not enough. A deterministic 2D contour can produce a closed blob, but it cannot infer unseen anatomy or generate a true rig. Research across Animated Drawings, SAM 2, Stable Fast 3D, TRELLIS, Hunyuan3D, and post-hoc riggers led to AniGen, which generates shape, skeleton, and skinning together.

The second challenge was privacy. Real single-image 3D requires a learned GPU model, so “nothing leaves the browser” would be false. WallAlive instead minimizes data before consent: only the isolated character can be sent, after a second visible human approval, while the camera and room frame remain inaccessible to the agent.

The third challenge was reliability. The free ZeroGPU Space can queue or reject work. We added clear progress/error states, preserved results in local Blob URLs, documented a dedicated-GPU production path, and shipped a binary-verified rigged judge fixture.

## Accomplishments

- A real drawing → neural mesh + skeleton + skin weights → camera AR loop
- A colored 159,930-vertex GLB with 20 bones, parsed and quality-gated in tests
- Generated bone-branch animation instead of whole-object-only motion
- Real WebXR surface hit testing plus universal camera overlay
- Eight WebMCP tools sharing one mutation layer with the human UI
- Visible attribution for every human, agent, and system action
- Human-only authority for camera, capture, and isolated-image upload
- A quota-free one-click judge demo

## What is next

- Self-host AniGen on a dedicated 18 GB+ NVIDIA GPU
- Run a curated multi-category set: child line art, humanoid, quadruped, bird, plant, and machine
- Add optional SAM 2 point/box refinement for difficult backgrounds
- Retarget richer motion clips to generated skeletons
- Multi-character classroom stories with teacher-controlled privacy
- Persistent WebXR anchors and accessible motion descriptions

## Built with

WebMCP, AniGen, Hugging Face Gradio, WebXR, Three.js, React, TypeScript, Canvas API, WebGL, Cloudflare Workers, ChatGPT Sites

## Suggested tags

WebMCP, image-to-3D, rigging, WebXR, augmented reality, Three.js, children’s art, browser agents, privacy, human-agent collaboration

## Final submission checklist

- [x] Working judge-accessible public URL
- [x] Text explains WebMCP fit, UX improvement, human-agent collaboration, and implementation
- [x] Demo video is under three minutes and contains narration
- [ ] Demo video uploaded to YouTube with **Public** visibility
- [x] Public GitHub repository
- [x] Functional source, assets, attribution, and instructions included
- [x] MIT license detected by GitHub
- [x] Repository About section has description and live homepage
