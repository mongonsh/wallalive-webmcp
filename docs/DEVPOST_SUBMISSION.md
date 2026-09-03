# WallAlive — Devpost submission packet

Deadline: **September 3, 2026 at 1:00 PM PDT** / **September 4, 2026 at 5:00 AM JST**

## Project name

WallAlive

## Tagline

Draw it. Wake it. Play.

## One-line pitch

A shared drawing-to-story world where friends create a cast together and a WebMCP agent safely directs its 3D quests and creator collection.

## Links

- Live app: https://wallalive-webmcp.mungunshagai-tb.chatgpt.site
- Public repository: https://github.com/mongonsh/wallalive-webmcp
- Public YouTube demo: `PASTE_PUBLIC_YOUTUBE_URL_HERE`
- Prepared video: `outputs/WallAlive-WebMCP-demo.mp4`

## Inspiration

Children already treat drawings as living characters. A few lines can have a name, a voice, and an entire world behind them. WallAlive asks: **what if a child’s drawing could step off the wall into the room—and a browser agent could become its director?**

## What it does

1. Friends join a Cloudflare D1-backed room with guest usernames and paint one compact vector wall together; invite links contain no private session token.
2. A child may also explicitly open the camera, tap one character, and capture it.
3. Local Canvas processing separates the drawing from smooth lighting, page borders, text, and foreground clutter.
4. Seven local ONNX graphs isolate the selected figure, recognize exact face/body pixels, estimate depth, and decode an eight-family variable skeleton. A high-resolution private relief preview preserves artwork while semantic-outline-clipped weights prevent one limb from deforming unrelated pixels. It is not presented as full unseen-view reconstruction.
5. A second visible approval explains that only the isolated drawing—not the live camera or room frame—may go to AniGen, which jointly predicts richer full geometry, an arbitrary skeleton, and skinning weights.
6. WallAlive loads generated GLBs as Three.js `SkinnedMesh` assets. Unsupported branch actions stay locked instead of inventing a skeleton.
7. Four original PBR worlds contain raycastable objects and progress-bearing activities: mini movie, firefly hide-and-seek, cooperation spell, and living-gallery curation.
8. Nineteen goal-level WebMCP tools inspect rooms, per-figure readiness, rigs, 3D paint state, quest objects, and the private Story Passport; request visible human repair; stage a child-controlled paint adventure; adapt the next learning challenge; prepare username invites; stage and direct stories; touch real scene objects; recommend products; stage a creator-credited Shopify handoff; and read attributed history.
9. WebXR hit testing places the same character on a real surface when supported; every other browser gets the camera-overlay experience.

The child-facing outcome is a compact learning loop: **Imagine → Sequence → Perform → Reflect**. The learner starts with their own artifact, helps shape a beginning–middle–end story, approves the agent's plan, performs it, and then explains or revises what happened. A private Story Passport records completed beats, shared activity, the learner's retell, and one chosen revision; the agent can inspect that structured evidence to adapt its next scaffold without grading the child.

The one-click judge demo uses the exact supplied drawing and loads its precomputed full neural reconstruction immediately: 68,326 vertices, 136,648 triangles, seven active semantic bones, a complete watertight back, normalized weights, and restored approved front color without copying facial marks onto the rear. Judging never depends on shared public GPU quota. A separate official AniGen reference fixture remains in the automated evidence suite.

## Why WallAlive is a strong fit for WebMCP

WebMCP is the collaboration layer, not a chat box attached to AR. The child authors the input and keeps sensor authority. The agent reads the real capabilities of every character, assigns compatible roles and actions, and submits a structured multi-character production plan inside the same visible world.

The shared-authority boundary is structural: **the camera and approval are not tools**. No registered tool can open the camera, capture a frame, approve external processing, retrieve pixels, or approve a staged show. `request_rigged_3d_cast` can surface the visible 3D choice, then stops. `stage_magic_show` validates and displays choreography, then stops until the human presses **Approve & play**.

Without WebMCP, an agent would have to infer multiple rigs and coordinate dozens of controls through screenshots and brittle clicks. With WebMCP, it can identify the exact figure that needs repair without receiving its pixels, bring the human to that visible repair surface, re-inspect the result, avoid impossible movements, and stage the next evidence-based learning challenge for human approval.

## WebMCP implementation

WallAlive registers nineteen imperative tools with `document.modelContext.registerTool()` in the top-level document:

- `inspect_creative_scene`
- `inspect_learning_progress`
- `stage_next_learning_challenge`
- `inspect_character_capabilities`
- `inspect_reconstruction_readiness`
- `request_character_repair`
- `request_rigged_3d_cast`
- `stage_magic_show`
- `direct_live_ensemble`
- `orchestrate_spatial_cinematics`
- `recommend_creator_products`
- `stage_shopify_import_kit`
- `inspect_shopify_import_kit`
- `inspect_shared_room`
- `prepare_room_invite`
- `interact_story_world`
- `list_collaboration_history`

Each tool uses narrow JSON Schema, nested `additionalProperties: false`, bounded arrays and strings, cancellation signals, `readOnlyHint` / `untrustedContentHint` annotations, shared capability validation, and verification-rich results. Story playback and Creator Drop export are separately staged for human approval. No tool publishes to Shopify, purchases, charges, or returns image pixels.

## How we built it

- React 19 + TypeScript, vinext, Cloudflare Workers, and ChatGPT Sites
- Local target-aware drawing isolation with connected components, clutter/border rejection, morphology, and flood fill
- AniGen `ss_flow_solo` + `slat_flow_auto` through a lazy `@gradio/client` connection
- Three.js `GLTFLoader`, real `SkinnedMesh` assets, generated-bone actions, lighting, shadow, and 360° interaction
- Seven same-origin ONNX/WASM graphs for point-prompted target isolation, nine-part segmentation, high-resolution face parsing, eight-family variable topology, 17-joint pose, and distinct front/back depth
- An 80,486-parameter SketchDepth compact U-Net trained on 6,144 balanced analytic shapes and selected without opening its 768-example sealed test
- Identity-preserving sketch-to-render conditioning + local TripoSR + WallAlive variable-graph skinning for the exact-drawing quota-free neural asset
- WebXR `immersive-ar` with `hit-test`
- Imperative WebMCP registration through `document.modelContext`
- Human-only camera and isolated-image approval boundary
- Automated tests that parse the shipped GLB and verify its mesh, skeleton, bone branches, skin indices, and skin weights

## Challenges

The first version proved that a thick silhouette is not enough. A deterministic 2D contour can produce a closed blob, but it cannot infer unseen anatomy or generate a true rig. Research across Animated Drawings, DrawingSpinUp, SAM 3D Objects, Stable Fast 3D, TripoSR, TRELLIS, Hunyuan3D, post-hoc riggers, and AniGen led to a two-tier design: a learned local asymmetric-depth rig for instant privacy, then full neural reconstruction when approved.

The second challenge was privacy. Real single-image 3D requires a learned GPU model, so “nothing leaves the browser” would be false. WallAlive instead minimizes data before consent: only the isolated character can be sent, after a second visible human approval, while the camera and room frame remain inaccessible to the agent.

The third challenge was reliability. The free ZeroGPU Space can queue or reject work. We added clear progress/error states, preserved results in local Blob URLs, documented a dedicated-GPU production path, and shipped a binary-verified rigged judge fixture.

The fourth challenge was anatomy. A good silhouette still gave straight, misplaced limbs. We built an independent 490-character Meta Amateur Drawings benchmark and trained a 161,133-parameter browser pose model. On 75 untouched test drawings it reaches 0.7969 PCK@5% and 0.8643 PCK@10%. The pose graph bends limbs through elbows and knees, while high-resolution segmentation remains authoritative for the child’s eye, ear, mouth, cheek, outline, and color.

## Accomplishments

- A real drawing → neural mesh + skeleton + skin weights → camera AR loop
- The exact drawing reconstructed at 256³ extraction resolution as a smoothed, colored 68,326-vertex / 136,648-triangle watertight GLB with seven active semantic bones
- A learned asymmetric depth model with 0.03636 sealed normalized surface MAE, 0.91988 correlation, and ONNX agreement within 2.26e-6
- Eight post-split real drawing families classified correctly and exported as closed, colored, actively skinned local GLBs
- A locally trained 17-joint drawing pose model whose ONNX export exactly reproduces the untouched-test result
- Generated bone-branch animation instead of whole-object-only motion
- Real WebXR surface hit testing plus universal camera overlay
- Nineteen goal-level WebMCP tools sharing the same live room, Story Passport, child-controlled 3D paint studio, per-figure repair state, interactive 3D, and Creator Shop state as the human UI
- Persistent username rooms and compact collaborative vector drawing through Cloudflare D1
- Touchable quest objects with visible progress in all four original worlds
- Artwork-aware product recommendations plus Shopify draft CSV, storefront blueprint, print artwork, and adult handoff exports
- Visible attribution for every human, agent, and system action
- Human-only authority for camera, capture, and isolated-image upload
- A quota-free one-click judge demo

## What is next

- Self-host AniGen on a dedicated 18 GB+ NVIDIA GPU
- Expand the eight-family held-out benchmark to hundreds of photographed child drawings with human-reviewed 3D preference scores
- Run a teacher-led pilot measuring time to first story, beginning–middle–end coherence, retelling vocabulary, turn-taking, and revision behavior; no learning-gain claim is made before that evidence exists
- Add optional SAM 2 point/box refinement for difficult backgrounds
- Retarget richer motion clips to generated skeletons
- Teacher-managed room roles, verified identity, and moderation beyond current guest handles
- Persistent WebXR anchors and accessible motion descriptions

## Built with

WebMCP, Shopify-ready import kit (no connected store), Cloudflare D1, ChatGPT Sites, Chrome, AniGen, TripoSR, ONNX Runtime Web, WebXR, Three.js, React, TypeScript, Canvas API, WebGL

## Suggested tags

WebMCP, image-to-3D, rigging, WebXR, augmented reality, Three.js, children’s art, browser agents, privacy, human-agent collaboration

## Final submission checklist

- [x] Working judge-accessible public URL
- [x] Text explains WebMCP fit, UX improvement, human-agent collaboration, and implementation
- [ ] Final 2–2.5 minute video re-recorded from the current public build with narration, the real Browser Agent prompt, the guided-demo label, and the learning loop visible
- [ ] Demo video uploaded to YouTube with **Public** visibility
- [x] Public GitHub repository
- [x] Functional source, assets, attribution, and instructions included
- [x] MIT license detected by GitHub
- [x] Repository About section has description and live homepage
