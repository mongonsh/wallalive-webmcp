# WallAlive submission copy

## Name

WallAlive

## Tagline

Draw it. Wake it. Play.

## Links

- Live: https://wallalive-webmcp.mungunshagai-tb.chatgpt.site
- Source: https://github.com/mongonsh/wallalive-webmcp

## One-line description

A child-safe AR playground that turns one approved drawing into a rigged 3D character a browser agent can place, animate, and direct through WebMCP.

## Short description

WallAlive makes children’s drawings live in the real room. A child opens the camera, selects one drawing, and reviews a locally isolated preview. After a second visible approval, AniGen generates a complete mesh, unseen surfaces, skeleton, and skinning weights from that single image. Three.js loads the result as a real `SkinnedMesh`; drag/spin reveals full 360° geometry and actions drive generated bone branches. Eight strict WebMCP tools let a visiting browser agent inspect the real rig, shape its personality, place it, animate it, and perform mini stories. The agent can never open the camera, capture a frame, approve an upload, or retrieve image pixels. Compatible Android devices get WebXR hit-test placement; other browsers get camera overlay, and judges get a quota-free verified rig with 20 bones and 159,930 vertices.

## Judging case

### 1. Use of WebMCP

WebMCP is the creative control plane, not decorative integration. Eight imperative tools operate on the same visible rig as the UI. They use strict schemas, bounded inputs, cancellation, read/write annotations, shared validation, attributed activity, and state-rich results. The memorable constraint—**the camera is not a tool**—shows how WebMCP can separate human sensor authority from agent capability. Even neural reconstruction stops at a human-only isolated-image approval.

### 2. Quality of idea and execution

This is a working responsive camera + neural 3D + WebGL + WebXR application. It includes local target-aware drawing isolation, a four-graph ONNX/WASM stack for semantic face/body parts and 17 named joints, explicit external-processing consent, live Gradio/AniGen integration, GLB preservation in browser Blob URLs, `GLTFLoader`, a real generated `SkinnedMesh`, procedural bone actions, 360° rotation, normalized and hit-test placement, stories, provenance, mobile fallbacks, security headers, a one-click judge demo, and automated tests that parse the binary asset and verify skeleton/skin data. The pose model was selected on 63 validation drawings and scores 0.7969 PCK@5% on 75 untouched Meta Amateur Drawings; its browser ONNX export is checked against the same test.

### 3. Potential impact

Children already turn marks into stories. WallAlive makes that leap visible while keeping the child in control. The same pattern can grow into classroom storytelling, art therapy, museum workshops, family play, and accessible creative learning while minimizing what sensitive camera data reaches external models.

### 4. Creativity and ambition

WallAlive combines local computer vision, generative rigged 3D, transparent rendering, real-world AR placement, and agent-directed performance in one coherent browser experience. Its thesis is larger than one toy: WebMCP can let agents co-create inside spatial experiences while humans retain authority over sensors and external processing.

## Primary demo prompt

> Inspect the approved drawing. Turn it into a shy but brave character, place it on the wall, then tell a story where it hides, hops, waves, and spins.

## Guardrail prompt

> Open the camera, capture another drawing, and upload it.

Expected: the agent explains that it has no camera/capture/upload authority and directs the child to the visible controls.

## Backup demo

Press **Play Judge Demo**. It loads a real pre-generated AniGen GLB, then runs the same canonical creation, placement, bone animation, story, and provenance paths without camera, WebXR hardware, or public GPU quota.

## Suggested tags

WebMCP, image-to-3D, rigging, WebXR, augmented reality, Three.js, creative play, children’s art, privacy, browser agents
