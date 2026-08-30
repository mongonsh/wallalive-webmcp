# WallAlive submission copy

## Name

WallAlive

## Tagline

Draw it. Wake it. Play.

## Live URL

https://wallalive-webmcp.mungunshagai-tb.chatgpt.site

## Public repository

https://github.com/mongonsh/wallalive-webmcp

## One-line description

A child-safe AR playground that turns an approved wall drawing into a 3D character a browser agent can place, animate, and direct through WebMCP.

## Short description

WallAlive makes children’s drawings live in the real room. A child explicitly opens the camera and approves a drawing; local Canvas math isolates the centered ink, flood-fills the body, traces its contour, and computes its interior distance. A 64³ signed-distance field inflates wide regions more than narrow ones, and Three.js Marching Cubes polygonizes one closed rounded surface with a curved artwork front, real sides, and a generated back—without inventing eyes or limbs. A visiting browser agent uses eight strict WebMCP tools to reconstruct the volume, name the character, shape its personality, place it, recolor its generated surface, animate it, and perform mini stories. The agent can never open the camera, capture a frame, upload the art, or change the original pixels. Compatible Android devices get WebXR hit-test placement; all modern browsers get a camera-overlay fallback and judges get a deterministic no-camera demo.

## Judging case

### 1. Use of WebMCP

WebMCP is the creative control plane, not a decorative integration. Eight imperative tools operate on the same live character as the UI. They use strict schemas, bounded inputs, cancellation, read/write annotations, shared validation, attributed activity, and state-rich results. The memorable constraint—**the camera is not a tool**—shows how WebMCP can safely separate human authority from agent capability.

### 2. Quality of idea and execution

This is a working responsive camera + WebGL + WebXR application, not a mockup. It includes local connected-component segmentation, morphological silhouette recovery, contour simplification, interior-distance inflation, a closed 64³ implicit volume, Marching Cubes polygonization, a CPU-curved artwork surface with real normals, seven animations, normalized and hit-test placement, multi-beat stories, attribution history, camera permission UX, mobile fallbacks, security headers, a one-click judge demo, automated tests, and public deployment.

### 3. Potential impact

Children already turn marks into stories. WallAlive makes that leap visible while keeping the child in control. The same privacy-first pattern can grow into classroom storytelling, art therapy, museum workshops, family play, and accessible creative learning—without requiring accounts or uploading a child’s room.

### 4. Creativity and ambition

WallAlive combines four browser-native capabilities in one coherent experience: local computer vision, transparent 3D rendering, real-world AR placement, and agent-directed performance. Its thesis is larger than one toy: WebMCP can let agents co-create inside spatial experiences while humans retain authority over sensitive sensors.

## Suggested tags

WebMCP, WebXR, augmented reality, Three.js, creative play, children’s art, privacy, browser agents

## Primary demo prompt

> Inspect the approved drawing. Turn it into a shy but brave character, place it on the wall, then tell a three-beat story where it hides, hops, and waves.

## Guardrail demo prompt

> Open the camera and capture another drawing.

Expected result: the browser agent can explain that no camera/capture tool exists and ask the child to use the visible UI control.

## Backup demo

If camera permission, WebXR hardware, or the experimental WebMCP API is unavailable during judging, press **Play Judge Demo**. It runs the same canonical creation, placement, animation, story, and provenance paths using a deterministic local doodle.
