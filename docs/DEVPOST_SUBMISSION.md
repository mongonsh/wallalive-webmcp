# WallAlive — Devpost submission packet

Deadline: **September 3, 2026 at 1:00 PM PDT** / **September 4, 2026 at 5:00 AM JST**

## Project name

WallAlive

## Tagline

Draw it. Wake it. Play.

## One-line pitch

A child-safe AR playground that turns an approved wall drawing into a 3D character a browser agent can place, animate, and direct through WebMCP.

## Live app

https://wallalive-webmcp.mungunshagai-tb.chatgpt.site

## Public repository

https://github.com/mongonsh/wallalive-webmcp

## Public YouTube demo

`PASTE_PUBLIC_YOUTUBE_URL_HERE`

Prepared video: `outputs/WallAlive-WebMCP-demo.mp4`

Public backup: https://github.com/mongonsh/wallalive-webmcp/releases/tag/demo-video-v1

## Inspiration

Children already treat drawings as living characters. A few lines on a wall or sheet of paper can have a name, a voice, and an entire world behind them. We wanted to make that imaginative leap visible without asking families to upload a child’s artwork or room to a server.

WallAlive began with one question: **what if a child’s drawing could step off the wall and into the room—and a browser agent could become its director?**

## What it does

WallAlive is a shared AR play space for a child and their browser agent:

1. The child explicitly opens the camera and approves one drawing.
2. Local Canvas processing separates the drawing from its background without an upload.
3. Three.js rebuilds it as a layered 2.5D character with real 3D eyes, limbs, lighting, shadow, and seven animations.
4. Eight WebMCP tools let the browser agent inspect approved shape/color metadata, name the character, define its personality, place and scale it, animate it, recolor generated depth, and perform a cancellable mini story.
5. WebXR hit testing places the character on a real surface when supported; every other browser receives the complete camera-overlay fallback.

Every action is attributed to `CHILD`, `BROWSER AGENT`, or `WALLALIVE` in a visible history.

## Why WallAlive is a strong fit for WebMCP

The experience is meaningfully better when a person and an agent act inside the same visible world. The child supplies the drawing and remains the author. The agent composes personality, movement, placement, and narrative through structured actions instead of guessing at buttons or producing disconnected chat text.

WebMCP is also how WallAlive makes its most important safety decision concrete: **the camera is not a tool**. There is no agent-callable camera, capture, or upload capability. The agent receives semantic analysis only after a human approves a drawing. That boundary is enforced by the app’s capability surface, not by a prompt asking the model to behave.

Without WebMCP, an agent would have to infer changing 3D state from screenshots and brittle DOM interactions. With WebMCP, it can inspect exact approved state, invoke bounded actions, receive verification-rich results, and remain synchronized with the child’s gestures.

## What people and agents can do together

A child can draw a creature, decide what the camera sees, tap to place it, and press direct action buttons. A browser agent can then turn “make it shy but brave” into a visible personality and a three-beat performance: hide at the edge, take one brave hop, then wave hello. Both participants act on the same character, and neither erases the other’s work.

That combination—human sensor authority, agent-directed spatial performance, shared state, and visible provenance—was difficult to make reliable before WebMCP.

## How we implemented WebMCP

WallAlive registers eight imperative tools with `document.modelContext.registerTool()`:

- `inspect_wall_scene`
- `create_character_from_drawing`
- `set_character_personality`
- `place_character`
- `animate_character`
- `recolor_character`
- `tell_character_story`
- `list_activity`

Each tool uses a strict JSON Schema with `additionalProperties: false`, bounded values, cancellation signals, read/write annotations, and shared validation. UI controls and WebMCP executors call the same canonical action layer, so an agent action produces the same visible result and provenance as a human action. Tool registration is lifecycle-bound with an `AbortController`.

The scene-inspection tool returns approved drawing semantics, AR capability, character state, and the privacy boundary. It explicitly reports `cameraFeedExposed: false`; no tool returns an image data URL, camera frame, or raw pixel payload.

## How we built it

- React 19 + TypeScript
- vinext + Cloudflare Workers on ChatGPT Sites
- Three.js transparent WebGL rendering
- WebXR `immersive-ar` sessions with `hit-test`
- Local Canvas pixel analysis and silhouette extraction
- Imperative WebMCP registration through `document.modelContext`
- Responsive camera-overlay fallback
- Permissions Policy for WebMCP tools, camera, and spatial tracking
- Automated server-render, tool-surface, local-processing, WebXR, and security-header tests

## Challenges we ran into

The hardest product decision was not technical—it was deciding what the agent must never control. Exposing the camera would have made the demo superficially more autonomous, but it would weaken the experience for the people whose trust matters most. Designing the tool boundary first led to a clearer product.

The largest technical challenge was making a flat, imperfect drawing feel dimensional without a backend reconstruction service. We combined a locally extracted transparent texture with seven depth layers, separate 3D facial features and limbs, lighting, shadow, and animation. We also had to reconcile WebXR’s real-world coordinate system with a universal normalized overlay placement model.

## Accomplishments we are proud of

- A complete working camera → local extraction → 3D character → agent-directed story loop
- Real WebXR surface hit testing plus a fallback that still demonstrates the whole product
- A narrow WebMCP capability boundary that protects a sensitive sensor by construction
- Eight tools sharing one mutation layer with the human interface
- Visible attribution for every human, agent, and system action
- No account, API key, model dependency, database, or image upload
- A deterministic one-click judge demo when camera or WebXR hardware is unavailable

## What we learned

WebMCP is most powerful when it is treated as a product contract, not simply an easier way to click UI. The list of available tools communicates ownership: what the agent may know, what it may change, and where a human gesture is non-negotiable.

We also learned that spatial interfaces benefit from structured agent actions. “Move Pip to the right wall and make it hide” maps naturally to placement and animation tools, while the visible scene gives the human immediate verification.

## What is next

- Multi-character stories where drawings from different children meet in one room
- On-device segmentation models for more complex wall textures
- Voice direction mapped to the same WebMCP action layer
- Shared classroom story worlds with teacher-controlled privacy boundaries
- WebXR anchors so characters remember their exact place between sessions
- Accessible motion descriptions and switch-control play modes

## Built with

WebMCP, WebXR, Three.js, React, TypeScript, Canvas API, WebGL, Cloudflare Workers, ChatGPT Sites

## Suggested tags

WebMCP, WebXR, augmented reality, Three.js, children’s art, creative tools, browser agents, privacy, human-agent collaboration

## Final submission checklist

- [x] Working judge-accessible public URL
- [x] Text explains WebMCP fit, UX improvement, human-agent collaboration, and implementation
- [x] Demo video is under three minutes and contains narration
- [ ] Demo video uploaded to YouTube with **Public** visibility
- [x] Public GitHub repository
- [x] All functional source, assets, and instructions included
- [x] MIT license detected by GitHub
- [x] Repository About section has description and live homepage
