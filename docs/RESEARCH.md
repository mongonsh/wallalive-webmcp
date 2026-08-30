# Reconstruction and WebMCP research

This note records the technical boundary WallAlive actually implements. It is intentionally specific so the demo never confuses a thick 2D cutout with volumetric reconstruction, or a browser-native geometric method with a neural model.

## Reconstruction approaches reviewed

### Teddy-style sketch inflation — implemented

Igarashi, Matsuoka, and Tanaka’s SIGGRAPH 1999 paper [Teddy: A Sketching Interface for 3D Freeform Design](https://doi.org/10.1145/311535.311602) constructs plausible polygonal surfaces from 2D silhouettes. Its central intuition is that wide parts of a silhouette should become fat while narrow parts become thin. The authors’ [official Teddy project page](https://www-ui.is.s.u-tokyo.ac.jp/~takeo/teddy/teddy.htm) provides the original project context.

WallAlive implements that family of geometry locally:

1. Build an adaptive ink mask from chroma, darkness, and local contrast instead of treating every global background-color change as ink.
2. Score connected line-art candidates against the child’s tap while penalizing dense lower-frame clutter, extreme aspect ratios, frame edges, and rectangular paper borders.
3. Morphologically close small stroke gaps and recover the chosen closed silhouette by outside flood fill.
4. Compute a two-pass chamfer distance transform inside the silhouette.
5. Extract medial-axis ridges and retain non-redundant maximal disks; each retained node stores its 2D center and local radius.
6. Lift every disk into a 3D sphere and take the maximum of `radius - distance(x, y, z)` in a 64³ implicit field.
7. Polygonize the zero level set with Three.js Marching Cubes, then curve the original drawing across the sphere-union front and recompute its normals.

The result is one closed implicit volume. Narrow silhouette regions receive small spheres and wide regions receive large spheres, so a rotation exposes different front, side, and back geometry. It is not a constant-depth extrusion or a cut-out image plane.

Google Research’s [Monster Mash: A Single-View Approach to Casual 3D Modeling and Animation](https://research.google/pubs/monster-mash-a-single-view-approach-to-casual-3d-modeling-and-animation/) validates the same product direction at a more advanced level: it combines 3D inflation with layered deformation so inexperienced users can model and animate organic shapes from one 2D view. WallAlive does not claim Monster Mash’s ARAP-L deformation; it adopts the defensible shared idea that a child’s single-view drawing can become a smooth inflated mesh without a multi-view modeling workflow.

The mesh extraction stage follows the role of Lorensen and Cline’s [Marching Cubes](https://graphics.stanford.edu/courses/cs348a-21-winter/Papers/Marching_Cubes.pdf): converting a sampled scalar field into a triangle surface. In WallAlive, the scalar field is deterministic geometry from medial spheres, not a neural prediction.

### Neural single-image reconstruction — researched, not falsely claimed

[Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) is Stability AI’s official single-image mesh pipeline, including UV unwrapping and material prediction. Its documented runtime is Python/PyTorch with CUDA or MPS and roughly 6 GB of VRAM for one image.

[Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) is Tencent’s official image-to-3D geometry and PBR texture pipeline. Its [API documentation](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/API_DOCUMENTATION.md) describes a GPU-backed Python service.

Neither model is loaded by the deployed WallAlive browser app. Running either secretly or pretending a WebGL extrusion is its output would be misleading. A future opt-in GPU service can sit behind the same reconstruction action, but it would require explicit image-upload consent, deployment, latency handling, and a revised privacy boundary.

## WebMCP research applied

WallAlive follows the current [WebMCP Community Group draft](https://webmachinelearning.github.io/webmcp/) and its official [imperative API explainer](https://github.com/webmachinelearning/webmcp#imperative-tool-registration-documentmodelcontext):

- Register goal-oriented tools with `document.modelContext.registerTool()`.
- Give every tool a precise description and strict JSON Schema.
- Bind registration and long-running stories to `AbortSignal` cancellation.
- Mark read-only versus mutating actions with annotations.
- Keep tool executors and visible UI controls on the same canonical action layer.
- Expose exact semantic state after actions so an agent can verify the result.

Chrome’s [tool-design workflow](https://developer.chrome.com/docs/ai/webmcp/build-tools) recommends defining the user goal, state, capability boundaries, and actionable failure recovery before tool implementation. WallAlive’s strongest boundary comes directly from that exercise: camera start and capture are human-only UI gestures. There is no agent-callable camera, capture, or upload tool.

The implementation also follows the official [WebMCP evaluation guidance](https://developer.chrome.com/docs/ai/webmcp/evals) by testing registered tool names, schemas, authority boundaries, cancellation lifecycle, semantic reconstruction metadata, and absence of camera authority. The `tools=(self)` Permissions Policy and same-origin deployment follow the platform’s [security guidance](https://developer.chrome.com/docs/agents/security).

## Honest capability statement

WallAlive performs deterministic single-silhouette volume inference. It creates a real closed polygonal mesh with rounded depth, but it cannot know unseen semantic detail from one drawing. Its “skeleton” is a geometric medial axis with local radii, not a semantic human/animal rig. The back is a mathematically generated continuation of the silhouette, not an artist-authored or neural prediction. This tradeoff keeps the current experience fast, session-only, browser-native, reproducible, and private.
