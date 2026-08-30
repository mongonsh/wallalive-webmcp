# Drawing-to-rigged-3D and WebMCP research

This note records what WallAlive actually does, why the previous silhouette method was insufficient, and the boundary between local privacy work and neural reconstruction.

## Root cause of the previous result

The earlier app projected drawing pixels onto a Marching Cubes body derived from the 2D contour. That can create a closed rounded surface, but it has no learned 3D prior and no real skinned-mesh asset type. Every input therefore became a version of its front silhouette. More contour tuning could not infer unseen anatomy, separate semantic parts reliably, or produce skeleton and skin weights.

That path now appears only as **rough private preview** when a user declines neural processing.

## Systems evaluated

| System | What it provides | Why it is or is not the main path |
| --- | --- | --- |
| [Meta Animated Drawings](https://github.com/facebookresearch/AnimatedDrawings) | Detector, segmentation, pose estimation, and 2D character animation | Excellent precedent for children’s drawings, but its output is a 2D articulated deformation rather than 360° geometry. |
| [SAM 2](https://ai.meta.com/research/sam2/) | Promptable image/video object masks | Useful segmentation, but it cannot invent an unseen back, mesh, skeleton, or skin weights. |
| [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) | Fast textured single-image GLB generation | Produces a real mesh but not an animation skeleton or skinning weights. |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Image-conditioned 3D representations and meshes | Strong geometry prior, but no coherent animation rig in its primary output. |
| [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Geometry and PBR texture generation through a GPU service | Real 3D, but rigging remains a separate problem. |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) / [RigAnything](https://github.com/Isabella98Liu/RigAnything) | Post-hoc automatic rigging | Viable after a mesh generator, but a sequential pipeline compounds geometry/rig errors and deployment cost. |
| [AniGen](https://github.com/VAST-AI-Research/AniGen) | Single-image mesh, skeleton, and skinning weights generated jointly | Selected. It directly matches WallAlive’s requirement for animate-ready 360° assets across humanoids, animals, and machinery. |

AniGen represents shape, skeleton, and skinning as mutually consistent fields and generates them together. Its official repository reports generalization across animals, humanoids, and machinery; the official Space includes example conditions for a child drawing, dog, owl, plant, whale, T-rex, lamp, and machine arm. This is a far more defensible “varied drawing” prior than hand-coded eye/limb heuristics.

## Implemented neural path

1. The browser locally isolates one human-selected drawing into a transparent PNG.
2. A visible approval card explains the external processing boundary.
3. `@gradio/client` connects directly to the official `VAST-AI/AniGen` Space.
4. `/prepare_input_for_generation` stores the processed RGBA condition and returns its session path.
5. `/generate_preview` runs `ss_flow_solo` + `slat_flow_auto` with automatic joint count and returns a colored rigged mesh GLB plus a skeleton GLB.
6. WallAlive immediately downloads the temporary mesh into a browser Blob URL.
7. Three.js loads it with `GLTFLoader`, normalizes it, verifies mesh/bone/vertex counts, and drives generated bone branches.

The preview GLB is already a real `SkinnedMesh`. The optional final extraction endpoint adds simplification and texture baking, but it is not necessary to prove geometry, skeleton, skinning, 360° rotation, or bone deformation.

## Verified evidence

- The official Space accepted the WallAlive demo PNG and completed its CPU preprocessing path.
- A complete AniGen generation was downloaded and parsed locally.
- The fixture contains exactly one mesh, one `SkinnedMesh`, 20 bones, and 159,930 vertices.
- Blender rendered its front/back geometry and preserved vertex color.
- The deployed test suite parses the binary GLB and fails if it regresses into a non-skinned or low-detail asset.
- Desktop/mobile browser QA verified the approval UI, lazy Gradio client, successful CORS requests, session creation, file upload, preprocessing call, and friendly public-quota failure handling.

The current free ZeroGPU quota prevented repeated end-to-end generations across every example category in one session. The model and live pipeline are real; a dedicated endpoint is still required for reliable high-volume category testing and production use.

## WebMCP design

WallAlive follows the [WebMCP Community Group draft](https://webmachinelearning.github.io/webmcp/) and the official [imperative API explainer](https://github.com/webmachinelearning/webmcp#imperative-tool-registration-documentmodelcontext):

- Register goal-oriented tools with `document.modelContext.registerTool()`.
- Use precise descriptions and strict JSON Schema.
- Bind registration and stories to `AbortSignal` cancellation.
- Mark read-only and mutating actions.
- Share the same validated action functions between UI controls and tools.
- Return exact provider, model, asset, phase, and geometry metadata for verification.

The important safety capability is structural: no registered tool can open the camera, capture a frame, or approve an external upload. `reconstruct_rigged_3d_character` can surface the approval card, then stops. This makes human agency enforceable in code instead of relying on a prompt.

## Honest capability statement

WallAlive now uses a real learned single-image 3D prior and a real skinned skeleton. It does not claim perfect recovery of artist-authored unseen surfaces. Single-view reconstruction is inherently ambiguous, public GPU capacity is not reliable, and child privacy requires explicit external-processing consent. For hackathon judging, the bundled verified rig proves the full 3D/WebMCP/animation path without depending on quota; for production, self-host AniGen and run a curated multi-category evaluation set before promising quality levels.
