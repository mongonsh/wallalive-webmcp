# Drawing-to-rigged-3D and WebMCP research

This note records what WallAlive actually does, why the previous silhouette method was insufficient, and the boundary between local privacy work and neural reconstruction.

## Root cause of the previous result

The earlier app projected drawing pixels onto a Marching Cubes body derived from the 2D contour. That can create a closed rounded surface, but it has no learned 3D prior and no real skinned-mesh asset type. Every input therefore became a version of its front silhouette. More contour tuning could not infer unseen anatomy, separate semantic parts reliably, or produce skeleton and skin weights.

That path now appears only as **rough private preview** when a user declines neural processing.

## Systems evaluated

| System | What it provides | Why it is or is not the main path |
| --- | --- | --- |
| [Meta Animated Drawings](https://github.com/facebookresearch/AnimatedDrawings) | Detector, segmentation, pose estimation, and 2D character animation | Excellent precedent for children’s drawings, but its output is a 2D articulated deformation rather than 360° geometry. |
| [ChildlikeSHAPES / CharSegNet](https://arxiv.org/abs/2504.08022) | Hierarchical drawing parsing with 25 semantic classes | Closest semantic-recognition precedent. Its coarse-to-fine design directly informed WallAlive v2; its reported dataset/model were not available as a browser-ready checkpoint, so WallAlive trains and ships a compact local model. |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Promptable mobile image/video masks | A strong local proposal model, but it needs a point or box and does not name eyes, cheeks, or limbs. Raw masks also selected paper/clutter in drawing tests, so it is not allowed to replace target-aware drawing extraction blindly. |
| [SAM 2](https://ai.meta.com/research/sam2/) | Promptable image/video object masks | Useful segmentation, but it cannot invent an unseen back, mesh, skeleton, or skin weights. |
| [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) | Fast textured single-image GLB generation | Produces a real mesh but not an animation skeleton or skinning weights. |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Image-conditioned 3D representations and meshes | Strong geometry prior, but no coherent animation rig in its primary output. |
| [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Geometry and PBR texture generation through a GPU service | Real 3D, but rigging remains a separate problem. |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) / [RigAnything](https://github.com/Isabella98Liu/RigAnything) | Post-hoc automatic rigging | Viable after a mesh generator, but a sequential pipeline compounds geometry/rig errors and deployment cost. |
| [AniGen](https://github.com/VAST-AI-Research/AniGen) | Single-image mesh, skeleton, and skinning weights generated jointly | Selected. It directly matches WallAlive’s requirement for animate-ready 360° assets across humanoids, animals, and machinery. |

AniGen represents shape, skeleton, and skinning as mutually consistent fields and generates them together. Its official repository reports generalization across animals, humanoids, and machinery; the official Space includes example conditions for a child drawing, dog, owl, plant, whale, T-rex, lamp, and machine arm. This is a far more defensible “varied drawing” prior than hand-coded eye/limb heuristics.

The two Kivicube examples supplied for comparison demonstrate authored AR artwork and an authored AR game: the artist prepares the target, content, and interactions ahead of time. WallAlive has the harder requirement of accepting a previously unseen drawing from a live camera. It therefore needs both drawing-specific recognition and generative 3D reconstruction; an AR target tracker alone does not solve that problem.

## Learned local part recognition

WallAlive now includes a task-specific semantic segmenter rather than asking a general object mask to guess anatomy. `WallAlive Hierarchical PartUNet v2` is a 246,508-parameter U-Net with four coarse context channels (foreground, face, upper appendage, lower appendage) and nine semantic outputs: body, eye, cheek, mouth, ear, arm, hand, leg, and foot. It is exported to a 976 KB ONNX file and runs in-browser through a lazy-loaded WebAssembly runtime.

Training is reproducible in `ml/train_part_detector.py`. The generator creates filled and outlined blobs, ellipses, rounded and spiky bodies; upright, tall, horizontal, quadruped, side-faced and multi-arm layouts; zero to three eyes; asymmetric/missing limbs; varied ears, cheeks, mouths, hands and feet; colored or gray ink; paper grids; folds/shadows; rotations, shear, blur, JPEG damage, noise, and unrelated strokes. The checked-in run used 6,000 generated training examples plus 1,176 augmentations of 392 genuine Amateur Drawings. Evaluation used 700 generated examples and 98 untouched real drawings. Generated IoU was 0.927 body, 0.783 eye, 0.732 cheek, 0.708 mouth, 0.706 ear, 0.657 arm, 0.861 hand, 0.665 leg, and 0.762 foot; real foreground IoU was 0.738.

This architecture follows the central finding in ChildlikeSHAPES: broad character context should be estimated before small semantic parts. The foreground mask is predicted by the coarse branch; face and appendage context is fed back into the semantic refiner. Unlike a single-stage model, a tiny eye or hand is therefore interpreted relative to the whole figure rather than by local appearance alone.

Browser QA found that v2’s stronger real-drawing foreground transfer could still miss an arm/hand pair that v1 recognized. WallAlive therefore retains both sub-megabyte checkpoints as a conservative local ensemble: hierarchical v2 supplies the primary hints, and v1 may add only missing, spatially non-overlapping instances up to a bounded per-class maximum. This preserves a useful specialist without averaging incompatible masks or replacing exact contour evidence.

The model is intentionally not the geometry source of truth. Its low-resolution masks identify what a region means; WallAlive then snaps each prediction to the nearest high-resolution isolated pixel component. This preserves the drawing’s measured center, dimensions, rotation, outline, and sampled color. Existing medial-skeleton limbs are retained and only confirmed or supplemented by high-confidence learned hints.

The private supplied drawing was evaluated locally without uploading it: v2 found the body, eyes, cheeks, central mouth, ears, legs, and feet. The arm remained below the learned threshold, so its position continued to come from the exact contour/medial skeleton. On six separate public character textures from Meta Animated Drawings, mean body-mask IoU improved from 0.637 with v1 to 0.675 with v2 at the browser threshold. Four examples scored 0.806–0.912; a six-arm character and quadruped remain difficult at 0.387 and 0.293. Those failure cases are kept visible because the small set is evidence of transfer, not a universal-quality claim.

EdgeTAM was also tested locally as an optional high-recall proposal generator. Its 14 MB quantized ONNX pair ran in roughly 0.7 seconds on the local CPU, but automatically selecting among its masks averaged only 0.234 IoU on six public drawings (0.459 with an oracle candidate choice). A promptable “segment anything” mask is not a semantic part recognizer, so the smaller drawing-specific model was selected for the default path.

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
- Browser QA verified lazy same-origin delivery of the PartUNet ONNX file and WebAssembly runtime, with nine semantic kinds present in the demo and no third-party request for image recognition.
- The private preview now creates shallow 3D eye lenses, pupils, and contour-following feature tubes, so detected face parts have lighting, parallax, and animation instead of remaining a flat crop.

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

WallAlive now uses a learned local semantic model, a real learned single-image 3D prior, and a real skinned skeleton. It does not claim perfect recognition of every drawing or perfect recovery of artist-authored unseen surfaces. Single-view reconstruction is inherently ambiguous, the local part model has only a small public-transfer benchmark so far, public GPU capacity is not reliable, and child privacy requires explicit external-processing consent. For hackathon judging, the bundled verified rig proves the full 3D/WebMCP/animation path without depending on quota; for production, self-host AniGen and expand the curated multi-category evaluation set before promising quality levels.
