# Drawing-to-rigged-3D and WebMCP research

This note records what WallAlive actually does, why the previous silhouette method was insufficient, and the boundary between local privacy work and neural reconstruction.

## Root cause of the previous result

The earlier app projected drawing pixels onto a Marching Cubes body derived from the 2D contour. That can create a closed rounded surface, but it has no learned 3D prior and no real skinned-mesh asset type. Every input therefore became a version of its front silhouette. More contour tuning could not infer unseen anatomy, separate semantic parts reliably, or produce skeleton and skin weights.

That path now appears only as **rough private preview** when a user declines neural processing.

## Systems evaluated

| System | What it provides | Why it is or is not the main path |
| --- | --- | --- |
| [Meta Animated Drawings](https://github.com/facebookresearch/AnimatedDrawings) | Detector, segmentation, pose estimation, and 2D character animation | Excellent precedent for children’s drawings, but its output is a 2D articulated deformation rather than 360° geometry. |
| [ChildlikeSHAPES / CharSegNet](https://arxiv.org/abs/2504.08022) | 16,075 annotated drawings and hierarchical parsing with 25 semantic categories | Selected as the real-data source for v3. WallAlive trains its own browser-sized student because the research architecture/checkpoint is not a mobile WebAssembly artifact. |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Promptable mobile image/video masks | A strong local proposal model, but it needs a point or box and does not name eyes, cheeks, or limbs. Raw masks also selected paper/clutter in drawing tests, so it is not allowed to replace target-aware drawing extraction blindly. |
| [SAM 2](https://ai.meta.com/research/sam2/) | Promptable image/video object masks | Useful segmentation, but it cannot invent an unseen back, mesh, skeleton, or skin weights. |
| [SAM 3](https://github.com/facebookresearch/sam3) | Promptable instance segmentation from text and visual prompts | Strong general proposals, but a large prompt model is not a browser-sized drawing anatomy parser and still does not create a rigged 3D asset. |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | Single-image shape, texture, pose, and layout reconstruction | Strong current 3D alternative, but its primary output is a static reconstructed object rather than an animate-ready skeleton and skinning field. |
| [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) | Fast textured single-image GLB generation | Produces a real mesh but not an animation skeleton or skinning weights. |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Image-conditioned 3D representations and meshes | Strong geometry prior, but no coherent animation rig in its primary output. |
| [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Geometry and PBR texture generation through a GPU service | Real 3D, but rigging remains a separate problem. |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) / [RigAnything](https://github.com/Isabella98Liu/RigAnything) | Post-hoc automatic rigging | Viable after a mesh generator, but a sequential pipeline compounds geometry/rig errors and deployment cost. |
| [AniGen](https://github.com/VAST-AI-Research/AniGen) | Single-image mesh, skeleton, and skinning weights generated jointly | Selected. It directly matches WallAlive’s requirement for animate-ready 360° assets across humanoids, animals, and machinery. |

AniGen represents shape, skeleton, and skinning as mutually consistent fields and generates them together. Its official repository reports generalization across animals, humanoids, and machinery; the official Space includes example conditions for a child drawing, dog, owl, plant, whale, T-rex, lamp, and machine arm. This is a far more defensible “varied drawing” prior than hand-coded eye/limb heuristics.

The two Kivicube examples supplied for comparison demonstrate authored AR artwork and an authored AR game: the artist prepares the target, content, and interactions ahead of time. WallAlive has the harder requirement of accepting a previously unseen drawing from a live camera. It therefore needs both drawing-specific recognition and generative 3D reconstruction; an AR target tracker alone does not solve that problem.

## Learned local part recognition

WallAlive now includes a task-specific hierarchical segmenter rather than asking a general object mask to guess anatomy. `WallAlive ChildlikeSHAPES PartUNet v3` has a 288,109-parameter whole-character network with five coarse context channels (foreground, head, torso, upper appendage, lower appendage) and nine semantic outputs: body, eye, cheek/facial mark, mouth, ear, arm, hand, leg, and foot. A 109,832-parameter v3 face U-Net and a 435,624-parameter 128² residual-SE v4 face U-Net reprocess the enlarged crop located by the coarse head mask. The three primary ONNX graphs total 3.23 MiB and run in-browser through a lazy-loaded WebAssembly runtime.

Training is reproducible in `ml/train_childlike_detector.py`, `ml/train_face_detector_v4.py`, and `ml/train_component_gate_v5.py`. The checked-in runs use 12,992 official ChildlikeSHAPES training drawings, a disjoint deterministic 1,000-drawing validation split, and the complete 1,986-drawing official test split only after selection. V3 adds 3,000 synthetic examples. V4 adds 3,500 synthetic examples, samples 7,000 balanced drawings per epoch, and targets rare cheeks/accessories and ears with residual-SE blocks, Dice/Tversky/boundary loss, graph-paper augmentation, and label hard negatives. Its first three epochs use high positive-class weights for rare-part recall; epochs four through eight reduce those weights to suppress false positives. `ml/evaluate_face_ensemble.py` selects per-class blend weights and thresholds from validation histograms. V5 uses 700 validation drawings to fit a standardized per-component logistic gate and 300 disjoint validation drawings for regularization/threshold selection; the gate reads 22 local logit/geometry features and adds no network download.

This architecture implements the central finding in ChildlikeSHAPES rather than merely citing it: broad character context is estimated before small semantic parts, and the face is enlarged and parsed separately. The foreground/head/torso/appendage logits feed the semantic refiner; the head logit determines the second crop from the original high-resolution isolated drawing, not from the downsampled first pass. In the browser, v3 face logits are bilinearly aligned to 128² and blended with v4 logits using validation-selected v4 weights of 0.5 eye, 0.3 cheek, 0.6 mouth, and 0.5 ear.

On the untouched official test set, calibrated body/limb IoU is 0.901 body, 0.603 arm, 0.540 hand, 0.696 leg, and 0.656 foot. The face ensemble improves every class over the v3 crop: eye 0.6419→0.6578, cheek/facial accessory 0.1652→0.1962, mouth 0.5688→0.5753, and ear 0.4494→0.4791. Macro face IoU rises from 0.4563 to 0.4771 (+4.6% relative). The exact browser decoder initially measured cheek precision/recall/F1 of 0.332/0.731/0.457 and ear 0.623/0.794/0.698. A first high-precision gate overcorrected on the live demo, so the accepted version enforces calibration recall within 0.03 for cheek and 0.05 for ear of the ungated detector. It changes official metrics to cheek 0.382/0.685/0.490 and ear 0.742/0.713/0.727, while count-exact accuracy rises 0.700→0.748 and 0.782→0.833. Absent-image false-positive rates fall 0.240→0.181 and 0.145→0.075. Eye and mouth gates are explicitly disabled, preserving their 0.960 and 0.928 component F1. The heterogeneous accessory label remains the weakest class, so that limitation is reported rather than hidden behind a generic “SAM-like” claim.

An experimental 70,518-parameter image-level presence head was also trained with asymmetric multi-label loss after a browser audit showed rare-label imbalance. It did not pass the sealed acceptance check: cheek F1 improved only 0.0013 and ear regressed 0.0096. It is not shipped. The component gate succeeded because it rejects an individual bad island without deleting a correct instance elsewhere.

The ensemble decodes up to six face instances and ten appendages per class. Predictions from the enlarged face and whole-character passes are merged only when their centers and measured extents indicate the same instance. Uncalibrated whole-image and legacy cheek/ear masks cannot bypass the v5 gate. The older v2/v1 checkpoints stay off the mobile critical path and download only if the primary stack misses an eye, mouth, arm, or leg anchor; they may restore a missing common-part mate but cannot flood the rig with optional facial marks.

The model is intentionally not the geometry source of truth. Its low-resolution masks identify what a region means; WallAlive then snaps each prediction to the nearest high-resolution isolated pixel component. This preserves the drawing’s measured center, dimensions, rotation, outline, and sampled color. Existing medial-skeleton limbs are retained and only confirmed or supplemented by high-confidence learned hints.

The private supplied drawing is evaluated locally without uploading it. The corrected browser ensemble recognizes two eyes, one cheek/facial mark, one mouth, two ears, two legs, and two feet from the selected character crop in 168 ms from a cold page and 106 ms warm, while preserving the measured pink line color (`#d29ea8`), 55-point contour, and 71-point medial skeleton. On six separate public character textures from Meta Animated Drawings, the body model scores 0.803, 0.808, 0.860, 0.935, 0.939, and 0.952 IoU (mean 0.883), including the previous six-arm and quadruped failures. This transfer evidence supplements the much larger untouched official test; it is not a universal-quality claim.

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
- Browser QA verifies lazy same-origin delivery of all three primary ONNX graphs and the WebAssembly runtime, with older checkpoints fetched only on uncertainty and no third-party request for image recognition. The final v3+v4 stack processes eleven varied local QA images without runtime errors at 101 ms mean warm latency; only the annotated drawing benchmarks are treated as accuracy evidence.
- Exact high-resolution eye/cheek/mouth outlines and sampled colors are projected onto the generated surface even when their semantic name came from ML; the previous renderer mistakenly limited projection to heuristic `image-region` parts.
- All generated arm/leg bone branches are retained in arrays and driven during dance/walk, rather than silently reducing multi-arm characters to one left and one right branch.

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
