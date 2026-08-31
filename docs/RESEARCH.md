# Drawing-to-rigged-3D and WebMCP research

This note records what WallAlive actually does, why the previous silhouette method was insufficient, and the boundary between local privacy work and neural reconstruction.

## Root cause of the frightening result

The failure had three independent causes. First, the target cutout checkpoint had only 350 Meta Amateur Drawings and roughly 0.61 sealed-test IoU, so it was not a reliable primary segmenter for printed books, painted faces, textured paper, faint pencil, and cluttered phone photos. Second, the analytic-ellipsoid depth prior was allowed to inflate uncertain silhouettes into face-like eggs. Third, guessed eyes, pupils, cheeks, mouths, and markings were converted into raised spheres and contour tubes. A false semantic label therefore became frightening visible anatomy. The interface then made the problem worse by displaying uncalibrated “clean cutout,” part-count, and topology claims.

The v29 repair changes the decision boundary instead of applying another renderer patch. Point-local line-art extraction now runs first, the compact drawing-specific checkpoint runs second, and general MagicTouch is only a final proposal. After the six local graphs run, `character-quality.ts` requires facial, articulated-part, pose, or variable-topology evidence and separately measures rectangularity, axis-aligned perimeter, and oversized mask coverage. Generic segmentation confidence is explicitly not character evidence.

Even an accepted mask stops at a transparent cutout review. The app initializes no Three.js renderer, enables no action, and makes no 3D claim until a real skinned GLB exists. The child can continue or try again; AniGen capacity failure leaves this safe state intact. The earlier learned-depth Marching Cubes relief remains useful offline reconstruction research, but it is no longer presented to a child as the recovered character.

The optional AniGen path remains the route to generated unseen geometry, skeleton, and skin weights. Its public GPU quota is not a reliable production service, so the local fallback must remain safe and honest when full generation is unavailable.

## Systems evaluated

| System | What it provides | Why it is or is not the main path |
| --- | --- | --- |
| [MediaPipe Interactive Segmenter / MagicTouch](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/tasks/web/vision) | Browser-ready positive/negative point and stroke segmentation | Retained only as a last-resort proposal. A general object mask can confidently select paper, a screen, or a camera patch, so drawing-aware extraction and the post-recognition character gate must authorize it. |
| [Meta Animated Drawings](https://github.com/facebookresearch/AnimatedDrawings) | Detector, segmentation, pose estimation, and 2D character animation | Excellent precedent for children’s drawings, but its output is a 2D articulated deformation rather than 360° geometry. |
| [ChildlikeSHAPES / CharSegNet](https://arxiv.org/abs/2504.08022) | 16,075 annotated drawings and hierarchical parsing with 25 semantic categories | Selected as the real-data source for v3. WallAlive trains its own browser-sized student because the research architecture/checkpoint is not a mobile WebAssembly artifact. |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) | Promptable mobile image/video masks | A strong local proposal model, but it needs a point or box and does not name eyes, cheeks, or limbs. Raw masks also selected paper/clutter in drawing tests, so it is not allowed to replace target-aware drawing extraction blindly. |
| [SAM 2](https://ai.meta.com/research/sam2/) | Promptable image/video object masks | Useful segmentation, but it cannot invent an unseen back, mesh, skeleton, or skin weights. |
| [SAM 3](https://github.com/facebookresearch/sam3) | Promptable instance segmentation from text and visual prompts | Strong general proposals, but a large prompt model is not a browser-sized drawing anatomy parser and still does not create a rigged 3D asset. |
| [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) | Single-image shape, texture, pose, and layout reconstruction | Strong current 3D alternative, but its primary output is a static reconstructed object rather than an animate-ready skeleton and skinning field. |
| [DrawingSpinUp](https://github.com/LordLiang/DrawingSpinUp) | Drawing-specific contour removal, multiview synthesis, 3D reconstruction, thinning, and Mixamo animation | The closest sketch-specific reference pipeline and especially strong for bipeds, but its published workflow requires per-character optimization/training and a CUDA/Linux stack; Mixamo remains a post-hoc humanoid rigging stage. |
| [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) | Fast textured single-image GLB generation | Produces a real mesh but not an animation skeleton or skinning weights. |
| [TripoSR](https://github.com/VAST-AI-Research/TripoSR) | MIT-licensed learned single-image mesh reconstruction | A raw sparse cat drawing produced a thin poor shape. Conditioning it with an identity-preserving straight-on 3D-aware concept render of the supplied drawing produced a materially better 68,326-vertex / 136,648-triangle watertight mesh at 256³ extraction resolution on Apple M4 in 14.7 seconds. WallAlive then applies Taubin smoothing, front-only approved color projection, and its variable graph rigger. TripoSR itself still has no native rig. |
| [TRELLIS](https://github.com/microsoft/TRELLIS) | Image-conditioned 3D representations and meshes | Strong geometry prior, but no coherent animation rig in its primary output. |
| [Hunyuan3D 2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Geometry and PBR texture generation through a GPU service | Real 3D, but rigging remains a separate problem. |
| [UniRig](https://github.com/VAST-AI-Research/UniRig) / [RigAnything](https://github.com/Isabella98Liu/RigAnything) | Post-hoc automatic rigging | Viable after a mesh generator, but a sequential pipeline compounds geometry/rig errors and deployment cost. |
| [AniGen](https://github.com/VAST-AI-Research/AniGen) | Single-image mesh, skeleton, and skinning weights generated jointly | Selected. It directly matches WallAlive’s requirement for animate-ready 360° assets across humanoids, animals, and machinery. |

AniGen represents shape, skeleton, and skinning as mutually consistent fields and generates them together. Its official repository reports generalization across animals, humanoids, and machinery; the official Space includes example conditions for a child drawing, dog, owl, plant, whale, T-rex, lamp, and machine arm. This is a far more defensible “varied drawing” prior than hand-coded eye/limb heuristics.

The two Kivicube examples supplied for comparison demonstrate authored AR artwork and an authored AR game: the artist prepares the target, content, and interactions ahead of time. WallAlive has the harder requirement of accepting a previously unseen drawing from a live camera. It therefore needs both drawing-specific recognition and generative 3D reconstruction; an AR target tracker alone does not solve that problem.

## Learned local part recognition

WallAlive now includes a task-specific hierarchical segmenter rather than asking a general object mask to guess anatomy. `WallAlive ChildlikeSHAPES PartUNet v3` has a 288,109-parameter whole-character network with five coarse context channels (foreground, head, torso, upper appendage, lower appendage) and nine semantic outputs: body, eye, cheek/facial mark, mouth, ear, arm, hand, leg, and foot. A 109,832-parameter v3 face U-Net and a 435,624-parameter 128² residual-SE v4 face U-Net reprocess the enlarged crop located by the coarse head mask. A separate 161,133-parameter AmateurPose v6 graph predicts 17 named joints. Topology-v10 and SketchDepth-v1 complete a six-graph, 5.72 MiB primary stack that runs in-browser through a lazy-loaded WebAssembly runtime.

Training is reproducible in `ml/train_childlike_detector.py`, `ml/train_face_detector_v4.py`, and `ml/train_component_gate_v5.py`. The checked-in runs use 12,992 official ChildlikeSHAPES training drawings, a disjoint deterministic 1,000-drawing validation split, and the complete 1,986-drawing official test split only after selection. V3 adds 3,000 synthetic examples. V4 adds 3,500 synthetic examples, samples 7,000 balanced drawings per epoch, and targets rare cheeks/accessories and ears with residual-SE blocks, Dice/Tversky/boundary loss, graph-paper augmentation, and label hard negatives. Its first three epochs use high positive-class weights for rare-part recall; epochs four through eight reduce those weights to suppress false positives. `ml/evaluate_face_ensemble.py` selects per-class blend weights and thresholds from validation histograms. V5 uses 700 validation drawings to fit a standardized per-component logistic gate and 300 disjoint validation drawings for regularization/threshold selection; the gate reads 22 local logit/geometry features and adds no network download.

The pose work uses a second real-drawing domain, not another synthetic validation loop. `ml/prepare_amateur_benchmark.py` streams Meta's full COCO annotation JSON and creates filename-hashed train/validation/test manifests from 490 valid characters in a reproducible image sample. `ml/evaluate_amateur_benchmark.py` recreates the browser's 512² isolated-character layout and measures the existing 96² semantic models. `ml/train_amateur_pose_v6.py` trains only on 352 training drawings with rotation, translation, scale, shear, mirroring, camera blur, brightness, contrast, and sensor-noise augmentation; epoch selection uses 63 validation drawings. The 75 test drawings are not instantiated until after checkpoint selection and are evaluated once. `ml/evaluate_amateur_pose_onnx.py` then runs the exported browser graph over that exact test set and fails unless it reproduces the PyTorch metrics.

This architecture implements the central finding in ChildlikeSHAPES rather than merely citing it: broad character context is estimated before small semantic parts, and the face is enlarged and parsed separately. The foreground/head/torso/appendage logits feed the semantic refiner; the head logit determines the second crop from the original high-resolution isolated drawing, not from the downsampled first pass. In the browser, v3 face logits are bilinearly aligned to 128² and blended with v4 logits using validation-selected v4 weights of 0.5 eye, 0.3 cheek, 0.6 mouth, and 0.5 ear.

On the untouched official test set, calibrated body/limb IoU is 0.901 body, 0.603 arm, 0.540 hand, 0.696 leg, and 0.656 foot. The face ensemble improves every class over the v3 crop: eye 0.6419→0.6578, cheek/facial accessory 0.1652→0.1962, mouth 0.5688→0.5753, and ear 0.4494→0.4791. Macro face IoU rises from 0.4563 to 0.4771 (+4.6% relative). The exact browser decoder initially measured cheek precision/recall/F1 of 0.332/0.731/0.457 and ear 0.623/0.794/0.698. A first high-precision gate overcorrected on the live demo, so the accepted version enforces calibration recall within 0.03 for cheek and 0.05 for ear of the ungated detector. It changes official metrics to cheek 0.382/0.685/0.490 and ear 0.742/0.713/0.727, while count-exact accuracy rises 0.700→0.748 and 0.782→0.833. Absent-image false-positive rates fall 0.240→0.181 and 0.145→0.075. Eye and mouth gates are explicitly disabled, preserving their 0.960 and 0.928 component F1. The heterogeneous accessory label remains the weakest class, so that limitation is reported rather than hidden behind a generic “SAM-like” claim.

An experimental 70,518-parameter image-level presence head was also trained with asymmetric multi-label loss after a browser audit showed rare-label imbalance. It did not pass the sealed acceptance check: cheek F1 improved only 0.0013 and ear regressed 0.0096. It is not shipped. The component gate succeeded because it rejects an individual bad island without deleting a correct instance elsewhere.

The ensemble decodes up to six face instances and ten appendages per class. Predictions from the enlarged face and whole-character passes are merged only when their centers and measured extents indicate the same instance. Uncalibrated whole-image and legacy cheek/ear masks cannot bypass the v5 gate. The older v2/v1 checkpoints stay off the mobile critical path and download only if the primary stack misses an eye, mouth, arm, or leg anchor; they may restore a missing common-part mate but cannot flood the rig with optional facial marks.

The model is intentionally not the geometry source of truth. Its low-resolution masks identify what a region means; WallAlive then snaps each prediction to the nearest high-resolution isolated pixel component. This preserves the drawing’s measured center, dimensions, rotation, outline, and sampled color. Existing medial-skeleton limbs are retained and only confirmed or supplemented by high-confidence learned hints.

TopologyNet v10 adds a separate 402,052-parameter learned graph prior. It was trained with synthetic endpoint/junction supervision plus 10,241 disjoint real Quick, Draw! stroke records spanning biped, quadruped, winged, aquatic, radial, branched, machine, and chain families. Its thick centerline band is now Zhang-Suen thinned before graph-degree endpoint and junction recovery; this changed the radial QA drawing from a collapsed 2-node graph to 12 nodes / 11 branches and the tree to 13 nodes / 12 branches.

The private supplied drawing is evaluated locally without uploading it. On the current exact crop, the model reports biped topology at 0.859 confidence and preserves body, two eyes, two cheeks/facial marks, mouth, ears, arm/hand, leg/foot, and the measured pale-pink line color. The offline SketchDepth evaluator can produce a 26,054-vertex, 52,104-triangle continuous surface and a ten-bone graph GLB, but that analytic-domain result is not displayed in the live capture flow. On six separate public character textures from Meta Animated Drawings, the body model scores 0.803, 0.808, 0.860, 0.935, 0.939, and 0.952 IoU (mean 0.883). This is transfer evidence for local annotation and gating, not a universal 3D-quality claim.

SketchDepth-v1 is an 80,486-parameter compact U-Net with three input channels—foreground, interior distance, and authored/internal contour ink—and two independent outputs for front and hidden-surface depth. `ml/train_sketch_depth_v1.py` synthesizes analytic unions of articulated ellipsoids with exact surface supervision. Training, validation, and sealed test contain 6,144 / 768 / 768 examples, balanced equally across eight topology families; no child drawings or user pixels are used. Selection uses validation only. The sealed test reaches 0.036363 normalized surface MAE, 0.91988 surface correlation, and 0.042699 front/back asymmetry MAE; ONNX maximum error is 2.26e-6. This is a real learned depth model, but its real-image hidden surface cannot be ground-truthed from one view.

The broader 75-drawing Amateur test split exposed the next root cause precisely. Its body-mask IoU is 0.8975 and 96.31% of visible annotated joints fall inside the isolated body, but the semantic decoder's named-joint hit rate is only 0.6333 eye, 0.0733 ear, 0.8200 arm, 0.5267 hand, 0.8733 leg, and 0.5933 foot. AmateurPose v6 improves the actual rig signal to 0.7969 PCK@5%, 0.8643 PCK@10%, and 5.578 mean input pixels of error. Strict per-joint PCK is 0.76–0.8133 for eyes, 0.80–0.8133 for ears, 0.7467–0.8267 for arm joints, and 0.72–0.88 for leg joints. The ONNX browser export matches the selected PyTorch checkpoint exactly.

Pose is deliberately a geometry refiner, not a second face renderer. The existing masks remain authoritative for eyes, cheeks, mouths, ears, local color, and contours. When the semantic detector finds one or two arms and one or two legs and the predicted shoulder/hip/knee/ankle ordering is plausible, the pose graph bends each arm through shoulder→elbow→wrist and each leg through hip→knee→ankle. Unusual animals, machinery, and multi-limbed characters retain their instance-aware semantic skeleton instead of being forced into a humanoid template.

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

The quota-free judge path uses the same exact supplied drawing but does not pretend the local depth surface is photorealistic. A strict identity-preserving, straight-on sketch-to-render pass created a 3D-aware condition with small restrained eyes, cheek, mouth, pink markings, two ears, side arm, and feet. Local TripoSR then produced a watertight full volume at 256³ extraction resolution. `scripts/rig-neural-glb.py` applies volume-preserving smoothing, projects the approved colors only onto the visible front, keeps the sides/rear neutral, converts glTF vertex colors to linear space, faces the authored front toward the app camera, transfers the learned biped graph into ear/arm/leg bones, and emits normalized skin weights. This path is precomputed evidence, not a browser-time API claim; new live inputs still use the explicit AniGen approval path.

## Verified evidence

- Eight post-split held-out human drawings classify correctly: biped snowman, quadruped cat, winged bird, aquatic fish, radial octopus, branched tree, machine car, and chain snake.
- Their learned-depth graph-skinned GLBs contain 28,624–63,408 triangles, 3–9 active bones, 17.9–42.6% relative depth, nonzero front/back asymmetry, normalized weights, complete color data, zero boundary/non-manifold edges, and a closed true volume.
- The exact supplied drawing's full neural judge asset contains 68,326 vertices, 136,648 triangles, seven bones with all seven active, 12.76% branch-influence coverage, 70.77% relative depth, complete vertex color, normalized weights, and zero boundary/non-manifold edges. Its rear is intentionally feature-free: approved front marks are never projected onto hidden surfaces.

- The official Space accepted the WallAlive demo PNG and completed its CPU preprocessing path.
- A complete AniGen generation was downloaded and parsed locally.
- The older independent AniGen reference fixture contains exactly one mesh, one `SkinnedMesh`, 20 bones, and 159,930 vertices; it is no longer presented as the supplied drawing.
- Blender rendered its front/back geometry and preserved vertex color.
- The deployed test suite parses the binary GLB and fails if it regresses into a non-skinned or low-detail asset.
- Desktop/mobile browser QA verified the approval UI, lazy Gradio client, successful CORS requests, session creation, file upload, preprocessing call, and friendly public-quota failure handling.
- Browser QA verifies lazy same-origin delivery of all six primary ONNX graphs and the WebAssembly runtime, with older checkpoints fetched only on uncertainty and no third-party request for local image recognition. Only annotated or sealed benchmarks are treated as accuracy evidence.
- Exact high-resolution eye/cheek/mouth outlines and sampled colors are projected onto the generated surface even when their semantic name came from ML; the previous renderer mistakenly limited projection to heuristic `image-region` parts.
- All generated arm/leg bone branches are retained in arrays and driven during dance/walk, rather than silently reducing multi-arm characters to one left and one right branch.

The current free ZeroGPU quota prevented repeated end-to-end generations across every example category in one session. The model and live pipeline are real; a dedicated endpoint is still required for reliable high-volume category testing and production use.

## WebMCP design

WallAlive follows the [WebMCP Community Group draft](https://webmachinelearning.github.io/webmcp/) and the official [imperative API explainer](https://github.com/webmachinelearning/webmcp#imperative-tool-registration-documentmodelcontext):

- Register six goal-level tools with `document.modelContext.registerTool()` in the top-level page.
- Expose shared workflow state and separate per-character action capabilities before mutation.
- Use narrow descriptions, strict nested JSON Schema, bounded cast/beat arrays, and cancellation.
- Validate every proposed move against verified pose/topology branches rather than silently animating the wrong geometry.
- Share the same state and per-character action channels between human UI and tool executors.
- Separate staging from execution: `stage_magic_show` displays an inert plan, while **Approve & play** remains a human-only UI action.
- Return the validated plan or exact performed actions, final state, and camera-data exclusion for verification.

The important safety capability is structural: no registered tool can open the camera, capture a frame, approve external processing, or approve a staged show. `request_rigged_3d_cast` and `stage_magic_show` can surface human decisions, then stop. This makes human agency enforceable in code instead of relying on a prompt.

## Honest capability statement

WallAlive uses learned local semantic, topology, and pose models to verify and annotate a transparent cutout, then uses real full-volume neural reconstruction for a playable skinned skeleton. It does not claim perfect recognition or a true artist-authored unseen back from one view. Single-view reconstruction is ambiguous, the independent tests are finite, and public GPU capacity is not reliable. The safe local state is reviewable 2D, not counterfeit 3D. For production, self-host AniGen on a dedicated GPU and expand the accepted Meta/ChildlikeSHAPES training set plus real paper-photo hard negatives before promising broad quality levels.
