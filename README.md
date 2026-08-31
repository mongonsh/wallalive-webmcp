# WallAlive

**Draw it. Wake it. Play.**

**Live demo:** https://wallalive-webmcp.mungunshagai-tb.chatgpt.site

**Public source:** https://github.com/mongonsh/wallalive-webmcp

WallAlive turns one human-approved drawing into a colored, 360° **3D character** in the camera view. Google MediaPipe MagicTouch first performs point-guided object isolation, with local drawing-specific fallbacks for line art. The private first result is an identity-preserving, shallow closed relief: the exact artwork appears only on the front, the hidden side is bounded, and uncertain eye/limb labels cannot create or deform visible geometry. [AniGen](https://github.com/VAST-AI-Research/AniGen) remains the visibly approved upgrade for jointly generated unseen surfaces, skeleton, and skinning weights. WebMCP can inspect, name, place, animate, recolor, and direct either result without receiving camera authority.

## Why this needs WebMCP

- Human and agent actions update the same visible character and AR scene.
- The agent composes personality, placement, movement, and story through eight narrow tools.
- The camera is deliberately absent from the tool surface.
- A WebMCP tool may request reconstruction, but only a visible human action can approve the isolated-image upload.
- Every action is attributed to `CHILD`, `BROWSER AGENT`, or `WALLALIVE`.
- `inspect_wall_scene` exposes the real provider, model, GLB type, mesh/bone counts, generation phase, and privacy boundary.

## The real 3D loop

1. **Scan or upload and isolate locally:** the child can capture with the camera or choose an existing drawing photo. MediaPipe MagicTouch uses the child’s tap as a positive prompt and the frame corners as negative prompts. A point-local line extractor and the compact WallAlive cutout model remain fallbacks. Tiny/faint results are rejected instead of being displayed as the wrong character.
2. **Build private 3D immediately:** WallAlive maps the exact transparent artwork to the front triangles of one shallow, closed Marching Cubes relief. Learned depth is clamped inside the silhouette distance envelope with a 0.16 maximum half-depth. The back and sides use neutral sampled body color. Semantic labels remain review-only and never add raised face meshes or local deformation bones.
3. **Play or upgrade:** the private result is rotatable and actionable without an upload. A separate approval can send only the isolated drawing—not the live camera or room frame—to AniGen.
4. **Generate jointly when approved:** AniGen’s `ss_flow_solo` + `slat_flow_auto` path predicts full geometry, a skeleton of arbitrary complexity, and smooth skinning weights together. Three.js loads the returned GLB into a tab-local Blob URL.
5. **Enter AR:** Android/WebXR devices can place either result with surface hit testing; iPhone and other browsers use the camera-overlay experience.

**Play Judge Demo** loads the supplied drawing and its precomputed full neural reconstruction immediately, without spending public GPU quota. An identity-preserving orthographic sketch-to-render pass supplied a 3D-aware condition, TripoSR reconstructed a watertight surface at 256³ extraction resolution, and WallAlive smoothed it, restored approved front colors without copying marks onto the rear, and transferred the learned variable drawing graph into semantic skin weights. The exact fixture is a 68,326-vertex, 136,648-triangle `SkinnedMesh` with seven active bones, 70.8% depth-to-height, normalized weights, and zero boundary or non-manifold edges. The older official AniGen fixture remains as a separate reference test.

## WebMCP tool surface

| Tool | Mode | Purpose |
| --- | --- | --- |
| `inspect_wall_scene` | Read | Returns approved drawing state, actual reconstruction provider/type/counts, AR capability, and privacy boundary. |
| `reconstruct_rigged_3d_character` | Write | Creates the identity-safe local closed relief immediately, or requests `neural-full` and surfaces human approval. It cannot upload or approve by itself. |
| `set_character_personality` | Write | Changes performance intent without modifying the original drawing. |
| `place_character` | Write | Places and scales the character in the visible scene. |
| `animate_character` | Write | Drives one safe action on the generated skeleton. |
| `recolor_character` | Write | Changes a generated presentation accent, not the original pixels. |
| `tell_character_story` | Write | Performs a cancellable one-to-four-beat story. |
| `list_activity` | Read | Returns recent attributed actions without camera or image data. |

All tools use strict schemas, `additionalProperties: false`, cancellation signals, bounded inputs, read/write annotations, and explicit error results. There is no camera, capture, or upload tool.

## Privacy boundary

- Camera start and capture are human-only UI gestures.
- Point-guided segmentation, optional semantic suggestions, and closed-relief reconstruction happen locally.
- The 5.72 MiB six-model primary ONNX stack is served from WallAlive’s own origin and runs with WebAssembly; it does not send the image to an inference API. Body parts, the 96²/128² face ensemble, variable topology, optional 17-joint pose, and front/back depth run concurrently. A learned closed-form component gate rejects false cheek/ear islands without another model download. Older fallback checkpoints load only when the primary stack misses a basic eye, mouth, or limb anchor.
- The agent never receives live frames or raw image pixels.
- Neural generation requires a separate visible human approval.
- Only the isolated drawing is sent to the selected AniGen Space.
- Returned GLBs are copied to tab-local Blob URLs and released when replaced or when the tab closes.

## Browser and inference support

| Capability | Support |
| --- | --- |
| Camera capture + 3D overlay | Modern mobile/desktop browsers over HTTPS with WebGL |
| Private identity-safe closed 3D | In-browser MediaPipe/ONNX/WASM + Three.js, no image upload |
| Generative unseen geometry | Optional AniGen public Space; dedicated AniGen GPU recommended for production |
| Immersive room placement | WebXR `immersive-ar` + `hit-test`, usually compatible Android Chrome devices |
| iPhone/iPad and non-WebXR browsers | Camera overlay; immersive hit testing is not claimed |
| No-camera judging | Bundled exact-drawing neural GLB and one-click story |

The public Hugging Face ZeroGPU service is capacity-limited and can reject or queue live generations. A reliable public product must self-host AniGen on an NVIDIA GPU with at least 18 GB VRAM or configure a dedicated endpoint. No single-view model can guarantee a perfect artist-authored back; quality varies with occlusion, ambiguity, and input clarity.

## Architecture

```text
human camera gesture + tap
        │
        ▼
MediaPipe point-guided drawing isolation
        │
        ▼
WallAlive ChildlikeSHAPES local ONNX/WASM stack
        ├── 96² full character: body + head/torso + limbs
        ├── 96² TopologyNet v10: foreground + centerline + endpoints + junctions
        │       └── variable minimum-spanning graph for any supported family
        ├── 96² AmateurPose: 17 named joints + humanoid applicability gate
        ├── 64² SketchDepth v1: distinct front + hidden-surface depth
        └── validation-calibrated face ensemble
            ├── 96² v3 enlarged head crop
            └── 128² v4 rare-feature head crop
                └── learned cheek/ear component confidence gate
        │
        ├── exact pixel snap: position + outline + color
        ├── joint paths: variable graph branches or applicable humanoid chains
        └── bounded signed-distance relief → Marching Cubes → one continuous local SkinnedMesh
                └────────────────────────────────────► private identity-safe 360° character
        │
        ├── visible isolated-image approval (human only)
        ▼
AniGen: image condition → shape + skeleton + skin weights
        │
        ▼
rigged GLB → local Blob URL → Three.js GLTFLoader / SkinnedMesh
        │                              ▲
        ├── 360° camera overlay / AR   └── WebMCP place / animate / story
        └── WebXR hit-test placement
```

- React 19 + TypeScript, built with vinext for Cloudflare/Sites
- Three.js `GLTFLoader`, `SkinnedMesh`, procedural bone actions, and WebXR hit testing
- WallAlive local stack: a 288,109-parameter whole-character network, 109,832-parameter 96² and 435,624-parameter 128² face networks, a 402,052-parameter spatial variable-topology network, a 161,133-parameter optional 17-joint pose network, and an 80,486-parameter compact U-Net front/back depth prior, loaded through `onnxruntime-web`
- Instance-aware decoding preserves multiple same-side arms/legs and separate facial features; a 22-feature standardized logistic gate judges individual cheek/ear masks from model agreement, probability, size, shape, fill, and position before every accepted region snaps back to high-resolution pixel geometry
- Original eyes, cheeks, mouth, color, and line quality stay in the front texture. The local renderer deliberately adds no raised semantic face geometry.
- Semantic and topology outputs remain inspectable and manually editable, but only the full neural asset uses generated articulation. This prevents false anatomy from warping a child’s drawing.
- Lazy-loaded `@gradio/client` connection to AniGen
- `document.modelContext.registerTool()` imperative WebMCP integration
- One canonical action layer shared by UI and tool executors
- Deterministic contour preview remains available when the user declines upload

## Local model training

`ml/train_childlike_detector.py` trains the checked-in v3 models from the official pixel-labeled ChildlikeSHAPES release. `ml/train_face_detector_v4.py` adds a 128² residual-SE face parser with rare-class sampling, boundary/Tversky loss, graph-paper augmentation, label hard negatives, a three-epoch high-recall warmup, and five moderate-weight refinement epochs. `ml/evaluate_face_ensemble.py` searches blend weights and thresholds on validation only. `ml/train_component_gate_v5.py` then splits those 1,000 validation drawings into 700 meta-training and 300 calibration drawings to learn per-component cheek/ear confidence; eyes and mouths are deliberate no-ops. `ml/prepare_amateur_benchmark.py`, `ml/evaluate_amateur_benchmark.py`, `ml/train_amateur_pose_v6.py`, and `ml/evaluate_amateur_pose_onnx.py` build, measure, train, and export-check the independent Meta Amateur Drawings pose benchmark. The official test splits remain outside all fitting and threshold selection.

On the untouched official ChildlikeSHAPES test set, the body/limb model scores 0.901 body, 0.603 arm, 0.540 hand, 0.696 leg, and 0.656 foot IoU. The frozen v3+v4 face ensemble improves every face class over v3 alone: eye 0.642→0.658, cheek/facial accessory 0.165→0.196, mouth 0.569→0.575, and ear 0.449→0.479. Macro face IoU rises from 0.4563 to 0.4771. At the browser component level, the recall-preserving v5 gate raises cheek precision 0.332→0.382 and F1 0.457→0.490, while ear precision rises 0.623→0.742 and F1 0.698→0.727. Cheek/ear absent-image false-positive rates fall 0.240→0.181 and 0.145→0.075; eye and mouth behavior is unchanged.

The independent Amateur Drawings run contains 352 training, 63 validation, and 75 untouched test characters. The baseline browser pipeline already covered 96.31% of visible joints with its body mask but localized named semantic parts poorly—especially ears, hands, and feet. AmateurPose v6 reaches 0.7969 PCK@5%, 0.8643 PCK@10%, and 5.578 mean pixels of error at 96² on the untouched test split. The checked-in ONNX export reproduces those metrics exactly. Its output only refines named joint paths after a part-and-geometry applicability gate; it cannot invent eyes, ears, mouths, or cheeks. Exact segmentation evidence still decides face position, shape, outline, and sampled color. These numbers are regression evidence, not a claim that single-view recognition or an inferred unseen back can be perfect.

`ml/train_topology_v10.py` jointly learns four graph fields from exact procedural graph labels and an eight-family spatial class head from 10,241 real Google Quick, Draw! records. Real strokes supervise class only, never pseudo endpoints or junctions. On disjoint sealed tests it reaches 0.9915 one-pixel-tolerant centerline F1, 0.5600 endpoint IoU, 0.6436 junction IoU, and 0.9587 real-drawing family accuracy; the checked-in ONNX reproduces the report exactly.

`ml/train_sketch_depth_v1.py` trains the checked-in 80,486-parameter compact U-Net on analytic unions of articulated ellipsoids. The 6,144 training, 768 validation, and 768 sealed-test examples are balanced across the same eight families and contain no child drawings or user pixels. Model selection uses validation only; the sealed test reports 0.03636 normalized surface MAE, 0.91988 surface correlation, and a nonzero 0.04270 front/back asymmetry MAE. ONNX agrees with the selected PyTorch checkpoint within 2.26e-6. These numbers validate the learned depth task on analytic ground truth; an unseen back from a real single view remains a plausible prior, not measured fact.

`ml/prepare_varied_quickdraw_benchmark.py` now preserves eight attributed source records after line 2400—beyond topology-v10's complete training/validation/test window ending before line 1960. The set covers biped, quadruped, winged, aquatic, radial, branched, machine, and chain families. Topology-v10 classifies all eight correctly. The offline browser-math mirror then emits eight colored graph-skinned GLBs: every result has nonzero learned front/back asymmetry, 17.9–42.6% relative depth, normalized weights on every vertex, active branch influence, zero boundary/non-manifold edges, and a closed true volume. The contact sheet and machine-readable results are in `eval/varied-drawings/`.

For difficult real photos, `ml/evaluate_single_drawing.py` reproduces the browser face ensemble and writes per-part overlay evidence. `ml/prepare_single_drawing_mesh.py` mirrors the learned asymmetric depth volume at higher resolution, retains the exact front texture and sampled palette, and writes semantic/rig metadata. `scripts/export-semantic-glb.py` exports a colored, closed, skinned GLB without Blender; `scripts/rig-neural-glb.py` can smooth a full neural surface, project approved color only onto its visible front, and transfer the learned drawing graph. `scripts/evaluate-anigen.mjs --inspect` validates geometric closure after welding legitimate UV/normal seams, true volume, normalized weights, active joints, branch influence, and color materials.

## Local development

Requires Node.js `>=22.13.0`.

```bash
npm ci
npm run dev
```

Validation:

```bash
npm run lint
npm test
npm audit --omit=dev
```

Optional exact-input 3D regression (requires the Python packages in `ml/requirements.txt`; Blender is not required):

```bash
python ml/evaluate_single_drawing.py --image drawing.png --output /tmp/wallalive-face-evidence
python ml/prepare_single_drawing_mesh.py --image isolated-drawing.png --output /tmp/wallalive-mesh --resolution 256
python scripts/export-semantic-glb.py --input-dir /tmp/wallalive-mesh --output /tmp/wallalive.glb
node scripts/evaluate-anigen.mjs --inspect /tmp/wallalive.glb
```

## Attribution and license

WallAlive is MIT licensed. V3 is trained on the official [ChildlikeSHAPES](https://arxiv.org/abs/2504.08022) release under CC-BY-4.0; dataset images are not redistributed in this repository. The retained fallback models use a 490-image slice of Meta’s public [Amateur Drawings Dataset](https://github.com/facebookresearch/AnimatedDrawings#amateur-drawings-dataset), released under MIT. TopologyNet v10 uses the official [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset); the eight small benchmark records retain key IDs and source attribution under CC BY 4.0. The optional live reconstruction path uses [AniGen](https://github.com/VAST-AI-Research/AniGen), whose project and model card are MIT licensed; its repository separately notes research/non-commercial restrictions on a bundled CUBVH-derived component, which must be reviewed before commercial self-hosting. The exact-drawing judge surface uses the official MIT-licensed [TripoSR](https://github.com/VAST-AI-Research/TripoSR) implementation plus WallAlive's own graph rigger. The older official AniGen `brickbob.png` result is retained only as independent reference evidence. See [docs/RESEARCH.md](./docs/RESEARCH.md) for the evaluated alternatives and exact capability boundary.
