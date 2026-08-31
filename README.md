# WallAlive

**Draw it. Wake it. Play.**

**Live demo:** https://wallalive-webmcp.mungunshagai-tb.chatgpt.site

**Public source:** https://github.com/mongonsh/wallalive-webmcp

WallAlive turns one human-approved drawing into a colored, 360° **rigged 3D character** in the camera view. Compact locally trained models recognize visible semantic parts and decode a variable endpoint/junction graph for bipeds, quadrupeds, winged and aquatic forms, radial creatures, plants, machines, and chains. It uses [AniGen](https://github.com/VAST-AI-Research/AniGen) to generate a coherent mesh, unseen surfaces, skeleton, and skinning weights from the isolated image, then loads the result as a Three.js `SkinnedMesh`. WebMCP lets a browser agent name, place, animate, recolor, and direct that same visible character.

The old contour inflation is retained only as a clearly labeled **rough private preview**. It is never presented as neural reconstruction.

## Why this needs WebMCP

- Human and agent actions update the same visible character and AR scene.
- The agent composes personality, placement, movement, and story through eight narrow tools.
- The camera is deliberately absent from the tool surface.
- A WebMCP tool may request reconstruction, but only a visible human action can approve the isolated-image upload.
- Every action is attributed to `CHILD`, `BROWSER AGENT`, or `WALLALIVE`.
- `inspect_wall_scene` exposes the real provider, model, GLB type, mesh/bone counts, generation phase, and privacy boundary.

## The real 3D loop

1. **Scan or upload and recognize locally:** the child can capture with the camera or choose an existing drawing photo. WallAlive separates one drawing from paper edges, text, and foreground clutter in browser memory. Uploaded photos use a wider full-character search window so top ears and lower feet are not removed before inference. WallAlive PartUNet proposes nine visible semantic part classes, TopologyNet v10 decodes variable endpoints, junctions, and branch paths for non-human drawings, and AmateurPose v6 estimates 17 named joints only for applicable humanoids. Semantic masks snap back to original pixel regions, so face position, outline, and sampled color come from the child’s drawing.
2. **Approve minimally:** a second visible approval explains that only the isolated drawing—not the live camera or room frame—will be sent to AniGen.
3. **Generate jointly:** AniGen’s `ss_flow_solo` + `slat_flow_auto` path predicts full geometry, a skeleton of arbitrary complexity, and smooth skinning weights together.
4. **Load and play:** the browser downloads the returned GLB into a tab-local Blob URL. Three.js loads its `SkinnedMesh`, classifies arm/leg bone branches, and drives wave, dance, walk, hop, hide, and spin actions.
5. **Enter AR:** Android/WebXR devices can place the same model with surface hit testing; iPhone and other browsers use the camera-overlay experience.

**Play Judge Demo** loads a previously generated AniGen reference result immediately, without spending public GPU quota. The fixture is a colored 159,930-vertex, 319,836-triangle `SkinnedMesh` with 20 bones, 26.6% depth-to-width, normalized weights, and zero boundary or non-manifold edges; tests parse the binary GLB and enforce those invariants.

## WebMCP tool surface

| Tool | Mode | Purpose |
| --- | --- | --- |
| `inspect_wall_scene` | Read | Returns approved drawing state, actual reconstruction provider/type/counts, AR capability, and privacy boundary. |
| `reconstruct_rigged_3d_character` | Write | Uses an already approved AniGen rig. If absent, surfaces the human approval UI but cannot upload or approve it. |
| `set_character_personality` | Write | Changes performance intent without modifying the original drawing. |
| `place_character` | Write | Places and scales the character in the visible scene. |
| `animate_character` | Write | Drives one safe action on the generated skeleton. |
| `recolor_character` | Write | Changes a generated presentation accent, not the original pixels. |
| `tell_character_story` | Write | Performs a cancellable one-to-four-beat story. |
| `list_activity` | Read | Returns recent attributed actions without camera or image data. |

All tools use strict schemas, `additionalProperties: false`, cancellation signals, bounded inputs, read/write annotations, and explicit error results. There is no camera, capture, or upload tool.

## Privacy boundary

- Camera start and capture are human-only UI gestures.
- Segmentation, semantic-part recognition, and the first preview happen locally.
- The 5.39 MiB five-model ONNX stack is served from WallAlive’s own origin and runs with WebAssembly; it does not send the image to an inference API. The 96² and 128² face logits are validation-calibrated and blended locally; variable topology and an optional 17-joint pose graph run in parallel with the part model. A learned closed-form component gate rejects false cheek/ear islands without another model download. Older fallback checkpoints load only when the primary stack misses a basic eye, mouth, or limb anchor.
- The agent never receives live frames or raw image pixels.
- Neural generation requires a separate visible human approval.
- Only the isolated drawing is sent to the selected AniGen Space.
- Returned GLBs are copied to tab-local Blob URLs and released when replaced or when the tab closes.

## Browser and inference support

| Capability | Support |
| --- | --- |
| Camera capture + 3D overlay | Modern mobile/desktop browsers over HTTPS with WebGL |
| Rigged single-image 3D | AniGen public Space for the demo; dedicated AniGen GPU recommended for production |
| Immersive room placement | WebXR `immersive-ar` + `hit-test`, usually compatible Android Chrome devices |
| iPhone/iPad and non-WebXR browsers | Camera overlay; immersive hit testing is not claimed |
| No-camera judging | Bundled verified AniGen GLB and one-click story |

The public Hugging Face ZeroGPU service is capacity-limited and can reject or queue live generations. A reliable public product must self-host AniGen on an NVIDIA GPU with at least 18 GB VRAM or configure a dedicated endpoint. No single-view model can guarantee a perfect artist-authored back; quality varies with occlusion, ambiguity, and input clarity.

## Architecture

```text
human camera gesture + tap
        │
        ▼
local target-aware drawing isolation
        │
        ▼
WallAlive ChildlikeSHAPES local ONNX/WASM stack
        ├── 96² full character: body + head/torso + limbs
        ├── 96² TopologyNet v10: foreground + centerline + endpoints + junctions
        │       └── variable minimum-spanning graph for any supported family
        ├── 96² AmateurPose: 17 named joints + humanoid applicability gate
        └── validation-calibrated face ensemble
            ├── 96² v3 enlarged head crop
            └── 128² v4 rare-feature head crop
                └── learned cheek/ear component confidence gate
        │
        ├── exact pixel snap: position + outline + color
        ├── joint paths: shoulder→elbow→wrist / hip→knee→ankle
        └─────────────────────────────────────────────► rough private preview
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
- WallAlive local stack: a 288,109-parameter whole-character network, 109,832-parameter 96² and 435,624-parameter 128² face networks, a 402,052-parameter spatial variable-topology network, and a 161,133-parameter optional 17-joint pose network, loaded through `onnxruntime-web`
- Instance-aware decoding preserves multiple same-side arms/legs and separate facial features; a 22-feature standardized logistic gate judges individual cheek/ear masks from model agreement, probability, size, shape, fill, and position before every accepted region snaps back to high-resolution pixel geometry
- Raised face meshes and detected contour tubes add real eye/cheek/mouth parallax to the private preview instead of leaving the features flat
- High-confidence topology endpoints now create actual articulated ear/arm/leg branches with hand/foot children when the fixed humanoid pose gate is not applicable; this is what lets variable non-human drawings drive the preview rig instead of merely displaying a class label
- Lazy-loaded `@gradio/client` connection to AniGen
- `document.modelContext.registerTool()` imperative WebMCP integration
- One canonical action layer shared by UI and tool executors
- Deterministic contour preview remains available when the user declines upload

## Local model training

`ml/train_childlike_detector.py` trains the checked-in v3 models from the official pixel-labeled ChildlikeSHAPES release. `ml/train_face_detector_v4.py` adds a 128² residual-SE face parser with rare-class sampling, boundary/Tversky loss, graph-paper augmentation, label hard negatives, a three-epoch high-recall warmup, and five moderate-weight refinement epochs. `ml/evaluate_face_ensemble.py` searches blend weights and thresholds on validation only. `ml/train_component_gate_v5.py` then splits those 1,000 validation drawings into 700 meta-training and 300 calibration drawings to learn per-component cheek/ear confidence; eyes and mouths are deliberate no-ops. `ml/prepare_amateur_benchmark.py`, `ml/evaluate_amateur_benchmark.py`, `ml/train_amateur_pose_v6.py`, and `ml/evaluate_amateur_pose_onnx.py` build, measure, train, and export-check the independent Meta Amateur Drawings pose benchmark. The official test splits remain outside all fitting and threshold selection.

On the untouched official ChildlikeSHAPES test set, the body/limb model scores 0.901 body, 0.603 arm, 0.540 hand, 0.696 leg, and 0.656 foot IoU. The frozen v3+v4 face ensemble improves every face class over v3 alone: eye 0.642→0.658, cheek/facial accessory 0.165→0.196, mouth 0.569→0.575, and ear 0.449→0.479. Macro face IoU rises from 0.4563 to 0.4771. At the browser component level, the recall-preserving v5 gate raises cheek precision 0.332→0.382 and F1 0.457→0.490, while ear precision rises 0.623→0.742 and F1 0.698→0.727. Cheek/ear absent-image false-positive rates fall 0.240→0.181 and 0.145→0.075; eye and mouth behavior is unchanged.

The independent Amateur Drawings run contains 352 training, 63 validation, and 75 untouched test characters. The baseline browser pipeline already covered 96.31% of visible joints with its body mask but localized named semantic parts poorly—especially ears, hands, and feet. AmateurPose v6 reaches 0.7969 PCK@5%, 0.8643 PCK@10%, and 5.578 mean pixels of error at 96² on the untouched test split. The checked-in ONNX export reproduces those metrics exactly. Its output only refines named joint paths after a part-and-geometry applicability gate; it cannot invent eyes, ears, mouths, or cheeks. Exact segmentation evidence still decides face position, shape, outline, and sampled color. These numbers are regression evidence, not a claim that single-view recognition or an inferred unseen back can be perfect.

`ml/train_topology_v10.py` jointly learns four graph fields from exact procedural graph labels and an eight-family spatial class head from 10,241 real Google Quick, Draw! records. Real strokes supervise class only, never pseudo endpoints or junctions. On disjoint sealed tests it reaches 0.9915 one-pixel-tolerant centerline F1, 0.5600 endpoint IoU, 0.6436 junction IoU, and 0.9587 real-drawing family accuracy; the checked-in ONNX reproduces the report exactly. `ml/prepare_varied_quickdraw_benchmark.py` separately preserves five attributed source records after line 2400—beyond the complete training/validation/test window ending before line 1960—and v10 classifies all five filled cutouts correctly. This benchmark validates recognition and conditioning only; a live AniGen GPU run is still required to validate each newly generated GLB.

For difficult real photos, `ml/evaluate_single_drawing.py` reproduces the browser face ensemble and writes per-part overlay evidence. `ml/prepare_single_drawing_mesh.py` then builds a higher-resolution signed-distance volume from the isolated pixels, retains the exact front texture and sampled palette, and writes semantic/rig metadata. `scripts/build-semantic-glb.py` consumes that evidence in Blender to export a colored, closed, skinned GLB with a plain inferred back and topology-derived appendage bones. `scripts/evaluate-anigen.mjs --inspect` validates geometric closure after welding legitimate UV/normal seams, volume, normalized skin weights, bone count, and color materials.

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

Optional exact-input 3D regression (requires the Python packages in `ml/requirements.txt` and Blender):

```bash
python ml/evaluate_single_drawing.py --image drawing.png --output /tmp/wallalive-face-evidence
python ml/prepare_single_drawing_mesh.py --image isolated-drawing.png --output /tmp/wallalive-mesh --resolution 256
blender --background --python scripts/build-semantic-glb.py -- --input-dir /tmp/wallalive-mesh --output /tmp/wallalive.glb
node scripts/evaluate-anigen.mjs --inspect /tmp/wallalive.glb
```

## Attribution and license

WallAlive is MIT licensed. V3 is trained on the official [ChildlikeSHAPES](https://arxiv.org/abs/2504.08022) release under CC-BY-4.0; dataset images are not redistributed in this repository. The retained fallback models use a 490-image slice of Meta’s public [Amateur Drawings Dataset](https://github.com/facebookresearch/AnimatedDrawings#amateur-drawings-dataset), released under MIT. TopologyNet v10 uses the official [Google Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset); the five small benchmark records retain key IDs and source attribution under CC BY 4.0. The neural reconstruction path uses [AniGen](https://github.com/VAST-AI-Research/AniGen), whose project and model card are MIT licensed; AniGen’s repository separately notes research/non-commercial restrictions on a bundled CUBVH-derived component, which must be reviewed before commercial self-hosting. The bundled judge fixture was generated by the official AniGen Space from its `brickbob.png` reference input and is included for reproducible, quota-free evaluation. See [docs/RESEARCH.md](./docs/RESEARCH.md) for the evaluated alternatives and exact capability boundary.
