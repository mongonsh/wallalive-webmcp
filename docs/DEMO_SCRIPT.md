# WallAlive demo script (2:25 target)

## 0:00–0:18 — Open with undeniable 3D

Start on a child’s paper drawing, cut to WallAlive, and press **Play Judge Demo**.

**Voiceover:** “Children already imagine their drawings are alive. WallAlive makes that leap visible—and lets a browser agent become the director.”

Hold on BrickBob loading in the room. Drag to a side and back view, then press **Spin**.

**Voiceover:** “This is not a cut-out or inflated picture. AniGen generated a colored 159,930-vertex SkinnedMesh with 20 real bones and unseen back geometry.”

## 0:18–0:42 — Show the drawing path and privacy boundary

Press **No camera? Try a demo doodle**, then **Generate real 3D**. Hold on the approval card.

**Voiceover:** “A child controls the camera and capture. WallAlive isolates one drawing locally. Real 3D requires a GPU, so a second human approval sends only this transparent character—never the live camera or room frame.”

Open **Privacy** and point to the three human-only gestures. If recording a successful live generation, continue; otherwise return to **Play Judge Demo** so quota cannot interrupt the video.

## 0:42–1:05 — Explain WebMCP

Open **Tools**.

**Voiceover:** “Eight strict WebMCP tools turn the browser agent into a creative playmate. The agent can inspect exact model and rig state, but there is no camera, capture, or upload tool.”

Prompt:

> Inspect the approved drawing. Make the character brave on the outside but shy on the inside.

Show `inspect_wall_scene` and `reconstruct_rigged_3d_character` in activity history. Inspection should expose AniGen, `glTF SkinnedMesh`, `neuralModelUsed: true`, and the generated bone/vertex counts.

## 1:05–1:30 — Share the same world

Prompt:

> Put BrickBob on the right side of the wall, then make them hide.

Show the character move and hide. Tap a different point manually, then press **Hop**.

**Voiceover:** “Human gestures and agent tools control the same live rig. Every action has an author and a visible result.”

On compatible Android hardware, press **Enter real AR**, wait for the hit-test ring, and tap a surface. Otherwise identify camera overlay as the universal fallback.

## 1:30–2:00 — The story moment

Prompt:

> Tell a story called “BrickBob finds their courage”: hide at the edge, take one brave hop, wave hello, then spin.

Show captions and movements without cutting away. Hold on the wave so viewers can see the arm bone deform the mesh.

**Voiceover:** “One tool call becomes a cancellable sequence of visible actions—not hidden generated output.”

## 2:00–2:15 — Prove the guardrail

Ask:

> Open the camera, capture another drawing, and upload it.

Show that no such tool exists.

**Voiceover:** “WebMCP makes authority concrete. The agent directs the character; only the child chooses what a sensor sees or what image may leave the tab.”

## 2:15–2:25 — Close

Show the full UI with the rig waving, AniGen Rig DNA, and visible history.

**Voiceover:** “WallAlive. Draw it. Wake it. Play.”

## Recording notes

- Keep the final public YouTube video under three minutes.
- Use **Play Judge Demo** as the primary take; it is deterministic and quota-free.
- Record one successful camera-to-AniGen generation separately if public GPU capacity allows, then edit it into the privacy section.
- Record at 1440×900 or a modern phone resolution and make tool names readable.
- Show one full 360° turn and one visible bone-driven wave.
- Hide bookmarks, personal tabs, notifications, and room details.
- Put the public live URL, repository, AniGen attribution, and GPU-capacity note in the video description.
