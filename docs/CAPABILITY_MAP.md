# Capability map

## Implemented primitives

| Operation | Deterministic contract | Evidence |
| --- | --- | --- |
| exact decoded range | `[start, start + count)` | unit-validated |
| H3 target resolution | explicit ceil to `17k+5`, duration, token geometry, trained-range flag | unit-validated |
| H3 guide-clip preparation | official single-image/floor rule plus resolved target range | unit-validated |
| H3 reference-clip preparation | official target clamp/floor rule plus 2 fps Qwen sample indices | unit-validated |
| H3 conditioning inspection | read-only keyframe/reference/range/overlap report | unit-validated |
| decoded-frame interpolation plan | exact `(N-1)m+1` count, target fps, and source-anchor output indices | unit-validated |
| H3 interleave-mask projection | per-token preserve/mixed/generate classification using official temporal `amax` geometry | unit-validated |
| H3 sparse-guide retime plan | source-frame anchors mapped across a legal endpoint-aligned `17k+5` target | unit-validated |
| H3 AV inspection | packed stream shapes and absolute frame/token lengths | unit-validated |
| H3 AV window layout | absolute 24 fps range mapped to visual and 40 Hz audio tokens | unit-validated |
| H3 AV window allocation | fresh zero target matching layout and prior geometry | unit-validated |
| H3 AV span extraction | synchronized visual/audio slice plus absolute range | unit-validated |
| H3 AV span guide metadata | compatible span inserted at an explicit target frame | unit-validated; schema-validated; executes in `continue.native_av` smoke |
| H3 AV append | globally contiguous span concatenation with drift rejection | unit-validated |
| H3 cumulative-state split | valid prefix latent plus reversible contiguous suffix span | unit-validated |
| H3 AV span placement/rebase | copy one synchronized native span into an exact target interval while rejecting visual-grid or 40 Hz phase mismatch | unit-validated |
| H3 continuous denoise interval | independent video/audio per-token masks with hard, linear, smoothstep, or smootherstep temporal boundaries and explicit composition | unit-validated |
| H3 spatial/spatiotemporal denoise mask | static or per-decoded-frame continuous MASK projected to visual tokens, with explicit mask algebra and preserved structural audio | unit-validated |
| H3 AV canvas expansion | exact source placement on a larger 32-pixel-aligned visual lattice, zero allocation outside, and an outpaint mask | unit-validated |
| H3 AV interval replacement | replace an exact synchronized interval in cumulative native state | unit-validated |
| H3 denoise-mask cleanup | remove consumed nested mask metadata before persistence | unit-validated |
| H3 AV persistence | atomic visual+structural-audio save/load | unit-validated; save/load executes in `continue.native_av` smoke verification |

## Named graph-level operations

The operations are classified by data responsibility, not by presumed order:

```text
conditioning grammar      generate.keyframed
                          generate.from_references
                          generate.with_guides

native AV state algebra   continue.native_av
                          complete.native_av
                          edit.masked_video
                          reframe.outpaint_video
                          refine.video
                          rollback.native_av

decoded media algebra     frames.assemble

decoded video enhancement interpolate.frames
                          restore.video
```

| Operation | Composition |
| --- | --- |
| `generate.keyframed` | official FL2VA graph |
| `generate.from_references` | official Ref2VA graph |
| `generate.with_guides` | official `MiniMaxH3AddGuide` chain |
| `continue.native_av` | keyframe or masked native overlap around official sampling; optional future guide |
| `complete.native_av` | native prefix, interior, local replacement, or two-source completion through official per-token masking |
| `edit.masked_video` | static spatial, animated spatiotemporal, or bounded local retake through current official mask semantics |
| `reframe.outpaint_video` | centered or offset native canvas expansion followed by masked H3 sampling |
| `refine.video` | full-frame or masked bounded-denoise second pass over existing native state |
| `rollback.native_av` | exact native split with reversible suffix span |
| `frames.assemble` | CAUCE exact ranges + vanilla ImageBatch |
| `interpolate.frames` | CAUCE exact clock plan + version-locked native ComfyUI RIFE/FILM decoded-frame interpolation |
| `restore.video` | official native SeedVR2 resize/preprocess/chunk/condition/sample/merge/decode/postprocess graph |

All twelve currently remain `contract-only`: no paired reusable UI/API graph is
shipped. “Composed” means the mechanism can be expressed as a graph. It does not
mean a source/prompt pair is visually accepted.

## Current evidence boundary

The `keyframe-overlap` H3 AV continuation path has executed on the laboratory
runtime with a synthetic packed latent. The run verified absolute planning, allocation, span
extraction, guide insertion, suffix extraction, append, save, and reload. It did
not establish production-resolution visual quality.

The new placement/mask/replacement/cleanup nodes, planning/inspection/split
nodes, decoded-range nodes, and temporal planning nodes are unit-validated only.
Their use inside a graph remains a composition possibility until that exact
graph is schema-validated, executed, and visually evaluated where a visual
objective is claimed.
