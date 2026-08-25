# Capability map

## Implemented primitives

| Operation | Deterministic contract | Evidence |
| --- | --- | --- |
| exact decoded range | `[start, start + count)` | unit-validated |
| H3 AV inspection | packed stream shapes and absolute frame/token lengths | unit-validated |
| H3 AV window layout | absolute 24 fps range mapped to visual and 40 Hz audio tokens | unit-validated |
| H3 AV window allocation | fresh zero target matching layout and prior geometry | unit-validated |
| H3 AV span extraction | synchronized visual/audio slice plus absolute range | unit-validated |
| H3 AV span guide metadata | compatible span inserted at an explicit target frame | unit-validated; schema-validated; executes in `continue.native_av` smoke |
| H3 AV append | globally contiguous span concatenation with drift rejection | unit-validated |
| affine/analytic/perspective/displacement maps | inverse coordinate field | unit-validated |
| vector-field advection | Euler/RK2/RK4 map integration | unit-validated |
| depth camera reprojection | map plus disocclusion validity | unit-validated |
| map modulation/composition | one resampling-ready map | unit-validated |
| image warp | decoded reference-media generation | unit-validated |
| H3 AV persistence | atomic visual+structural-audio save/load | unit-validated; save/load executes in `continue.native_av` smoke verification |

## Named graph-level operations

| Operation | Composition |
| --- | --- |
| `generate.keyframed` | official FL2VA graph |
| `generate.from_references` | official Ref2VA graph |
| `generate.with_guides` | official `MiniMaxH3AddGuide` chain |
| `continue.native_av` | CAUCE AV plan/allocate/extract/guide/append around official sampling |
| `connect.two_sided_guides` | exact decoded ranges + two official AddGuide nodes + vanilla ImageBatch |
| `reference.transform` | CAUCE decoded-media coordinate maps and image warp |
| `frames.assemble` | CAUCE exact ranges + vanilla ImageBatch |

All seven currently remain `contract-only`: no paired reusable UI/API graph is
shipped. “Composed” means the mechanism can be expressed as a graph. It does not
mean a source/prompt pair is visually accepted.

## Current evidence boundary

The complete six-node H3 AV path has executed on the laboratory runtime with a
synthetic packed latent. The run verified absolute planning, allocation, span
extraction, guide insertion, suffix extraction, append, save, and reload. It did
not establish production-resolution visual quality.

The decoded-range and motion-map families are unit-validated. Their use inside
a graph remains a composition possibility until that exact graph is
schema-validated, executed, and visually evaluated where a visual objective is
claimed.
