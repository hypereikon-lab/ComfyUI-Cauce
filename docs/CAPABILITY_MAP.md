# Capability map

## Implemented primitives

| Operation | Deterministic contract | Evidence |
| --- | --- | --- |
| exact decoded range | `[start, start + count)` | unit-validated |
| H3 AV inspection | packed stream shapes and absolute frame/token lengths | unit-validated |
| H3 AV window layout | absolute 24 fps range mapped to visual and 40 Hz audio tokens | unit-validated |
| H3 AV window allocation | fresh zero target matching layout and prior geometry | unit-validated |
| H3 AV span extraction | synchronized visual/audio slice plus absolute range | unit-validated |
| H3 AV span guide metadata | compatible span inserted at an explicit target frame | unit-validated; live schema/execution pending |
| H3 AV append | globally contiguous span concatenation with drift rejection | unit-validated |
| affine/analytic/perspective/displacement maps | inverse coordinate field | unit-validated |
| vector-field advection | Euler/RK2/RK4 map integration | unit-validated |
| depth camera reprojection | map plus disocclusion validity | unit-validated |
| map modulation/composition | one resampling-ready map | unit-validated |
| image warp | decoded reference-media generation | unit-validated |
| H3 AV persistence | atomic visual+structural-audio save/load | unit-validated at path layer; live tensor path pending |

## Workflow-level compositions

| Operation | Composition |
| --- | --- |
| first/last-frame generation | official FL2VA graph |
| reference-conditioned generation | official Ref2VA graph |
| decoded temporal guides | official `MiniMaxH3AddGuide` chain |
| native latent-tail continuation | CAUCE AV plan/allocate/extract/guide/append around official sampling |
| two-sided decoded guide window | exact decoded ranges + two official AddGuide nodes + vanilla ImageBatch |
| procedural motion reference | CAUCE image warp -> official Ref2VA/AddGuide |

“Composed” means the mechanism can be expressed as a graph. It does not mean a
source/prompt pair is visually accepted.
