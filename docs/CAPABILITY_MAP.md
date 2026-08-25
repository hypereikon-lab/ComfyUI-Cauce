# Capability map

## Implemented in CAUCE

| Operation | Deterministic contract | Result status |
| --- | --- | --- |
| exact decoded range | `[start, start + count)` | unit-validated |
| H3 two-sided guide-window plan | two guide clips, two frame indices, accepted generated range | unit-validated; visual result pending per run |
| two-sided guide-window assembly | `A + accepted generated range + B` | unit-validated |
| affine/analytic/perspective/displacement maps | inverse coordinate field | unit-validated |
| vector-field advection | Euler/RK2/RK4 map integration | unit-validated |
| depth camera reprojection | map plus disocclusion validity | unit-validated |
| map modulation/composition | one resampling-ready map | unit-validated |
| image warp | reference-media generation | unit-validated |
| H3 AV latent persistence | atomic visual+structural-audio save/load | unit-validated at file/path layer; live tensor path requires ComfyUI |

## Composed from existing nodes

| Operation | Composition |
| --- | --- |
| image-to-video and first/last-frame video | official FL2VA graph |
| reference-video motion transfer | official Ref2VA graph |
| arbitrary temporal guides | official `MiniMaxH3AddGuide` chain |
| native tail continuation | validated external continuation nodes plus official H3 graph |
| generate a range between two source clips | CAUCE guide selection + two official AddGuide nodes + CAUCE assembly |
| primitive/simulation motion conditioning | CAUCE image warp -> video reference -> official Ref2VA/AddGuide |

“Composed” means the mechanism exists and the graph can be constructed. It is
not a guarantee that a particular prompt/source pair will be visually accepted.
