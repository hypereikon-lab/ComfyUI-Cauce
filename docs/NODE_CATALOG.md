# Node catalog

## CAUCE/Assembly

### `CauceAcceptDecodedRange`

Returns an exact `[start_frame, start_frame + frame_count)` IMAGE slice and its
count. Workflow meaning such as “guide,” “accepted center,” or “edit” remains
outside the node.

## CAUCE/H3 AV Latent

| Node | Operation |
| --- | --- |
| `CauceH3InspectAVLatent` | validate and report packed AV shape, absolute frames, and token lengths |
| `CauceH3PlanAVWindow` | calculate one absolute overlap+extension layout on both H3 clocks |
| `CauceH3AllocateAVWindow` | allocate the layout's zero target using prior latent geometry |
| `CauceH3ExtractAVSpan` | extract synchronized visual/audio tokens as `CAUCE_H3_AV_SPAN` |
| `CauceH3AddAVSpanGuide` | add one compatible span to H3 positive conditioning at a frame index |
| `CauceH3AppendAVSpan` | append one globally contiguous span to a cumulative H3 latent |
| `CauceH3SplitAVLatent` | split an origin-zero cumulative state into a valid prefix and contiguous suffix span |

No node owns prompt, seed, sampler, scheduler, denoise, decode, or a
continuation preset.

## CAUCE/H3 Planning

| Node | Operation |
| --- | --- |
| `CauceH3ResolveTargetShape` | expose target ceil-to-`17k+5`, duration, visual/audio tokens, and trained-range status |
| `CauceH3PrepareGuideClip` | make official AddGuide single-image/floor clipping and resolved placement explicit |
| `CauceH3PrepareReferenceClip` | make Ref2VA target clamp, floor clipping, and 2 fps Qwen samples explicit |
| `CauceH3InspectConditioning` | report active H3 keyframes, references, ranges, and overlaps without mutation |

Planning nodes do not encode media, create conditioning, or sample. Their
accepted IMAGE output is passed to the corresponding official H3 node.

## CAUCE/Motion Maps

| Node | Operation |
| --- | --- |
| `CauceAffineMotionMap` | translation, scale, rotation, pivot, easing |
| `CauceAnalyticMotionMap` | swirl, pinch, wave, radial wave, tunnel, kaleidoscope |
| `CaucePerspectiveMotionMap` | four-corner projective pullback |
| `CauceDisplacementMotionMap` | import arbitrary RG displacement data |
| `CauceModulateMotionMap` | temporal envelope and optional spatial mask |
| `CauceVectorField` | uniform, rotation, radial, vortex, curl, wave fields |
| `CauceIntegrateAdvection` | Euler, RK2, or RK4 field integration |
| `CauceDepthCameraMotionMap` | depth-based camera reprojection and validity |
| `CauceComposeMotionMaps` | compose maps before media sampling |
| `CauceWarpImage` | sample IMAGE media through a map |

## CAUCE/Persistence

### `CauceSaveAVLatent`

Atomically saves both streams of one packed H3 latent to an indexed
`safetensors` artifact inside the ComfyUI output root.

### `CauceLoadAVLatent`

Loads an explicit or latest indexed CAUCE H3 AV latent. Paths are constrained
to the output root.
