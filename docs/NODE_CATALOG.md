# Node catalog

## CAUCE/Assembly

### `CauceAcceptDecodedRange`

Returns an exact `[start_frame, start_frame + frame_count)` IMAGE slice and its
count. Workflow meaning such as “guide,” “accepted center,” or “edit” remains
outside the node.

### `CauceRestoreDecodedAnchors`

Restores each decoded source frame at positions `0, factor, 2*factor, ...` in a
densified generated batch and crops legal H3 tail padding. Only the intervening
decoded frames remain generative. This is an exact delivery operation after
H3/VAE decode; it does not claim that alternating decoded frames are separately
maskable inside H3's temporally compressed latent lattice.

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
| `CauceH3PlaceAVSpan` | copy one synchronized span into an exact target interval, optionally rebasing only when both clocks align |
| `CauceH3SetAVDenoiseInterval` | attach independent continuous video/audio token masks for one frame interval; `1` generates and `0` preserves |
| `CauceH3ApplyVideoDenoiseMask` | project one static or per-frame continuous `MASK` onto the H3 visual-token lattice and compose it with existing mask state |
| `CauceH3ExpandAVCanvas` | copy native video state onto a larger 32-pixel-aligned canvas, preserve the interior, and mask newly allocated regions |
| `CauceH3ReplaceAVSpan` | replace one globally aligned synchronized interval and discard spent mask metadata |
| `CauceH3ClearAVDenoiseMask` | remove a consumed nested AV denoise mask without changing latent samples |
| `CauceH3DilateVisualTokens` | map source visual tokens monotonically onto a longer legal H3 lattice, allocate missing tokens, and attach a continuous temporal-inpaint mask |
| `CauceH3ResizeAVLatent` | resize only native H3 visual state and attach separate video/audio denoise strengths for a same-model high-resolution pass |
| `CauceH3ReplaceVisualStream` | graft a duration-compatible H3-VAE visual latent onto an existing packed AV carrier for a pixel/VAE second pass |
| `CauceH3ExtractVisualStream` | expose a cloned visual-only latent plus the untouched AV carrier so external visual tools require an explicit graft back into H3 state |

No node owns prompt, seed, sampler, scheduler, decode, or a continuation/
completion preset. The mask node owns only deterministic per-token denoise
metadata; the official sampler owns how that metadata affects inference. Mask
projection uses decoded-frame `amax` inside each H3 visual token and preserves
continuous spatial values until the current core's own patch-grid processing.

## CAUCE/H3 Model

### `CauceH3DomemasterCoordinates`

Clones an H3 `MODEL` and installs a reversible inference-time coordinate warp
on `PackedLayout.position_ids`. Target-video rows always receive the warp;
FL2VA keyframe rows may receive the same warp. Text, audio, Ref2VA reference
rows, pixels, latent values, weights, prompts, and schedules remain unchanged.

The only current profile maps samples inside a square equidistant 180-degree
domemaster support to the x/y components of their front-hemisphere camera ray.
`strength=0` reproduces the stock H3 grid, and `strength=1` applies the complete
coordinate transform. This node is an experimental inference ablation, not a
trained lens adapter and not evidence that H3 preserves calibrated projection.
See [H3 domemaster coordinates](H3_DOMEMASTER_COORDINATES.md).

## CAUCE/H3 Planning

| Node | Operation |
| --- | --- |
| `CauceH3ResolveTargetShape` | expose target ceil-to-`17k+5`, duration, visual/audio tokens, and trained-range status |
| `CauceH3PrepareGuideClip` | make official AddGuide single-image/floor clipping and resolved placement explicit |
| `CauceH3PrepareReferenceClip` | make Ref2VA target clamp, floor clipping, documented 2–15 s status, and 2 fps Qwen samples explicit |
| `CauceH3InspectConditioning` | report active H3 keyframes, references, ranges, and overlaps without mutation |
| `CauceH3PlanControlClip` | report the official Fun Control truncate/repeat-last temporal policy and bilinear center-fit geometry without changing frames |
| `CauceH3InspectPackedSequence` | count exact H3 packed rows across text, guides, references, target audio, and target video; keep the memory estimate explicitly calibrated and heuristic |

Planning nodes do not encode media, create conditioning, or sample. Their
accepted IMAGE output is passed to the corresponding official H3 node.

## CAUCE/Temporal Planning

| Node | Operation |
| --- | --- |
| `CaucePlanH3GuideRetime` | map sparse source frames to endpoint-aligned official arbitrary-frame guides on a legal H3 target |

This planning node does not sample H3. Native temporal densification lives in
the packed AV-state nodes above; official H3 remains the visible graph stage
that performs inference. See
[Temporal and spatial video enhancement](TEMPORAL_SPATIAL_ENHANCEMENT.md).

## CAUCE/Persistence

### `CauceSaveAVLatent`

Atomically saves both streams of one packed H3 latent to an indexed
`safetensors` artifact inside the ComfyUI output root.

### `CauceLoadAVLatent`

Loads an explicit or latest indexed CAUCE H3 AV latent. Paths are constrained
to the output root.
