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
| `CauceH3PlaceAVSpan` | copy one synchronized span into an exact target interval, optionally rebasing only when both clocks align |
| `CauceH3SetAVDenoiseInterval` | attach independent continuous video/audio token masks for one frame interval; `1` generates and `0` preserves |
| `CauceH3ReplaceAVSpan` | replace one globally aligned synchronized interval and discard spent mask metadata |
| `CauceH3ClearAVDenoiseMask` | remove a consumed nested AV denoise mask without changing latent samples |

No node owns prompt, seed, sampler, scheduler, decode, or a continuation/
completion preset. The mask node owns only deterministic per-token denoise
metadata; the official sampler owns how that metadata affects inference.

## CAUCE/H3 Planning

| Node | Operation |
| --- | --- |
| `CauceH3ResolveTargetShape` | expose target ceil-to-`17k+5`, duration, visual/audio tokens, and trained-range status |
| `CauceH3PrepareGuideClip` | make official AddGuide single-image/floor clipping and resolved placement explicit |
| `CauceH3PrepareReferenceClip` | make Ref2VA target clamp, floor clipping, documented 2–15 s status, and 2 fps Qwen samples explicit |
| `CauceH3InspectConditioning` | report active H3 keyframes, references, ranges, and overlaps without mutation |

Planning nodes do not encode media, create conditioning, or sample. Their
accepted IMAGE output is passed to the corresponding official H3 node.

## CAUCE/Persistence

### `CauceSaveAVLatent`

Atomically saves both streams of one packed H3 latent to an indexed
`safetensors` artifact inside the ComfyUI output root.

### `CauceLoadAVLatent`

Loads an explicit or latest indexed CAUCE H3 AV latent. Paths are constrained
to the output root.
