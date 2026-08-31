# Operation data contracts

Semantic operation types describe graph-level data, not additional ComfyUI
socket types. A materialized graph resolves them into concrete nodes and links.

## Media reference

Project bindings may identify media directly or through another invocation:

```json
"input://frames/start.png"
```

```json
{
  "invocation": "reference-transform-0004",
  "output": "reference_clip",
  "artifact_hash": "optional lowercase sha256"
}
```

The second form makes graph composition explicit and content-addressable
without making operations aware of editorial meaning.

## `CAUCE_H3_GENERATION_BINDINGS`

The common inference configuration contains:

```text
model files
width / height / target_frames
seed
sampler / scheduler / steps / denoise
output prefix
```

An explicit `MiniMaxH3SigmaShift` node is an optional graph experiment rather
than part of `CAUCE_H3_GENERATION_BINDINGS`. When used, its video and
structural-audio shifts belong in the experiment receipt and require a direct
model-path control.

Operation-specific inputs such as prompt, endpoint frames, references, guides,
overlap, or extension remain outside this common block. Model paths and enum
values must be rebound against the live runtime before execution.

## `CAUCE_DECODED_GUIDE_SET`

An ordered list of decoded media references and exact pixel-frame indices:

```json
[
  { "media": "input://guides/a.png", "frame_index": 0 },
  { "media": "input://guides/b.mp4", "frame_index": 102 }
]
```

The materialized topology contains one official `MiniMaxH3AddGuide` node per
item. A different guide count is a different static graph topology.

Before the official node, `CauceH3PrepareGuideClip` can emit a
`CAUCE_H3_GUIDE_PLAN`. The plan records the source frame count, accepted
single-image or `17k+5` clip, discarded tail, resolved negative index, and
half-open target range. It changes no conditioning.

## H3 target and reference plans

`CAUCE_H3_TARGET_PLAN` records the requested and resolved target frame counts,
exact rational duration, target dimensions, visual/audio token counts, and
whether the result lies inside the documented approximate trained range.

`CAUCE_H3_REFERENCE_PLAN` records the source and resolved target counts, exact
accepted `17k+5` prefix, discarded tail, and the frame indices/timestamps shown
to Qwen at 2 fps. It distinguishes the current ComfyUI runtime minimum from the
model's documented 2–15 second reference-video range. The accepted IMAGE batch
remains decoded media and is passed to official Ref2VA conditioning.

## `CAUCE_FRAME_RANGE_SET`

An ordered list of media/range pairs using half-open decoded frame intervals:

```json
[
  { "media": "artifact://left", "range": [0, 240] },
  { "media": "artifact://center", "range": [22, 102] }
]
```

The graph contains one `CauceAcceptDecodedRange` per item and enough vanilla
`ImageBatch` nodes to concatenate them.

## Native AV state

`LATENT` outputs named `native_av_latent` or `continued_native_av_latent` refer
to H3's packed visual and structural-audio state. Cross-run references must
retain the exact CAUCE save artifact and its hash. Decoded video is a different
data product and cannot substitute for native state in `continue.native_av`.

`CauceH3SplitAVLatent` operates only on an origin-zero cumulative state. It
returns a complete prefix `LATENT` and a globally contiguous
`CAUCE_H3_AV_SPAN`; immediately appending that span reconstructs the original
state exactly. This supports rollback and branching without resetting a
nonzero 40 Hz phase.

`CauceH3PlaceAVSpan` accepts an explicit target origin and frame index. It may
rebase the span descriptor only when its visual-token length and absolute
40 Hz structural-audio phase remain exact. Incompatible placement is rejected;
there is no fractional-token interpolation.

## Native AV denoise mask

Current official H3 sampling accepts `noise_mask` with the same nested video/
structural-audio shape as the packed latent. CAUCE stores it as:

```text
latent["noise_mask"] = nested(video_mask, audio_mask)
```

Both values are continuous in `[0, 1]`: `1` requests generation and `0`
preserves the supplied token. Video and audio profiles are evaluated on their
own token centers from one pixel-frame interval. Optional linear, smoothstep,
or smootherstep fades may extend before/after the fully generated interval.
Composition is explicit: replace, maximum, minimum, or multiply.

The mask is transient inference metadata. Clear it before persisting a completed
native state unless a later run deliberately reuses that exact mask.

## Soundtrack boundary

The fixed production soundtrack is not part of these operation inputs. H3's
internal structural-audio tensor remains present wherever required by the model
and native continuation contract.
