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
video and structural-audio sigma shifts
output prefix
```

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

## Soundtrack boundary

The fixed production soundtrack is not part of these operation inputs. H3's
internal structural-audio tensor remains present wherever required by the model
and native continuation contract.
