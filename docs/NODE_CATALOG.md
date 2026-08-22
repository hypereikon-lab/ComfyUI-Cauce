# Node catalog

## Media

- `CAUCE · Select Image Frame`

Selects one opaque frame from any `IMAGE` batch while retaining a one-frame
batch. Negative indices count from the end, so `-1` is a generated segment's
endpoint. No image analysis or semantic label is introduced.

## Timeline

- `CAUCE · Project`
- `CAUCE · Timeline Point`
- `CAUCE · Media Span`
- `CAUCE · Compile Window`
- `CAUCE · Empty Timeline`
- `CAUCE · Append Timeline Item`
- `CAUCE · Decode Domain`

`Timeline Point` contains absolute time and an arbitrary prompt. It does not
contain descriptive or semantic fields. `Media Span` is metadata for any
opaque image, video, audio, mask, or latent interval.

## Plates

- `CAUCE · Plate Canvas`
- `CAUCE · Plate Layer`
- `CAUCE · Domemaster Preview`
- `CAUCE · Attach Point Image`
- `CAUCE · Export Plate`

Plate layers support position, scale, rotation, opacity, an optional mask,
feathering, and normal/screen/multiply/add blending. Export writes a normal
Comfy PNG plus `.prompt.txt` and `.json` sidecars.

## H3

- `CAUCE · H3 FL2VA`
- `CAUCE · Add H3 Image Reference`
- `CAUCE · Add H3 Video Reference`
- `CAUCE · Add H3 Audio Reference`
- `CAUCE · H3 Ref2VA`
- `CAUCE · H3 Timed Guide`

Reference nodes construct an ordered runtime set without embedding media
tensors in JSON. Ref2VA validates per-kind counts, total count, and declared
temporal durations before invoking the official node. The audio-reference node
is retained for upstream compatibility but is not used by the current fixed-
soundtrack production workflows.

## Masks

- `CAUCE · Time Field Span`
- `CAUCE · Compile H3 AV Mask`

Fields use explicit polarity: `1 = generate`, `0 = preserve`. Video and H3's
internal audio rows are projected independently. Production workflows freeze
the internal audio rows; multiple masks intersect and preservation wins.

## Audio

- `CAUCE · Empty Audio Track`
- `CAUCE · Exact Audio Slice`
- `CAUCE · Place Audio`
- `CAUCE · Authoritative Audio`

All placements use sample indices derived from rational time. Authoritative
audio is the final master slice for muxing and is not passed through H3's VAE.

## Continuity

- `CAUCE · Prepare H3 Continuation`
- `CAUCE · Resolve Parent Latent`
- `CAUCE · Prepare H3 AV Bridge`
- `CAUCE · Accept Decoded Window`

The parent node retains the H3 causal origin and only removes the post-accept
tail. The decoded node removes context/prefix frames for delivery. `Generation
Window` exposes `nearest_run`, `floor_run`, `ceil_run`, `exact_frames`, and
`full_render`; only phase-safe endpoints can become continuation parents.

CAUCE intentionally does not expose latent concatenation: independent H3
latents have incompatible causal token phases at a naïve join.

For runtimes predating H3 clip guides, a continuation can decode its parent,
select the final accepted image, and connect it as the next FL2VA `first_frame`.
The image endpoint and the inherited latent tail are complementary constraints.

The bridge node copies a left visual parent tail and a right visual parent head
into the target, protects both endpoints, freezes H3's internal audio rows, and
leaves only the visual middle denoisable. It is the native latent path for
temporal gap filling between already-generated states.

## Seams

- `CAUCE · Build Confluence Window`
- `CAUCE · Confluence Fields`
- `CAUCE · H3 Confluence Guides`
- `CAUCE · Prepare H3 Seam Repair`
- `CAUCE · Apply Confluence Patch`

Confluence operates on two already-decoded, opaque video batches. The first
node extracts equal tail/head contexts, adds symmetric duplicate guards until
the batch is an exact H3 run, and returns both the seam plan and its matching
`CAUCE_WINDOW`. At the default 24 fps settings this is
`2 guards + 60 A + 60 B + 2 guards = 124` frames.

The default one-second request snaps to the nearest symmetric H3 token interval:
`[51,73)`, 22 frames. `H3 Confluence Guides` anchors the preserved 22 frames
immediately before and after that interval. `Confluence Fields` emits exact
sampling/acceptance masks and a continuous decoded opacity. The prepare node
injects the source latent, verifies official per-token mask support, and fails
closed on older ComfyUI cores. The apply node returns `A + B` with exactly the
original combined frame count; exterior frames are copied without change.

Workflow 60 uses the official standard sampler. H3's internal audio latent is
zero-masked and discarded; the fixed production audio never enters these nodes.

## Artifacts

- `CAUCE · Run Receipt`
- `CAUCE · Save AV Latent`
- `CAUCE · Load AV Latent`
- `CAUCE · Save Receipt`
- `CAUCE · Compare Receipts`

Receipts include window/profile values, model hashes, seed, sampler, scheduler,
steps, CFG, workflow hash, and parents.

## Runtime

- `CAUCE · Execution Profile`
- `CAUCE · Preflight`

Preflight is read-only. A failure reports missing models, unexpected sizes,
disk reserve, and current torch/CUDA/device information; it never repairs the
environment.
