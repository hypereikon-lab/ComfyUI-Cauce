# Node catalog

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
temporal durations before invoking the official node.

## Masks

- `CAUCE · Time Field Span`
- `CAUCE · Compile H3 AV Mask`

Fields use explicit polarity: `1 = generate`, `0 = preserve`. Video and audio
are projected independently onto H3's visual support and 40 Hz grids. Multiple
masks intersect: preservation wins.

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

The bridge node copies a left parent tail and a right parent head into the
target, protects both endpoints, and leaves only the middle denoisable. It is
the native latent path for temporal gap filling between already-generated
states.

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
