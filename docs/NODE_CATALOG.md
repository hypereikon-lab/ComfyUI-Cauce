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

## Temporal inpainting

- `CAUCE · Build Temporal Inpaint Window`
- `CAUCE · Temporal Inpaint Fields`
- `CAUCE · H3 Temporal Guide Clips`
- `CAUCE · Prepare H3 Temporal Inpaint`
- `CAUCE · Splice Temporal Inpaint Patch`
- `CAUCE · Build Native Latent Seam`
- `CAUCE · Prepare Native Latent Inpaint`
- `CAUCE · Assemble Native Two-Clip Loop`

Temporal inpainting operates on two already-decoded, opaque video batches. The first
node extracts equal tail/head contexts, adds symmetric duplicate guards until
the batch is an exact H3 run, and returns both the seam plan and its matching
`CAUCE_WINDOW`. At the default 24 fps settings this is
`2 guards + 60 A + 60 B + 2 guards = 124` frames.

The production example requests three seconds and resolves to the symmetric H3
token interval `[26,98)`, 72 frames. `H3 Temporal Guide Clips` anchors the
preserved ranges `[4,26)` and `[98,120)`, 22 frames each. `Temporal Inpaint
Fields` emits exact
sampling/acceptance masks and a continuous decoded opacity. The prepare node
injects the source latent, verifies official per-token mask support, and fails
closed on older ComfyUI cores. The apply node returns `A + B` with exactly the
original combined frame count; exterior frames are copied without change.

Workflow 50 uses the official standard sampler. H3's internal audio latent is
zero-masked and discarded; the fixed production audio never enters these nodes.

The three native-latent nodes implement workflow 60. They preserve phase-matched
tail/head rows from the two final H3 AV latents, expose only the central rows to
the official masked sampler, and apply both decoded proposals to a closed
two-clip loop without changing either source duration. The production preset
uses 22 protected source frames and an explicit 22-frame guide clip on each
side. H3 samples the 80-frame center, while the decoded splice accepts only its
inner 72 frames (three seconds), leaving four frames of temporal overscan at
each boundary.

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

## Maintenance

- `CAUCE · Storage Inventory`
- `CAUCE · Storage Cleanup`

Inventory recursively scans one explicit Comfy root: `input/` or `output/`.
It produces a hashed plan containing only relative paths, sizes, and modification
times. It supports a relative subfolder, include/exclude globs, minimum age, and
marker preservation. Symlinks, absolute paths, and parent traversal are rejected.

Cleanup is inert while `armed = false` and stages that exact plan under Comfy's
user directory. When armed, it requires the staged plan plus its exact
confirmation code and rechecks every planned file. New, missing,
modified, symlinked, or otherwise mismatched files are never deleted. Successful
operations write a receipt under Comfy's user directory, outside both cleaned
roots. No maintenance node can address models, workflows, custom nodes, or any
arbitrary filesystem path.
