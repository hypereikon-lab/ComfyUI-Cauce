# Temporal and spatial video enhancement

This document separates three operations that are often all called “upscale”
but have different clocks, failure modes, and owners:

1. decoded-frame interpolation increases sample rate without asking H3 to
   reinterpret the shot;
2. H3 guide retiming regenerates motion at H3's fixed 24 fps and is creative,
   not pixel-exact;
3. SeedVR2 restoration increases spatial definition while preserving frame
   count and frame rate.

CAUCE plans and audits these operations. ComfyUI's native RIFE/FILM path,
official H3, and official SeedVR2 own their respective model inference.

## Why alternating empty H3 frames is not frame interpolation

The intuitive construction is:

```text
frame 0, empty, frame 1, empty, frame 2, ...
mask  0,     1,       0,     1,       0, ...
```

That would work only if the model had one independently maskable latent token
per decoded frame. H3 does not.

The current official H3 implementation has two temporal reductions:

- the causal video VAE downsamples time by a factor of four;
- the DiT uses the repeating decoded-frame coverage pattern
  `(1, 4, 4, 4, 4)` for each five-token / seventeen-frame cycle.

When an animated decoded-frame mask is projected to that token lattice, the
official path takes the temporal maximum within each token span. A four-frame
token containing any empty/generated frame therefore receives mask strength
`1` for the whole token. Known and missing frames inside that span are sampled
together; the known frames are not guaranteed to survive exactly.

For a 124-frame H3 clip and a requested 2x interleave:

```text
exact VFI target       (124 - 1) * 2 + 1 = 247 frames
next legal H3 target   260 frames
H3 video tokens        77
```

`CauceInspectH3InterleaveProjection` reports every token as
`preserve-only`, `mixed-known-and-missing`, or `generate-only`. Mixed tokens
are the proof that the proposed mask is creative regeneration, not exact
frame insertion.

## Pipeline A: conservative temporal upscale

Use decoded-frame interpolation after the H3 shot is visually accepted.

```text
accepted 24 fps VIDEO
  -> GetVideoComponents
  -> CAUCE Plan Frame Interpolation
  -> native Frame Interpolation Model Loader
  -> native Frame Interpolate, multiplier 2
  -> CreateVideo at 48 fps
```

For `N` source samples and integer multiplier `m`:

```text
target samples = (N - 1) * m + 1
target fps     = source fps * m
source i       = output index i * m
```

The first-to-last sample span is invariant:

```text
(target samples - 1) / target fps
  = ((N - 1) * m) / (source fps * m)
  = (N - 1) / source fps
```

The exact output is one sample shorter than the informal `N * m` shorthand.
ComfyUI 0.34.0's native `FrameInterpolate` copies source samples into the
output buffer and synthesizes only intermediate slots.

### RIFE default

Use native `FrameInterpolate` at 2x first with
`rife_v4.26.safetensors`. The graph is locked to ComfyUI 0.34.0 commit
`12d5279438bfefc058a269eae805ceab6047777f`; no interpolation custom-node
repository is required. The model is the official Comfy-Org repack of RIFE.

Initial characterization matrix:

```text
multiplier       2
model            rife_v4.26.safetensors, 22.7 MB
source anchoring every second output frame
```

Judge occlusion reveals, thin branches, texture shimmer, fast zooms, and any
source-frame alteration. Prefer staged 2x tests before 4x.

### FILM comparison

Use the same two native nodes with `film_net_fp16.safetensors` at 2x only when
RIFE fails on large displacement. FILM is a comparison path, not an
unconditional second pass after RIFE. Its official Comfy-Org model payload is
68.9 MB.

## Pipeline B: creative H3 retiming

This path does not increase FPS. It asks H3 to compose a longer or shorter
24 fps shot from sparse still guides:

```text
decoded source frames
  -> select endpoint plus periodic source frames
  -> CAUCE Plan H3 Guide Retime
  -> official MiniMaxH3AddGuide chain
  -> official H3 sampler at legal 17k+5 target
```

`CaucePlanH3GuideRetime` maps the selected source frames across the resolved H3
timeline, always pinning the two endpoints. Because target length snaps upward
to `17k+5`, the resolved duration may be longer than requested. This operation
can improve or alter perceived motion, but it can also change appearance,
geometry, and timing between guides. It is a generative shot revision.

Use it when new motion is desired. Do not label it FPS conversion or use it as
a hidden replacement for deterministic interpolation.

## Pipeline C: conservative spatial-temporal restoration

Use ComfyUI's native SeedVR2 video topology:

```text
VIDEO -> components -> spatial resize -> SeedVR2 preprocess
  -> tiled VAE encode -> temporal chunk
  -> SeedVR2 conditioning -> one-step sampler
  -> temporal merge -> tiled decode -> SeedVR2 postprocess
```

This operation changes spatial resolution, not frame count or fps. SeedVR2's
video model restores detail across a temporal window, so it is preferable to
independent per-frame image upscaling when temporal texture stability matters.

For the laboratory RTX 5090 with 32 GB VRAM and limited disk, the initial
target is:

```text
model   seedvr2_3b_nvfp4.safetensors            1.997 GB
VAE     ema_vae_fp16.safetensors                about 0.5 GB
steps   1
chunk   auto first
color   lab first; compare wavelet only if necessary
```

The exact installed filenames and `object_info` inputs are live gates. Model
size alone does not prove activation memory will fit; chunk size and temporal
overlap must be characterized on the real 32 GB runtime.

## Recommended production composition

The compute-aware default is:

```text
accepted H3 shot, 24 fps
  -> SeedVR2 restore at 24 fps
  -> RIFE 2x at restored resolution
  -> 48 fps master
```

SeedVR2 is the heavier operation. Running it before interpolation means it
processes the original number of frames rather than almost twice as many. RIFE
then estimates motion using the restored spatial signal.

The empirical gate must compare four outputs from the same source range:

1. source control;
2. RIFE-only 24 to 48 fps;
3. SeedVR2-only spatial restoration at 24 fps;
4. SeedVR2 followed by RIFE.

Run the reverse order, RIFE then SeedVR2, only as a bounded characterization
test. It roughly doubles SeedVR2's temporal workload and may cause SeedVR2 to
reinforce interpolation artifacts.

## Deliberate exclusions

- AMT is excluded from the production dependency set because its official
  repository is CC BY-NC 4.0.
- Current learned H3 latent upscaler repositories were not promoted because
  the audited code/weight releases do not publish a usable license. Their
  temporal dimension also remains unchanged; they are spatial upscalers.
- Deterministic latent interpolation or affine manipulation does not create
  learned temporal detail and is not presented as temporal super-resolution.
- No audio model, audio encoding, training, LoRA, streaming, or second UI is
  part of these pipelines.

## Evidence boundary

The planning mathematics and topology dossiers are unit/schema validated.
They are not yet paired executable UI/API workflows and have not yet received
a visual verdict on the lab GPU. Promotion requires live `object_info`, paired
workflow export, exact count/fps assertions, successful saved media, and human
visual assessment.
