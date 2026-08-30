# Temporal and spatial video enhancement

This document separates three operations that are often all called “upscale”
but have different clocks, failure modes, and owners:

1. decoded-frame interpolation increases sample rate without asking H3 to
   reinterpret the shot;
2. H3 guide retiming regenerates motion at H3's fixed 24 fps and is creative,
   not pixel-exact;
3. SeedVR2 restoration increases spatial definition while preserving frame
   count and frame rate.

CAUCE plans and audits these operations. RIFE/FILM, official H3, and official
SeedVR2 own their respective model inference.

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
  -> RIFE VFI, multiplier 2
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
The original RIFE/FILM node implementations copy source samples into the output
buffer and synthesize only intermediate slots.

### RIFE default

Use `RIFE VFI` at 2x first. The locked public integration is
`ThunderFun/ComfyUI-RIFE-FILM-Only` commit
`3469aad35d6774bb7943bf30d5ae7b5acd9039bc`, MIT licensed.

Initial characterization matrix:

```text
multiplier       2
precision        bf16 on RTX 5090, with fp32 control
scale_factor     1.0
fast_mode        true
ensemble         false first; compare true where supported
```

Judge occlusion reveals, thin branches, texture shimmer, fast zooms, and any
source-frame alteration. Prefer staged 2x tests before 4x.

### FILM comparison

Use `FILM VFI` at 2x only when RIFE fails on large displacement. FILM is a
midpoint model; larger multipliers recursively subdivide intervals and can
accumulate artifacts. It is a comparison path, not an unconditional second
pass after RIFE.

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
model   seedvr2_ema_3b_fp16_nvfp4.safetensors   about 2.0 GB
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
