# H3-native temporal densification and spatial regeneration

This document defines enhancement paths that use the same MiniMax H3 model,
its native packed audiovisual state, its VAE, and ordinary ComfyUI sampling.
No auxiliary interpolation, restoration, or learned upscaler model belongs to
these operations.

The two graph-level functions are:

```text
densify.temporal    more generated motion samples, same editorial duration
regenerate.spatial  more spatial samples/detail, same frame count and fps
```

Both remain visually unassessed until their exact graphs execute on the target
runtime. The deterministic state transforms are unit-validated; that is not a
claim of useful image quality.

## Native H3 geometry

H3 generation runs at 24 fps. Legal decoded lengths satisfy:

```text
frames = 17k + 5
```

The visual VAE does not create one independent latent per decoded frame. Its
repeating decoded-frame coverage is:

```text
1, 4, 4, 4, 4 frames per visual token
```

The packed state contains:

```text
visual: [B, 24, Tv, H/16, W/16]
audio:  [B, 32, 2, Ta]
```

Current ComfyUI H3 core accepts independent continuous noise masks for both
streams and reinjects preserved state during sampling. CAUCE therefore works
on native visual tokens. It does not create an alternating decoded-frame mask,
because known and missing decoded frames would often collapse into the same
four-frame token.

## Temporal densification

### Exact delivery clock

For `N` source frames and integer factor `m`:

```text
delivery_frames = (N - 1)m + 1
delivery_fps    = 24m
```

The first-to-last sample duration remains:

```text
(N - 1) / 24 seconds
```

H3 itself still sees a slower target at 24 fps. CAUCE snaps
`delivery_frames` upward to the next legal `17k+5` count, samples that target,
decodes it, crops only the tail padding, and labels the accepted frame batch
with `delivery_fps`.

Example for 124 source frames at 2x:

```text
source decoded frames       124
source visual tokens         37
delivery frames             247
H3 target frames            260
H3 target visual tokens      77
tail crop                     13
delivery fps                  48
```

### Token placement

`CauceH3DilateVisualTokens` maps source-token centres monotonically to target
token centres in stretched model time. Every source token is copied once. All
other target tokens start empty and are marked for H3 generation.

For source token span `Si` and target token span `Tj`:

```text
source centre ci = (start(Si) + end(Si) - 1) / 2
desired centre   = m * ci
anchor j         = nearest available target centre, preserving order
```

The visual denoise profile is defined by distance to the nearest anchor:

```text
d(j) = min |j - anchor|
u(j) = clamp(d(j) / feather_tokens, 0, 1)
mask(j) = anchor_denoise + curve(u(j)) * (gap_denoise - anchor_denoise)
```

Supported curves are linear, smoothstep, and smootherstep. The conservative
first experiment uses:

```text
factor          2
anchor_denoise  0
gap_denoise     1
feather_tokens  1
audio_denoise   1
```

H3 receives preserved visual tokens on both temporal sides of each gap. This
is bidirectional temporal inpainting over native state. Preservation is exact
at the packed visual-token input, not guaranteed pixel-exact after the model
and VAE decode.

### Windowing

The released model documents a trained range of 124–362 frames. Target
windows should remain inside it. Practical maximum source-window sizes are:

```text
2x: 175 source frames -> 350 desired samples before legal snapping
3x: 118 source frames -> 352 desired samples before legal snapping
4x:  89 source frames -> 353 desired samples before legal snapping
```

Use smaller source windows with native overlap, retain the sampled native AV
state, accept only the non-overlap centre/suffix, and assemble exact decoded
ranges. Factor 2 is the production candidate. Factors 3 and 4 are bounded
experiments until drift and runtime are characterized.

### Prompt and structural audio

The graph rebuilds ordinary official H3 conditioning for the larger target.
Use a neutral continuity prompt first. Prompt ablation is mandatory:

```text
empty/minimal continuity prompt
source generation prompt
explicit slow-motion or continuous-motion prompt
```

The production soundtrack is fixed and is not model conditioning. H3's
structural-audio latent may be generated only because the released model packs
the modalities jointly; discard generated audio at delivery and remux the
fixed soundtrack later.

## Spatial regeneration

MiniMax documents H3-Regenerate-2K as a same-model second pass that regenerates
its lower-resolution result in context rather than using a dedicated super-
resolution model. The implementation has not been released, so the following
variants are explicit, inspectable approximations built from current public H3
and ComfyUI behavior.

### Variant A: native latent-hires second pass

```text
source packed AV state
  -> resize only visual latent H/W
  -> keep time and structural audio unchanged
  -> video denoise mask = bounded value
  -> audio denoise mask = 0
  -> rebuild H3 conditioning at target W/H
  -> ordinary H3 partial-denoise sample
  -> decode
```

`CauceH3ResizeAVLatent` supports bicubic, bilinear, nearest-exact, and area
resize through PyTorch. It never resizes the time axis. This is the fastest
candidate but interpolated latent state may not match the H3 VAE manifold as
well as Variant B.

### Variant B: pixel upscale, H3-VAE encode, second pass

```text
source packed AV state
  -> H3 VAE decode
  -> deterministic pixel resize
  -> H3 VAE encode
  -> replace only visual stream on source AV carrier
  -> video denoise mask = bounded value
  -> audio denoise mask = 0
  -> rebuild H3 conditioning at target W/H
  -> ordinary H3 partial-denoise sample
  -> decode
```

`CauceH3ReplaceVisualStream` requires `[B,24,T,H,W]`, exact source duration,
H3 patch-grid alignment, and compatible batch size. This round trip introduces
VAE loss but begins the second pass on a state produced by the actual H3 VAE.

### Variant C: overlapping tiled pixel/VAE regeneration

This variant bounds VRAM at larger output sizes:

```text
decode -> deterministic global upscale
  -> overlapping spatial tiles, each retaining the complete time window
  -> H3-VAE encode each tile
  -> same H3 partial-denoise each tile
  -> decode tiles
  -> smooth overlap-weighted fusion
  -> encode the fused result for retained native state
```

Every tile must share the same source global prior, prompt, denoise schedule,
and deterministic seed rule. Simple independent tiles are likely to disagree
on structures crossing tile boundaries. The useful research direction is a
low-resolution global prior plus overlapping local regeneration and weighted
fusion, as in tiled diffusion research; this topology remains experimental
until a live implementation proves lower seam energy than the full-frame
baseline.

### Denoise characterization

Do not assume one universal value. Run a fixed-seed ladder for each variant:

```text
0.15  preservation-heavy
0.25
0.35  initial balanced candidate
0.50
0.65  regeneration-heavy
```

Evaluate spatial detail, object/texture identity, temporal flicker, geometry
drift, and prompt drift separately. A sharper still frame is not sufficient if
motion coherence is worse.

## Ordering and repeated passes

The initial factorial test should compare:

```text
source
temporal only
spatial only: latent variant
spatial only: pixel/VAE variant
temporal -> spatial
spatial -> temporal
```

`temporal -> spatial` is the likely compute-efficient default because the
spatial pass sees the final denser motion, but it also processes more frames.
`spatial -> temporal` gives temporal inpainting stronger spatial anchors but
may amplify spatial-pass drift. Evidence, not naming, decides.

Repeated H3 passes are allowed as graph composition. Each pass must retain a
separate native state and receipt. Avoid an unbounded enhancement loop:

```text
pass 0 source
pass 1 bounded regenerate
pass 2 lower-denoise polish only if pass 1 has a measured defect
```

## LoRA and fine-tuning boundary

Training does not belong inside CAUCE. CAUCE exposes the inference-time native
state and masks. Dataset manifests, caching, training recipes, checkpoints,
and evaluation belong in the production repository.

Current DiffSynth-Studio documentation supports H3 LoRA training with staged
cache/train execution and separate FL2VA/Ref2VA tasks. Candidate experiments:

```text
spatial regeneration LoRA
  input: degraded or lower-resolution video context
  target: higher-quality native video
  objective: recover detail while preserving time and geometry

temporal completion LoRA
  input: token-dilated native video state with randomly missing intervals
  target: original dense native state/video
  objective: improve motion completion under CAUCE's actual mask distribution
```

A full 33B fine-tune is outside a single 32 GB RTX 5090 target. NF4/LoRA may be
possible only after disk, RAM, optimizer state, cache size, and actual runtime
are measured. A recipe is not evidence that training fits.

## Experimental matrix and acceptance

Record for every run:

```text
source artifact/hash
source native AV artifact/hash
H3 checkpoint and LoRA hashes
operation variant
factor or target geometry
mask parameters
prompt
seed/sampler/scheduler/sigmas/shifts
runtime, peak VRAM, peak RAM, disk delta
decoded output and retained native state
```

Temporal acceptance:

```text
exact delivery frame count and fps
no duplicate-frame cadence posing as new motion
lower jerk/acceleration discontinuity than source retime baseline
bounded drift at source-token anchor neighbourhoods
```

Spatial acceptance:

```text
exact target geometry and unchanged temporal count
measurable detail gain
no unacceptable flicker or edge shimmer
bounded semantic/geometry drift
no visible tile seams for tiled variant
```

## Sources and present evidence

- [MiniMax H3 official repository](https://github.com/MiniMax-AI/MiniMax-H3)
  documents native lengths and the H3-Regenerate-2K same-model direction.
- [ComfyUI H3 implementation](https://github.com/Comfy-Org/ComfyUI) owns
  official conditioning, packed H3 state, mask semantics, sampling, and VAE.
- [ComfyUI-Continuity](https://github.com/roadmaus/ComfyUI-Continuity) provides
  community evidence for spatial-only H3 latent resize and second-pass sampling.
- [H3 latent upscaler](https://github.com/rockerBOO/h3-latent-upscaler) explores
  resized H3 latent and conditioning geometry.
- [DiffSynth-Studio H3 training documentation](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/MiniMax-H3.md)
  documents present LoRA training paths.
- [FrescoDiffusion](https://arxiv.org/abs/2603.17555) motivates a global-prior,
  overlap-tile, weighted-fusion research path; it is not yet evidence for the
  CAUCE H3 topology.

CAUCE currently proves deterministic geometry, shapes, masks, and contracts.
It does not yet prove that either H3-native enhancement operation executes or
improves a real shot on the laboratory runtime.
