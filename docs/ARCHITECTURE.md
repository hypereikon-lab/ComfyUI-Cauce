# CAUCE architecture

## Boundary

CAUCE is a media/time compiler. It does not know what an image depicts. It
stores only information required to place, condition, preserve, generate,
accept, decode, and reproduce media.

The production soundtrack is a fixed master clock and delivery asset. It is
not a generative target, latent parent, or mandatory H3 reference. When H3
requires a nested audiovisual latent internally, CAUCE supplies and preserves
an empty audio stream as structural scaffolding and discards it after sampling.
The master soundtrack is never encoded through AudioVAE by temporal inpainting.

```text
MediaAsset / actual Comfy media tensors
              |
TimelinePoint + MediaSpan + TimeField
              |
GenerationWindow
              |
H3 compatibility adapter
              |
CONDITIONING + nested AV LATENT + nested AV MASK
              |
sampler
              |
phase-safe parent latent + accepted decoded range -> DecodeDomain
              |
ArtifactReceipt
```

Creative state and execution state never share a schema. A point remains the
same point when a different model, sampler, precision, or GPU is selected.

## Versioned contracts

| Schema | Purpose |
|---|---|
| `cauce.project/1` | Project metadata and one canonical clock |
| `cauce.timeline/1` | Ordered points, spans, and windows |
| `cauce.point/1` | Absolute time, arbitrary prompt, media versions |
| `cauce.span/1` | Opaque media over `[start,end)` |
| `cauce.field/1` | Scalar generate/preserve values over time |
| `cauce.window/1` | Context, render, acceptance, prefix, and discard plan |
| `cauce.decode-domain/1` | Accepted decoded artifacts assembled together |
| `cauce.execution-profile/1` | Hardware/model/runtime decisions |
| `cauce.receipt/1` | Reproducibility and parentage |
| `cauce.storage-plan/1` | Hashed, root-scoped physical file deletion plan |
| `cauce.storage-receipt/1` | Exact deleted/skipped paths and reclaimed bytes |

Contract values are plain dictionaries so Comfy can carry them as custom
sockets and workflows can persist their constructor widgets. Time values are
stored as `{numerator, denominator}` plus derived convenience seconds.

Storage plans are intentionally outside the media/timeline ontology. They may
name only files under Comfy's resolved `input/` or `output/` root. Plans contain
relative paths plus size and modification time; cleanup recomputes the plan hash
and requires the same plan to have been staged by an earlier unarmed run. It
restats every file before unlinking it. A plan cannot become a general file
browser or address model, workflow, user, custom-node, or operating-system paths.

## Clock model

The master coordinate is rational time. H3 projections are:

```text
pixel frames       N  = 17k + 5, 24 fps
visual latents     Tv = 5k + 2
audio latents      Ta = round(N * 40 / 24)
audio samples      round(time * sample_rate)
```

Visual latent support is not a uniform `/4` operation:

```text
(1, 4, 4, 4, 4) repeated
```

CAUCE reduces a visible-frame spatial mask over every actual support interval
with `amax`. A one-frame mark therefore cannot disappear because an arbitrary
nearest temporal sample missed it.

## Windows and decode domains

Four ranges are distinct:

1. `context_range` — already-existing media exposed to the next generation.
2. `duplicate_prefix_range` — decoder/model prefix not accepted as new media.
3. `accepted_range` — new media committed to the master timeline.
4. `decode_domain` — accepted decoded spans assembled on one delivery clock.

Context and duplicate prefix can overlap. CAUCE uses the longest hidden head,
not their sum.

A production continuation boundary only needs to satisfy the visual H3 grid:
`5, 22, 39, 56, ...` frames. Earlier CAUCE versions additionally required the
40 Hz audio grid and therefore admitted only `39, 90, 141, ...`; that constraint
was removed when generated audio left the production contract. H3's internal
audio rows are frozen rather than inherited.

Every window also declares an acceptance policy:

- `nearest_run` (default) selects the closest phase-safe `17k+5` endpoint.
- `floor_run` selects the previous phase-safe endpoint.
- `ceil_run` selects the following phase-safe endpoint.
- `exact_frames` preserves the requested visible-frame boundary; use decoded
  acceptance, but it cannot become a latent continuation parent unless it also
  happens to be a valid H3 run.
- `full_render` accepts everything after the hidden head.

The contract stores both the requested time range and the resolved integer
frame range. No downstream node has to reconstruct that decision from floats.

## H3 adapter

The adapter lazily imports the current official Comfy classes only when a node
executes. CAUCE calls their public `execute` methods and unwraps their standard
`NodeOutput`; no official model/node implementation is copied.

The adapter fails closed if current Comfy no longer exposes the expected H3
classes. This makes upstream change visible instead of silently producing a
different conditioning layout.

## Continuation

`Prepare H3 Continuation`:

1. Validates that previous and target visual latent geometry match.
2. Extracts a phase-aligned `5/22/39/56/...` frame visual tail.
3. Copies only the video tail into the new latent head.
4. Preserves that visual context with mask zero and freezes the complete
   internal audio stream with mask zero.
5. Uses masked continuation as the runtime-compatible latent baseline.
6. Can pair that baseline with the previous decoded endpoint as FL2VA's native
   `first_frame`; this is the shipped compatibility path for ComfyUI 0.33.1.
7. Exposes `mask_plus_guide` only when the runtime contains the official H3
   clip-guide implementation; older runtimes fail with a targeted message.
8. Returns the exact decoded head trim.

`Resolve Parent Latent` crops only the post-accept tail. It retains the causal
origin and context head, producing another legal H3 latent from which the next
phase-aligned tail can be copied. CAUCE never removes a non-five-token prefix
and then pretends that the remaining latent begins at phase zero.

Likewise, independent H3 AV latents are not concatenated before VAE decode.
Every parent latent is decoded in its own causal domain; `Accept Decoded
Window` then removes the repeated head and snapped tail on exact frame/sample
boundaries. Accepted decoded spans are assembled afterward.

Generated H3 audio is not accepted into production. The fixed master soundtrack
uses `Authoritative Audio` and bypasses AudioVAE entirely.

## Localized temporal inpainting

Continuation generates an unknown future from one parent. Temporal inpainting
instead receives two already-decoded clips and regenerates a localized interval
around their visible join. It does not
concatenate independent H3 latents. The source pixels are first assembled into
one causal VAE domain:

```text
duplicate guard + tail(A) + head(B) + duplicate guard
  → one H3 video latent
  → left/right 22-frame H3 guide clips
  → official H3 per-token mask over the requested center
  → standard H3 sampler
  → decode one repaired working domain
  → accept only the central patch
  → unchanged A prefix + patch + unchanged B suffix
```

At the default 24 fps geometry, 2.5 seconds from each side gives 120 real
frames. Two duplicate guards at each edge make a legal 124-frame H3 run. The
cut is frame 62 of that working domain. The production three-second request
resolves to the symmetric token boundaries `[26,98)`: 36 frames from each
source, 72 frames total. Preserved guide clips occupy `[4,26)` and `[98,120)`.
The final frame count is always `len(A) + len(B)`.

Temporal inpainting deliberately separates three fields:

1. `sampling_support` is binary and controls the official H3 per-token denoise
   mask. Its boundaries are exact visual-token boundaries.
2. `hard_acceptance` is binary and defines the only decoded interval that can
   enter the result. It is not passed as an opacity heuristic.
3. `output_opacity` is a second continuous field used after decode to merge the
   accepted patch against the two original clips. Its default transition is
   four frames by default.

For a transition of `r` frames, the default field is

```text
w(d) = 0.5 - 0.5 cos(pi * clamp(d / r, 0, 1))
```

where `d` is distance from the nearest repair boundary. The first and last
accepted frames therefore have opacity zero, the interior reaches one, and the
derivative vanishes at both ends.

The continuous gradient belongs only to decoded compositing. H3 preservation
is governed by a token mask and by the model's own timestep labels. ComfyUI
v0.33.1 accepted the nested mask at the sampler boundary but did not yet pass
per-row mask timesteps into MiniMax H3; it therefore regenerated rows that the
graph appeared to preserve. CAUCE now inspects the native mask hooks and
`MiniMaxH3AddGuide` before sampling, and rejects an incomplete runtime.

The two guide clips are causal context, not replacement media. The left clip
shows incoming position and velocity; the right clip shows the outgoing state.
Together they condition the regenerated interval from both temporal directions
while leaving media opaque.

This is video-only by design. H3's structurally required nested audio stream is
zero-masked and its output is discarded. The fixed soundtrack remains on
CAUCE's authoritative sample clock. Long masters should not be encoded or
loaded into temporal inpainting; repair local clips and place their accepted outputs on
the visual timeline.

## Persistence

AV latent files store two tensors in one safetensors file:

```text
video [B,24,T,H,W]
audio [B,32,2,T40]
```

An optional canonical receipt is embedded in metadata. Files are written to a
temporary path in the same directory, flushed by safetensors, and atomically
replaced. Indexed slots overwrite rejected rerolls instead of allowing “newest
file” to accidentally become a rejected continuation parent.

Latent files resume between completed sampling runs. They are not partial
sampler checkpoints: a true mid-step resume would additionally require RNG,
solver history, sigmas, scheduler, conditioning, and noise-mask state.
