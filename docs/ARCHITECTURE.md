# CAUCE architecture

## Boundary

CAUCE is a media/time compiler. It does not know what an image depicts. It
stores only information required to place, condition, preserve, generate,
accept, decode, and reproduce media.

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

Contract values are plain dictionaries so Comfy can carry them as custom
sockets and workflows can persist their constructor widgets. Time values are
stored as `{numerator, denominator}` plus derived convenience seconds.

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

A joint AV continuation boundary must satisfy both clocks: a valid H3 video
run (`17k+5`) and an integer position on the 40 Hz audio grid. The resulting
sequence is `39, 90, 141, 192, ...` frames. Video-only boundaries such as 22 or
56 are deliberately rejected for copied AV context.

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

1. Validates that previous and target AV latent geometry match.
2. Extracts a phase-aligned `39/90/141/...` frame AV tail.
3. Copies the video/audio tail into the new latent head.
4. Preserves video with mask zero and optionally releases audio through a short
   half-cosine feather.
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

Generated-audio continuation still needs empirical decoder-duration conformance
and seam testing. Final master music should use `Authoritative Audio`, which
bypasses AudioVAE drift.

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
