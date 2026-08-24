# Technical language and experiment reports

CAUCE is the name of the node package. Operations, nodes, workflows, outputs,
and experiment reports use descriptive technical names rather than metaphors.

## Canonical terms

- **Temporal inpainting:** regenerate a bounded time interval inside an encoded
  source video while preserving temporal context outside it.
- **Join / cut:** the exact edit boundary between the tail of one clip and the
  head of another. It is an input location, not a generation method.
- **Working window:** the bounded source-video domain encoded and sampled in one
  H3 run.
- **Inpaint interval:** the visible-frame interval selected for regeneration.
- **Source latent:** the VAE encoding of the existing working video. Temporal
  inpainting starts from this latent, not from an empty middle.
- **Temporal guide clips:** preserved clips immediately before and after the
  inpaint interval, attached through the official H3 guide conditioning.
- **Per-token denoise mask:** the binary H3 latent mask. `1` regenerates a token
  through the selected sigma schedule; `0` preserves the encoded source token.
- **Decoded patch:** the generated visible frames corresponding to the inpaint
  interval after VAE decode.
- **Decoded opacity feather:** a post-decode crossfade used only at patch edges.
  It does not alter denoise strength or model conditioning.
- **Frame interpolation:** synthesis of intermediate frames by an interpolation
  algorithm such as optical-flow or learned interpolation. CAUCE temporal
  inpainting does not currently use frame interpolation.
- **FL2VA / Ref2VA:** MiniMax H3 model families and conditioning backends. These
  names identify the model path, not the temporal operation performed by CAUCE.
- **Continuation:** generate future frames from a previous state. This is not
  equivalent to temporal inpainting, which has known context on both sides.
- **Motion map / pullback:** target-to-source normalized coordinate grid used
  to sample an image or H3 visual latent.
- **Vector field:** velocity data integrated through time into a motion map.
- **Map composition:** evaluate several coordinate transforms as one pullback
  and sample media once.
- **Sequential pass:** sample or denoise an intermediate result before applying
  another operation; not equivalent to composed-map evaluation.
- **Disocclusion validity:** confidence that a depth reprojection has a visible
  source sample at a target coordinate; it is not a semantic mask.
- **Warped H3 noise:** seeded H3 visual noise spatially correlated by a motion
  map before denoising. It does not modify model weights.

Terms such as “Confluence”, “bridge”, “polish”, and “redo” are not node,
workflow, or algorithm names. When an informal description is useful, the
experiment report still names the exact operation and parameters.

## Production temporal-inpainting contract

```text
two existing 24 fps videos
→ tail/head source context
→ one 124-frame working window
→ VAE encode existing video
→ binary H3 per-token denoise mask
→ 22-frame temporal guide clip before the interval
→ 22-frame temporal guide clip after the interval
→ H3 sampling only inside the mask
→ VAE decode
→ duration-preserving patch splice
```

The verified preset uses 2.5 seconds of source context per side, a 3-second
inpaint interval (`[26,98)`, 72 visible frames), guide clips `[4,26)` and
`[98,120)`, 20 `res_multistep` steps with the `simple` scheduler, denoise `1.0`
inside the binary mask, and a four-frame cosine opacity feather after decode.
No production audio enters this process.

## Required experiment report

Every run is described with the same fields:

```text
operation
input media and ordering
model backend and exact weights
source-latent origin
conditioning inputs
working window and inpaint interval
denoise mask and guide ranges
sampler, scheduler, steps, denoise, seed
decode and patch-insertion method
output frame count, fps, duration, and path
measured and visual result
promotion status: verified / executes but rejected / blocked
```

A completed ComfyUI job proves execution only. Production promotion additionally
requires inspection of the target interval, both patch edges, and preservation
outside the mask.
