# H3 workflow wiring

## FL2VA first-to-last

```text
CAUCE Timeline Point (A) -> plate/result IMAGE --------------------┐
CAUCE Timeline Point (B) -> plate/result IMAGE ------------------┐ |
CAUCE Compile Window -------------------------------------------┐| |
CAUCE Execution Profile (FL2VA) -------------------------------┐|| |
CLIP + video VAE ---------------------------------------------┐||||
                                                             vvvvv
                                                      CAUCE H3 FL2VA
                                                             |
                                                       positive + latent
                                                             |
                                             ModelSamplingMiniMaxH3
                                                             |
                                                          sampler
```

The CAUCE node delegates conditioning/latent construction to the official
`MiniMaxH3ImageToVideo` implementation.

## Ref2VA motion reference

```text
Load Video frames
    -> CAUCE Add H3 Video Reference
optional additional image/video reference nodes
    -> CAUCE H3 Ref2VA
```

Prompt tags are returned by the reference-chain nodes. CAUCE preserves the
official ordering performed by the current native H3 implementation.

## Timed guide

```text
FL2VA or Ref2VA positive + latent
CAUCE window
absolute master_seconds
IMAGE or VIDEO
    -> CAUCE H3 Timed Guide
    -> positive
```

The guide node resolves absolute time to `frame_idx` and invokes the official
`MiniMaxH3AddGuide`. Several guide nodes can be chained.

## Temporal preserve/generate field

```text
CAUCE Time Field Span (0 = preserve)
    -> optional more spans
    -> CAUCE Compile H3 AV Mask
                           ^
                    H3 latent + window
```

The returned latent contains a nested `noise_mask` consumed by the normal
Comfy sampler.

## Continuation

Run 1:

```text
H3 latent -> sampler -> CAUCE Resolve Parent Latent
                              |
                    CAUCE Save AV Latent (index 1)
```

Run 2:

```text
CAUCE Load AV Latent (index 1) --------------------------┐
new H3 positive + empty target latent ----------------┐ |
                                                      v v
                                      CAUCE Prepare H3 Continuation
                                                      |
                                           positive + masked latent
                                                      |
                                                   sampler
                                                      |
                                      CAUCE Resolve Parent Latent
                                                      |
                                      CAUCE Save AV Latent (index 2)
```

Use `39` frames as the already-tested initial visual context candidate. Every
H3 visual boundary `5`, `22`, `39`, `56`, ... is valid. The parent video tail is
copied; the internal audio stream is frozen and discarded.

Decode each resolved parent separately, then route its images through `CAUCE ·
Accept Decoded Window` with the same `CAUCE_WINDOW`. It removes both the hidden
head and any snapped tail. The fixed master audio remains outside H3.

Do not concatenate independent H3 latents before VAE decode. Their causal
`1,4,4,4,4` temporal phases do not restart safely at a naïve join. Assemble the
accepted decoded spans (or encoded per-window files) on the master clock.

Use the window's default `nearest_run` mode for a reusable latent parent.
Choose `exact_frames` only for decoded-only endpoints; `Resolve Parent Latent`
will reject an exact endpoint that is not a legal H3 run.

## Ordering masks and continuation

CAUCE intersects existing nested masks, so either order is supported:

```text
Compile AV Mask -> Prepare Continuation
Prepare Continuation -> Compile AV Mask
```

In both cases the copied continuation head remains preserved.
