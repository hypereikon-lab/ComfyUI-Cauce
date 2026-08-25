# Operations guide

This guide describes graph-level composition without fixing a particular
prompt, model filename, resolution, or editorial structure.

## 1. Vanilla-first H3 baseline

Build the simplest model path with official nodes:

```text
model/text/VAE loaders
-> MiniMaxH3ImageToVideo or MiniMaxH3ReferenceToVideo
-> official sampler and scheduler
-> VAE decode
-> video save
```

Establish a seed-matched baseline before adding a CAUCE operation.

## 2. Continuation

```text
previous sampled H3 latent -------------------------┐
new official H3 target latent -> PrepareContinuation├-> sampler
                                                   └-> decode

sampled latent -> ResolveParentLatent(accepted_end_frame)
decoded images -> AcceptDecodedRange(start_frame, frame_count)
```

Rules:

- source and target use the same H3 latent geometry;
- context length is one of `5, 22, 39, 56, ...` visible frames;
- context must be shorter than the target;
- parent endpoints use the `17k+5` grid;
- prompts and any official guide conditioning remain explicit graph inputs.

## 3. Temporal inpainting across a cut

### 3.1 Build the working domain

Decode or load two normalized 24 fps clips. Connect them to
`CauceBuildSeamWindow`.

The default measured configuration resolves to:

```text
working batch: 124 frames
repair:        [26,98) = 72 frames
left guide:    [4,26)  = 22 frames
right guide:   [98,120) = 22 frames
```

Read these values from `seam_json`; do not re-derive them from approximate
seconds elsewhere in the graph.

### 3.2 Prepare conditioning

First run a masked-only control without guide clips. Then use official
batch-range/image slicing nodes to extract both guide clips from
`working_images` and apply two official `MiniMaxH3AddGuide` nodes at the frame
indices declared by the seam plan. This separates the effect of the preserved
main latent from the effect of duplicate guide conditioning.

CAUCE does not wrap the official guide node.

### 3.3 Encode and mask

```text
working_images -> official VAE encode -> encoded_video_latent
official H3 target latent --------------------------┐
encoded_video_latent -> PrepareH3TemporalInpaint ---┴-> masked latent
```

`CauceTemporalInpaintFields` supplies the binary baseline. For the continuous
comparison, use `CauceBuildTemporalDenoiseField`, set
`CaucePrepareH3TemporalInpaint.mask_mode = continuous`, and connect its
`denoise_strength` output as `generation_support`.

For a spatially animated field, combine the temporal strength with any standard
Comfy `MASK[T,H,W]` using the native `Combine Masks` node in `multiply` mode.
CAUCE then projects time to the H3 visual-token grid; official ComfyUI performs
the final `2×2` latent-patch max pooling. Structural audio remains zero-masked.

### 3.4 Sample, decode, splice

Sample the masked latent with the official sampler, decode the entire working
batch, then call `CauceApplySeamPatch` with both original decoded clips.

The final splice:

- replaces only the repair interval;
- preserves frames outside the interval exactly;
- applies a decoded cosine/smoothstep/linear opacity feather;
- preserves the combined source duration.

The complete mathematical and experimental contract is documented in
[TEMPORAL_INPAINTING.md](TEMPORAL_INPAINTING.md). Construct W0/W1/W2 as matched
comparisons from [WORKFLOW_CONTRACTS.md](WORKFLOW_CONTRACTS.md).

## 4. Motion maps

### Affine and analytic

Create a `CauceAffineMotionMap` or `CauceAnalyticMotionMap`, optionally modulate
it, and preview with `CauceWarpImage`.

### External displacement

`CauceDisplacementMotionMap` accepts an arbitrary RG image batch. This can be
optical flow, a simulation, a hand-authored field, or another numeric source.

### Advection

```text
CauceVectorField
-> CauceIntegrateAdvection
-> optional CauceModulateMotionMap
-> CauceWarpImage
```

### Depth camera

Provide a scalar depth image to `CauceDepthCameraMotionMap`. Inspect its
validity field: disocclusions are part of the operation and should be handled
explicitly downstream.

### Composition

```text
map A --┐
        ├-> CauceComposeMotionMaps -> CauceWarpImage
map B --┘
```

Compose maps before image sampling to avoid repeated resampling blur.

## 5. Latent persistence

Use `CauceSaveAVLatent` immediately after a sampled latent when later work needs
the native H3 state. Use indexed paths for deterministic reruns.

`CauceLoadAVLatent` reads only relative paths inside ComfyUI output and validates
the CAUCE H3 AV format marker.

Latent persistence avoids an unnecessary decode/encode boundary, but it does
not make independently generated latents phase-compatible.

## 6. Research operations

Research nodes are opt-in. Every run should include:

- the same official baseline;
- an identity/zero CAUCE control;
- one active magnitude;
- unique output prefixes;
- structural and perceptual comparison.

Start conservatively:

```text
warped noise correlation: 0.05
motion modulation:        0.15
sigma transport strength: 0.10
sigma padding:             border
```

Do not chain several active interventions before establishing the causal effect
of each one.

## 7. Evidence record

For every meaningful run record:

```text
source hashes or filenames
model and quantization
resolution and frame count
prompt
seed
sampler, scheduler, steps, denoise
CAUCE operation parameters
output paths
runtime duration and peak memory when available
structural checks
visual or measured result
promotion state
```

Use the states in [TECHNICAL_LANGUAGE.md](TECHNICAL_LANGUAGE.md).
