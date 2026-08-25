# Node catalog

CAUCE registers 26 nodes: 20 stable and six experimental.

## Continuity

### `CaucePrepareContinuation`

Inputs: target H3 latent, previous H3 latent, phase-safe context length.

Outputs: masked target latent and context-frame count.

Copies a phase-aligned visual tail to the target head, protects it from
denoising, and freezes structural audio.

### `CauceResolveParentLatent`

Inputs: sampled H3 latent and phase-safe accepted endpoint.

Output: cropped parent latent.

Crops visual and structural-audio streams at a common H3 endpoint and removes
the sampling mask.

### `CauceAcceptDecodedRange`

Inputs: decoded image batch, start frame, frame count.

Outputs: exact image slice and accepted count.

## Temporal inpainting

### `CauceBuildSeamWindow`

Builds one H3-compatible tail/head batch and a `CAUCE_SEAM` plan. The plan
reports repair, sampling, guide, and splice ranges.

### `CauceTemporalInpaintFields`

Returns:

- visible sampling support;
- hard decoded acceptance;
- soft decoded output opacity;
- numerical report.

### `CauceBuildTemporalDenoiseField`

Builds a spatially uniform continuous denoise-strength field on exact H3 visual
tokens. `shoulder_tokens = 0` reproduces hard temporal support; larger values
apply linear, smoothstep, or cosine shoulders while leaving decoded acceptance
unchanged.

For animated spatial control, multiply this output by a standard Comfy mask
batch with the official `Combine Masks` node. CAUCE intentionally does not
duplicate native mask composition.

### `CaucePrepareH3TemporalInpaint`

Injects an encoded source video into an H3 target latent, projects visible
support to H3 visual tokens, and freezes structural audio. `binary` is the
backward-compatible default. `continuous` preserves fractional strength and
requires an explicit `generation_support` input.

### `CauceApplySeamPatch`

Extracts the generated patch, applies decoded opacity feathering, and replaces
the corresponding tail/head frames while preserving total duration.

## Motion maps

Every builder returns `CAUCE_MAP`, a validity mask, and a report unless noted.

| Node | Operation |
| --- | --- |
| `CauceAffineMotionMap` | translation, scale, rotation, pivot, easing |
| `CauceAnalyticMotionMap` | swirl, wave, radial, polygonal and related analytic fields |
| `CaucePerspectiveMotionMap` | projective four-corner pullback |
| `CauceDisplacementMotionMap` | arbitrary RG displacement input |
| `CauceModulateMotionMap` | temporal envelope and optional spatial mask |
| `CauceVectorField` | time-varying velocity field |
| `CauceIntegrateAdvection` | integrate a vector field into a pullback |
| `CauceDepthCameraMotionMap` | depth-based camera reprojection and disocclusion |
| `CauceComposeMotionMaps` | compose two inverse maps before sampling |
| `CauceWarpImage` | sample an image or frame batch through a map |

## Persistence

### `CauceSaveAVLatent`

Atomically writes visual and structural-audio tensors to an indexed
safetensors file inside ComfyUI output.

### `CauceLoadAVLatent`

Loads an explicit file or indexed/latest file from a relative output folder.
Rejects files without the current CAUCE H3 AV format marker.

## Research

All nodes in this section use `CATEGORY = CAUCE/Research`.

| Node | Hypothesis |
| --- | --- |
| `CauceBuildNativeLatentSeam` | build protected context from native source latents |
| `CaucePrepareH3NativeLatentInpaint` | denoise only the latent center between protected sources |
| `CauceWarpH3Latent` | impose a coordinate pullback before repair sampling |
| `CauceWarpedH3Noise` | bias motion through weak temporal noise correlation |
| `CauceSigmaMotionSampler` | transport visual latent state during deterministic sampling |
| `CauceH3FlowLatentInjectionSampler` | partially replace one visual clean estimate during deterministic Euler sampling while preserving the implied noise endpoint and structural audio |

These nodes require matched baselines and measured validation for every claimed
effect.
