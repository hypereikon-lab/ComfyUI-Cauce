# H3 extension map

This document separates current CAUCE operations from workflow-only H3 usage,
isolated experiments, and deferred integrations. Community precedent is
evidence that a graph is worth testing, not evidence that it is visually
accepted on the laboratory runtime.

## Implemented deterministic layer

The following operations now have typed contracts, offline topology dossiers,
registered low-level nodes where vanilla ComfyUI lacks the required explicit
data transform, and unit tests:

| Operation | Deterministic CAUCE contribution | Live state |
| --- | --- | --- |
| `edit.masked_video` | continuous MASK projection and composition over native H3 visual tokens | unit-validated; live execution pending |
| `reframe.outpaint_video` | aligned larger-canvas allocation, exact source placement, generated-region mask | unit-validated; live execution pending |
| `refine.video` | reuse of continuous interval and optional spatial masks for bounded second-pass H3 sampling | unit-validated; live execution pending |

No operation changes the H3 sampler. Official conditioning, model, guider,
scheduler, sampler, and VAE decode stay visible in every graph. The native
`MiniMaxH3SigmaShift` patch is not part of the canonical sampling spine: the
current official template and the characterized lab workflow connect the
loaded model directly to the guider and scheduler. Keep a sigma-shift node only
in an explicitly named experiment.

## Workflow-only patterns

These capabilities are already expressible with official nodes and should not
be wrapped by new CAUCE nodes:

### Structured multishot generation

Use an ordinary H3 generation graph with a time-ordered prompt describing
shots, cuts, or transitions. It is prompt data, not a new model operation.

### Window-aligned reference conditioning

For a serial sequence, bind the matching reference-video range to each target
window. The production plan records that mapping; each executable graph still
contains a concrete `MiniMaxH3ReferenceToVideo` input.

### Multiple visual anchors

Chain official `MiniMaxH3AddGuide` nodes for first, last, interior image, or
interior legal-length clip guides. Negative frame indices remain official node
behavior.

## Isolated experiments

### Learned H3 visual-latent upscale

Candidate implementation:

```text
native AV state
  -> vanilla Separate AV Latent
  -> community learned 24-channel visual-latent upscaler
  -> vanilla Concat AV Latent with unchanged structural-audio state
  -> bounded refine.video
```

The fp16 weight is approximately 691 MB. Adoption requires an isolated install,
schema capture, lineage audit, and an A/B comparison against direct-resolution
H3. It is not part of CAUCE and is not a production dependency.

Source: <https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler>

### Local in-context high-resolution regeneration

The official dedicated `H3-Regenerate-2K` checkpoint is not available for local
use. Community graphs approximate the mechanism with the base model and an
existing generation as context. Treat single-window reproduction as an
experiment; do not present chunked long-form refinement as established.

Sources:

- <https://github.com/MiniMax-AI/MiniMax-H3>
- <https://github.com/ckinpdx/ComfyUI-MMH3Tools>

## Deferred integrations

### H3 Fun ControlNet

Depth, Canny, HED, MLSD, pose, and video-inpaint control are highly relevant,
but current ComfyUI support is an open draft that requires a core patch. Do not
install it into the shared laboratory runtime until the integration is merged
or an isolated compatibility branch has passed the complete runtime gate.

Tracking source: <https://github.com/Comfy-Org/ComfyUI/pull/15860>

## Excluded scope

- generative audio or audio conditioning;
- training and LoRAs;
- streaming or acceleration work;
- sampler monkey patches;
- motion forcing transplanted from unrelated latent architectures;
- contact-sheet or turnaround workflows that require a LoRA.

The production soundtrack remains a fixed editorial clock and is muxed after
visual assembly.
