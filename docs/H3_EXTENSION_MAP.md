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
| `generate.with_control` | exact reporting of official control-clip fitting plus packed-sequence inspection around the official model patch | unit-validated planning; runtime/model gates pending |

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

Gated implementation:

```text
native AV state
  -> CAUCE Extract H3 Visual Stream
  -> community learned 24-channel visual-latent upscaler
  -> CAUCE Replace H3 Visual Stream on unchanged structural-audio carrier
  -> bounded same-H3 regeneration
```

The fp16 weight is approximately 691 MB. The external node and weight remain a
separately owned dependency; only the visual/AV adapters belong to CAUCE.
Adoption requires an isolated install, schema capture, exact source/weight
hashes, and an identical-seed A/B against native latent resize and pixel/VAE
re-encoding. Upstream reports of FL2VA shape mismatch and temporal flicker are
explicit rejection gates, not minor caveats.

Source: <https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler>

### Local in-context high-resolution regeneration

The official dedicated `H3-Regenerate-2K` checkpoint is not available for local
use. Community graphs approximate the mechanism with the base model and an
existing generation as context. Treat single-window reproduction as an
experiment; do not present chunked long-form refinement as established.

Sources:

- <https://github.com/MiniMax-AI/MiniMax-H3>
- <https://github.com/ckinpdx/ComfyUI-MMH3Tools>

## Official structural control

H3 Fun ControlNet support is now merged into ComfyUI through
`MiniMaxH3FunControlNetApply` and `ModelPatchLoader`. The union checkpoint
supports Canny, depth, HED, MLSD, pose, and video inpainting. CAUCE does not
reimplement the control model: `generate.with_control` composes the official
nodes and only exposes their otherwise implicit input fitting and packed-row
cost.

Do not update the shared runtime merely because the node is merged. Deployment
still requires live `object_info`, a pinned core commit, a model-patch hash, and
verification of the relevant compatibility/correctness gates:

- [ComfyUI #16020](https://github.com/Comfy-Org/ComfyUI/pull/16020): merged references/keyframes with control and dynamic-VRAM prefetch correctness; the laboratory capture predates it;
- [ComfyUI #15988](https://github.com/Comfy-Org/ComfyUI/pull/15988): denoise-mask velocity conversion;
- [ComfyUI #15978](https://github.com/Comfy-Org/ComfyUI/issues/15978) and [#15981](https://github.com/Comfy-Org/ComfyUI/issues/15981): mask regressions requiring live verification.

See [H3 structural control](STRUCTURAL_CONTROL.md).

## Studied, not adopted

- `TwoAbove/ComfyUI-H3VideoOutpaint`: useful evidence for phase-aligned,
  overlap-window chronological outpaint; its monolithic node overlaps CAUCE,
  and current H3 context windows do not encode absolute global position.
- `nazgut/ComfyUI-MiniMaxH3-CLSS`: WIP port from another model family; its
  anchor bank, AdaIN and calibrated re-noise are not yet H3-native evidence.
- `KJNodes` MiniMax token counter: confirms official `PackedLayout`; CAUCE now
  counts rows without importing or patching ComfyUI. Generic KJNodes mask
  authoring and preview nodes may compose around CAUCE through standard Comfy
  datatypes. KJ attention/FFN chunking remains an optional runtime experiment
  only if baseline memory measurements demand it; it is not CAUCE functionality.
- modality-LoRA loaders, TensorRT VAE, alternative attention, and mega-packs:
  outside current training/acceleration scope or too broad for the shared lab.

## Excluded scope

- generative audio or audio conditioning;
- training and LoRAs;
- streaming or acceleration work;
- sampler monkey patches;
- motion forcing transplanted from unrelated latent architectures;
- contact-sheet or turnaround workflows that require a LoRA.

The production soundtrack remains a fixed editorial clock and is muxed after
visual assembly.
