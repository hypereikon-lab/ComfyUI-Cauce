# Research basis

CAUCE is informed by official source, primary papers, and measured runtime
behavior. Community repositories and discussions are leads, not implementation
authority.

## Primary technical anchors

- official ComfyUI MiniMax H3 nodes and model runtime;
- current MiniMax H3 paper/model documentation;
- PyTorch `grid_sample` coordinate semantics;
- causal video VAE frame/token geometry;
- temporal inpainting and masked video diffusion literature;
- optical flow, semi-Lagrangian advection, and depth-image reprojection;
- deterministic diffusion-sampler state and multistep-history behavior.

Record upstream commit hashes when a capability depends on current source.
Mathematical reimplementation is preferred over copying third-party code; do
not import GPL implementation code into this MIT repository.

## Temporal-inpainting source audit — 2026-08-24

Primary implementation sources:

- MiniMax H3 official repository and model documentation:
  <https://github.com/MiniMax-AI/MiniMax-H3>
- ComfyUI H3 conditioning nodes:
  <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py>
- ComfyUI H3 model runtime:
  <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/minimax/model.py>
- native per-token AV noise masks, merged as ComfyUI PR `#15375` on
  2026-08-18, merge commit `ff6c8a8af144fc9e9e7bc436b1b202f9316848d8`:
  <https://github.com/Comfy-Org/ComfyUI/pull/15375>
- arbitrary clip guides, merged as ComfyUI PR `#15439`:
  <https://github.com/Comfy-Org/ComfyUI/pull/15439>

Relevant algorithmic references:

- RePaint: known-region reinjection and diffusion resampling:
  <https://github.com/Yidan-Zhang/RePaint-Inpainting-using-Denoising-Diffusion-Probabilistic-Models>
- VACE: unified video condition units for creation and masked editing:
  <https://arxiv.org/abs/2503.07598>
- AVID: overlapping temporal windows and middle-frame attention for long video
  inpainting: <https://arxiv.org/abs/2312.03816>
- Flow-Guided Diffusion for Video Inpainting: latent propagation and flow-guided
  interpolation: <https://arxiv.org/abs/2311.15368>
- ProPainter: bidirectional completed-flow propagation and sparse temporal
  refinement: <https://arxiv.org/abs/2309.03897>
- VideoCanvas: causal-VAE temporal ambiguity in arbitrary timestamp control:
  <https://arxiv.org/abs/2510.08555>

Consequences for CAUCE:

1. native ComfyUI row masking is the canonical temporal-inpainting sampler
   path;
2. preserved latent context, AddGuide conditioning, prompt, and decoded opacity
   are separate interventions and require ablations;
3. sampling support and decoded acceptance are separate domains;
4. motion continuity must be evaluated with flow/registration, not inferred
   from a visually soft crossfade;
5. causal-VAE frame/token phase is a structural constraint, not an editorial
   convenience.

## Current research questions

### Native-latent temporal inpainting

Can phase-matched source H3 latents preserve more motion and texture information
than decoded video re-encoding at a seam? Structural correctness is established;
perceptual benefit is not.

### Latent pullbacks

Which small coordinate transforms remain inside H3's learned latent manifold,
and can a repair pass preserve their intended direction without erasing it?

### Motion-correlated noise

Can weak transported noise bias gesture while preserving the expected Gaussian
statistics? Strong correlations have failed decode-quality gates.

### Sigma transport

Can incremental visual-latent transport produce measurable directional control
at magnitudes below the tearing threshold? Current evidence proves integration
and nonzero influence, not control fidelity.

## Experiment template

```text
hypothesis
official baseline
identity/zero control
single active intervention
fixed sources/prompt/seed/model/sampler
structural checks
requested quantitative measurement
visual inspection
resource record
promotion decision
```

Negative results remain valuable: they define the usable envelope and prevent
repeating expensive GPU experiments.
