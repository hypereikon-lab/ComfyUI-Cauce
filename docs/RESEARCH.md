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
- current H3 mask implementation, including `1/256` strength quantization,
  per-row timesteps, and `2×2` latent-patch max pooling:
  <https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py>
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
6. fractional denoise strength is a native runtime capability; CAUCE should
   compile fields into it rather than patching the model or wrapping the
   sampler.

## AnimateDiff-to-H3 sampler audit — 2026-08-24

Source snapshots:

- ComfyUI `5f0c4e18cb7e98f0e7c46c2c7ce928d641351e67`;
- MiniMax H3 `d21241f0a4b3acbb34c97dae47fa417b7065e438`;
- H3 Context Noise `7e5531233b42dadd19c40d86770521a36508c358`;
- H3 Context Loop `0e6109ba956625f22dd18ab779fdd1d490b11d8c`;
- H3 Motion Context MultiRef
  `87de57ba619297503fa49c9594c0c021d5b0c261`;
- Untwisting RoPE `299d4c56a3f057a97b3140d2136189bcd1e7d6bb`.
- ComfyUI-MAINodes Motion Lab
  `68a8cb68e569bf2770b6f84e7646c9324b23b538`.

AnimateDiff Evolved established a useful experimental vocabulary: sliding
context windows, scheduled context policies, FreeNoise, FreeInit, image
injection during sampling, advanced ControlNet scheduling, and strict neutral
controls. Its implementation assumptions do not transfer literally. AnimateDiff
puts video time in the batch of a 2D latent diffusion model plus a temporal
module. H3 uses a causal `f16t4d24` visual VAE, a separate structural-audio
latent, rectified-flow sampling, and a 33B packed single-stream transformer with
3D MM-RoPE.

Current H3-relevant mechanisms divide cleanly:

1. Official H3 references and `MiniMaxH3AddGuide` already own image, video, and
   audio anchors. CAUCE must not duplicate them.
2. H3 Motion Context and Context Loop carry a previous visual latent or decoded
   tail into continuation. Context Noise adds tapered corruption to reduce
   copied chroma/texture residue. These are continuation interventions.
3. Untwisting RoPE scales selected frequency channels of selected native
   reference keys after H3's Q/K normalization and rotary embedding. It leaves
   target rows, Q, V, audio, guides, and the 32 unrotated head channels native.
   This is a mature MIT external pack and should be composed, not copied.
4. ComfyUI's generic context-window implementation understands packed
   multimodal latents only when the model maps primary-window indices to every
   modality. At the audited commit `MiniMaxH3` does not implement that mapping;
   therefore an H3 wrapper would fail rather than provide an AnimateDiff-style
   long-context path. CAUCE does not add one speculatively.
5. MiniMax-H3-Fun-ControlNet-Union support exists as draft ComfyUI PR `#15860`
   with separate trained weights. It is the proper future path for learned
   pose/depth/canny-style residual control, not a reason to create a CAUCE
   ControlNet wrapper.
6. MAINodes Motion Lab retimes decoded frames, VAE-encodes that time-smear,
   wraps it as an H3 V2V initialization, and runs a truncated schedule before
   selecting the original clock back out. Its trajectory bank/load and paired
   `x0` diagnostics are useful experimental infrastructure, but they are not
   direct latent-coordinate motion operators. The project is GPL-3.0; CAUCE is
   MIT and must compose with it or re-derive independent mathematics rather
   than copy its implementation.

The missing, bounded hypothesis is state-space injection. CAUCE implements one
visual clean-estimate substitution after an Euler flow transition:

```text
x' = x + a * M * (1 - sigma) * (guide - x0_hat)
```

It preserves the current implied noise endpoint, copies structural audio, and
requires a later H3 evaluation. This is analogous in purpose to AnimateDiff
image injection, but derived from H3's actual rectified-flow and packed-AV
contracts. See `H3_FLOW_LATENT_INJECTION.md`.

Primary and implementation references:

- AnimateDiff Evolved:
  <https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved>
- FreeInit: <https://arxiv.org/abs/2312.07537>
- Untwisting RoPE paper: <https://arxiv.org/abs/2602.05013>
- H3 Untwisting implementation:
  <https://github.com/xmarre/ComfyUI-Untwisting-RoPE>
- H3 Context Noise:
  <https://github.com/beijinren/ComfyUI-H3-Context-Noise>
- H3 Context Loop:
  <https://github.com/ethanfel/ComfyUI-MiniMaxH3-Contex-Loop>
- H3 Motion Lab, trajectory banking, and `x0` diagnostics:
  <https://github.com/matlowai/ComfyUI-MAINodes>
- draft H3 Fun ControlNet support:
  <https://github.com/Comfy-Org/ComfyUI/pull/15860>
- rectified-flow feature injection/editing reference, FREE-Edit:
  <https://arxiv.org/abs/2603.01164>
- rectified-flow inversion and feature sharing, RF-Edit:
  <https://arxiv.org/abs/2411.04746>

## Current research questions

### Continuous spatiotemporal denoise fields

Do token-aligned fractional shoulders reduce motion discontinuity relative to a
binary mask without weakening preservation? After the temporal-only ablation,
can an independently animated spatial field direct where repair propagates
without introducing `32×32` patch-grid artifacts? These are separate tests.

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
