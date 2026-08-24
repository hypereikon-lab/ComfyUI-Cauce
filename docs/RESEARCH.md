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
