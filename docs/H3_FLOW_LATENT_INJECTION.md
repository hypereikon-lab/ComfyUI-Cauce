# H3 flow latent injection

`CauceH3FlowLatentInjectionSampler` is an experimental sampler adapter for one
architecture-specific question: can a same-geometry visual latent impose a
small, controlled bias on an H3 generation without reducing the guide to a
decoded video reference?

It is not an H3 reference node, ControlNet, motion LoRA, context-window system,
or generic AnimateDiff image-injection port.

## Why the operation is H3-specific

Current native H3 sampling packs two generated streams into one rectified-flow
state:

```text
visual latent             [B,24,Tv,H,W]
structural-audio latent   [B,32,Ta,F]
```

The transformer separately packs text, reference rows, guides, target audio,
and target video into one attention sequence. References and guides already
have official conditioning paths. CAUCE therefore intervenes only in the
generated visual flow state and copies the packed audio state unchanged.

For the H3 flow coordinate

```text
x_sigma = sigma * epsilon + (1 - sigma) * x0
```

an Euler model evaluation supplies a clean estimate `x0_hat`. After advancing
to `sigma_next`, CAUCE applies once:

```text
x' = x + a * M * (1 - sigma_next) * (guide - x0_hat)
```

where:

- `a` is `strength` in `[0,1]`;
- `M` is a continuous visual-token mask;
- `guide` is a VAE-space visual latent with the exact target geometry.

At `a*M = 1`, this substitutes the guide for the current clean estimate while
preserving the Euler step's implied noise endpoint. The local Euler derivative
changes, as it must when the clean endpoint changes. Smaller values interpolate.
The node always leaves at least one later model evaluation so H3 can repair the
intervention toward its learned audiovisual manifold.

## Node contract

Inputs:

```text
base_sampler       official deterministic Euler SAMPLER
sigmas             the exact SIGMAS also connected to SamplerCustomAdvanced
guide_latent       same-geometry H3 visual LATENT
flow_progress      target clean weight 1-sigma_next in H3 flow space
strength           clean-estimate substitution fraction
mask_projection    mean | maximum
mask               optional visible-frame MASK[T,H,W]
```

Outputs:

```text
sampler             SAMPLER for SamplerCustomAdvanced
report_json         exact operation and safety contract
```

The adapter fails closed when:

- the base sampler is not deterministic Euler;
- the runtime does not expose packed H3 AV geometry;
- runtime sigmas differ from the schedule used to configure the adapter;
- guide and target video geometry differ;
- mask frame count is neither one nor the exact H3 decoded span;
- fewer than two sampler transitions remain.

The first version intentionally excludes RES and other multistep solvers. Their
history stores prior derivatives or clean estimates; changing only the current
state would make the intervention internally inconsistent.

`flow_progress` is deliberately not a linear step index. With H3's default
video shift of 12, step 10 of a 20-step simple schedule lands at approximately
`sigma = 0.923`, so its clean weight is only `0.077`. The adapter searches the
actual supplied schedule for the transition whose `1-sigma_next` is closest to
the requested value and reports the selected step, both sigmas, and the
effective guide-delta weight `strength * (1-sigma_next)`.

## Mask geometry

H3's causal visual VAE maps visible frames to latent-token spans with a special
first token and repeated four-frame groups. The node projects a standard Comfy
`MASK[T,H,W]` onto those exact spans and preserves fractional spatial values.

`mean` answers “what fraction of this token's visible interval is selected?”
and is the conservative default. `maximum` selects an entire token when any
visible frame in its causal span is active. Native Comfy mask nodes remain the
owner of drawing, animation, multiplication, and other spatial composition.

## What transferred from AnimateDiff Evolved

The useful transfer is experimental method, not tensor layout:

- make interventions schedule-aware;
- provide an exact neutral control;
- keep the intervention independently maskable;
- change one dimension at a time;
- run an additional denoising/model pass after perturbing state;
- treat context windows, noise construction, reference control, and latent
  injection as separate mechanisms.

AnimateDiff represented time as a frame batch around a 2D diffusion model plus
a motion module. H3 stores time inside a causal 3D visual latent, jointly
samples structural audio, uses rectified flow, and applies 3D MM-RoPE in a
single packed transformer. Reusing AnimateDiff's frame indexing, context
windows, or diffusion re-noising equations directly would be incorrect.

## First empirical matrix

Use workflow contract W5. Fix source, guide, prompt, seed, model, scheduler,
steps, target geometry, and decode. Record:

| Case | Adapter | Strength | Purpose |
| --- | --- | ---: | --- |
| A | none | — | official Euler baseline |
| B | connected | 0.00 | exact implementation identity |
| C1 | connected | 0.05 | small active effect |
| C2 | connected | 0.10 | second dose only after C1 is stable |

Then vary only one axis:

```text
flow_progress: 0.20, 0.45, 0.60
mask: full, temporal band, localized animated field
guide: source latent, coordinate-warped source latent, alternative native latent
```

An acceptable result requires more than divergence from baseline. Inspect
motion continuity, guide-direction obedience, texture locking, temporal reset,
spatial tearing, audio-stream integrity, runtime, and memory. Failed settings
belong in `LAB_RESULTS.md`.

A repeated still is only a structural-attraction control. It cannot establish
motion obedience. A causal motion test must encode a same-geometry sequence
whose transform is known analytically, then measure the generated trajectory
against both that guide and a matched static-guide branch.

## Adjacent mechanisms kept separate

- Official H3 image/video/audio references and arbitrary guides already own
  conditioning-time anchors.
- Untwisting RoPE changes only selected native-reference key frequencies. It is
  a composable external mechanism, not latent injection.
- H3 Context Noise corrupts carried continuation context to reduce appearance
  residue. It answers a different continuation problem.
- H3 Fun ControlNet introduces trained residual control weights. It should be
  evaluated through upstream ComfyUI support rather than wrapped by CAUCE.
- CAUCE sigma transport warps the current latent repeatedly. This node instead
  performs one flow-consistent clean-estimate substitution.

These can eventually be crossed in a factorial experiment, but none should be
enabled while validating another one's identity control.
