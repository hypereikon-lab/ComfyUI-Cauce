# Sigma-conditioned H3 latent transport

This experiment inserts coordinate transport inside one native ComfyUI sampler
call. It does not use a motion-reference video, decode intermediate frames, or
restart the solver at every step.

## Audited runtime contract

The laboratory runtime is based on ComfyUI commit
[`924743af`](https://github.com/Comfy-Org/ComfyUI/commit/924743af083c151296cc16f925aeab113b6484e8).
Its [`KSAMPLER.sample`](https://github.com/Comfy-Org/ComfyUI/blob/924743af083c151296cc16f925aeab113b6484e8/comfy/samplers.py#L977-L1007)
keeps the selected solver inside one call. Deterministic
[`res_multistep`](https://github.com/Comfy-Org/ComfyUI/blob/924743af083c151296cc16f925aeab113b6484e8/comfy/k_diffusion/sampling.py#L1394-L1464)
performs exactly one model evaluation per nonterminal sigma and retains its
previous denoised estimate internally.

ComfyUI packs H3's nested audiovisual latent before sampling and records the
original stream shapes on the prepared model. CAUCE retains ComfyUI's
deterministic RES equations but owns the small solver loop required for the
operator split. At a transport step it unpacks both the current state and the
retained previous denoised estimate, applies the same visual pullback to both,
repacks them, and only then requests the next prediction.

This covariance is not optional. The first live smoke test transported only
the current state while leaving RES's second-order history in its old
coordinate frame. The graph executed successfully, but the decoded video
showed strong horizontal colour bands. That implementation was rejected. The
current contract transports state and history together so every finite
difference compares registered fields.

## Operator split

Let `x_i` be the noisy state at sigma step `i`, `M` a target-to-source motion
map, and `a_i` the cumulative sigma envelope. CAUCE computes

```text
delta_i = a_i - a_(i-1)
grid_i  = identity + delta_i * (M - identity)
x_i*    = grid_sample(x_i.video, grid_i)
h_(i-1)*= grid_sample(d_(i-1).video, grid_i)
d_i     = H3(x_i*, sigma_i, conditioning)
x_(i+1) = res_multistep_update(x_i*, d_i, h_(i-1)*, sigma_i, sigma_(i+1))
```

The audio stream is copied when either packed state is rebuilt. This is
first-order transport/diffusion operator splitting with covariant solver
history. It does not pretend that a nonlinear map interpolated by displacement
is an exact diffeomorphic exponential. Strength must therefore remain small.

## Sigma envelopes

Percent zero is the highest-sigma evaluation and percent one the lowest.

- `accumulate`: zero before `start`, eases to `strength`, then retains it.
- `pulse`: rises to `strength` at the window midpoint and returns to identity
  at `end`.

Available easing functions are `linear`, `smoothstep`, and `cosine`. Negative
strength applies the first-order inverse direction.

Early transport can influence global organization while the state is noisy.
Middle transport tests gesture and camera structure. Late transport acts on a
more resolved latent and is expected to carry the highest risk of texture
tearing or endpoint drift.

## Compatibility and fail-closed behaviour

The current implementation accepts deterministic `res_multistep`,
`res_multistep_cfg_pp`, and first-order `euler`. Euler is the minimal diagnostic
path because it retains no denoised history; comparing it to RES distinguishes
multistep covariance failures from H3 manifold sensitivity. Ancestral and
arbitrary multi-evaluation solvers are rejected because one sigma step may
contain a different number of internal model evaluations. Runtime also verifies
that the observed model-call count equals `len(sigmas)-1`.

The node is a normal `SAMPLER → SAMPLER` wrapper:

```text
KSamplerSelect(res_multistep)
          + CAUCE_MAP
          ↓
CAUCE Sigma-Conditioned H3 Transport
          ↓
SamplerCustomAdvanced
```

Workflow 74 is the initial matched ablation. A valid decode proves wiring, not
motion fidelity. Promotion additionally requires optical-flow agreement,
endpoint stability and blind visual comparison against its baseline.

Workflow 75 is the causal motion probe. It supplies only a first image (`A →`),
uses a neutral shared prompt and compares official Euler against horizontal,
zoom and rotation fields with every stochastic and conditioning input held
fixed. Each map starts at identity in visible video time and grows
monotonically. The sigma envelope is `accumulate`, so the imposed coordinate
frame is retained instead of being undone before the final solver step.

The first live Euler control on 2026-08-24 produced byte-identical MP4s for the
official sampler and the CAUCE wrapper at strength zero. The original stress
preset (`32%` horizontal map × `0.25` strength, approximately `8%` retained
translation) produced severe mosaic and chroma-band corruption. A matched
horizontal sweep with border padding found `8% × 0.10` (approximately `0.8%`)
visually coherent, while `16% × 0.10` and `24% × 0.10` showed increasing latent
tearing. The clean branch diverged from the baseline over time, but integer
frame registration did not recover a directional horizontal displacement.
Therefore this validates the runtime path and a conservative operating region,
not motion-field obedience. The shipped example uses the conservative regime.
