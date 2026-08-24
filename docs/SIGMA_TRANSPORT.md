# Sigma-conditioned latent transport

Sigma transport is experimental. It inserts small coordinate pullbacks into a
deterministic sampler while preserving H3's packed structural-audio stream.

## Schedule

Let normalized progress be `p` and active interval `[a,b]`. CAUCE computes an
envelope `E(p)` and the per-step increment:

```text
delta_i = strength * (E(p_i) - E(p_{i-1}))
```

`accumulate` ends at the requested total strength. `pulse` returns to identity
by the end of the active interval.

## Packed H3 state

The sampler presents a packed tensor. CAUCE unpacks the visual and
structural-audio streams, transports only `[B,C,T,H,W]`, then repacks the
untouched structural-audio tensor.

Multistep solvers retain denoised history. Any retained visual history must be
transported into the same coordinate frame as the current state. Euler is the
minimal diagnostic because it has no multistep denoised history.

## Supported diagnostics

```text
euler
res_multistep
res_multistep_cfg_pp
```

Other solvers fail closed.

## Observed envelope

The zero-strength Euler path was bit-identical to the official baseline. On the
tested H3 configuration:

- a retained displacement near `0.8%` decoded coherently;
- around `1.6%` and `2.4%` produced increasing tearing;
- an `8%` horizontal displacement produced severe mosaic and chroma corruption.

The coherent `0.8%` result proved nonzero influence but did not establish
reliable directional camera control.

Start with:

```text
strength:     0.10
padding_mode: border
solver:       euler for diagnosis
```

Always compare official baseline, zero control, and one small active field with
the same seed and conditioning.
