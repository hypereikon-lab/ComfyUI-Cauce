# H3 Zenith RoPE experiment

`CauceH3ZenithRoPE` changes selected spatial RoPE phases during the native H3
forward. It does not resample latents, train weights, or inject a control video.
It is an experimental hypothesis, not a production geometry guarantee.

## Projection and scale

The fixed contract is square Zenith-180, equidistant 180 degrees, a full
short-edge circle, no authored radial remap. Zenith-spatial reference commit:
`a069463fa3748d5557eefb4a395466503320ebbd`,
`src/kernels/projection/fisheye.ts`. For token centers `(j+.5)/N,(i+.5)/N`,
set `x=2u-1`, `y=1-2v`, `r=hypot(x,y)`, `theta=pi*r/2` and
`d=(x*sin(theta)/r, cos(theta), y*sin(theta)/r)`.
Center is +Y; right is +X; image-up is +Z. Exterior centers retain stock phases.
Cells overlapping the disc boundary are represented by their center rays;
fractional coverage and VAE receptive fields are not modeled in this ablation.

H3 uses 16 frequency pairs per axis and duplicated `[t,h,w]` phase halves.
Choose the lowest actual absolute inverse frequencies, keeping at least one
spatial band native. The first screen uses eight of sixteen per spatial axis.
Time and all unselected bands remain bit-identical.

For alternating sign `s=+1,-1`, the selected h/w coordinate features are
`h = c + (16/(pi/2))*(-dz+s*(dy-1))` and
`w = c + (16/(pi/2))*( dx-s*(dy-1))`, with native grid mean
`c=16*(N-1)/N`. This includes all three ray components while matching the
native tangent derivatives at the center. It is an anisotropic feature basis,
not a proof of geodesic or rotationally invariant attention.
Blend phases as `stock + strength*(ray_phase-stock_phase)`.
No ERP periodicity or wrap is imposed. Exterior/interior phase discontinuities
remain possible at the disc rim and are a specific visual failure to inspect.

## Runtime contract

- Scope: target video and optionally FL2VA keyframe `cond` rows sharing the
  exact native grid; text, audio and Ref2VA reference rows remain unchanged.
- The phase hook verifies that native `rope_freqs` is actually consumed and
  emits `CAUCE_ZENITH_ROPE` with selected frequencies and maximum phase delta.
- Strength zero returns the original MODEL object without adding a wrapper.
- Nonzero execution uses a per-instance temporary method override, restored
  in `finally`, including error paths. No global classes are patched.
- Standard serial Comfy execution only. Concurrent forwards on one diffusion
  model and compiled/cached alternate forward implementations are unsupported.
- Do not stack with coordinate warps; unexpected spatial grids fail closed.

## Evidence and experiment

Projection vectors, phase locality, same-time geometry, keyframe scope,
strength scaling, and restoration after exceptions have deterministic tests.
Live runs must compare identical image/prompt/seed/sampler settings. Save and
inspect generated videos; phase change and exterior blackness do not establish
geometric improvement. Repeat across images/seeds before claiming reliability.

References: [SpheRoPE](https://arxiv.org/abs/2606.32033) motivates selective
spectral adaptation in other backbones/ERP. The tilted ray basis here is an
unvalidated adaptation for Zenith/H3, not that paper's released algorithm.
