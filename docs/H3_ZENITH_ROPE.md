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

## Initial live result — 2026-09-04 (Chile)

The node registered and executed on ComfyUI 0.34.0, RTX 5090, native H3 FL2VA
INT8 ConvRot with its existing PDD-8 LoRA. Three F2V source images were compared
at strength 0/0.5/1, 768 square, 124 frames at 24 fps, seed 20260903.
All nine executions completed. Real H3 logs confirmed a 24x24 spatial token
grid and eight selected inverse frequencies 0.01 through 0.00017782794.
Maximum phase changes were 0.07630947 rad (strength 0.5) and 0.15261894 rad (1).

Decoded outputs changed, but the screen did not show a clear improvement in
projection preservation across sources. On A all three variants eventually
abandoned the initial lens appearance; C/D retained similar trajectories and
did not demonstrate a geometric advantage from the phase patch.

A separate matched A test added explicit fixed-lens domemaster language. With
the patch bypassed, the output preserved the circular fisheye appearance much
better. Eight-band strength 1 offered no clear extra benefit. Increasing to
twelve bands changed phases by up to 1.52619 rad and visibly degraded support
preservation. No post-mask or control video was used in these tests.

**Not promoted as a geometry solution.** Default remains zero. These results
are a negative/inconclusive test of this particular untrained phase basis,
not a disproof of learned geometry adapters or of spherical RoPE generally.
Visual frame-sequence inspection and decoded diagnostics do not establish
calibrated angular accuracy; no reliability claim is made from one seed.
