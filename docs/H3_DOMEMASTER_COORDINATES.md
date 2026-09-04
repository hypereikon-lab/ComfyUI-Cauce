# H3 domemaster coordinate experiment

## Status and claim boundary

`CauceH3DomemasterCoordinates` is a reversible, inference-only MiniMax H3
ablation. It asks one narrow question: does changing H3's spatial token address
from a square pixel grid toward an equidistant front-hemisphere ray grid improve
domemaster geometry without retraining?

It is **not** a calibrated lens adapter, a ControlNet, a LoRA, a support mask,
or a claim of projection preservation. A visual result must pass a controlled
comparison against the exact stock graph and seed before the patch earns any
status beyond `experimental`.

## Geometry

For a square image with disc-centred normalized coordinates `(u,v)`:

```text
rho   = sqrt(u^2 + v^2)
theta = rho * pi/2
phi   = atan2(v,u)

ray = (
  sin(theta) * cos(phi),
  sin(theta) * sin(phi),
  cos(theta)
)
```

The mapping is defined only for `rho <= 1`, the visible front hemisphere of an
equidistant 180-degree domemaster. H3 exposes three RoPE axes `(t,h,w)` but both
time and the two native spatial axes must remain usable. The first experiment
therefore maps `(h,w)` to `(ray_y,ray_x)` while `ray_z` remains implicit: on the
front hemisphere it is uniquely determined by x/y.

This is not the only possible spherical encoding. It is the smallest reversible
intervention that fits H3's current packed layout without modifying attention
weights or allocating another learned branch.

## Scope

The patch changes only `PackedLayout.position_ids[:,1:]` for:

- target `video` rows;
- optionally FL2VA `cond` keyframe rows, so anchors and generated tokens share
  the same coordinate convention.

It intentionally leaves these rows stock:

- text and audio;
- Ref2VA images/videos, whose source projection may differ from the target;
- pixels and VAE latents;
- model weights and sampler schedules.

Rows outside the unit disc default to the stock grid. The black exterior is a
separate support invariant and should be handled by VAE state plus native H3
continuous denoise masks. Collapsing the exterior onto the rim is exposed only
as an explicit ablation (`outside_disc=rim`).

## Controlled use

```text
LoadDiffusionModel
  -> CauceH3DomemasterCoordinates
  -> optional MiniMaxH3SigmaShift
  -> ordinary H3 guider / sampler
```

Run at 768 x 768 and compare the same graph, inputs, prompt, seed, schedule, and
sampler at `strength = 0, 0.25, 0.5, 0.75, 1`. `strength=0` is the implementation
control and must match the unpatched graph. Do not change prompting in the same
matrix.

Evaluate support leakage, circle/rim drift, endpoint error, radial/tangential
motion, temporal stability, and ordinary image quality independently. A useful
patch should improve geometric measures without merely freezing the image or
destroying motion.

## Why this precedes training

Recent spherical and ray-coordinate work supports two distinct routes:

1. training-free positional transforms can expose geometric priors already in
   a pretrained DiT;
2. robust camera/lens control often keeps native RoPE intact and adds a small,
   zero-initialized learned ray branch to attention.

This node tests the first route cheaply. If it produces a reproducible positive
signal, it supplies an informed baseline for a learned H3 domemaster adapter.
If it does not, it should be removed rather than preserved as folklore.

## Compatibility contract

The runtime wrapper requires current ComfyUI MiniMax H3 behavior:

- a prebuilt `minimax_payload["layout"]`;
- `PackedLayout.position_ids` and `PackedLayout.segments`;
- diffusion-model wrappers in `comfy.patcher_extension`.

The node fails closed if the layout is absent or a selected visual segment no
longer matches the target H3 spatial-row count. Any upstream layout change must
be re-audited before use.
