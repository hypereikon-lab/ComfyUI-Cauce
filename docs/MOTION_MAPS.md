# Motion-map mathematics

## Coordinate convention

A CAUCE motion map is an inverse pullback:

```text
output(x, t) = source(map(x, t))
```

The grid uses PyTorch normalized `align_corners=False` coordinates. A positive
forward translation therefore samples from the opposite direction in the
pullback grid.

The identity pixel-center coordinate for width `W` is:

```text
x_norm = 2 * (x + 0.5) / W - 1
```

and likewise for height.

## Contract

```text
grid:       float32 [T,H,W,2]
validity:   float32 [T,H,W]
fps:        float
operation:  string
parameters: object
tensor_hash:string
```

Validity is a first-class output. It marks samples that remain supported by the
source geometry and propagates through resizing and composition.

## Affine pullback

For translation `d`, scale `s`, rotation `R`, and pivot `p`, the forward map is:

```text
y = p + R * s * (x - p) + d
```

Sampling needs its inverse:

```text
x = p + (R * s)^-1 * (y - p - d)
```

Parameters may vary over time through linear, smoothstep, cosine, or looping
envelopes.

## Projective pullback

The perspective operation solves a homography from four endpoint corner
offsets. The homography is inverted before sampling so the result remains an
inverse map rather than a forward splat.

## Arbitrary displacement

An RG image can represent displacement in either centered `[0,1]` or signed
form. After spatial and temporal resampling, its magnitude is converted into
normalized coordinate units and added to identity.

No semantic interpretation is attached to the field.

## Modulation

Given identity `I`, map `M`, temporal strength `a(t)`, and optional spatial mask
`m(x,t)`:

```text
M_mod = I + a(t) * m(x,t) * (M - I)
```

This permits a single reusable map to be gated in time and space.

## Composition

For pullbacks `A` then `B`, the composed source lookup is:

```text
C(x) = A(B(x))
```

CAUCE samples the coordinates of `A` through `B`, combines validity, and emits
one map. Applying `C` once normally preserves more image detail than applying
two image warps.

## Vector-field advection

A field stores velocity `v(x,t)`. Integration solves a backward characteristic
from output time to source time:

```text
dx/dt = -v(x,t)
```

Euler, midpoint/RK2, and RK4 integration are available. A sine-loop temporal
mode constructs a closed envelope whose final map returns to identity.

## Depth-camera reprojection

Depth is converted to camera-space points, transformed through translation and
rotation, projected forward with a depth/confidence selection, and converted
back to an inverse sampling grid.

The resulting holes are disocclusions, not errors to hide. They appear in the
validity field and require a downstream fill, inpaint, or crop policy.

## Image sampling

`CauceWarpImage` accepts one image or a batch matching the map length. Sampling
uses bilinear `grid_sample` with `border`, `reflection`, or `zeros` padding.

## Experimental H3 use

The Research surface can resample H3 visual latents using motion maps. The
structural-audio stream is copied unchanged. Because the learned visual latent
space is not guaranteed to be equivariant to arbitrary affine or advective
warps, decode integrity and directional obedience must be measured separately.

Warped noise transports one shared Gaussian anchor across time and mixes it
with independent H3 noise:

```text
n = sqrt(rho) * warped_anchor + sqrt(1-rho) * independent
```

Per-token spatial standard deviation is renormalized afterward. Begin near
`rho = 0.05`; large correlations can leave the expected noise manifold.
