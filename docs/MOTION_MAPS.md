# CAUCE motion-map algebra

This subsystem expresses motion as data. It does not describe the contents of
an image and it does not assume that a field represents a camera, an object, a
fluid, or a gesture. Those interpretations remain outside the contract.

## Contract

`CAUCE_MAP` has schema `cauce.motion-map/1` and carries:

```text
grid       float32 [visible_frames, map_height, map_width, 2]
validity   float32 [visible_frames, map_height, map_width]
direction  target_to_source
coords     PyTorch normalized, align_corners = false
fps        exact declared map rate
hash       SHA-256 over grid + validity
metadata   operation, parameters, provenance
```

For a target coordinate `x`, the map stores the source lookup `M(x)`. An image
or latent is sampled as

```text
output(x) = source(M(x))
```

This inverse or pullback convention is the native convention of
`torch.nn.functional.grid_sample`. It avoids holes for ordinary inverse warps.
Operations that begin as forward geometry, notably depth-camera reprojection,
are forward-splatted with nearest-depth visibility and converted back into the
same pullback contract.

`validity` is not cosmetic opacity. It records how much trustworthy source
support exists at each target coordinate. A downstream H3 operation may use
`1 - validity` as candidate generation support, while decoded compositing can
make a separate aesthetic decision.

## Generators implemented

| Generator | Parameters | Useful family of motion |
|---|---|---|
| Affine map | translation, scale, rotation, pivot, temporal easing | pans, zooms, rotations, feedback transforms |
| Perspective map | four target-corner offsets | corner pin, keystone, projective drift, pseudo-camera tilt |
| Analytic map | swirl, pinch, wave, radial wave, tunnel, kaleidoscope | polar and nonlinear coordinate transforms |
| Vector field | uniform, rotation, radial, vortex, curl-sine, wave | velocities rather than final coordinates |
| Advection integrator | Euler, RK2, RK4 | flow map integrated through a time-varying field |
| Depth camera map | scalar depth, FOV, XYZ translation, yaw/pitch/roll | 2.5D parallax with occlusion/disocclusion support |
| Displacement import | arbitrary RG field | optical flow, simulations, particles, external renderers, future methods |

The displacement importer is the escape hatch that keeps the system
generalizable. Any future method that can produce a two-channel field can enter
the same algebra without adding a semantic entity to CAUCE.

## Algebra implemented

### Composition before sampling

If `A` and `B` are pullbacks, applying A to the source and then B to that result
is equivalent geometrically to

```text
C(x) = A(B(x))
```

`CAUCE Compose Motion Maps` calculates this map and propagates both validity
fields. The image or latent is then sampled once. This is the default when the
goal is maximum detail retention.

### Sequential passes

Sampling after every map is intentionally different:

```text
Y1 = sample(X, A)
Y2 = sample(Y1, B)
```

Repeated interpolation changes the signal, accumulates diffusion, and can
reproduce the material behaviour of early feedback animation. CAUCE supports
this explicitly by connecting multiple `Warp Image` or `Warp H3 Latent` nodes,
but does not call it equivalent to a composed map.

### Modulation

For a strength field `s(t,x)`, modulation interpolates from identity:

```text
M_s(t,x) = x + s(t,x) [M(t,x) - x]
```

`s` combines a temporal envelope with an optional arbitrary spatial `MASK`.
Negative and greater-than-one strengths are allowed for research, while the
validity calculation remains bounded.

### Closed motion

`sine_loop` maps begin and end at the same state. Vector fields use a periodic
velocity envelope whose integrated displacement closes at the final frame.
Numerical closure is tested for RK4. Visual loop quality still depends on H3,
the endpoints, and the sampling topology.

## Sampling domains

### Image domain

`CAUCE Warp Image` accepts one plate or one frame per map frame. A single plate
is broadcast through time. This is useful for previews, motion-reference video,
plate-space transformations, and decoded feedback passes.

### H3 latent domain

`CAUCE Warp H3 Latent` only changes the spatial axes of H3's visual stream. For
a 124-frame 768×512 run, H3 uses a visual latent of approximately
`[B,24,37,32,48]`. CAUCE evaluates the visible-frame map at the center of every
real causal VAE support interval rather than treating time as a uniform `/4`.
The structural audio stream is copied and frozen.

Three mask policies are explicit:

| Policy | Visual sampling mask | Intended use |
|---|---|---|
| `none` | omitted | deterministic latent transformation or decode preview |
| `holes` | generate unsupported pixels only | depth disocclusions and out-of-frame areas |
| `all` | regenerate the complete visual latent | low-denoise second-pass experiment |

Spatial latent warping preserves more native information than decode → image
warp → encode, but it is not lossless semantic geometry. H3 channels are a
learned spatiotemporal representation, and sufficiently strong coordinate
changes can leave the model manifold.

### H3 noise domain

`CAUCE Warped H3 Noise` generates normal seeded H3 noise, transports one shared
visual Gaussian anchor through the map, and mixes it with independent token
noise using an explicit `temporal_correlation` in `[0,1]`. Unsupported regions
retain independent noise and per-frame spatial variance is normalized.
The audio noise stream is unchanged.

`temporal_correlation` is a target covariance, not a simple opacity: the
transported anchor enters with amplitude `sqrt(correlation)`. Live H3 testing
therefore uses `0.05` as the conservative starting point together with a
modest motion-map envelope (`0.15` in workflow 71). A stress preset of `0.85`
plus a `0.7` map envelope completed numerically but produced a chromatically
corrupt decode, while the conservative preset and a seed-matched normal-noise
baseline remained on-manifold. Treat higher values as deliberate material
experiments and increase them in small steps.

This is a training-free H3 experiment related to the warped-noise strategy in
[Go-with-the-Flow](https://openaccess.thecvf.com/content/CVPR2025/html/Burgert_Go-with-the-Flow_Motion-Controllable_Video_Diffusion_Models_Using_Real-Time_Warped_Noise_CVPR_2025_paper.html):
that work also fine-tunes its video model on structured warped noise. CAUCE does
not train or modify H3, so zero-shot transfer is substantially less tolerant of
strong correlation. It remains an empirical control, not a guarantee that H3
will reproduce the field.

## Possibility matrix

The following are combinations of the implemented primitives, not separate
hard-coded effects.

| Construction | Map/data path | H3 path | Main question |
|---|---|---|---|
| Controlled pan/zoom | affine | warped noise or latent | Does native spatial bias survive denoising? |
| Projective camera drift | perspective | warped noise | Can endpoint constraints coexist with keystone motion? |
| Polar tunnel | analytic tunnel + swirl | composed warped noise | Can a closed map yield a closed gesture? |
| Fluid motion | curl field → RK4 advection | noise or image reference | How much field topology remains legible? |
| Layer-local motion | any map × spatial mask | latent `all` at low denoise | Can one region move while context stays stable? |
| 2.5D push | depth camera map | latent `holes` | Can H3 synthesize only disocclusions? |
| 2.5D + atmosphere | depth map ∘ advection | composed once | Does one resample preserve detail better? |
| Optical-flow steering | external RG flow | warped noise | Does measured motion steer a new generation? |
| Motion continuation | terminal flow field + envelope | low-denoise latent pass | Can velocity, not only appearance, cross a window? |
| Closed loop refinement | sine-loop map | latent pass, then temporal inpainting | Can spatial closure reduce work at the temporal seam? |
| Feedback texture | sequential image/latent warps | repeated low-denoise passes | When does accumulation become useful material? |
| Hybrid transition | affine ∘ polar ∘ imported flow | one map, one sample | Which scale of control survives best? |

## Sequential-pass families

The most useful comparison is not “one or many passes” in isolation. It is a
matrix in which the source, field, noise, and denoise remain recorded:

```text
A. normal H3 baseline
B. same seed + warped visual noise
C. baseline latent + deterministic warp + decode
D. baseline latent + warp + 0.15 denoise
E. baseline latent + warp + 0.35 denoise
F. composed map in one second pass
G. same constituent maps across separate second/third passes
H. decoded image warp used as Ref2VA video reference
I. depth warp with only invalid regions regenerated
J. spatial warp followed by temporal inpainting at the loop seam
```

For every comparison, save the pre-pass AV latent and the map hash. A second
pass that looks better but cannot identify its parent latent, seed, sigma
schedule, map, and mask is not a reproducible result.

## Conservative first matrix for the lab

The initial RTX 5090 sweep should keep the existing 124-frame 768×512 envelope:

| Variable | Values |
|---|---|
| control domain | image preview / visual noise / H3 latent |
| map | affine / curl advection / depth camera |
| normalized strength | 0.25 / 0.5 / 0.75 |
| second-pass denoise | 0.15 / 0.35 / 0.55 |
| map evaluation | composed once / sequential passes |
| padding | border / reflection |
| seed policy | same seed baseline / one fixed alternate seed |

Do not cross the full Cartesian product initially. Run the CPU/image previews,
reject geometrically bad maps, then promote only selected rows to the H3 matrix.

## Failure modes and measurements

| Failure | Likely cause | Measurement |
|---|---|---|
| motion disappears | model conditioning overwhelms weak noise bias | optical-flow agreement with intended map |
| tearing/folding | map Jacobian changes sign or magnitude too rapidly | determinant and gradient statistics |
| duplicated edges | depth forward splat collision | validity/disocclusion area and z-order |
| smeared texture | too many sequential bilinear samples | high-frequency energy against one-sample composite |
| identity drift | second-pass denoise too high | endpoint perceptual distance |
| temporal shimmer | map support and H3 token support disagree | flow acceleration and temporal spectrum |
| loop pulse | easing closes position but not velocity | first/last velocity and acceleration mismatch |
| latent collapse | warp leaves H3's learned manifold | decoder artifacts before and after denoise |

Execution is only Gate 1. Visual promotion additionally requires endpoint
preservation, field agreement, absence of local folding, temporal smoothness,
and comparison against the unmodified baseline.

## Relation to prior work

- The [official MiniMax H3 repository](https://github.com/MiniMax-AI/MiniMax-H3)
  defines the native audiovisual latent architecture used here.
- The [official ComfyUI H3 nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py)
  remain CAUCE's conditioning backend; their model implementation is not copied.
- [LatentWarp](https://arxiv.org/abs/2311.00353) and
  [flow-guided video diffusion](https://arxiv.org/abs/2311.15368) demonstrate
  related latent/flow alignment strategies for temporal consistency.
- [Deforum's animation implementation](https://github.com/deforum-art/deforum-stable-diffusion/blob/main/helpers/animation.py)
  and [PyTTI's transformation settings](https://pytti-tools.github.io/pytti-book/Settings.html)
  document earlier decoded-feedback and 2D/3D transform practices. CAUCE makes
  the distinction between a composed pullback and sequential feedback explicit.

These references motivate experiment families. They do not prove that a given
parameterization will transfer unchanged to MiniMax H3; every H3 claim remains
subject to the laboratory matrix.
