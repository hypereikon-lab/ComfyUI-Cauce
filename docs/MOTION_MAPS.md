# Motion-reference maps

CAUCE motion maps describe where each output pixel samples its source:

```text
source_coordinate = map(output_coordinate, time)
```

This inverse/pullback form matches PyTorch `grid_sample` and makes composition
exact at the coordinate level. Compose maps first and sample the image once to
avoid repeated interpolation loss.

## Available construction

- affine translation, scale, rotation, and pivots;
- projective corner pinning;
- analytic swirl, pinch, wave, radial wave, tunnel, and kaleidoscope maps;
- arbitrary RG displacement imports;
- depth-based camera reprojection with a disocclusion-validity field;
- uniform, rotational, radial, vortex, curl-like, and wave vector fields;
- Euler, RK2, and RK4 advection integration;
- temporal envelopes and arbitrary spatial masks;
- map composition and image sampling.

## Role in H3 workflows

The output is reference media, not an H3 latent intervention. A generated grid,
primitive animation, depth push, or simulation can be given to official Ref2VA
or `MiniMaxH3AddGuide`. The model decides how to translate that visible motion
into a generated scene.

Validity masks identify out-of-domain samples and depth disocclusions. They are
data for downstream compositing or guide design; they are not automatically an
H3 denoise mask.

## Reproducibility

Every map contains geometry, fps, operation parameters, normalized-coordinate
convention, validity, and a tensor hash. Reports should be stored with the
workflow when a reference is used in production.
