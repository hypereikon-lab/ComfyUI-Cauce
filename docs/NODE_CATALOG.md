# Node catalog

## CAUCE/Assembly

### `CauceAcceptDecodedRange`

Returns an exact `[start_frame, start_frame + frame_count)` IMAGE slice and its
count.

## CAUCE/Native H3

### `CauceBuildH3GuideBridge`

Inputs two matching 24 fps IMAGE batches. Returns the left tail guide, right
head guide, a hashed bridge plan, target frame count, right-guide frame index,
and JSON report. Defaults: 22-frame guides in a 124-frame target.

### `CauceApplyH3GuideBridge`

Consumes the original sources, decoded H3 target, and bridge plan. Discards the
two conditioning intervals, returns the complete joined video and isolated
generated center, and reports exact accepted ranges.

## CAUCE/Motion Maps

| Node | Operation |
| --- | --- |
| `CauceAffineMotionMap` | translation, scale, rotation, pivot, easing |
| `CauceAnalyticMotionMap` | swirl, pinch, wave, radial wave, tunnel, kaleidoscope |
| `CaucePerspectiveMotionMap` | four-corner projective pullback |
| `CauceDisplacementMotionMap` | import arbitrary RG displacement data |
| `CauceModulateMotionMap` | temporal envelope and optional spatial mask |
| `CauceVectorField` | uniform, rotation, radial, vortex, curl, wave fields |
| `CauceIntegrateAdvection` | Euler, RK2, or RK4 field integration |
| `CauceDepthCameraMotionMap` | depth-based camera reprojection and validity |
| `CauceComposeMotionMaps` | compose maps before media sampling |
| `CauceWarpImage` | sample IMAGE media through a map |

## CAUCE/Persistence

### `CauceSaveAVLatent`

Atomically saves the visual and structural-audio tensors of one H3 latent to an
indexed `safetensors` artifact inside the ComfyUI output root.

### `CauceLoadAVLatent`

Loads an explicit or latest indexed CAUCE H3 audiovisual latent from the output
root. Paths are checked against traversal outside that root.
