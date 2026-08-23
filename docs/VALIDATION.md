# Laboratory validation matrix

Local unit tests establish contracts, clocks, generated workflow integrity, and
runner materialization; they cannot establish model quality. The seven visual
graphs load without unknown nodes in the live lab frontend, and both API
templates match the live `/object_info` signatures. Validate the following on
the RTX 5090 before promoting advanced workflows to production defaults.

## Gate 1 — load and compatibility

- CAUCE imports with no startup exception.
- All 41 nodes appear under `CAUCE/*`.
- Preflight identifies the existing models without modifying files.
- FL2VA and Ref2VA wrappers locate current official Comfy H3 classes.
- No CUDA, PyTorch, ComfyUI, Manager, or model update is triggered.

## Gate 2 — official conditioning matrix

Use a fixed seed and 124 frames.

| Case | Expected |
|---|---|
| FL2VA no image | T2VA completes |
| FL2VA first | First image anchors frame 0 |
| FL2VA last | Last image anchors frame 123 |
| FL2VA first+last | Both endpoints respected |
| Ref2VA image | `<Picture 1>` resolves |
| Ref2VA video | Motion reference resolves |
| Timed image guide | Exact requested `frame_idx` |

## Gate 3 — clock impulses

Build visible one-frame flashes and fixed-master timeline markers at:

- window start;
- frame 1;
- visual support boundaries 1, 5, 9, 13, 17, 18, 22;
- accepted-range start;
- final visible frame.

Compare compiled metadata, generated mask tensors, decoded frames, and master
sample indices. The master is never passed through AudioVAE and no cumulative
float conversion is allowed.

## Gate 4 — masks

- One static spatial mask.
- One moving frame-by-frame mask.
- One-frame mark located between apparent `/4` sampling points.
- Soft values 0.25, 0.5, and 0.75.
- Existing mask intersected with continuation head in both node orders.

Checksum preserved source regions where a deterministic identity is expected.

## Gate 5 — continuation

For `22`, `39`, and `56` frame visual contexts where the target is long enough:

- Compare latent tail vs decoded/re-encoded context.
- Confirm token-cycle validation.
- Confirm copied target head is byte-identical before sampling.
- Confirm mask values are zero through the whole copied head.
- Measure visible seam after accepted trim.
- Reroll clip 2 and verify slot 1 remains the parent while slot 2 is replaced.

The production default should remain 39 only after these tests confirm it on
the current model/runtime.

The first live `mask_only` run completed but failed the visible-seam gate. The
shipped hybrid—identical 39-frame latent mask plus the decoded last accepted
frame connected as native FL2VA `first_frame`—passed the first visual join
comparison. Repeat it over heterogeneous material before making it a production
preset. Completion alone is not a pass.

## Gate 6 — decode domain

Decode every phase-safe parent independently, accept its decoded range, then
assemble those visible spans. Inspect the five frames on each side of every
seam. Confirm that no workflow attempts to concatenate independent H3 latents
before causal VAE decode.

## Gate 7 — resource profile

For every profile record:

- peak VRAM;
- peak host RAM;
- first-run and warm-run duration;
- output and temporary disk growth;
- behaviour after several sequential windows;
- behaviour after browser/tunnel disconnect while Comfy keeps running.

Profiles are promoted from “candidate” to “verified” only after these results
are committed as receipts and documented measurements.

## Gate 8 — sequence recovery

- Materialize runner templates without submission.
- Run one window against localhost.
- Interrupt after a completed window and resume.
- Interrupt while a later window is running and inspect Comfy history.
- Confirm complete windows are skipped.
- Confirm failed windows remain explicitly failed.
- Confirm remote mode fails closed without a valid Cloudflare service token.

## Gate 9 — localized temporal inpainting

- Reject either input unless its reported frame rate is exactly 24 fps.
- Reject either input when it is shorter than the requested context.
- Confirm the production plan is 124 working frames, cut 62, and token-aligned
  sampling/acceptance `[26,98)` for the three-second request.
- Confirm guide clips are exactly `[4,26)` and `[98,120)`, 22 frames each.
- On ComfyUI v0.33.1, confirm temporal inpainting fails before sampling and reports the
  missing official AddGuide/per-token mask capabilities.
- On the updated core, confirm both guide clips appear as native H3 keyframes.
- Confirm the VAE-encoded composite and FL2VA target have identical shapes.
- Confirm `sampling_support` is binary and has no nonzero values outside
  `[sampling_start,sampling_end)`.
- Confirm `hard_acceptance` is binary and exactly matches the replacement range.
- Confirm `output_opacity` is independent from generation strength and also has
  exact zero endpoints.
- Confirm `cover` and `majority` projections both remain binary after causal
  token projection.
- Confirm the H3 internal audio mask is all zero, the master audio is never
  encoded, and no generated audio is accepted.
- Confirm the splice replaces 72 frames and returns exactly `len(A) + len(B)`.
- Checksum every frame outside the replacement range.
- Compare position, velocity, acceleration, optical flow, and perceptual seam
  energy over five frames on both patch edges and around the original cut.
- Compare decoded blend widths of 2, 4 and 6 frames while keeping the 72-frame
  accepted repair, guides, source, prompt, seed and profile fixed.
- Compare 22- and 39-frame guide clips only after the 72-frame interval passes.
- Verify graph execution synthetically, then promote only after the exact real
  gesture pairs that rejected v1 pass blind visual inspection.
