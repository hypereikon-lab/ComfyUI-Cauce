# Laboratory validation matrix

Local unit tests establish contracts, clocks, generated workflow integrity, and
runner materialization; they cannot establish model quality. The six visual
graphs load without unknown nodes in the live lab frontend, and both API
templates match the live `/object_info` signatures. Validate the following on
the RTX 5090 before promoting advanced workflows to production defaults.

## Gate 1 — load and compatibility

- CAUCE imports with no startup exception.
- All 36 nodes appear under `CAUCE/*`.
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
| Ref2VA audio/video | Both media streams resolve |
| Timed image guide | Exact requested `frame_idx` |
| Timed AV guide | Video/audio begins together |

## Gate 3 — clock impulses

Build visible one-frame flashes and one-sample/click events at:

- window start;
- frame 1;
- visual support boundaries 1, 5, 9, 13, 17, 18, 22;
- accepted-range start;
- final visible frame.

Compare compiled metadata, generated mask tensors, decoded frames, and final
sample indices. No cumulative float conversion is allowed.

## Gate 4 — masks

- One static spatial mask.
- One moving frame-by-frame mask.
- One-frame mark located between apparent `/4` sampling points.
- Video preserve / audio generate.
- Video generate / audio preserve.
- Soft values 0.25, 0.5, and 0.75.
- Existing mask intersected with continuation head in both node orders.

Checksum preserved source regions where a deterministic identity is expected.
Listen to every audio boundary; do not rely only on waveform metrics.

## Gate 5 — continuation

For `39`, `90`, and `141` frame contexts where the target is long enough:

- Compare latent tail vs decoded/re-encoded context.
- Confirm token-cycle validation.
- Confirm copied target head is byte-identical before sampling.
- Confirm mask values are zero through the whole copied head.
- Measure visible seam after accepted trim.
- Measure audio cross-correlation and click energy around the join.
- Reroll clip 2 and verify slot 1 remains the parent while slot 2 is replaced.

The production default should remain 39 only after these tests confirm it on
the current model/runtime.

The first live `mask_only` run completed but failed the visible-seam gate. The
next required comparison is the shipped hybrid: identical 39-frame latent mask
plus the decoded last accepted frame connected as native FL2VA `first_frame`.
Completion alone is not a pass.

For the two-ended bridge, confirm that both copied endpoint tensors are
byte-identical before sampling, neither context overlaps the middle, and the
right endpoint enters at the intended frame/audio tick.

## Gate 6 — decode domain

Decode every phase-safe parent independently, accept its decoded range, then
assemble those visible spans. Inspect the five frames and 250 ms on each side
of every seam. Confirm that no workflow attempts to concatenate independent H3
latents before causal VAE decode.

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
