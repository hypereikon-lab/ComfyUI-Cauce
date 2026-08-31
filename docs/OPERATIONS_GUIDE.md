# Operations guide

## Compose `continue.native_av`

1. Keep the completed packed H3 AV latent before decoding.
2. Inspect it with `CauceH3InspectAVLatent`.
3. Plan an absolute window with `CauceH3PlanAVWindow`.
4. Allocate that target with `CauceH3AllocateAVWindow`.
5. Extract the previous tail using `CauceH3ExtractAVSpan`.
6. Build normal official H3 positive conditioning for `window_frames`, without
   first/last images.
7. Choose one explicit overlap transport:
   - keyframe: insert the span with `CauceH3AddAVSpanGuide` at frame zero;
   - mask: place it with `CauceH3PlaceAVSpan`, then mask only the suffix with
     `CauceH3SetAVDenoiseInterval`.
8. Optionally add one decoded future image/clip through official
   `MiniMaxH3AddGuide` in the masked future-guide variant.
9. Run the ordinary official guider and sampler against the allocated target.
10. Extract only the new suffix from the sampled window using the returned
   timeline origin and exact overlap/extension values.
11. Append that suffix with `CauceH3AppendAVSpan`.
12. Save the cumulative latent only when persistence is useful; decode normal
   visual outputs separately.

Keep prompt, model, resolution, seed, sampler, scheduler, steps, overlap,
extension, and output prefix explicit in the workflow/run receipt. If an
experiment deliberately inserts the native `MiniMaxH3SigmaShift` patch, record
both shifts and compare it against the direct-model control; it is not a
canonical default.

## Compose `complete.native_av`

1. Create or load one complete target lattice with explicit absolute origin.
2. Extract every known native context as `CAUCE_H3_AV_SPAN`.
3. Place each known span with `CauceH3PlaceAVSpan` at its exact target frame.
4. Fail if placement reports incompatible visual geometry or audio phase; do
   not round or approximate the 40 Hz boundary.
5. Mark only the unknown interval with `CauceH3SetAVDenoiseInterval`. Start with
   hard `inside=1`, `outside=0`, then test ramps as a separate comparison.
6. Build the ordinary official H3 prompt conditioning for the same complete
   target shape; optional decoded anchors use official `MiniMaxH3AddGuide`.
7. Sample normally and extract the generated interval explicitly.
8. For local replacement, pass that span to `CauceH3ReplaceAVSpan` on the
   original cumulative state.
9. Clear consumed mask metadata with `CauceH3ClearAVDenoiseMask` before save.
10. Inspect every boundary before assigning visual evidence.

Prefix completion, two-sided infill, local replacement, and two-source
connection are separate graph variants. They share the same low-level nodes;
there is no hidden “completion” preset inside CAUCE.

## Compose `rollback.native_av`

1. Load the exact cumulative native AV artifact.
2. Choose a legal prefix boundary.
3. Split with `CauceH3SplitAVLatent`.
4. Persist the prefix under a new branch/checkpoint name.
5. Retain the synchronized suffix when exact reconstruction may be needed.
6. Start any alternate continuation from the prefix; never overwrite the
   source checkpoint implicitly.

## Controlled iteration

Change one variable per comparison. Start with the characterized continuation
layout `22 + 119 = 141`; then vary overlap transport, prompt, overlap,
extension, mask boundary, or sampling parameters independently. A hard mask and
a continuous mask are different experiments and require separate receipts.

## Persistence

Use `CauceSaveAVLatent` only for packed H3 latents that must survive a graph or
process boundary. Use explicit indexed paths for reproducible runs and `latest`
only during exploration.
