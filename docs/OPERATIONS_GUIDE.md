# Operations guide

## Compose `continue.native_av`

1. Keep the completed packed H3 AV latent before decoding.
2. Inspect it with `CauceH3InspectAVLatent`.
3. Plan an absolute window with `CauceH3PlanAVWindow`.
4. Allocate that target with `CauceH3AllocateAVWindow`.
5. Extract the previous tail using `CauceH3ExtractAVSpan`.
6. Build normal official H3 positive conditioning for `window_frames`, without
   first/last images.
7. Insert the span with `CauceH3AddAVSpanGuide` at frame zero.
8. Run the ordinary official guider and sampler against the allocated target.
9. Extract only the new suffix from the sampled window using the returned
   timeline origin and exact overlap/extension values.
10. Append that suffix with `CauceH3AppendAVSpan`.
11. Save the cumulative latent only when persistence is useful; decode normal
   visual outputs separately.

Keep prompt, model, resolution, sigma shifts, seed, sampler, scheduler, steps,
overlap, extension, and output prefix explicit in the workflow/run receipt.

## Compose `connect.two_sided_guides`

1. Normalize A and B to matching geometry and 24 fps.
2. Select A's tail and B's head with `CauceAcceptDecodedRange`.
3. Create a fresh official H3 target.
4. Add the tail at frame zero and the head at `target_frames-guide_frames` with
   two official `MiniMaxH3AddGuide` nodes.
5. Sample and decode normally.
6. Select only `[guide_frames, target_frames-guide_frames)` from the target.
7. Chain vanilla `ImageBatch` nodes to assemble `A + accepted + B`.
8. Inspect both boundaries before assigning visual evidence.

## Controlled iteration

Change one variable per comparison. Start with the characterized continuation
layout `22 + 119 = 141`; then vary prompt, overlap, extension, or sampling
parameters independently.

## Persistence

Use `CauceSaveAVLatent` only for packed H3 latents that must survive a graph or
process boundary. Use explicit indexed paths for reproducible runs and `latest`
only during exploration.
