# Operations guide

## Prepare a native bridge

1. Load source A and B as IMAGE batches.
2. Normalize both to the same width, height, channels, and 24 fps.
3. Add `CauceBuildH3GuideBridge`.
4. Start with `guide_frames = 22` and `target_frames = 124`.
5. Record the returned plan hash and inspect both guide batches.

## Build the official H3 section

1. Use the official H3 node to create a fresh target with the returned frame
   count and matching resolution.
2. Chain `MiniMaxH3AddGuide` for the left guide at frame 0.
3. Chain a second `MiniMaxH3AddGuide` for the right guide at the returned frame
   index.
4. Wire the normal official guider, sampler, and decoder.
5. Keep all inference parameters explicit.

## Assemble and review

1. Feed the decoded target, both original sources, and plan to
   `CauceApplyH3GuideBridge`.
2. Save both `generated_bridge` and `joined_images` with distinct prefixes.
3. Verify frame count and fps.
4. Watch the A-to-center and center-to-B boundaries at normal speed and frame by
   frame.
5. Mark the run `visually accepted` or `rejected`; queue completion alone means
   only `executes`.

## Controlled iteration

Change one variable per comparison. Recommended order:

1. prompt wording;
2. guide length (22, then 39, then 56 frames);
3. target length;
4. sampler parameters.

Keep source media, model, quantization, resolution, frame count, seed, sampler,
scheduler, steps, decode, and output handling fixed whenever they are not the
active variable.

## Motion-reference workflow

Build or import a coordinate map, compose all desired maps, warp an image or
primitive batch once, then pass the resulting video as an ordinary H3 reference
through official nodes. Review the reference itself before spending inference
time.

## Persistence

Use `CauceSaveAVLatent` only for packed H3 latents that must survive a graph or
process boundary. Artifacts live below the ComfyUI output root. Use indexed
filenames; resolve explicit versions for reproducible production runs and
`latest` only during exploration.
