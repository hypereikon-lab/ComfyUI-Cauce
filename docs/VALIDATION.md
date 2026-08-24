# Validation

Validation is layered. Local unit tests establish deterministic contracts;
live H3 runs establish runtime compatibility; visual and measured gates establish
whether an operation works for its intended purpose.

## Gate 1 — package

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

Confirm 24 registered nodes, matching display mappings, 19 stable categories,
and exactly five `CAUCE/Research` nodes.

## Gate 2 — H3 runtime

Verify:

- current CAUCE commit is installed;
- representative `/object_info` requests return the expected schemas;
- official H3 model nodes are available;
- native H3 denoise-mask hooks from ComfyUI PR `#15375` pass the CAUCE
  capability probe, including clean-latent reinjection and model forward mask
  arguments;
- queue is idle before test submission.

## Gate 3 — temporal geometry

Unit-test:

- legal `17k+5` working lengths;
- token-aligned repair boundaries;
- guide ranges outside the generated interval;
- sampling mask binary values;
- exact duration-preserving splice ranges;
- unchanged frames outside the patch.

Run the matched W0/W1/W2 comparison. W1 isolates native masked sampling; W2
adds only the two official guide clips. Do not accept W2 without proving an
improvement over W1.

## Gate 4 — continuation

Confirm:

- source latent is a complete H3 run;
- selected tail begins on the correct token-cycle phase;
- source and target latent geometry match;
- copied context is zero-masked;
- structural audio is zero-masked;
- resolved parent endpoint is phase-safe;
- decoded accepted range is exact.

## Gate 5 — motion maps

Pure mathematics:

- identity is exact;
- translation direction matches pullback convention;
- loop envelopes close;
- composition matches sequential coordinate sampling;
- depth reprojection begins at identity and marks disocclusion;
- hashes are deterministic.

Live visual checks:

- output follows the requested field direction;
- validity behaves at boundaries;
- map composition avoids unnecessary repeated sampling.

## Gate 6 — persistence

Confirm:

- path remains inside ComfyUI output;
- write is atomic;
- safetensors metadata contains the exact current format marker;
- visual and structural-audio tensors round-trip with identical shapes and
  values;
- indexed/latest resolution is deterministic.

## Gate 7 — Research

Every active Research run includes:

1. official baseline;
2. identity or zero control;
3. one small active intervention;
4. identical seed, source, prompt, model, sampler, and decode;
5. structural checks;
6. operation-specific measurement;
7. visual inspection.

For latent motion, measure optical flow or registration tied to the requested
direction. Do not promote based only on output divergence.

## Gate 8 — resource envelope

Record resolution, frame count, model variant, steps, peak VRAM when available,
system-memory pressure, runtime, and storage impact. Do not infer that a clean
short run generalizes to a longer working domain.
