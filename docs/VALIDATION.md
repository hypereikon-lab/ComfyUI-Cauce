# Validation

Validation has four independent gates.

## Gate 1 — deterministic code

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

Tests cover exact ranges, guide-window geometry and hashing, source extraction,
accepted-range assembly, motion-map algebra, H3 timebase math, persistence paths,
and the 15-node registry.

Passing this gate does not evaluate H3 output quality.

## Gate 2 — live schema

After a targeted install/restart:

```text
GET /object_info/CaucePrepareH3TwoSidedGuideWindow
GET /object_info/CauceAssembleH3TwoSidedGuideWindow
GET /queue
```

Validate every official H3 node/socket against live `/object_info`. Browser
workflow JSON and API prompt JSON are separate schemas.

## Gate 3 — execution

Queue one graph. Confirm:

- no validation or import errors;
- expected target frame count and resolution;
- two guide clips placed at the planned indices;
- isolated accepted generated range has the planned count;
- joined output equals `len(A) + len(accepted) + len(B)`;
- output routes resolve through `/history` and `/view`.

This gate earns only `executes`.

## Gate 4 — visual objective

Inspect:

- outgoing motion from A into the generated interval;
- incoming motion from the generated interval into B;
- identity, geometry, and texture stability;
- duplicate or frozen guide frames;
- discontinuity at both accepted-range boundaries;
- normal-speed rhythm against the fixed soundtrack timeslot.

Record prompt, seed, model, quantization, resolution, frame count, guide length,
sampler, scheduler, steps, plan hash, and exact output path. Only an inspected
result can be called `visually accepted`.
