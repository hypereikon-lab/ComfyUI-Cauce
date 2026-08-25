# Validation

Validation has four independent gates.

## Gate 1 — deterministic code

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

Tests cover operation catalog/schema/ownership/artifact invariants, decoded
ranges, absolute 24→40 Hz token boundaries, window layout, allocation,
synchronized span extraction, latent-guide metadata, drift-safe append,
motion-map algebra, persistence paths, and the 19-node registry.

## Gate 2 — live schema

After a targeted install/restart, verify:

```text
GET /object_info/CauceH3InspectAVLatent
GET /object_info/CauceH3PlanAVWindow
GET /object_info/CauceH3AllocateAVWindow
GET /object_info/CauceH3ExtractAVSpan
GET /object_info/CauceH3AddAVSpanGuide
GET /object_info/CauceH3AppendAVSpan
GET /queue
```

Validate official H3 nodes against the same `/object_info` snapshot.

## Gate 3 — execution

Queue a minimal `continue.native_av` graph. Confirm:

- input latent reports the expected packed AV geometry;
- `22 + 119` resolves to a 141-frame target;
- the target contains 42 visual tokens and its globally aligned audio length;
- the tail guide is on the active positive-conditioning edge;
- only the 119-frame suffix is appended;
- the cumulative video/audio boundaries remain synchronized;
- outputs resolve through exact `/history/{prompt_id}` and `/view` routes.

This gate earns only `executes`.

## Gate 4 — visual objective

Compare continued motion, identity, geometry, texture, freeze/duplication, and
the decoded boundary at normal speed and frame by frame. Record the complete
graph and runtime parameters. Only an inspected result can be called `visually
accepted`.
