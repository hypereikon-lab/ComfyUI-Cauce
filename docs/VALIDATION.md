# Validation

Validation has four independent gates.

## Gate 1 — deterministic code

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

The ordinary suite can skip tensor-dependent cases when the selected Python
does not provide NumPy. That reduced result is useful for contract-only checks
but is not complete validation. Before release, use a pre-existing runtime that
already provides NumPy and require zero skips:

```bash
python3 tools/verify_full.py
```

`verify_full.py` fails closed when NumPy is absent or any test is skipped. Do
not install or upgrade the laboratory GPU environment merely to satisfy this
offline gate.

Tests cover operation catalog/schema/ownership/artifact invariants, decoded
ranges, absolute 24→40 Hz token boundaries, window layout, allocation,
synchronized span extraction, latent-guide metadata, drift-safe append,
reversible state split, exact span placement/rebase, independent continuous AV
denoise masks, exact interval replacement, mask cleanup, official target/
guide/reference preprocessing rules, conditioning inspection,
persistence paths, the 18-node
registry, complete offline topology coverage, and every topology port that
touches a CAUCE node.

## Gate 2 — live schema

After a targeted install/restart, verify:

```text
GET /object_info/CauceH3InspectAVLatent
GET /object_info/CauceH3PlanAVWindow
GET /object_info/CauceH3AllocateAVWindow
GET /object_info/CauceH3ExtractAVSpan
GET /object_info/CauceH3AddAVSpanGuide
GET /object_info/CauceH3AppendAVSpan
GET /object_info/CauceH3SplitAVLatent
GET /object_info/CauceH3PlaceAVSpan
GET /object_info/CauceH3SetAVDenoiseInterval
GET /object_info/CauceH3ReplaceAVSpan
GET /object_info/CauceH3ClearAVDenoiseMask
GET /object_info/CauceH3ResolveTargetShape
GET /object_info/CauceH3PrepareGuideClip
GET /object_info/CauceH3PrepareReferenceClip
GET /object_info/CauceH3InspectConditioning
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

Masked continuation/completion is a separate execution gate. Confirm the live
core includes H3 per-token mask support, the nested mask shapes match both AV
streams, placed spans preserve their exact audio phase, and the consumed mask
is absent from the persisted result. Passing the older keyframe smoke does not
establish these masked variants.

## Gate 4 — visual objective

Compare continued motion, identity, geometry, texture, freeze/duplication, and
the decoded boundary at normal speed and frame by frame. Record the complete
graph and runtime parameters. Only an inspected result can be called `visually
accepted`.
