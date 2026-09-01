# Canonical contract bundle

`operations/contract-bundle.json` is generated from the authoritative CAUCE
sources. It contains:

- current operation versions and hashes;
- exact historical `(operation, version, hash, source commit)` tuples;
- every topology key and structural signature;
- the complete graph-archetype catalog;
- every registered CAUCE class type, display name, and category.

It contains no production media, runtime credentials, concrete project
bindings, model filenames, or acceptance verdicts. Those remain downstream.

Regenerate after a contract or registry change:

```bash
uv run --no-project --with numpy -- python tools/export_contract_bundle.py
```

CI and downstream consumers use the fail-closed check:

```bash
uv run --no-project --with numpy -- python tools/export_contract_bundle.py --check
```

Historical contracts are full immutable specs under `operations/history`, not
version ranges and not arbitrary syntactically valid hashes. A past execution
is valid only when its exact tuple exists in that archive.
