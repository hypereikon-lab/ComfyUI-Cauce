# CAUCE engineering and laboratory runbook

This file applies to the complete repository. Read it before changing code,
deploying the node pack, or operating the laboratory ComfyUI instance. User
instructions remain authoritative.

## 1. Mission

CAUCE owns only operations for which it adds an explicit mathematical,
reproducibility, or safety contract:

- deterministic decoded-media selection and assembly;
- preparation of decoded guide clips for official H3 conditioning nodes;
- generic image-space coordinate maps and reference-media warps;
- bounded persistence of packed H3 audiovisual latents.

Official ComfyUI/MiniMax nodes own H3 conditioning, latent construction,
sampling, and decoding. Compose those nodes directly in workflows.

CAUCE does not own a second UI, production scheduling, remote authentication,
model management, semantic image descriptions, generative audio, training,
LoRAs, acceleration, or streaming. The production soundtrack is fixed and
stays outside H3 conditioning.

## 2. Sources of truth

Use this order:

1. `git status --short`, current branch, and recent commits;
2. this file;
3. `README.md` and `docs/INDEX.md`;
4. architecture, workflow, node, and validation documents;
5. code and tests;
6. live `/object_info` and actual outputs for live-runtime claims.

Do not reconstruct state from conversational memory. When a user supplies a
Codex JSONL, search it narrowly for exact tool calls, commit hashes, routes,
workflow signatures, and outputs. Never print cookies, Cloudflare credentials,
or unrelated conversation content.

Prefer an official node plus graph composition over a wrapper that merely
renames an upstream operation or provides defaults.

## 3. Repository invariants

Dependency direction:

```text
ComfyUI graph
  -> cauce_nodes bindings
      -> cauce operations
          -> NumPy/PyTorch or narrow official runtime adapters
```

The registry is assembled from:

```text
cauce_nodes/assembly.py
cauce_nodes/two_sided_window.py
cauce_nodes/motion.py
cauce_nodes/persistence.py
```

Core operations must not know about browser tabs, Cloudflare, Manager, or a
production project. Bindings must remain thin. The package version in
`pyproject.toml` and `cauce.__version__` must match.

CAUCE ships no executable workflow JSON until both the browser graph and API
prompt have been validated against live node schemas. Documentation may state a
graph recipe without pretending it is an import-tested artifact.

Preserve dirty-worktree changes. Do not use destructive checkout, reset, broad
cleanup, or force-push.

Local verification:

```bash
python3 -m compileall -q cauce cauce_nodes
python3 -m unittest discover -s tests -v
git diff --check
```

Do not install or upgrade global GPU packages to satisfy tests.

## 4. H3-native rules

- Production video is 24 fps.
- Legal H3 frame counts follow `17k + 5`.
- Prefer trained-range targets from 124 through 362 frames.
- H3 state contains visual and structural-audio streams.
- Use `cauce/timebase.py` for frame/token arithmetic.
- Use official `MiniMaxH3ImageToVideo`, `MiniMaxH3AddGuide`, Ref2VA/FL2VA
  conditioning, guider, sampler, and decoder nodes directly.
- Treat upstream implementations as dependencies or graph components; do not
  copy their internals into CAUCE without a separate reason and validation.

A CAUCE two-sided window plan selects guide media and accepted output ranges. It never
claims to determine the model's internal transition. Every window report must
retain `quality_status = requires_visual_validation`.

## 5. Evidence language

Use these states precisely:

- `unit-validated`: deterministic code passed local tests;
- `schema-validated`: a graph matches current live `/object_info`;
- `executes`: ComfyUI completed it without runtime error;
- `visually accepted`: the requested behavior was inspected and accepted;
- `rejected`: it executed but did not satisfy the visual objective;
- `blocked`: an external layer prevents the next check.

Never promote an idea because it changed pixels or completed the queue. A
metric must measure the requested property. A failed visual result is not a
capability and should not appear in current-state documentation.

## 6. Workflow construction

Before creating a graph, inspect live schemas for every nontrivial node. Keep
browser-format workflow JSON and API prompt JSON distinct. Record:

```text
workflow label
source filenames and ordering
resolution, fps, frame count
official H3 nodes and model variant
prompt, seed, sampler, scheduler, steps
CAUCE plan hash
output prefix
```

For the H3 two-sided guide window:

1. normalize both sources to matching geometry and 24 fps;
2. run `CaucePrepareH3TwoSidedGuideWindow`;
3. create a fresh official H3 target of the returned length;
4. chain two official `MiniMaxH3AddGuide` nodes using returned clips and frame
   indices;
5. sample and decode with the normal official graph;
6. run `CauceAssembleH3TwoSidedGuideWindow`;
7. inspect the complete result and the isolated accepted generated range.

Do not describe this graph as successful before step 7.

## 7. Laboratory topology

The usual laboratory origin is:

```text
https://comfy.hypereikon.online/
```

Physical envelope:

```text
Windows portable ComfyUI
RTX 5090, 32 GB VRAM
64 GB RAM
Cloudflare Tunnel -> http://localhost:8188
Cloudflare Access in front of the hostname
```

The tunnel exposes the ComfyUI HTTP origin. It is not SSH, PowerShell, CMD,
RDP, arbitrary filesystem access, or a power controller. The hostname works
only while the tower, network, tunnel service, and ComfyUI process are healthy.

Use the authenticated in-app browser for live operations. Do not extract
cookies or embed credentials in commands, code, graphs, or documents.

## 8. Targeted deployment

Installing/updating custom code or restarting ComfyUI is a consequential live
action. Confirm it with the user at action time even when the code work itself
was already authorized.

Before deployment:

- intended changes are committed and pushed to the branch the installed clone
  follows;
- the queue is idle;
- the authenticated browser is on the laboratory origin;
- no core/runtime/model update is included.

Manager sequence:

```text
POST /manager/queue/reset
POST /manager/queue/update
POST /manager/queue/start
GET  /manager/queue/status
GET  /customnode/installed
```

Target only `ComfyUI-Cauce`. Python changes then require:

```text
POST /manager/reboot
```

A brief 502 is expected while ComfyUI restarts. Afterwards verify:

```text
GET /customnode/installed
GET /object_info/CaucePrepareH3TwoSidedGuideWindow
GET /object_info/CauceAssembleH3TwoSidedGuideWindow
GET /queue
```

Do not update CUDA, PyTorch, drivers, models, ComfyUI core, or unrelated nodes.
Do not reboot the physical tower.

## 9. Workflow-tab hygiene

Distinguish the browser page tab, the workflow tabs inside ComfyUI, and the
automation handle controlling the page. Before live work, keep a ledger:

```text
label | owner | purpose | signature | output prefix | state
```

Pre-existing or unidentified workflows are user-owned. Close only workflows
created by the current agent or explicitly identified by the user. Use at most
one active graph and one matched comparison. Pasting JSON can open a new
workflow instead of replacing the canvas; reconcile the ledger immediately.

Before queueing, verify the active workflow, source files, parameters, output
prefix, validation errors, and queue state. After completion, resolve exact
outputs from `/history` and authenticated `/view` routes.

## 10. Recovery

| Observation | Interpretation | Action |
| --- | --- | --- |
| brief 502 after Manager reboot | Python process restarting | wait and reload |
| persistent 502 | origin reachable but ComfyUI may be down | retry once, then request operator start |
| hostname unreachable | tower/network/tunnel layer | request smallest physical check |
| node absent from `/object_info` | import or dependency failure | inspect logs; do not mutate GPU stack |
| Manager reports old commit | installed clone/update mismatch | verify remote ref, then targeted update |
| queue unexpectedly busy | active or stalled job | inspect queue/history before restart |

Retry only the narrow failed layer. For long inferences poll every 20–40
seconds, communicate at least once per minute, and do not duplicate a job based
on an ambiguous progress display.

## 11. Finish checklist

- registry imports without ComfyUI;
- all local tests pass;
- `git diff --check` passes;
- documentation matches the current registry;
- no output is called successful without visual inspection;
- live deployment, if requested and confirmed, reports the intended commit and
  exposes both H3 two-sided guide-window nodes through `/object_info`.
