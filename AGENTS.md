# CAUCE engineering and laboratory runbook

This file applies to the complete repository. Read it before changing code,
deploying the node pack, or operating the laboratory ComfyUI instance. User
instructions remain authoritative.

## 1. Mission

CAUCE has two deliberately separate surfaces:

1. low-level custom nodes for operations that add an explicit mathematical,
   reproducibility, or safety contract;
2. named operation contracts and, once paired live validation exists, reusable
   UI/API graph templates composed from CAUCE, official H3, and vanilla nodes.

The low-level surface owns:

- deterministic decoded-media range selection;
- visible H3 target, guide-clip, and reference-clip planning;
- read-only inspection of active H3 conditioning structure;
- absolute H3 AV frame/video-token/audio-token layouts;
- synchronized packed-AV span extraction, placement, continuous denoise masks,
  exact replacement, native guide insertion, split, and append;
- bounded persistence of packed H3 audiovisual latents.

Official ComfyUI/MiniMax nodes own model loading, decoded-media conditioning,
sampling, and decoding. CAUCE may create or structurally transform a packed AV
target only when the operation has an explicit clock/layout contract. Compose
all operations explicitly in workflows.

A named CAUCE operation is a typed graph-level function over media or H3 state.
It does not imply that CAUCE owns every node in the graph. Operation contracts
must declare node ownership and artifact/evidence state explicitly. Never wrap
an official graph in a monolithic custom node merely to claim it as CAUCE.

CAUCE does not own a second UI, production scheduling, remote authentication,
model management, semantic image descriptions, generative audio, training,
LoRAs, acceleration, or streaming. The production soundtrack is fixed and
stays outside H3 conditioning.

## 2. Sources of truth

Use this order:

1. `git status --short`, current branch, and recent commits;
2. this file;
3. `README.md`, `docs/INDEX.md`, and `operations/catalog.json`;
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
cauce_nodes/av_latent.py
cauce_nodes/planning.py
cauce_nodes/persistence.py
```

Core operations must not know about browser tabs, Cloudflare, Manager, or a
production project. Bindings must remain thin. The package version in
`pyproject.toml` and `cauce.__version__` must match.

CAUCE ships no executable workflow JSON until both the browser graph and API
prompt have been paired, validated against live node schemas, and recorded in
the operation contract. A contract or evidence record may describe an exact
graph without masquerading as an importable template.

`operations/topologies/` contains non-executable offline design dossiers. Keep
them symbolic and in `offline-draft` state. Validate all CAUCE-owned ports
against the actual registry locally; validate official/vanilla ports against a
fresh live `/object_info` capture during materialization. Never promote a
topology file by renaming it into workflow JSON.

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
- Never calculate the 40 Hz stream from relative duration alone when a latent
  has a nonzero global frame origin. Use absolute frame boundaries.
- A subrange of an AV latent is `CAUCE_H3_AV_SPAN`, not a standalone `LATENT`,
  until an explicit append/allocation operation establishes its timeline.
- Use official `MiniMaxH3ImageToVideo`, `MiniMaxH3AddGuide`, Ref2VA/FL2VA
  conditioning, guider, sampler, and decoder nodes directly.
- Use inspected upstream implementations as references and test oracles. Do not
  create a runtime dependency when the useful mechanism is a small deterministic
  primitive already within CAUCE's ownership boundary.
- Do not add monolithic workflow-intent nodes such as “continue,” “bridge,” or
  “two-sided transition.” Expose layout/span/range operations; let the graph
  assign intent.

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
CAUCE layout/span hashes
output prefix
```

For native H3 AV continuation:

1. inspect the completed cumulative AV latent;
2. plan an absolute AV window;
3. allocate the target from that layout;
4. extract the prior synchronized tail as `CAUCE_H3_AV_SPAN`;
5. place it on the active official positive-conditioning edge;
6. sample with the ordinary official graph;
7. extract only the explicit new suffix from the sampled window;
8. append that globally contiguous span;
9. decode and inspect the result.

For native completion or replacement, place known synchronized spans into one
complete target lattice, set explicit per-stream denoise intervals, sample via
the ordinary official path, then clear mask metadata before persistence. A
prefix rebase is legal only when both the H3 visual grid and the absolute 40 Hz
audio phase align; fail closed otherwise.

Do not describe either graph as successful before visual inspection.

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
  exposes all eleven H3 AV state nodes and four H3 planning nodes through
  `/object_info`.
