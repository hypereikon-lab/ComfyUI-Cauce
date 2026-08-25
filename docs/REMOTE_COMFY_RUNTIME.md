# Remote ComfyUI runtime behind Cloudflare Access

## Purpose

This document describes the laboratory ComfyUI installation as a remotely
reachable, programmable system. It is independent of any custom-node package,
model family, or production workflow.

The useful abstraction is:

```text
authenticated client
  -> Cloudflare Access
  -> Cloudflare Tunnel
  -> one ComfyUI HTTP origin
  -> graph runtime + APIs + frontend + extensions + Manager
```

A manually edited workflow, a Python runner, an MCP server, a frontend
extension, a maintenance script, or a custom-node pack can use the same
surfaces without adopting another layer's schemas or assumptions.

This guide records:

- what the tunnel does and does not expose;
- the independent state planes that make up the system;
- the control surfaces available above mouse interaction;
- what has been exercised in the long development session;
- the live capabilities observed on 2026-08-24;
- safe operating and recovery procedures;
- neutral opportunities for deeper automation;
- which upstream and community patterns are worth borrowing without making
  them dependencies.

It does not define creative entities, timeline semantics, H3 algorithms, or a
new product boundary.

## 1. System topology

The current route is:

```text
authenticated browser or machine client
             |
             v
https://comfy.hypereikon.online
             |
             v
Cloudflare Access policy
             |
             v
Cloudflare edge and named Tunnel
             |
             v
cloudflared Windows service
             |
             v
http://localhost:8188
             |
             v
ComfyUI Python process
  |-- core HTTP routes
  |-- /ws execution channel
  |-- frontend bundle and extension JS
  |-- custom-node Python code and custom routes
  `-- ComfyUI-Manager routes
```

Cloudflare publishes the complete HTTP origin at `localhost:8188`, not only the
visible canvas. An authenticated request can therefore reach every route
registered on that ComfyUI server, including core APIs, Manager, and routes
added by custom nodes.

The same tunnel does **not** expose:

- PowerShell, CMD, SSH, RDP, or a general Windows terminal;
- arbitrary ports that were not added as tunnel routes;
- power controls for the physical tower;
- automatic relaunch of ComfyUI unless the laboratory configures a separate
  Windows startup mechanism;
- arbitrary filesystem access unless a Comfy route or installed extension
  explicitly implements it.

The tunnel and ComfyUI are independent processes. `cloudflared` may remain
healthy while ComfyUI is stopped, in which case Cloudflare can reach the tower
but has no origin to proxy.

## 2. Five independent state planes

Treating “Comfy” as one state caused several avoidable ambiguities. The runtime
actually has five planes with different lifecycles.

| Plane | Examples | Survives a browser reload | Survives a Comfy restart |
| --- | --- | ---: | ---: |
| Physical/OS | tower power, network, Windows, disks, GPU driver | yes | yes |
| Tunnel/auth | `cloudflared`, hostname, Access policy, session/token | usually | yes |
| Server/runtime | Python imports, node classes, models in memory, Manager | yes | no |
| Jobs/artifacts | queue, history, input/output files, saved workflows | yes | partly/usually |
| Frontend/workspace | active graph, unsaved workflow tabs, selection, panels | browser-dependent | browser-dependent |

Consequences:

- Closing an internal workflow tab does not delete its output files.
- Closing the browser does not cancel an already queued backend job.
- Restarting ComfyUI unloads Python modules and GPU state but does not normally
  stop the Windows tunnel service.
- Reloading the frontend can restore unsaved workflow drafts and tabs.
- A visible Assets panel is an index, not proof that the corresponding file
  exists or does not exist.
- A successful Manager update on disk is not proof that the restarted process
  imported that commit.

## 3. Control surfaces

Use the highest stable surface that fully expresses an operation. Lower layers
remain valuable for diagnosis and version-specific tooling.

| Surface | Best for | Mutation power | Expected stability |
| --- | --- | ---: | --- |
| Visible UI / semantic browser control | human review, Manager dialogs, graph inspection | medium | medium |
| Public frontend extension API | commands, panels, hooks, workflow-aware UI | medium | medium/high |
| Comfy core HTTP API | discovery, uploads, jobs, queue, history, outputs | high | high with negotiation |
| Comfy WebSocket | progress, execution events, previews, errors | low/read-mostly | high |
| Custom Python nodes/routes | tensors, files, domain operations, new protocols | potentially total | under our control |
| Manager HTTP API | install/update/remove, models, snapshots, reboot | administrative | version-sensitive |
| Frontend stores/services | tabs, drafts, current canvas, internal commands | high inside UI | internal/version-pinned |
| CDP/network inspection | source discovery, diagnostics, browser-only state | potentially high | diagnostic only |
| Local CLI/OS access | process launch, logs, packages, complete filesystem | total | unavailable through this tunnel |

### Selection rule

```text
documented API
  > documented extension hook or command
  > source-pinned internal service
  > semantic DOM interaction
  > raw CDP intervention
```

CDP is excellent for learning what the installed frontend really exposes. It
should not become the production protocol when an API or extension hook exists.

## 4. Capability discovery before action

ComfyUI core, the separately versioned frontend, Manager, and custom nodes can
all move at different speeds. Never infer capability from the latest upstream
documentation alone.

Begin every technical session with read-only negotiation:

```text
GET /features
GET /system_stats
GET /prompt
GET /queue
GET /object_info
GET /extensions
GET /models
GET /customnode/installed
GET /manager/version
GET /manager/queue/status
GET /v2/userdata?dir=workflows
```

Probe newer endpoints separately and accept `404` or a disabled response as a
normal version outcome:

```text
GET /api/jobs
GET /api/history_v2
GET /api/tasks
GET /api/workflows
GET /api/assets
GET /health
GET /internal/folder_paths
```

Do not use a failed optional probe as a reason to update ComfyUI. First select a
supported route or frontend service already present in the installed version.

### Live read-only audit: 2026-08-24

| Capability | Observed result |
| --- | --- |
| ComfyUI | `0.33.0`, local portable Windows runtime |
| Frontend package | `1.49.6` |
| Python / PyTorch | Python `3.12.10`, PyTorch `2.13.0+cu130` |
| Device | RTX 5090, 34,190,458,880 bytes reported VRAM |
| Core node schemas | 957 classes from `/object_info` |
| Browser LiteGraph registry | 961 registered graph node types |
| Frontend extensions | 57 scripts from `/extensions` |
| Manager | `V3.41`, v4 support declared |
| Manager queue | idle |
| Core queue | zero running, zero pending |
| Upload limit | 100 MiB declared by `/features` |
| Asset subsystem | `assets: false`; `/api/assets` returned disabled/503 |
| Saved user data | `/v2/userdata` available and able to enumerate workflows |
| Jobs v2 | `/api/jobs` available |
| Other current-upstream surfaces | `/api/history_v2`, `/api/tasks`, `/api/workflows`, and `/health` absent |
| Internal folder inventory | `/internal/folder_paths` available |
| Frontend command registry | 108 commands registered |
| Frontend workflow service | open/create/save/rename/reorder/close/delete/sync methods present |

This table is evidence for one moment, not a permanent contract.

One additional package drift was observed: a local custom-node repository was
at `e83f75f`, while `/customnode/installed` reported its live process at
`e225ffe`. That is not a tunnel problem. It demonstrates why repository state,
installed clone state, and imported runtime state must be checked separately.

## 5. What the long session has already exercised

The session JSONL is approximately 1.7 GiB and must not be ingested wholesale.
Targeted searches and repository records establish that the following paths
have actually been used:

### Authentication and reachability

- interactive Cloudflare Access login in the in-app browser;
- normal Comfy frontend and WebSocket operation through the hostname;
- expected short `502` intervals during Comfy process restarts;
- recovery after the origin returned without changing the tunnel;
- differentiation between tower/tunnel/origin failures.

### Runtime discovery

- `/system_stats` for runtime and device information;
- `/object_info` for node schemas and import verification;
- `/extensions` and the frontend node registry;
- `/models` and model-folder discovery;
- `/customnode/installed` for installed commit verification;
- Manager version, policy, and queue status.

### Workflow and job operation

- graph creation and loading in the frontend;
- API-format workflow validation against live node schemas;
- job submission and queue monitoring;
- long GPU jobs surviving frontend navigation;
- history inspection and exact output enumeration;
- output retrieval through authenticated `/view` URLs;
- visual inspection of MP4 outputs at selected times.

### Installation and deployment

- install through an approved Git URL;
- targeted Manager update of one custom-node pack;
- Manager queue reset/start/status polling;
- verification of the installed Git SHA;
- Comfy-only reboot through `/manager/reboot`;
- node-import verification after the process returned.

### Workspace behavior

- multiple unsaved workflow tabs restored by frontend state;
- full workflow paste opening another internal tab;
- explicit close/discard dialogs;
- discovery that Comfy keeps at least one workflow tab alive;
- replacement of accumulated experiments with one empty workflow;
- outputs and backend history remaining independent from tab closure.

### Bounded storage behavior

- uploads into Comfy input space;
- output generation and authenticated retrieval;
- distinction between frontend index, history record, and physical file;
- bounded custom-node inventory/cleanup for `input/` and `output/` only.

Not established by these actions:

- a general shell on the tower;
- safe arbitrary model deletion through Manager;
- physical power-on after a shutdown;
- guaranteed automatic Comfy launch on Windows boot;
- generic filesystem access outside routes deliberately implemented for it.

## 6. Authentication modes

### Interactive browser

Cloudflare Access checks requests using the browser's authenticated application
session. This is appropriate for:

- manual graph editing;
- authenticated UI automation;
- read-only source and runtime inspection;
- occasional Manager operations with direct user oversight.

Never export `CF_Authorization`, browser cookies, or profile state into scripts,
logs, repositories, or workflow JSON.

### Headless remote client

A durable runner should use a scoped Cloudflare Access service token and a
corresponding Service Auth policy. Cloudflare accepts:

```text
CF-Access-Client-Id: <client id>
CF-Access-Client-Secret: <client secret>
```

The token is an infrastructure credential, not a Comfy setting. It should be:

- created separately from human login;
- scoped to this Access application;
- stored in a secret manager or local environment variables;
- rotated and revocable;
- excluded from workflow files and run receipts;
- tested for both HTTP and WebSocket upgrade behavior in the selected client.

For clients that cannot attach headers to WebSocket upgrades, use HTTP polling
as a fallback or establish an authenticated application session through a
supported machine flow. Do not work around this by copying browser cookies.

### On-host client

A runner executed on the laboratory tower can use
`http://127.0.0.1:8188` directly and avoid Cloudflare entirely. This is the most
reliable location for unattended long sequences, but it requires an operator or
a separate deployment mechanism because the current tunnel does not provide a
terminal.

## 7. Core Comfy API

### Discovery

```text
GET /features
GET /system_stats
GET /object_info[/<NodeClass>]
GET /models[/<folder>]
GET /extensions
GET /workflow_templates
```

`/object_info` is the executable schema registry. It should drive validation,
input typing, node-pack dependency checks, and workflow generation. A node
visible in a README but absent from this endpoint is not available to the
running process.

### Upload and retrieval

```text
POST /upload/image
POST /upload/mask
GET  /view?filename=...&subfolder=...&type=input|output|temp
GET  /view_metadata/<folder>
```

The installed runtime currently declares a 100 MiB upload limit. Larger inputs
need chunking through a deliberately implemented route, an on-host transfer, or
operator assistance; retrying the browser upload does not change that limit.

### Execution

```text
POST /prompt
GET  /prompt
GET  /queue
POST /queue
POST /interrupt
GET  /history[/<prompt_id>]
POST /history
POST /free
```

A submitted prompt is an immutable queued snapshot. Later edits to the visible
graph do not alter that job.

The request body uses API-format JSON:

```json
{
  "client_id": "<uuid>",
  "prompt": {
    "12": {
      "class_type": "SomeNode",
      "inputs": {
        "value": 1,
        "source": ["4", 0]
      }
    }
  }
}
```

Keep UI workflow JSON and executable prompt JSON as separate artifacts:

- UI JSON preserves node positions, widgets, groups, notes, and subgraphs.
- API JSON preserves the executable dependency graph.

### WebSocket

`/ws?clientId=<uuid>` carries status, node execution, step progress, previews,
errors, and completion events. The same `client_id` should be sent with
`POST /prompt` so events can be associated with the submitting client.

Use WebSocket for responsive monitoring, but keep an HTTP reconciliation path:

```text
WebSocket event stream
  + GET /api/jobs or /queue
  + GET /history/<prompt_id>
  = durable job truth
```

A disconnected progress stream is not proof that the GPU job stopped.

### User data and saved workflows

```text
GET    /v2/userdata?dir=workflows
GET    /userdata/<file>
POST   /userdata/<file>
DELETE /userdata/<file>
POST   /userdata/<file>/move/<dest>
GET    /settings
POST   /settings[/<id>]
```

These routes can support a saved workflow library without browser downloads.
Deleting a user-data file is distinct from closing a frontend tab and must be
treated as a destructive filesystem action.

### Jobs route adapter

The live instance supports `/api/jobs`, while some other current-upstream APIs
are absent. A neutral client should implement fallbacks:

```text
list/cancel one job:
  prefer /api/jobs...
  fall back to /queue + /interrupt + /history

history:
  prefer /api/history_v2 when present
  fall back to /history

workflow library:
  prefer /api/workflows when present
  fall back to /v2/userdata and /userdata
```

This route selection belongs to the runtime client, not to any creative schema.

## 8. Frontend as a programmable workspace

The frontend is more than a DOM around the API. On the audited build,
`window.app` exposed:

- `graphToPrompt`, `loadGraphData`, `queuePrompt`, and `loadApiJson`;
- the current LiteGraph root and canvas;
- `app.api`, whose methods wrap queue, history, jobs, userdata, models, settings,
  logs, and memory operations;
- `app.extensionManager` stores for workflow, command, settings, dialogs,
  sidebars, bottom panels, and queue settings.

The workflow service exposed operations including:

```text
createNewTemporary
openWorkflow / openWorkflows
saveWorkflow / saveAs
renameWorkflow / reorderWorkflows
closeWorkflow / deleteWorkflow
syncWorkflows
activeWorkflow / workflows / modifiedWorkflows
```

The command registry contained reusable commands such as:

```text
Comfy.NewBlankWorkflow
Comfy.OpenWorkflow
Comfy.SaveWorkflow
Comfy.ExportWorkflowAPI
Comfy.ClearWorkflow
Comfy.QueuePrompt
Comfy.Interrupt
Workspace.CloseWorkflow
Comfy.RefreshNodeDefinitions
Comfy.Memory.UnloadModels
Comfy.Memory.UnloadModelsAndExecutionCache
```

This means many operations currently performed with clicks can be implemented
as a small Comfy frontend extension. Prefer officially registered commands and
extension hooks. Direct use of workflow stores is acceptable only behind a
frontend-version adapter and a capability check.

### Public extension surface

An extension can use `app.registerExtension(...)` to provide:

- commands, keybindings, and menu commands;
- settings;
- sidebar and bottom-panel tabs;
- selection-toolbox commands;
- node lifecycle hooks;
- graph-load hooks;
- custom widgets and node behavior;
- API event listeners.

This is the appropriate home for workspace-aware functions such as:

- enumerate and label open workflow tabs;
- close only agent-owned temporary workflows;
- validate the active graph before queueing;
- compare live graph state with a committed workflow;
- show installed/runtime drift;
- export UI and API representations together;
- record an experiment signature and output prefix;
- surface an explicit confirmation before administrative operations.

It does not require a second dashboard.

### Source-pinned internals

The current workflow store and command registry are useful, but internal. If a
feature needs them:

1. detect frontend package version through `/system_stats`;
2. verify required method names at runtime;
3. isolate calls in one adapter;
4. fail closed with a useful message;
5. test on the real frontend after every package update.

### CDP

CDP can inspect frontend services, issue same-origin diagnostic requests,
capture network events, pause videos at exact times, and inspect browser-console
failures. Use it for:

- research and reverse engineering;
- determining the actual command/store surface;
- debugging extension imports that only fail in the browser;
- validating DOM, canvas, or media behavior unavailable through HTTP.

Do not use CDP to persist secret state, bypass Access, or create an undocumented
production dependency when a custom extension can provide the same contract.

## 9. Custom Python nodes and routes

A custom-node package can contain three independent contributions:

```text
Python nodes       -> graph-executed compute
Python routes      -> HTTP control/data API
JavaScript files   -> frontend behavior
```

The backend route mechanism is straightforward:

```python
from aiohttp import web
from server import PromptServer

routes = PromptServer.instance.routes

@routes.get("/example/capabilities")
async def capabilities(request):
    return web.json_response({"schema": "example.capabilities/1"})
```

Custom messages can also be emitted over the existing Comfy WebSocket with
`PromptServer.instance.send_sync(...)`.

This makes a narrowly scoped runtime bridge possible without exposing a shell.
Every route must still be treated as code running with the permissions of the
ComfyUI process.

Good route contracts are:

- bounded and typed;
- capability-reporting;
- idempotent where possible;
- explicit about read versus mutation;
- path-rooted rather than accepting arbitrary absolute paths;
- auditable with request IDs and receipts;
- safe to retry after a tunnel interruption.

Avoid generic endpoints such as `run_command`, `eval_python`, or unrestricted
file read/write. Installing such a node would silently turn application access
into general machine access.

## 10. Manager as an administrative API

Manager V3.41 exposes an OpenAPI-described control plane for:

- installed-node inventory and mappings;
- install/update/reinstall/disable/uninstall/fix;
- model installation;
- update queues;
- snapshots;
- Comfy version switching;
- policy and DB mode;
- Comfy process reboot.

Administrative routes include:

```text
POST /manager/queue/reset
POST /manager/queue/install
POST /manager/queue/update
POST /manager/queue/uninstall
POST /manager/queue/install_model
POST /manager/queue/start
GET  /manager/queue/status
POST /manager/reboot
```

### Targeted update contract

For one installed pack:

```text
publish intended Git commit
  -> verify remote ref
  -> ensure GPU queue idle
  -> reset Manager queue
  -> enqueue only that pack
  -> start Manager queue
  -> wait for Manager queue idle
  -> verify installed SHA
  -> reboot ComfyUI if Python changed
  -> wait through expected short 502
  -> verify imported node schema
```

Never substitute visible `Update All` for a targeted update. The audited
Manager update policy is `nightly-comfyui`; a broad update can change core,
frontend expectations, Manager, and unrelated custom nodes at once.

### Security meaning

Installing a custom node is equivalent to authorizing Python code to execute as
the Windows account running ComfyUI. Manager's Git URL and pip surfaces are
therefore administrative even though they appear inside a web page.

Current Manager separates `allow_git_url_install` and `allow_pip_install` from
its general `security_level`. These flags default to deny and work only when
Comfy listens on loopback. That matches this architecture: Comfy remains on
`localhost:8188`, while Access protects the external hostname.

Do not widen Comfy's bind address merely to make an install feature work.

## 11. Operational playbooks

### Start a remote session

1. Confirm the hostname reaches Comfy rather than an Access error or persistent
   origin error.
2. Read `/features`, `/system_stats`, queue state, Manager state, and installed
   pack SHAs.
3. Snapshot open internal workflow tabs and identify their ownership.
4. Record the active graph signature and whether it is modified.
5. Confirm storage headroom before model or video work.
6. Select only capabilities reported by this runtime.

### Run a workflow

1. Resolve node schemas from `/object_info`.
2. Validate model filenames against `/models/<folder>`.
3. Assign a unique output prefix and client/job ID.
4. Verify zero frontend validation errors.
5. Confirm queue idle or intentionally scheduled.
6. Submit once.
7. Follow WebSocket progress and reconcile with job/queue HTTP state.
8. Read exact outputs from history/job assets.
9. Validate media, not merely the `success` status.
10. Save UI graph, API graph, parameters, runtime versions, and output paths.

### Restart ComfyUI

1. Verify no job is running.
2. Preserve any frontend state that must survive.
3. `POST /manager/reboot`.
4. Treat the immediate dropped request or `502` as expected.
5. Wait approximately 8–15 seconds before the first retry.
6. Re-read `/system_stats`, `/object_info/<representative node>`, Manager
   installed SHA, and queue state.
7. Do not reboot the physical tower.

### Diagnose a disconnection

| Observation | Likely failing plane | Action |
| --- | --- | --- |
| Brief 502 after reboot | Comfy process | wait and verify |
| Access page/log-in failure | auth/policy | reauthenticate; do not change Comfy |
| Hostname resolves, persistent 502 | origin process | ask operator to start/recover Comfy |
| Hostname unreachable | tower/network/tunnel | operator or Cloudflare diagnosis |
| UI loads, custom node absent | Python import | inspect schema/logs/Manager import state |
| Job disappeared from UI | frontend index | query jobs/queue/history directly |
| WebSocket stopped, queue still running | client connection | reconnect; do not resubmit |
| Manager reports old SHA | installed clone/update | rerun targeted update after checking refs |

### End a session

1. Reconcile jobs and outputs.
2. Persist reproducible workflow state.
3. Close only agent-owned temporary workflow tabs.
4. Leave one clean workflow if the frontend requires one.
5. Do not delete history or artifacts merely to make the UI look empty.
6. Record runtime/package drift and rejected experiments.
7. Leave queue idle unless an intentional unattended job is documented.

## 12. Artifact and provenance model

The frontend should not be the archive. A durable run record should contain:

```text
run id / prompt id / client id
timestamp
UI workflow hash
API workflow hash
node-pack SHAs
Comfy + frontend + Manager versions
model filenames and, when available, hashes
seed and execution parameters
input filenames/hashes
output filename/subfolder/type
job terminal state
validation result
```

Three representations of “the same workflow” should be kept distinct:

```text
editable UI graph
executable API graph
immutable submitted job snapshot
```

Likewise, distinguish:

```text
physical file
history record
asset-index entry
frontend thumbnail
authenticated /view reachability
```

These sets overlap but are not identical.

## 13. Security model for deeper automation

A useful neutral automation layer should borrow the following controls from
current secure Comfy/MCP projects:

- read-only discovery enabled by default;
- explicit classes for queue, filesystem, installation, and process mutations;
- dangerous-node inspection before submitting arbitrary workflows;
- absolute-path and `..` rejection;
- allowed roots for input/output operations;
- model download host and file-format allowlists;
- disk-space preflight;
- rate limiting for mutation endpoints;
- audit records with secret-field redaction;
- confirmation gates before install, uninstall, reboot, deletion, or paid API
  nodes;
- WebSocket monitoring with HTTP polling fallback;
- payload-size limits and pagination;
- no generic eval or shell route.

An Access login authorizes reaching the application. It does not make every
custom node or Manager operation safe.

## 14. Neutral opportunities beyond current use

These opportunities operate on the remote runtime itself and should not be
named or modeled as features of a creative node pack.

### Read-only runtime manifest

Produce one versioned capability document from:

```text
/features
/system_stats
/object_info
/extensions
/models
/customnode/installed
/manager/version
optional endpoint probes
```

It would let any future client answer “what can this machine do now?” without
assuming a repository or model family.

### Headless workflow executor

A small client can:

- authenticate with a service token;
- validate API JSON against live schemas;
- submit jobs;
- reconnect to progress;
- retrieve outputs;
- write reproducible receipts;
- resume a sequence without browser state.

It should be graph-agnostic. Specialized experiment or production runners can
remain users of the same lower-level client.

### Workspace bridge

A minimal frontend extension can expose safe workspace operations through
registered commands or narrowly scoped routes:

- list open workflow tabs with path/modified status;
- identify current graph and selected nodes;
- load, save, export, compare, or close a specified workflow;
- execute a registered frontend command;
- capture validation errors and browser-console extension failures;
- ask for user confirmation inside Comfy.

This would replace repeated DOM clicking while preserving the native Comfy
workspace. It should not introduce a second creative interface.

### Dependency resolver

Given a workflow, compare its node classes and model widget values against
`/object_info`, `/models`, and Manager mappings. Return a plan before offering
any install. This is more useful than allowing an agent to install reactively
after a graph fails.

### Experiment/run registry

Index submitted job snapshots, parameters, outputs, validations, and rejected
results independently of the Assets sidebar. The registry can remain a set of
JSON receipts before becoming a database.

### Safe administration adapter

Wrap Manager with:

- targeted operations only;
- queue-idle checks;
- before/after snapshots;
- intended and observed SHA verification;
- restart and import checks;
- rollback guidance;
- no `Update All` shortcut.

### Observability

Expose a compact health view from read-only APIs:

```text
Access reachable
Tunnel/origin response
Comfy version and uptime proxy
queue depth and active job
GPU/host memory
disk reserve via a bounded custom route
import failures
installed/repository drift
WebSocket freshness
```

This can be a CLI report or agent tool; it does not need a dashboard.

## 15. Avoiding premature product boundaries

Do not decide the package before the recurring operations are understood.

Prefer contracts such as:

```text
probe runtime
inspect graph
validate dependencies
submit job
observe job
retrieve artifact
record receipt
update one extension
restart origin
recover session
```

over early domain nouns such as:

```text
timeline engine
shot ontology
music-aware state
semantic media entity
universal creative controller
```

When a recurring operation becomes stable, decide its correct location:

| Recurring need | Natural home |
| --- | --- |
| model/tensor operation | custom Python node |
| reusable HTTP behavior | bounded custom route |
| live canvas/tab behavior | frontend extension |
| unattended job execution | external or on-host API client |
| installation/restart | guarded Manager adapter |
| authentication/reachability | Cloudflare configuration |
| project-specific media grammar | the relevant project or node package |

This keeps the remote system general and allows any creative package to shrink,
change, or be replaced without losing the operational infrastructure.

## 16. Ecosystem patterns reviewed

These repositories are research inputs, not selected dependencies.

### Official

- `Comfy-Org/ComfyUI` now carries a machine-readable OpenAPI specification and
  multiple job API generations.
- `Comfy-Org/ComfyUI_frontend` exposes public extension hooks plus a richer
  internal workflow/command architecture.
- `Comfy-Org/ComfyUI-Manager` exposes an OpenAPI-described administrative
  surface and explicit install security flags.
- `Comfy-Org/comfy-cli` demonstrates structured JSON output, API/UI workflow
  conversion, asynchronous run/wait/download commands, version selection,
  snapshots, and agent-friendly operation.

### Community patterns worth understanding

- `artokun/comfyui-mcp` treats Comfy as an agent control plane, with graph
  authoring, live panel integration, workflow diffing, rollback, WebSocket
  progress, and runtime-aware tools.
- `pytraveler/local-comfyui-mcp` highlights the key architectural gap: the
  backend API cannot see unsaved tab-memory graph state, so a browser/custom-node
  bridge is needed for live-canvas control.
- `hybridindie/comfyui_mcp` demonstrates path sanitization, dangerous-node
  auditing, rate limiting, payload limits, redacted audit logs, and explicit
  decisions not to expose some denial-of-service or destructive routes.
- `AIDC-AI/ComfyUI-Copilot` demonstrates workflow generation, debugging,
  rewriting, and parameter sweeps inside the native Comfy workspace.

The lesson is not “install an MCP server immediately.” The reusable lesson is
the separation between:

```text
backend job protocol
live-workspace bridge
knowledge/skills layer
administrative operations
security policy
```

## 17. Research snapshot and primary sources

Research was refreshed on 2026-08-24. Because these projects move quickly,
follow the links and re-probe the live instance before relying on a detail.

Pinned upstream snapshot:

- ComfyUI `180060c295da013c9bc834f575ae8142a4e6c38f`:
  [server routes](https://github.com/Comfy-Org/ComfyUI/blob/180060c295da013c9bc834f575ae8142a4e6c38f/server.py),
  [OpenAPI](https://github.com/Comfy-Org/ComfyUI/blob/180060c295da013c9bc834f575ae8142a4e6c38f/openapi.yaml),
  [user data](https://github.com/Comfy-Org/ComfyUI/blob/180060c295da013c9bc834f575ae8142a4e6c38f/app/user_manager.py).
- Frontend `02c7166f8c9a56dae3f5f60c96027c4bdbc4818d`:
  [workflow service](https://github.com/Comfy-Org/ComfyUI_frontend/blob/02c7166f8c9a56dae3f5f60c96027c4bdbc4818d/src/platform/workflow/core/services/workflowService.ts),
  [Comfy API client](https://github.com/Comfy-Org/ComfyUI_frontend/blob/02c7166f8c9a56dae3f5f60c96027c4bdbc4818d/src/scripts/api.ts),
  [extension interface](https://github.com/Comfy-Org/ComfyUI_frontend/blob/02c7166f8c9a56dae3f5f60c96027c4bdbc4818d/src/types/comfy.ts).
- Manager `f39cbd56fecae0b27a446c0cd450cd591f3a8bea`:
  [OpenAPI](https://github.com/Comfy-Org/ComfyUI-Manager/blob/f39cbd56fecae0b27a446c0cd450cd591f3a8bea/openapi.yaml),
  [security/configuration](https://github.com/Comfy-Org/ComfyUI-Manager/blob/f39cbd56fecae0b27a446c0cd450cd591f3a8bea/README.md).
- comfy-cli `77eb4865a24736904c9a1b56b683643f0200b5f2`:
  [README](https://github.com/Comfy-Org/comfy-cli/blob/77eb4865a24736904c9a1b56b683643f0200b5f2/README.md).

Official documentation:

- [Comfy server routes](https://docs.comfy.org/development/comfyui-server/comms_routes)
- [JavaScript extensions](https://docs.comfy.org/custom-nodes/js/javascript_overview)
- [Extension hooks](https://docs.comfy.org/custom-nodes/js/javascript_hooks)
- [Cloudflare Tunnel setup](https://developers.cloudflare.com/tunnel/setup/)
- [Cloudflare Access service tokens](https://developers.cloudflare.com/cloudflare-one/access-controls/service-credentials/service-tokens/)
- [Cloudflare authorization cookie](https://developers.cloudflare.com/cloudflare-one/access-controls/applications/http-apps/authorization-cookie/)
- [Windows tunnel service](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/as-a-service/windows/)

Community references:

- [artokun/comfyui-mcp](https://github.com/artokun/comfyui-mcp)
- [pytraveler/local-comfyui-mcp](https://github.com/pytraveler/local-comfyui-mcp)
- [hybridindie/comfyui_mcp](https://github.com/hybridindie/comfyui_mcp)
- [AIDC-AI/ComfyUI-Copilot](https://github.com/AIDC-AI/ComfyUI-Copilot)

## 18. Update discipline

Update this guide when any of the following changes:

- Cloudflare hostname, policy, or authentication mode;
- tunnel service topology;
- Comfy core or frontend package version;
- Manager version or security configuration;
- supported API surface;
- Windows startup behavior for ComfyUI;
- service-token decision;
- storage or hardware envelope;
- a new operational path is proven in the live instance;
- a documented path fails and a fallback is established.

Every update should mark claims as one of:

```text
documented upstream
observed live
implemented locally
proposed
rejected
```

That vocabulary prevents a promising possibility from being mistaken for an
available production capability.
