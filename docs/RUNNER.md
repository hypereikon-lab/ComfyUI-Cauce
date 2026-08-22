# CAUCE sequence runner

The runner is a small command-line companion, not a second creative interface.
It submits API-format Comfy workflows, writes state after every transition, and
resumes at the next incomplete window.

## Project format

```json
{
  "schema": "cauce.project/1",
  "server_url": "http://127.0.0.1:8188",
  "workflow_template": "api/h3_fl2va_window.template.json",
  "state_path": ".cauce/state.json",
  "receipts_dir": ".cauce/runs",
  "windows": [
    {"id": "w001", "seed": 1, "prompt": "..."},
    {
      "id": "w002",
      "workflow_template": "api/h3_continuation_window.template.json",
      "parent_latent": "cauce/latents/w001_00001.safetensors"
    }
  ]
}
```

The workflow is Comfy's API JSON, not browser workflow JSON. Placeholders can
appear anywhere:

```json
{
  "inputs": {
    "seed": "{{window.seed}}",
    "length": "{{window.frames}}",
    "filename_prefix": "cauce/{{window.id}}"
  }
}
```

When the entire string is one placeholder, its native JSON type is preserved.
`workflow_template` is a project default; an individual window can override it.
This lets one project use an FL2VA parent template for its first window and a
continuation template for later windows without branching the runner itself.

## Commands

```bash
python cauce_cli.py resume project.json --dry-run
python cauce_cli.py run project.json --once
python cauce_cli.py resume project.json
python cauce_cli.py status project.json
```

`--dry-run` writes every materialized workflow and state without contacting
ComfyUI.

## Cloudflare Access

The preferred production location is the laboratory machine using localhost.
Remote API execution through the tunnel requires a scoped Cloudflare Access
service token. CAUCE reads only these optional environment variables:

```text
CAUCE_CF_ACCESS_CLIENT_ID
CAUCE_CF_ACCESS_CLIENT_SECRET
```

It does not inspect browser cookies, Cloudflare sessions, password stores, or
the rest of the machine. Creating the service token and Access policy is a
separate infrastructure decision; it is not performed by this repository.
