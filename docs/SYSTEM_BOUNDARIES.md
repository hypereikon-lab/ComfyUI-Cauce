# System boundaries

The production system has five independent layers:

```text
laboratory infrastructure
  -> ComfyUI runtime
      -> official model nodes
          -> CAUCE operations
              -> a particular ComfyUI graph
```

## Laboratory infrastructure

Owns the physical tower, GPU, Windows process lifecycle, network, Cloudflare
Tunnel, Access policy, and storage capacity.

It does not own image or motion mathematics.

## ComfyUI runtime

Owns queue/history, WebSocket progress, input/output serving, workflow state,
custom-node discovery, Manager, and process restart.

The package-neutral operational reference is
[REMOTE_COMFY_RUNTIME.md](REMOTE_COMFY_RUNTIME.md).

## Official model nodes

Own H3 checkpoint loading, text conditioning, FL2VA, Ref2VA, guides, sampler
integration, VAE encode/decode, and model-specific sockets.

CAUCE must not shadow those nodes merely to set defaults or rename parameters.

## CAUCE

Owns only operations with reusable mathematics or explicit model translation:

- continuation geometry;
- temporal inpainting geometry and mask preparation;
- motion fields and inverse pullbacks;
- bounded H3 AV latent persistence;
- isolated experimental latent interventions.

CAUCE does not know where a clip belongs in a 35-minute work, what an image
depicts, or which artistic transition is desired.

## Graph

A graph selects source media, prompt, models, resolution, frame count,
conditioning mode, sampler, parameters, output path, and operation order.

Using CAUCE nodes in a graph does not make the whole graph part of CAUCE.

## Decision examples

| Requirement | Layer |
| --- | --- |
| expose `localhost:8188` privately | infrastructure |
| restart after a Python update | ComfyUI runtime |
| condition A→B with H3 | official model nodes + graph |
| compute a depth-driven pullback | CAUCE |
| splice a generated temporal patch without changing duration | CAUCE |
| choose which shot occurs at a musical moment | graph/editorial process |
| prove a latent intervention follows a vector field | CAUCE Research + validation |

When ownership is ambiguous, keep orchestration outside CAUCE and pass the
smallest explicit tensor or scalar contract into the operation.
