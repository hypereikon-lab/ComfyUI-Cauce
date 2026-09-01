# System boundaries

| Responsibility | Owner |
| --- | --- |
| H3 model loading, conditioning, sampling, decoding | official ComfyUI nodes |
| arbitrary image/video guides at frame positions | `MiniMaxH3AddGuide` |
| FL2VA and Ref2VA model behavior | MiniMax H3 / official integrations |
| explicit target/guide/reference preprocessing plans | CAUCE |
| read-only H3 conditioning structure inspection | CAUCE |
| exact decoded frame-range selection | CAUCE |
| absolute H3 AV window/span/split/append mathematics | CAUCE |
| insertion of a compatible native AV span into H3 conditioning | CAUCE |
| packed H3 latent save/load | CAUCE |
| typed semantic operation contracts | CAUCE operation catalog |
| official-only H3 operation implementation | official/vanilla graph declared by CAUCE contract |
| paired reusable UI/API operation templates | CAUCE, only after live paired validation |
| workflow graph, queue, history, outputs | ComfyUI |
| custom-node install/update/restart | Manager |
| authentication and HTTP publication | Cloudflare Access/Tunnel |
| physical power and Windows process startup | laboratory operator/configuration |

CAUCE does not claim sampler patches, procedural-motion warps, latent affine
control, prompt interpretation, or model behavior that official H3 does not
expose. Those earlier experimental directions are not part of the present
architecture.

CAUCE should add a node only when it owns reproducible transformation logic, a
small data contract, or a safety boundary. Defaults, labels, and wrappers alone
are not enough.

CAUCE may define a named operation without owning its model nodes. Ownership is
kept honest by the operation's `implementation_class` and per-stage `owner`.

The fixed production soundtrack is an editorial timebase, not a generative
input requirement. H3's structural-audio latent remains an internal model
stream wherever the official architecture requires it.
