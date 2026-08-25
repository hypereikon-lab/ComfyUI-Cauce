# System boundaries

| Responsibility | Owner |
| --- | --- |
| H3 model loading, conditioning, sampling, decoding | official ComfyUI nodes |
| arbitrary image/video guides at frame positions | `MiniMaxH3AddGuide` |
| FL2VA and Ref2VA model behavior | MiniMax H3 / official integrations |
| tail/head guide selection and accepted-center assembly | CAUCE |
| coordinate-map mathematics and warped reference media | CAUCE |
| packed H3 latent save/load | CAUCE |
| workflow graph, queue, history, outputs | ComfyUI |
| custom-node install/update/restart | Manager |
| authentication and HTTP publication | Cloudflare Access/Tunnel |
| physical power and Windows process startup | laboratory operator/configuration |

CAUCE should add a node only when it owns reproducible transformation logic, a
small data contract, or a safety boundary. Defaults, labels, and wrappers alone
are not enough.

The fixed production soundtrack is an editorial timebase, not a generative
input requirement. H3's structural-audio latent remains an internal model
stream wherever the official architecture requires it.
