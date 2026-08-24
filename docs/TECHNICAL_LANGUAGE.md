# Technical language

Use operation names that describe the actual transformation.

Preferred terms:

- continuation;
- first/last-frame conditioning;
- ordered reference conditioning;
- temporal inpainting;
- binary per-token denoise mask;
- decoded opacity feather;
- duration-preserving splice;
- affine or projective pullback;
- displacement field;
- vector-field advection;
- depth-camera reprojection;
- motion-map composition;
- image warp;
- native-latent seam;
- warped noise;
- sigma-conditioned latent transport.

Avoid metaphorical algorithm names in code, graphs, filenames, and reports.

## Evidence states

- `graph validated`: sockets resolve and the graph has no validation errors;
- `executes`: inference completed and artifacts exist;
- `executes but rejected`: tensor/runtime path worked but quality failed;
- `verified`: structural, measured, and perceptual gates passed;
- `blocked`: a named capability or external condition is missing.

## Required distinctions

- sampling support is not output opacity;
- nonzero pixel difference is not directional motion fidelity;
- clean decode is not production quality;
- H3 structural audio is not the production soundtrack;
- ComfyUI restart is not a physical tower reboot;
- Cloudflare Tunnel is not remote desktop or a shell;
- a node pack is not the complete graph or project.
