# ComfyUI compatibility boundary

CAUCE depends on a deliberately small observable runtime surface:

- `NestedTensor` to carry the H3 visual and structural-audio streams;
- current `PackedLayout` semantics for arbitrary-frame AV guides;
- immutable conditioning metadata updates;
- the configured ComfyUI output directory.

Every such import lives under `cauce.comfy_compat`. Mathematical modules and
operation contracts do not import ComfyUI. `probe_comfy_capabilities()` returns
`cauce.comfy-capabilities/1` and distinguishes three cases:

- `true`: the required structure was observed;
- `false`: an incompatible structure was observed;
- `null`: the capability could not be observed in this process.

The probe is diagnostic, not permission to infer untested model behavior. A
capability may enable construction while execution and visual evidence remain
separate gates.

CAUCE currently preserves the established ComfyUI node registration surface.
It does not migrate opportunistically to the evolving V3 API: that migration
will be a separate compatibility adapter with paired testing against the
specific ComfyUI revisions used by the laboratory.
