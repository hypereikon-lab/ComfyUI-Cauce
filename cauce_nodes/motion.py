"""ComfyUI-facing nodes for CAUCE's coordinate-map motion algebra."""

from __future__ import annotations

import json

from ..cauce.motion import (
    ANALYTIC_MAPS,
    EASINGS,
    INTEGRATORS,
    PADDING_MODES,
    VECTOR_FIELDS,
    WarpedH3Noise,
    affine_motion_map,
    analytic_motion_map,
    compose_motion_maps,
    depth_camera_motion_map,
    displacement_motion_map,
    integrate_advection,
    modulate_motion_map,
    motion_map_report,
    perspective_motion_map,
    vector_field,
    warp_h3_latent,
    warp_images,
)


MAP_CATEGORY = "CAUCE/Motion Maps"
H3_CATEGORY = "CAUCE/H3 Motion"


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _mask(value):
    import torch

    return torch.from_numpy(value).to(dtype=torch.float32)


def _geometry_inputs():
    return {
        "frames": ("INT", {"default": 124, "min": 1, "max": 3600, "step": 1}),
        "map_height": ("INT", {"default": 32, "min": 2, "max": 2048, "step": 1}),
        "map_width": ("INT", {"default": 48, "min": 2, "max": 2048, "step": 1}),
        "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}),
    }


class CauceAffineMotionMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_geometry_inputs(),
                "translate_x_start": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "translate_x_end": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "translate_y_start": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "translate_y_end": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "scale_start": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "scale_end": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "rotation_start": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "rotation_end": ("FLOAT", {"default": 0.0, "min": -10000.0, "max": 10000.0, "step": 0.1}),
                "pivot_x_percent": ("FLOAT", {"default": 50.0, "min": -500.0, "max": 500.0, "step": 0.1}),
                "pivot_y_percent": ("FLOAT", {"default": 50.0, "min": -500.0, "max": 500.0, "step": 0.1}),
                "easing": (list(EASINGS), {"default": "smoothstep"}),
            }
        }

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "build"
    CATEGORY = MAP_CATEGORY
    DESCRIPTION = "Build an exact target-to-source affine map; percentages refer to the frame extent."

    def build(self, frames, map_height, map_width, fps, **parameters):
        value = affine_motion_map(frames, map_height, map_width, fps=fps, **parameters)
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceAnalyticMotionMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_geometry_inputs(),
                "mode": (list(ANALYTIC_MAPS), {"default": "swirl"}),
                "amount_start": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "amount_end": ("FLOAT", {"default": 30.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "frequency": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "phase_cycles": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "sides": ("INT", {"default": 6, "min": 2, "max": 128, "step": 1}),
                "easing": (list(EASINGS), {"default": "smoothstep"}),
            }
        }

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "build"
    CATEGORY = MAP_CATEGORY

    def build(self, frames, map_height, map_width, fps, **parameters):
        value = analytic_motion_map(frames, map_height, map_width, fps=fps, **parameters)
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CaucePerspectiveMotionMap:
    @classmethod
    def INPUT_TYPES(cls):
        corner = ("FLOAT", {"default": 0.0, "min": -200.0, "max": 200.0, "step": 0.1})
        return {
            "required": {
                **_geometry_inputs(),
                "top_left_x_end": corner,
                "top_left_y_end": corner,
                "top_right_x_end": corner,
                "top_right_y_end": corner,
                "bottom_right_x_end": corner,
                "bottom_right_y_end": corner,
                "bottom_left_x_end": corner,
                "bottom_left_y_end": corner,
                "easing": (list(EASINGS), {"default": "smoothstep"}),
            }
        }

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "build"
    CATEGORY = MAP_CATEGORY
    DESCRIPTION = "Build a full projective corner-pin pullback from four endpoint offsets."

    def build(self, frames, map_height, map_width, fps, **parameters):
        value = perspective_motion_map(frames, map_height, map_width, fps=fps, **parameters)
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceDisplacementMotionMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "displacement_rg": ("IMAGE",),
                **_geometry_inputs(),
                "magnitude_percent": ("FLOAT", {"default": 10.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "encoding": (["centered_rg", "signed_rg"], {"default": "centered_rg"}),
            }
        }

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "build"
    CATEGORY = MAP_CATEGORY
    DESCRIPTION = "Import arbitrary RG vector data: optical flow, a simulation, or any external motion field."

    def build(self, displacement_rg, frames, map_height, map_width, fps, magnitude_percent, encoding):
        source = displacement_rg.detach().float().cpu().numpy()
        value = displacement_motion_map(
            source,
            frames,
            map_height,
            map_width,
            fps=fps,
            magnitude_percent=magnitude_percent,
            encoding=encoding,
        )
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceModulateMotionMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_map": ("CAUCE_MAP",),
                "strength_start": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_end": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "easing": (list(EASINGS), {"default": "smoothstep"}),
            },
            "optional": {"spatial_mask": ("MASK",)},
        }

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "modulate"
    CATEGORY = MAP_CATEGORY
    DESCRIPTION = "Apply temporal strength and an optional arbitrary spatial mask to any map."

    def modulate(self, motion_map, strength_start, strength_end, easing, spatial_mask=None):
        mask = None if spatial_mask is None else spatial_mask.detach().float().cpu().numpy()
        value = modulate_motion_map(
            motion_map,
            strength_start=strength_start,
            strength_end=strength_end,
            easing=easing,
            spatial_mask=mask,
        )
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceVectorField:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **_geometry_inputs(),
                "duration_seconds": ("FLOAT", {"default": 5.125, "min": 0.001, "max": 3600.0, "step": 0.001}),
                "kind": (list(VECTOR_FIELDS), {"default": "curl_sine"}),
                "speed_x_percent": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "speed_y_percent": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 30.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "spatial_scale": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "temporal_cycles": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "temporal_mode": (["forward", "sine_loop"], {"default": "forward"}),
            }
        }

    RETURN_TYPES = ("CAUCE_VECTOR_FIELD", "STRING")
    RETURN_NAMES = ("vector_field", "report_json")
    FUNCTION = "build"
    CATEGORY = MAP_CATEGORY

    def build(self, **parameters):
        value = vector_field(
            parameters.pop("frames"),
            parameters.pop("map_height"),
            parameters.pop("map_width"),
            **parameters,
        )
        report = {key: value[key] for key in ("schema", "kind", "frames", "height", "width", "fps", "duration_seconds", "temporal_mode", "tensor_hash")}
        report["parameters"] = value["parameters"]
        return value, _json(report)


class CauceIntegrateAdvection:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"vector_field": ("CAUCE_VECTOR_FIELD",), "integrator": (list(INTEGRATORS), {"default": "rk2"})}}

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "integrate"
    CATEGORY = MAP_CATEGORY

    def integrate(self, vector_field, integrator):
        value = integrate_advection(vector_field, method=integrator)
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceDepthCameraMotionMap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                **_geometry_inputs(),
                "fov_degrees": ("FLOAT", {"default": 50.0, "min": 1.0, "max": 178.0, "step": 0.1}),
                "near": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 10000.0, "step": 0.01}),
                "far": ("FLOAT", {"default": 10.0, "min": 0.002, "max": 100000.0, "step": 0.01}),
                "depth_mode": (["near_white", "near_black"], {"default": "near_white"}),
                "translate_x_end": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "translate_y_end": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "translate_z_end": ("FLOAT", {"default": 20.0, "min": -1000.0, "max": 1000.0, "step": 0.1}),
                "yaw_end": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1}),
                "pitch_end": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1}),
                "roll_end": ("FLOAT", {"default": 0.0, "min": -3600.0, "max": 3600.0, "step": 0.1}),
                "easing": (list(EASINGS), {"default": "smoothstep"}),
            }
        }

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "build"
    CATEGORY = MAP_CATEGORY
    DESCRIPTION = "Forward-splat a scalar depth plate into a camera pullback map with an explicit disocclusion mask."

    def build(self, depth, frames, map_height, map_width, fps, **parameters):
        scalar = depth[0].detach().float().mean(dim=-1).cpu().numpy()
        value = depth_camera_motion_map(scalar, frames, map_height, map_width, fps=fps, **parameters)
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceComposeMotionMaps:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"first": ("CAUCE_MAP",), "second": ("CAUCE_MAP",)}}

    RETURN_TYPES = ("CAUCE_MAP", "MASK", "STRING")
    RETURN_NAMES = ("motion_map", "validity", "report_json")
    FUNCTION = "compose"
    CATEGORY = MAP_CATEGORY
    DESCRIPTION = "Compose pullbacks as first(second(x)); sample the image or latent only once downstream."

    def compose(self, first, second):
        value = compose_motion_maps(first, second)
        return value, _mask(value["validity"]), _json(motion_map_report(value))


class CauceWarpImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "motion_map": ("CAUCE_MAP",), "padding_mode": (list(PADDING_MODES), {"default": "border"})}}

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "validity", "report_json")
    FUNCTION = "warp"
    CATEGORY = MAP_CATEGORY

    def warp(self, image, motion_map, padding_mode):
        warped, validity = warp_images(image, motion_map, padding_mode=padding_mode)
        report = motion_map_report(motion_map) | {"sampled_domain": "image", "padding_mode": padding_mode}
        return warped, validity, _json(report)


class CauceWarpH3Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",), "motion_map": ("CAUCE_MAP",), "padding_mode": (list(PADDING_MODES), {"default": "border"}), "mask_mode": (["none", "holes", "all"], {"default": "holes"})}}

    RETURN_TYPES = ("LATENT", "MASK", "STRING")
    RETURN_NAMES = ("latent", "validity", "report_json")
    FUNCTION = "warp"
    CATEGORY = H3_CATEGORY

    def warp(self, latent, motion_map, padding_mode, mask_mode):
        result, validity, report = warp_h3_latent(latent, motion_map, padding_mode=padding_mode, mask_mode=mask_mode)
        return result, validity, _json(report)


class CauceWarpedH3Noise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "motion_map": ("CAUCE_MAP",),
                "padding_mode": (list(PADDING_MODES), {"default": "reflection"}),
                "temporal_correlation": (
                    "FLOAT",
                    {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("NOISE", "STRING")
    RETURN_NAMES = ("noise", "report_json")
    FUNCTION = "build"
    CATEGORY = H3_CATEGORY
    DESCRIPTION = "Create H3 visual noise correlated by a CAUCE motion map; H3 audio noise remains unchanged."

    def build(self, seed, motion_map, padding_mode, temporal_correlation):
        report = motion_map_report(motion_map) | {
            "seed": int(seed),
            "padding_mode": padding_mode,
            "temporal_correlation": float(temporal_correlation),
            "output": "h3_warped_noise",
        }
        return (
            WarpedH3Noise(
                seed,
                motion_map,
                padding_mode,
                temporal_correlation=temporal_correlation,
            ),
            _json(report),
        )


NODE_CLASS_MAPPINGS = {
    "CauceAffineMotionMap": CauceAffineMotionMap,
    "CauceAnalyticMotionMap": CauceAnalyticMotionMap,
    "CaucePerspectiveMotionMap": CaucePerspectiveMotionMap,
    "CauceDisplacementMotionMap": CauceDisplacementMotionMap,
    "CauceModulateMotionMap": CauceModulateMotionMap,
    "CauceVectorField": CauceVectorField,
    "CauceIntegrateAdvection": CauceIntegrateAdvection,
    "CauceDepthCameraMotionMap": CauceDepthCameraMotionMap,
    "CauceComposeMotionMaps": CauceComposeMotionMaps,
    "CauceWarpImage": CauceWarpImage,
    "CauceWarpH3Latent": CauceWarpH3Latent,
    "CauceWarpedH3Noise": CauceWarpedH3Noise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceAffineMotionMap": "CAUCE Affine Motion Map",
    "CauceAnalyticMotionMap": "CAUCE Analytic Motion Map",
    "CaucePerspectiveMotionMap": "CAUCE Perspective Motion Map",
    "CauceDisplacementMotionMap": "CAUCE Displacement Motion Map",
    "CauceModulateMotionMap": "CAUCE Modulate Motion Map",
    "CauceVectorField": "CAUCE Vector Field",
    "CauceIntegrateAdvection": "CAUCE Integrate Advection",
    "CauceDepthCameraMotionMap": "CAUCE Depth Camera Motion Map",
    "CauceComposeMotionMaps": "CAUCE Compose Motion Maps",
    "CauceWarpImage": "CAUCE Warp Image",
    "CauceWarpH3Latent": "CAUCE Warp H3 Latent",
    "CauceWarpedH3Noise": "CAUCE Warped H3 Noise",
}
