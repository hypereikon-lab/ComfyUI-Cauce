#!/usr/bin/env python3
"""Build deterministic, loadable ComfyUI and API examples for CAUCE.

The visual schema mirrors ComfyUI frontend 1.49.x / workflow schema 0.4.  The
graphs intentionally use only comfy-core and ComfyUI-Cauce nodes.
"""

from __future__ import annotations

import json
from pathlib import Path
import uuid


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "examples" / "workflows"
API = ROOT / "examples" / "api"


def L(name, socket_type, optional=False):
    return (name, socket_type, "link", optional)


def W(name, socket_type):
    return (name, socket_type, "widget", False)


SPECS = {
    "LoadImage": (
        [W("image", "COMBO"), W("upload", "IMAGEUPLOAD")],
        [("IMAGE", "IMAGE"), ("MASK", "MASK")],
        [282, 102],
    ),
    "LoadVideo": (
        [W("file", "COMBO"), W("upload", "IMAGEUPLOAD")],
        [("VIDEO", "VIDEO")],
        [282, 82],
    ),
    "GetVideoComponents": (
        [L("video", "VIDEO")],
        [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"), ("bit_depth", "INT")],
        [220, 82],
    ),
    "ImageScale": (
        [
            L("image", "IMAGE"), W("upscale_method", "COMBO"),
            W("width", "INT"), W("height", "INT"), W("crop", "COMBO"),
        ],
        [("IMAGE", "IMAGE")],
        [270, 130],
    ),
    "CaucePlateCanvas": (
        [W("width", "INT"), W("height", "INT"), W("background", "STRING")],
        [("IMAGE", "IMAGE")],
        [270, 106],
    ),
    "CaucePlateLayer": (
        [
            L("canvas", "IMAGE"), L("layer", "IMAGE"), L("mask", "MASK", True),
            W("x_percent", "FLOAT"), W("y_percent", "FLOAT"), W("scale", "FLOAT"),
            W("rotation", "FLOAT"), W("opacity", "FLOAT"), W("blend_mode", "COMBO"),
            W("feather_pixels", "INT"),
        ],
        [("composite", "IMAGE"), ("layer_mask", "MASK")],
        [310, 282],
    ),
    "CauceTimelinePoint": (
        [W("point_id", "STRING"), W("master_seconds", "FLOAT"), W("prompt", "STRING")],
        [
            ("point", "CAUCE_POINT"), ("timeline_item", "CAUCE_ITEM"),
            ("prompt", "STRING"), ("master_seconds", "FLOAT"), ("timecode", "STRING"),
        ],
        [400, 216],
    ),
    "CauceAttachPointImage": (
        [L("point", "CAUCE_POINT"), L("image", "IMAGE"), W("version_id", "STRING"), W("source", "STRING")],
        [("point", "CAUCE_POINT"), ("image", "IMAGE")],
        [285, 102],
    ),
    "CauceExportPlate": (
        [L("image", "IMAGE"), L("point", "CAUCE_POINT"), W("prompt", "STRING")],
        [("image", "IMAGE"), ("png_path", "STRING"), ("prompt", "STRING")],
        [400, 200],
    ),
    "CauceGenerationWindow": (
        [
            W("window_id", "STRING"), W("accepted_start_seconds", "FLOAT"),
            W("accepted_duration_seconds", "FLOAT"), W("context_frames", "COMBO"),
            W("duplicate_prefix_frames", "COMBO"), W("snap_mode", "COMBO"),
            W("accept_mode", "COMBO"), W("maximum_frames", "INT"),
        ],
        [
            ("window", "CAUCE_WINDOW"), ("timeline_item", "CAUCE_ITEM"), ("length", "INT"),
            ("fps", "FLOAT"), ("accepted_offset_frames", "INT"),
            ("accepted_duration_seconds", "FLOAT"), ("accepted_end_seconds", "FLOAT"),
            ("summary", "STRING"), ("window_json", "STRING"),
        ],
        [323, 386],
    ),
    "CauceExecutionProfile": (
        [W("profile_name", "COMBO")],
        [
            ("profile", "CAUCE_PROFILE"), ("profile_name", "STRING"), ("family", "STRING"),
            ("width", "INT"), ("height", "INT"), ("tiled_vae", "BOOLEAN"),
            ("tile_size", "INT"), ("overlap", "INT"), ("temporal_size", "INT"),
            ("temporal_overlap", "INT"),
        ],
        [272, 238],
    ),
    "CaucePreflight": (
        [L("profile", "CAUCE_PROFILE"), W("minimum_free_reserve_gib", "FLOAT")],
        [("ready", "BOOLEAN"), ("report", "STRING"), ("preflight", "CAUCE_PREFLIGHT")],
        [315, 82],
    ),
    "CauceH3FL2VA": (
        [
            L("clip", "CLIP"), L("vae", "VAE"), L("window", "CAUCE_WINDOW"),
            L("profile", "CAUCE_PROFILE"), L("first_frame", "IMAGE", True),
            L("last_frame", "IMAGE", True), W("prompt", "STRING"),
        ],
        [("positive", "CONDITIONING"), ("latent", "LATENT"), ("window", "CAUCE_WINDOW")],
        [400, 238],
    ),
    "CauceH3ReferenceImage": (
        [L("image", "IMAGE"), L("references", "CAUCE_H3_REFS", True)],
        [("references", "CAUCE_H3_REFS"), ("tags", "STRING")],
        [266, 46],
    ),
    "CauceH3ReferenceVideo": (
        [L("video_frames", "IMAGE"), L("audio", "AUDIO", True), L("references", "CAUCE_H3_REFS", True)],
        [("references", "CAUCE_H3_REFS"), ("tags", "STRING")],
        [264, 66],
    ),
    "CauceH3Ref2VA": (
        [
            L("clip", "CLIP"), L("vae", "VAE"), L("audio_vae", "VAE"),
            L("window", "CAUCE_WINDOW"), L("profile", "CAUCE_PROFILE"),
            L("references", "CAUCE_H3_REFS", True), W("prompt", "STRING"),
            W("ref_image_size", "COMBO"),
        ],
        [
            ("positive", "CONDITIONING"), ("latent", "LATENT"),
            ("window", "CAUCE_WINDOW"), ("reference_tags", "STRING"),
        ],
        [400, 250],
    ),
    "CauceH3TimedGuide": (
        [
            L("positive", "CONDITIONING"), L("latent", "LATENT"), L("window", "CAUCE_WINDOW"),
            L("vae", "VAE", True), L("audio_vae", "VAE", True), L("image", "IMAGE", True),
            L("audio", "AUDIO", True), W("master_seconds", "FLOAT"),
        ],
        [("positive", "CONDITIONING"), ("frame_idx", "INT"), ("guide_json", "STRING")],
        [330, 182],
    ),
    "UNETLoader": (
        [W("unet_name", "COMBO"), W("weight_dtype", "COMBO")],
        [("MODEL", "MODEL")], [270, 82],
    ),
    "CLIPLoader": (
        [W("clip_name", "COMBO"), W("type", "COMBO"), W("device", "COMBO")],
        [("CLIP", "CLIP")], [270, 106],
    ),
    "VAELoader": ([W("vae_name", "COMBO")], [("VAE", "VAE")], [270, 58]),
    "MiniMaxH3SigmaShift": (
        [L("model", "MODEL"), W("shift_video", "FLOAT"), W("shift_audio", "FLOAT")],
        [("MODEL", "MODEL")], [277, 82],
    ),
    "KSamplerSelect": ([W("sampler_name", "COMBO")], [("SAMPLER", "SAMPLER")], [270, 58]),
    "BasicScheduler": (
        [L("model", "MODEL"), W("scheduler", "COMBO"), W("steps", "INT"), W("denoise", "FLOAT")],
        [("SIGMAS", "SIGMAS")], [270, 106],
    ),
    "BasicGuider": (
        [L("model", "MODEL"), L("conditioning", "CONDITIONING")],
        [("GUIDER", "GUIDER")], [180, 46],
    ),
    "RandomNoise": ([W("noise_seed", "INT")], [("NOISE", "NOISE")], [270, 82]),
    "SamplerCustomAdvanced": (
        [
            L("noise", "NOISE"), L("guider", "GUIDER"), L("sampler", "SAMPLER"),
            L("sigmas", "SIGMAS"), L("latent_image", "LATENT"),
        ],
        [("output", "LATENT"), ("denoised_output", "LATENT")], [230, 126],
    ),
    "CauceResolveParentLatent": (
        [L("latent", "LATENT"), L("window", "CAUCE_WINDOW")],
        [("LATENT", "LATENT")], [244, 46],
    ),
    "CaucePrepareContinuation": (
        [
            L("positive", "CONDITIONING"), L("target_latent", "LATENT"),
            L("previous_latent", "LATENT"), W("context_frames", "COMBO"),
            W("conditioning_mode", "COMBO"),
        ],
        [("positive", "CONDITIONING"), ("latent", "LATENT"), ("trim_frames", "INT")],
        [330, 142],
    ),
    "CaucePrepareBridge": (
        [
            L("positive", "CONDITIONING"), L("target_latent", "LATENT"),
            L("left_parent", "LATENT"), L("right_parent", "LATENT"),
            W("context_frames", "COMBO"), W("conditioning_mode", "COMBO"),
        ],
        [("positive", "CONDITIONING"), ("latent", "LATENT"), ("middle_frames", "INT")],
        [330, 162],
    ),
    "VAEDecode": ([L("samples", "LATENT"), L("vae", "VAE")], [("IMAGE", "IMAGE")], [160, 46]),
    "VAEDecodeAudio": ([L("samples", "LATENT"), L("vae", "VAE")], [("AUDIO", "AUDIO")], [170, 46]),
    "CauceAcceptDecodedWindow": (
        [L("images", "IMAGE"), L("window", "CAUCE_WINDOW"), L("audio", "AUDIO", True)],
        [("images", "IMAGE"), ("audio", "AUDIO"), ("accepted_frames", "INT")], [270, 66],
    ),
    "CauceSelectImageFrame": (
        [L("images", "IMAGE"), W("frame_index", "INT")],
        [("image", "IMAGE"), ("resolved_frame_index", "INT")], [260, 58],
    ),
    "VAEEncode": (
        [L("pixels", "IMAGE"), L("vae", "VAE")],
        [("LATENT", "LATENT")], [160, 46],
    ),
    "CauceBuildSeamWindow": (
        [
            L("left_frames", "IMAGE"), L("right_frames", "IMAGE"),
            L("left_fps", "FLOAT"), L("right_fps", "FLOAT"),
            W("context_seconds_per_side", "FLOAT"),
            W("repair_seconds_total", "FLOAT"),
            W("guide_frames", "INT"),
            W("maximum_frames", "INT"),
        ],
        [
            ("working_images", "IMAGE"), ("seam", "CAUCE_SEAM"),
            ("window", "CAUCE_WINDOW"), ("seam_json", "STRING"),
        ],
        [360, 154],
    ),
    "CaucePrepareH3SeamRepair": (
        [
            L("target_latent", "LATENT"), L("encoded_video_latent", "LATENT"),
            L("seam", "CAUCE_SEAM"), W("token_projection", "COMBO"),
            W("sampling_threshold", "FLOAT"),
            L("generation_support", "MASK", True),
        ],
        [("masked_latent", "LATENT"), ("mask_report", "STRING")], [350, 190],
    ),
    "CauceConfluenceFields": (
        [
            L("working_images", "IMAGE"), L("seam", "CAUCE_SEAM"),
            W("decoded_blend_frames", "INT"), W("curve", "COMBO"),
        ],
        [
            ("sampling_support", "MASK"), ("hard_acceptance", "MASK"),
            ("output_opacity", "MASK"), ("field_report", "STRING"),
        ],
        [350, 170],
    ),
    "CauceH3ConfluenceGuides": (
        [
            L("positive", "CONDITIONING"), L("target_latent", "LATENT"),
            L("working_images", "IMAGE"), L("seam", "CAUCE_SEAM"),
            L("vae", "VAE"),
        ],
        [("positive", "CONDITIONING"), ("guide_report", "STRING")],
        [360, 190],
    ),
    "CauceApplySeamPatch": (
        [
            L("left_frames", "IMAGE"), L("right_frames", "IMAGE"),
            L("repaired_working_images", "IMAGE"), L("seam", "CAUCE_SEAM"),
            W("decoded_feather_frames", "INT"), W("blend_curve", "COMBO"),
            L("blend_strength", "MASK", True),
        ],
        [("joined_images", "IMAGE"), ("repair_patch", "IMAGE"), ("splice_report", "STRING")],
        [360, 138],
    ),
    "CreateVideo": (
        [L("images", "IMAGE"), L("audio", "AUDIO", True), W("fps", "FLOAT"), W("bit_depth", "INT")],
        [("VIDEO", "VIDEO")], [270, 102],
    ),
    "SaveVideo": (
        [L("video", "VIDEO"), W("filename_prefix", "STRING"), W("format", "COMBO"), W("codec", "COMFY_DYNAMICCOMBO_V3")],
        [("video", "VIDEO")], [270, 106],
    ),
    "CauceRunReceipt": (
        [
            L("window", "CAUCE_WINDOW"), L("profile", "CAUCE_PROFILE"), W("artifact_id", "STRING"),
            W("seed", "INT"), W("sampler", "STRING"), W("scheduler", "STRING"),
            W("steps", "INT"), W("cfg", "FLOAT"), W("parents_json", "STRING"),
        ],
        [("receipt", "CAUCE_RECEIPT"), ("receipt_json", "STRING"), ("receipt_hash", "STRING")],
        [400, 296],
    ),
    "CauceSaveAVLatent": (
        [L("latent", "LATENT"), L("receipt", "CAUCE_RECEIPT", True), W("filename_prefix", "STRING"), W("artifact_index", "INT")],
        [("latent_path", "STRING")], [270, 102],
    ),
    "CauceLoadAVLatent": (
        [W("path_or_folder", "STRING"), W("artifact_index", "INT")],
        [("latent", "LATENT"), ("receipt", "CAUCE_RECEIPT"), ("resolved_path", "STRING")], [300, 82],
    ),
    "CauceSaveReceipt": (
        [L("receipt", "CAUCE_RECEIPT"), W("relative_path", "STRING")],
        [("receipt_path", "STRING")], [270, 58],
    ),
    "MarkdownNote": ([], [], [460, 220]),
}


class Workflow:
    def __init__(self, name, *, scale=0.55, offset=(150, 100)):
        self.name = name
        self.nodes = []
        self.links = []
        self.groups = []
        self.node_id = 0
        self.link_id = 0
        self.scale = scale
        self.offset = list(offset)

    def add(self, node_type, pos, widgets=None, *, size=None, title=None):
        inputs, outputs, default_size = SPECS[node_type]
        self.node_id += 1
        properties = {}
        if node_type != "MarkdownNote":
            properties["Node name for S&R"] = node_type
            if node_type.startswith("Cauce"):
                properties["aux_id"] = "hypereikon-lab/ComfyUI-Cauce"
            else:
                properties["cnr_id"] = "comfy-core"
        node = {
            "id": self.node_id,
            "type": node_type,
            "pos": list(pos),
            "size": list(size or default_size),
            "flags": {},
            "order": self.node_id - 1,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": properties,
        }
        if title:
            node["title"] = title
        for input_name, input_type, kind, optional in inputs:
            item = {"localized_name": input_name, "name": input_name, "type": input_type, "link": None}
            if kind == "widget":
                item["widget"] = {"name": input_name}
            if optional:
                item["shape"] = 7
            node["inputs"].append(item)
        for output_name, output_type in outputs:
            node["outputs"].append({
                "localized_name": output_name, "name": output_name,
                "type": output_type, "links": None,
            })
        if widgets is not None:
            node["widgets_values"] = list(widgets)
        if node_type == "MarkdownNote":
            node["color"] = "#24382a"
            node["bgcolor"] = "#132019"
        self.nodes.append(node)
        return node["id"]

    def node(self, node_id):
        return next(node for node in self.nodes if node["id"] == node_id)

    def connect(self, source_id, source_name, target_id, target_name):
        source = self.node(source_id)
        target = self.node(target_id)
        source_slot = next(i for i, item in enumerate(source["outputs"]) if item["name"] == source_name)
        target_slot = next(i for i, item in enumerate(target["inputs"]) if item["name"] == target_name)
        if target["inputs"][target_slot]["link"] is not None:
            raise ValueError(f"{target_id}.{target_name} already connected")
        source_type = source["outputs"][source_slot]["type"]
        target_type = target["inputs"][target_slot]["type"]
        if source_type != target_type:
            raise ValueError(f"socket mismatch {source_type} -> {target_type}")
        self.link_id += 1
        self.links.append([self.link_id, source_id, source_slot, target_id, target_slot, source_type])
        target["inputs"][target_slot]["link"] = self.link_id
        links = source["outputs"][source_slot]["links"]
        source["outputs"][source_slot]["links"] = (links or []) + [self.link_id]

    def group(self, title, bounding, *, color="#315f46"):
        self.groups.append({
            "id": len(self.groups) + 1,
            "title": title,
            "bounding": list(bounding),
            "color": color,
            "flags": {},
        })

    def data(self):
        return {
            "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://hypereikon.online/cauce/{self.name}")),
            "revision": 0,
            "last_node_id": self.node_id,
            "last_link_id": self.link_id,
            "nodes": self.nodes,
            "links": self.links,
            "groups": self.groups,
            "config": {},
            "extra": {"ds": {"scale": self.scale, "offset": self.offset}},
            "version": 0.4,
        }


def note(wf, pos, text, *, size=(520, 240), title=None):
    return wf.add("MarkdownNote", pos, [text], size=size, title=title)


def add_model_stack(wf, x, y, family):
    model_name = (
        "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
        if family == "FL2VA" else "minimax_h3_ref2va_pruned_fp8_scaled.safetensors"
    )
    unet = wf.add("UNETLoader", (x, y), [model_name, "default"])
    shifted = wf.add("MiniMaxH3SigmaShift", (x + 320, y), [12.0, 3.0])
    clip = wf.add(
        "CLIPLoader", (x, y + 150),
        ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
    )
    video_vae = wf.add("VAELoader", (x, y + 320), ["minimax_h3_video_vae_fp16.safetensors"])
    audio_vae = None
    if family == "Ref2VA":
        audio_vae = wf.add(
            "VAELoader", (x, y + 420), ["minimax_h3_audio_vae_fp32.safetensors"]
        )
    sampler = wf.add("KSamplerSelect", (x + 650, y), ["res_multistep"])
    scheduler = wf.add("BasicScheduler", (x + 650, y + 100), ["simple", 20, 1.0])
    wf.connect(unet, "MODEL", shifted, "model")
    wf.connect(shifted, "MODEL", scheduler, "model")
    return {
        "unet": unet, "model": shifted, "clip": clip, "video_vae": video_vae,
        "audio_vae": audio_vae, "sampler": sampler, "scheduler": scheduler,
    }


def add_sample_decode(wf, x, y, stack, conditioning, latent, window, *, seed, prefix):
    noise = wf.add("RandomNoise", (x, y), [seed, "fixed"])
    guider = wf.add("BasicGuider", (x, y + 130))
    sample = wf.add("SamplerCustomAdvanced", (x + 320, y + 60))
    parent = wf.add("CauceResolveParentLatent", (x + 610, y + 60))
    decode_v = wf.add("VAEDecode", (x + 900, y))
    accept = wf.add("CauceAcceptDecodedWindow", (x + 1120, y + 40))
    video = wf.add("CreateVideo", (x + 1430, y + 40), [24.0, 8])
    save = wf.add("SaveVideo", (x + 1750, y + 40), [prefix, "mp4", "auto"])
    wf.connect(stack["model"], "MODEL", guider, "model")
    wf.connect(conditioning[0], conditioning[1], guider, "conditioning")
    wf.connect(noise, "NOISE", sample, "noise")
    wf.connect(guider, "GUIDER", sample, "guider")
    wf.connect(stack["sampler"], "SAMPLER", sample, "sampler")
    wf.connect(stack["scheduler"], "SIGMAS", sample, "sigmas")
    wf.connect(latent[0], latent[1], sample, "latent_image")
    wf.connect(sample, "output", parent, "latent")
    wf.connect(window[0], window[1], parent, "window")
    wf.connect(parent, "LATENT", decode_v, "samples")
    wf.connect(stack["video_vae"], "VAE", decode_v, "vae")
    wf.connect(decode_v, "IMAGE", accept, "images")
    wf.connect(window[0], window[1], accept, "window")
    wf.connect(accept, "images", video, "images")
    wf.connect(video, "VIDEO", save, "video")
    return {
        "noise": noise,
        "sample": sample,
        "parent": parent,
        "accept": accept,
        "save": save,
    }


def build_plate():
    wf = Workflow("00_plate_sketch_handoff", scale=0.46, offset=(160, 120))
    note(wf, (-50, -360), """# CAUCE 00 · Plate sketch → PNG + prompt

Este flujo no usa GPU ni interpreta el contenido. Compone medios opacos, asocia
la imagen resultante a un punto del reloj maestro y exporta un PNG junto con el
prompt arbitrario para usar en Runway/GPT Image.

1. Sube los tres assets de `examples/assets/` al input de ComfyUI.
2. Ajusta posición, escala, opacidad y blend de cada capa.
3. Edita el prompt en el punto temporal.
4. Ejecuta para guardar el plate y su manifest.""", size=(700, 280))
    a = wf.add("LoadImage", (0, 0), ["cauce_forest_a.jpg", "image"], title="Plate source A")
    b = wf.add("LoadImage", (0, 280), ["cauce_forest_b.jpg", "image"], title="Plate source B")
    c = wf.add("LoadImage", (0, 560), ["cauce_forest_c.jpg", "image"], title="Plate source C")
    canvas = wf.add("CaucePlateCanvas", (350, 260), [1344, 768, "#10190B"])
    layer_a = wf.add("CaucePlateLayer", (700, 0), [50.0, 50.0, 1.0, 0.0, 1.0, "normal", 0])
    layer_b = wf.add("CaucePlateLayer", (1050, 240), [54.0, 48.0, 0.94, 0.0, 0.33, "screen", 48])
    layer_c = wf.add("CaucePlateLayer", (1400, 480), [46.0, 52.0, 0.92, 0.0, 0.24, "multiply", 48])
    point = wf.add("CauceTimelinePoint", (1780, 40), [
        "forest_threshold_001", 0.0,
        "Use the plate as the exact spatial composition. Preserve the path and watercourse as separate directional fields; retain humid Valdivian forest density, deep scale, and diffuse morning light.",
    ])
    attach = wf.add("CauceAttachPointImage", (1780, 340), ["plate_v001", "cauce_demo_assets"])
    export = wf.add("CauceExportPlate", (2140, 220), [""])
    wf.connect(canvas, "IMAGE", layer_a, "canvas")
    wf.connect(a, "IMAGE", layer_a, "layer")
    wf.connect(layer_a, "composite", layer_b, "canvas")
    wf.connect(b, "IMAGE", layer_b, "layer")
    wf.connect(layer_b, "composite", layer_c, "canvas")
    wf.connect(c, "IMAGE", layer_c, "layer")
    wf.connect(point, "point", attach, "point")
    wf.connect(layer_c, "composite", attach, "image")
    wf.connect(attach, "image", export, "image")
    wf.connect(attach, "point", export, "point")
    wf.connect(point, "prompt", export, "prompt")
    wf.group("OPAQUE MEDIA", (-40, -40, 350, 900))
    wf.group("PLATE COMPOSITION", (320, -40, 1430, 900))
    wf.group("POINT + HANDOFF", (1750, -40, 850, 760))
    return wf.data()


def build_fl2va():
    wf = Workflow("10_h3_fl2va_first_last", scale=0.30, offset=(110, 160))
    note(wf, (-50, -450), """# CAUCE 10 · H3 FL2VA first → last

Workflow de producción mínimo: dos imágenes opacas en puntos absolutos, una
ventana exacta H3, profile 5090, sampling visual, aceptación temporal, MP4,
latent y receipt. El audio generado por H3 se descarta; cambia imágenes, prompt
y seed sin cambiar la topología.""", size=(720, 250))
    first = wf.add("LoadImage", (0, 0), ["cauce_forest_a.jpg", "image"], title="First frame")
    last = wf.add("LoadImage", (0, 310), ["cauce_forest_b.jpg", "image"], title="Last frame")
    p0 = wf.add("CauceTimelinePoint", (330, 0), [
        "forest_motion_001_a", 0.0,
        "A single continuous shot through humid Valdivian temperate rainforest. Begin exactly from the first frame and arrive exactly at the last frame. The camera glides slowly forward while the two spatial directions—path and watercourse—exchange visual weight. Ferns move subtly in moist air; water keeps a coherent downstream flow. Natural diffuse light, no cuts.",
    ])
    p1 = wf.add("CauceTimelinePoint", (330, 310), ["forest_motion_001_b", 5.166666667, ""])
    a0 = wf.add("CauceAttachPointImage", (760, 0), ["frame_a_v001", "cauce_demo_assets"])
    a1 = wf.add("CauceAttachPointImage", (760, 310), ["frame_b_v001", "cauce_demo_assets"])
    window = wf.add("CauceGenerationWindow", (1090, 0), ["forest_window_001", 0.0, 5.0, "0", "0", "ceil", "nearest_run", 362])
    profile = wf.add("CauceExecutionProfile", (1090, 430), ["h3-5090-fl2va-640"])
    preflight = wf.add("CaucePreflight", (1090, 720), [35.0])
    stack = add_model_stack(wf, 1500, 0, "FL2VA")
    h3 = wf.add("CauceH3FL2VA", (2500, 0), [""])
    wf.connect(p0, "point", a0, "point"); wf.connect(first, "IMAGE", a0, "image")
    wf.connect(p1, "point", a1, "point"); wf.connect(last, "IMAGE", a1, "image")
    wf.connect(profile, "profile", preflight, "profile")
    wf.connect(stack["clip"], "CLIP", h3, "clip")
    wf.connect(stack["video_vae"], "VAE", h3, "vae")
    wf.connect(window, "window", h3, "window")
    wf.connect(profile, "profile", h3, "profile")
    wf.connect(a0, "image", h3, "first_frame")
    wf.connect(a1, "image", h3, "last_frame")
    wf.connect(p0, "prompt", h3, "prompt")
    result = add_sample_decode(
        wf, 3000, 0, stack, (h3, "positive"), (h3, "latent"), (window, "window"),
        seed=2026082101, prefix="cauce/demos/fl2va_first_last",
    )
    receipt = wf.add("CauceRunReceipt", (3650, 330), [
        "forest_window_001", 2026082101, "fixed", "res_multistep", "simple", 20, 1.0, "[]",
    ])
    save_latent = wf.add("CauceSaveAVLatent", (4110, 330), ["cauce/latents/forest_window", 1])
    save_receipt = wf.add("CauceSaveReceipt", (4110, 500), ["cauce/receipts/forest_window_001.json"])
    wf.connect(window, "window", receipt, "window"); wf.connect(profile, "profile", receipt, "profile")
    wf.connect(result["parent"], "LATENT", save_latent, "latent")
    wf.connect(receipt, "receipt", save_latent, "receipt")
    wf.connect(receipt, "receipt", save_receipt, "receipt")
    wf.group("PLATES + MASTER TIME", (-40, -40, 1400, 900))
    wf.group("MODELS + PROFILE", (1420, -40, 1050, 900))
    wf.group("H3 CONDITIONING", (2470, -40, 500, 500))
    wf.group("SAMPLE → ACCEPT → MP4", (2970, -40, 2130, 780))
    wf.group("PROVENANCE", (3620, 280, 820, 380))
    return wf.data()


def build_ref2va():
    wf = Workflow("20_h3_ref2va_motion_reference", scale=0.29, offset=(120, 180))
    note(wf, (-50, -450), """# CAUCE 20 · H3 Ref2VA: imagen + movimiento

Dos imágenes permanecen referencias visuales opacas. El MP4 demo aporta un
campo de movimiento/cámara; CAUCE no intenta describirlo ni clasificarlo.
Los tags compilados son `<Picture 1>`, `<Picture 2>`, `<Video 1>`.""", size=(720, 250))
    image_a = wf.add("LoadImage", (0, 0), ["cauce_forest_a.jpg", "image"])
    image_b = wf.add("LoadImage", (0, 280), ["cauce_forest_c.jpg", "image"])
    motion = wf.add("LoadVideo", (0, 560), ["cauce_motion_reference.mp4", "image"])
    components = wf.add("GetVideoComponents", (330, 560))
    ref_a = wf.add("CauceH3ReferenceImage", (620, 0))
    ref_b = wf.add("CauceH3ReferenceImage", (620, 220))
    ref_v = wf.add("CauceH3ReferenceVideo", (620, 500))
    window = wf.add("CauceGenerationWindow", (980, 0), ["ref_motion_001", 0.0, 5.0, "0", "0", "ceil", "nearest_run", 362])
    profile = wf.add("CauceExecutionProfile", (980, 430), ["h3-5090-ref2va-576x320"])
    preflight = wf.add("CaucePreflight", (980, 720), [35.0])
    stack = add_model_stack(wf, 1390, 0, "Ref2VA")
    h3 = wf.add("CauceH3Ref2VA", (2390, 0), [
        "Use <Picture 1> and <Picture 2> as the visual and spatial reference field. Use <Video 1> only as the movement and camera reference. Generate a continuous humid forest passage whose geometry remains coherent while path and stream trade perceptual dominance. Preserve the reference motion rhythm without copying literal image content from the motion clip.",
        "match",
    ])
    wf.connect(motion, "VIDEO", components, "video")
    wf.connect(image_a, "IMAGE", ref_a, "image")
    wf.connect(image_b, "IMAGE", ref_b, "image"); wf.connect(ref_a, "references", ref_b, "references")
    wf.connect(components, "images", ref_v, "video_frames"); wf.connect(ref_b, "references", ref_v, "references")
    wf.connect(profile, "profile", preflight, "profile")
    wf.connect(stack["clip"], "CLIP", h3, "clip")
    wf.connect(stack["video_vae"], "VAE", h3, "vae")
    wf.connect(stack["audio_vae"], "VAE", h3, "audio_vae")
    wf.connect(window, "window", h3, "window"); wf.connect(profile, "profile", h3, "profile")
    wf.connect(ref_v, "references", h3, "references")
    add_sample_decode(
        wf, 2890, 0, stack, (h3, "positive"), (h3, "latent"), (window, "window"),
        seed=2026082102, prefix="cauce/demos/ref2va_motion",
    )
    wf.group("OPAQUE REFERENCES", (-40, -40, 950, 900))
    wf.group("WINDOW + MODELS", (940, -40, 1420, 900))
    wf.group("REF2VA", (2360, -40, 500, 520))
    wf.group("SAMPLE → ACCEPT → MP4", (2860, -40, 2130, 760))
    return wf.data()


def build_timed_guide():
    wf = Workflow("30_h3_timed_guide", scale=0.29, offset=(120, 180))
    note(wf, (-50, -450), """# CAUCE 30 · Absolute-time guide

First/last definen los extremos. Una tercera imagen se ancla a 2,5 s del reloj
maestro mediante el AddGuide oficial. El tiempo no es un índice manual del
latent: CAUCE lo resuelve dentro de la ventana exacta.""", size=(720, 240))
    first = wf.add("LoadImage", (0, 0), ["cauce_forest_a.jpg", "image"])
    last = wf.add("LoadImage", (0, 280), ["cauce_forest_b.jpg", "image"])
    guide_image = wf.add("LoadImage", (0, 560), ["cauce_forest_c.jpg", "image"])
    point = wf.add("CauceTimelinePoint", (340, 0), [
        "guided_forest_001", 0.0,
        "One continuous forest shot. Begin at the first frame, pass through the guide composition at the specified master time, and arrive at the final frame. Motion remains slow, causal, and spatially coherent.",
    ])
    window = wf.add("CauceGenerationWindow", (780, 0), ["guided_window_001", 0.0, 5.0, "0", "0", "ceil", "nearest_run", 362])
    profile = wf.add("CauceExecutionProfile", (780, 430), ["h3-5090-fl2va-640"])
    stack = add_model_stack(wf, 1190, 0, "FL2VA")
    h3 = wf.add("CauceH3FL2VA", (2190, 0), [""])
    guide = wf.add("CauceH3TimedGuide", (2660, 0), [2.5])
    wf.connect(stack["clip"], "CLIP", h3, "clip"); wf.connect(stack["video_vae"], "VAE", h3, "vae")
    wf.connect(window, "window", h3, "window"); wf.connect(profile, "profile", h3, "profile")
    wf.connect(first, "IMAGE", h3, "first_frame"); wf.connect(last, "IMAGE", h3, "last_frame")
    wf.connect(point, "prompt", h3, "prompt")
    wf.connect(h3, "positive", guide, "positive"); wf.connect(h3, "latent", guide, "latent")
    wf.connect(window, "window", guide, "window"); wf.connect(stack["video_vae"], "VAE", guide, "vae")
    wf.connect(guide_image, "IMAGE", guide, "image")
    add_sample_decode(
        wf, 3070, 0, stack, (guide, "positive"), (h3, "latent"), (window, "window"),
        seed=2026082103, prefix="cauce/demos/timed_guide",
    )
    wf.group("MEDIA + TIME", (-40, -40, 800, 900))
    wf.group("WINDOW + MODELS", (750, -40, 1410, 900))
    wf.group("FL2VA + GUIDE", (2160, -40, 850, 520))
    wf.group("SAMPLE → ACCEPT → MP4", (3040, -40, 2130, 760))
    return wf.data()


def build_continuation():
    wf = Workflow("40_h3_two_window_continuation", scale=0.22, offset=(100, 180))
    note(wf, (-50, -500), """# CAUCE 40 · Two-window visual continuation

La ventana B hereda 39 frames latentes del parent A y usa además el
último frame decodificado de A como first-frame guide oficial. El latent asegura
contexto causal; la imagen refuerza la continuidad perceptual en runtimes H3 que
aún no incluyen clip guides. El audio interno se descarta. Ejecuta dos samples:
valida primero CAUCE 10.""", size=(790, 280))
    first = wf.add("LoadImage", (0, 0), ["cauce_forest_a.jpg", "image"])
    prompt_a = wf.add("CauceTimelinePoint", (320, 0), [
        "continuation_a", 0.0,
        "A single slow passage through humid Valdivian forest, beginning exactly from the image. Forward motion remains calm and coherent.",
    ])
    window_a = wf.add("CauceGenerationWindow", (760, 0), ["continuation_window_a", 0.0, 5.0, "0", "0", "ceil", "nearest_run", 362])
    window_b = wf.add("CauceGenerationWindow", (760, 440), ["continuation_window_b", 5.166666667, 3.541666667, "39", "0", "ceil", "nearest_run", 362])
    profile = wf.add("CauceExecutionProfile", (1120, 0), ["h3-5090-fl2va-640"])
    stack = add_model_stack(wf, 1480, 0, "FL2VA")
    cond_a = wf.add("CauceH3FL2VA", (2480, 0), [""])
    cond_b = wf.add("CauceH3FL2VA", (2480, 440), [
        "Continue the same visual event without a cut. Preserve direction, camera velocity, water flow, atmosphere, and causal motion from the inherited context. Let the forest density gradually increase.",
    ])
    wf.connect(stack["clip"], "CLIP", cond_a, "clip"); wf.connect(stack["video_vae"], "VAE", cond_a, "vae")
    wf.connect(window_a, "window", cond_a, "window"); wf.connect(profile, "profile", cond_a, "profile")
    wf.connect(first, "IMAGE", cond_a, "first_frame"); wf.connect(prompt_a, "prompt", cond_a, "prompt")
    wf.connect(stack["clip"], "CLIP", cond_b, "clip"); wf.connect(stack["video_vae"], "VAE", cond_b, "vae")
    wf.connect(window_b, "window", cond_b, "window"); wf.connect(profile, "profile", cond_b, "profile")
    result_a = add_sample_decode(
        wf, 3000, 0, stack, (cond_a, "positive"), (cond_a, "latent"), (window_a, "window"),
        seed=2026082104, prefix="cauce/demos/continuation_a",
    )
    endpoint = wf.add("CauceSelectImageFrame", (2480, 840), [-1])
    wf.connect(result_a["accept"], "images", endpoint, "images")
    wf.connect(endpoint, "image", cond_b, "first_frame")
    prepare_b = wf.add("CaucePrepareContinuation", (3000, 840), ["39", "mask_only"])
    wf.connect(cond_b, "positive", prepare_b, "positive")
    wf.connect(cond_b, "latent", prepare_b, "target_latent")
    wf.connect(result_a["parent"], "LATENT", prepare_b, "previous_latent")
    result_b = add_sample_decode(
        wf, 3400, 840, stack, (prepare_b, "positive"), (prepare_b, "latent"), (window_b, "window"),
        seed=2026082105, prefix="cauce/demos/continuation_b",
    )
    save_a = wf.add("CauceSaveAVLatent", (3650, 520), ["cauce/latents/continuation_a", 1])
    save_b = wf.add("CauceSaveAVLatent", (4050, 1360), ["cauce/latents/continuation_b", 1])
    wf.connect(result_a["parent"], "LATENT", save_a, "latent")
    wf.connect(result_b["parent"], "LATENT", save_b, "latent")
    wf.group("MEDIA + WINDOWS", (-40, -40, 1450, 960))
    wf.group("MODEL STACK", (1440, -40, 1000, 900))
    wf.group("WINDOW A", (2440, -40, 2700, 780))
    wf.group("WINDOW B · 39F INHERITED", (2440, 780, 3100, 840))
    return wf.data()


def build_bridge():
    wf = Workflow("50_h3_latent_bridge", scale=0.28, offset=(110, 180))
    note(wf, (-50, -430), """# CAUCE 50 · Latent bridge

Carga dos parents AV phase-safe ya guardados. Copia el tail izquierdo al head y
el head derecho al tail; únicamente el centro permanece generable. Antes de
ejecutar, ajusta ambos paths/índices a artifacts existentes.""", size=(700, 240))
    left = wf.add("CauceLoadAVLatent", (0, 0), ["cauce/latents/left_parent", 0])
    right = wf.add("CauceLoadAVLatent", (0, 180), ["cauce/latents/right_parent", 0])
    window = wf.add("CauceGenerationWindow", (360, 0), ["bridge_window_001", 10.0, 5.0, "0", "0", "ceil", "nearest_run", 362])
    profile = wf.add("CauceExecutionProfile", (360, 430), ["h3-5090-fl2va-640"])
    stack = add_model_stack(wf, 760, 0, "FL2VA")
    h3 = wf.add("CauceH3FL2VA", (1760, 0), [
        "Generate only the missing middle between the two inherited visual boundaries. Maintain causal camera motion, spatial direction, water flow, and atmosphere across both joins. No cut or camera reset.",
    ])
    bridge = wf.add("CaucePrepareBridge", (2220, 0), ["39", "mask_only"])
    wf.connect(stack["clip"], "CLIP", h3, "clip"); wf.connect(stack["video_vae"], "VAE", h3, "vae")
    wf.connect(window, "window", h3, "window"); wf.connect(profile, "profile", h3, "profile")
    wf.connect(h3, "positive", bridge, "positive"); wf.connect(h3, "latent", bridge, "target_latent")
    wf.connect(left, "latent", bridge, "left_parent"); wf.connect(right, "latent", bridge, "right_parent")
    result = add_sample_decode(
        wf, 2650, 0, stack, (bridge, "positive"), (bridge, "latent"), (window, "window"),
        seed=2026082106, prefix="cauce/demos/latent_bridge",
    )
    save = wf.add("CauceSaveAVLatent", (3300, 360), ["cauce/latents/bridge", 1])
    wf.connect(result["parent"], "LATENT", save, "latent")
    wf.group("PARENTS + WINDOW", (-40, -40, 760, 780))
    wf.group("MODEL STACK", (720, -40, 1000, 850))
    wf.group("BRIDGE CONDITIONING", (1720, -40, 880, 500))
    wf.group("SAMPLE → ACCEPT → MP4", (2610, -40, 2130, 760))
    return wf.data()


def build_confluence():
    wf = Workflow("60_h3_confluence_seam_repair", scale=0.22, offset=(90, 170))
    note(wf, (-50, -520), """# CAUCE 60 · Confluence bidireccional

Template para dos videos arbitrarios. Toma 2,5 s a cada lado del corte, añade
guards simétricos para formar 124 frames H3. El segundo interior solicitado se
ajusta a 22 frames exactos de la grilla temporal H3. Dos clips guía de 22 frames,
pegados a ambos bordes, entregan el movimiento entrante y saliente. Sólo ese
centro se genera; el exterior queda preservado por la máscara oficial de H3.

Requiere un ComfyUI oficial reciente con MiniMaxH3AddGuide y máscara temporal
por token. CAUCE se niega a ejecutar en runtimes anteriores. Sube
`gesture_a.mp4` y `gesture_b.mp4`; ambos necesitan al menos 60 frames a 24 fps.
El audio master no entra ni se reemplaza con audio generado.""", size=(820, 310))
    left_video = wf.add("LoadVideo", (0, 0), ["gesture_a.mp4", "image"], title="Gesture A")
    right_video = wf.add("LoadVideo", (0, 340), ["gesture_b.mp4", "image"], title="Gesture B")
    left_parts = wf.add("GetVideoComponents", (330, 0))
    right_parts = wf.add("GetVideoComponents", (330, 340))
    left_scale = wf.add("ImageScale", (590, 0), ["lanczos", 640, 640, "center"])
    right_scale = wf.add("ImageScale", (590, 340), ["lanczos", 640, 640, "center"])
    # Comfy preserves the original FLOAT widgets for left_fps/right_fps even
    # after those inputs are converted to links.  Keep their serialized values
    # so the remaining widgets do not shift left when the workflow is loaded.
    seam = wf.add(
        "CauceBuildSeamWindow",
        (930, 150),
        [24.0, 24.0, 2.5, 1.0, 22, 362],
    )
    first = wf.add("CauceSelectImageFrame", (1320, 0), [0], title="Working first frame")
    last = wf.add("CauceSelectImageFrame", (1320, 340), [-1], title="Working last frame")
    profile = wf.add("CauceExecutionProfile", (1680, 650), ["h3-5090-fl2va-640"])
    stack = add_model_stack(wf, 1720, 0, "FL2VA")
    condition = wf.add("CauceH3FL2VA", (2730, 0), [
        "Repair only the masked temporal join. Preserve framing, visual content, and motion outside it. Generate one continuous gesture whose position, velocity, and acceleration connect the left state to the right state without a cut. Do not introduce new subjects, objects, camera resets, or scene changes. Audio is not part of this repair."
    ])
    encode = wf.add("VAEEncode", (2730, 390))
    fields = wf.add("CauceConfluenceFields", (3100, 520), [4, "cosine"])
    prepare = wf.add(
        "CaucePrepareH3SeamRepair", (3100, 230), ["cover", 0.5]
    )
    noise = wf.add("RandomNoise", (3490, 0), [2026082201, "fixed"])
    guides = wf.add("CauceH3ConfluenceGuides", (3300, -170))
    guider = wf.add("BasicGuider", (3490, 150))
    sample = wf.add("SamplerCustomAdvanced", (3810, 100))
    decode = wf.add("VAEDecode", (4210, 100))
    apply = wf.add("CauceApplySeamPatch", (4470, 100), [4, "cosine"])
    joined_video = wf.add("CreateVideo", (4780, 0), [24.0, 8])
    joined_save = wf.add("SaveVideo", (5110, 0), [
        "cauce/demos/confluence_native_join", "mp4", "auto",
    ])
    patch_video = wf.add("CreateVideo", (4780, 260), [24.0, 8])
    patch_save = wf.add("SaveVideo", (5110, 260), [
        "cauce/demos/confluence_native_patch", "mp4", "auto",
    ])

    wf.connect(left_video, "VIDEO", left_parts, "video")
    wf.connect(right_video, "VIDEO", right_parts, "video")
    wf.connect(left_parts, "images", left_scale, "image")
    wf.connect(right_parts, "images", right_scale, "image")
    wf.connect(left_scale, "IMAGE", seam, "left_frames")
    wf.connect(right_scale, "IMAGE", seam, "right_frames")
    wf.connect(left_parts, "fps", seam, "left_fps")
    wf.connect(right_parts, "fps", seam, "right_fps")
    wf.connect(seam, "working_images", first, "images")
    wf.connect(seam, "working_images", last, "images")
    wf.connect(stack["clip"], "CLIP", condition, "clip")
    wf.connect(stack["video_vae"], "VAE", condition, "vae")
    wf.connect(seam, "window", condition, "window")
    wf.connect(profile, "profile", condition, "profile")
    wf.connect(first, "image", condition, "first_frame")
    wf.connect(last, "image", condition, "last_frame")
    wf.connect(seam, "working_images", encode, "pixels")
    wf.connect(stack["video_vae"], "VAE", encode, "vae")
    wf.connect(condition, "latent", prepare, "target_latent")
    wf.connect(encode, "LATENT", prepare, "encoded_video_latent")
    wf.connect(seam, "seam", prepare, "seam")
    wf.connect(seam, "working_images", fields, "working_images")
    wf.connect(seam, "seam", fields, "seam")
    wf.connect(fields, "sampling_support", prepare, "generation_support")
    wf.connect(condition, "positive", guides, "positive")
    wf.connect(condition, "latent", guides, "target_latent")
    wf.connect(seam, "working_images", guides, "working_images")
    wf.connect(seam, "seam", guides, "seam")
    wf.connect(stack["video_vae"], "VAE", guides, "vae")
    wf.connect(stack["model"], "MODEL", guider, "model")
    wf.connect(guides, "positive", guider, "conditioning")
    wf.connect(noise, "NOISE", sample, "noise")
    wf.connect(guider, "GUIDER", sample, "guider")
    wf.connect(stack["sampler"], "SAMPLER", sample, "sampler")
    wf.connect(stack["scheduler"], "SIGMAS", sample, "sigmas")
    wf.connect(prepare, "masked_latent", sample, "latent_image")
    wf.connect(sample, "output", decode, "samples")
    wf.connect(stack["video_vae"], "VAE", decode, "vae")
    wf.connect(left_scale, "IMAGE", apply, "left_frames")
    wf.connect(right_scale, "IMAGE", apply, "right_frames")
    wf.connect(decode, "IMAGE", apply, "repaired_working_images")
    wf.connect(seam, "seam", apply, "seam")
    wf.connect(fields, "output_opacity", apply, "blend_strength")
    wf.connect(apply, "joined_images", joined_video, "images")
    wf.connect(joined_video, "VIDEO", joined_save, "video")
    wf.connect(apply, "repair_patch", patch_video, "images")
    wf.connect(patch_video, "VIDEO", patch_save, "video")

    wf.group("SOURCES · NORMALIZE", (-40, -40, 890, 760))
    wf.group("CONFLUENCE WINDOW", (880, -40, 760, 600))
    wf.group("H3 MODEL + WORKING DOMAIN", (1640, -40, 1420, 960))
    wf.group("CONDITIONAL SEAM REPAIR", (3060, -40, 1380, 800))
    wf.group("DURATION-PRESERVING SPLICE", (4330, -40, 1110, 620))
    return wf.data()


def api_fl2va():
    return {
        "1": {"class_type": "CauceGenerationWindow", "inputs": {
            "window_id": "{{window.id}}", "accepted_start_seconds": "{{window.accepted_start_seconds}}",
            "accepted_duration_seconds": "{{window.accepted_duration_seconds}}", "context_frames": "{{window.context_frames}}",
            "duplicate_prefix_frames": "{{window.duplicate_prefix_frames}}", "snap_mode": "ceil",
            "accept_mode": "{{window.accept_mode}}", "maximum_frames": 362,
        }},
        "2": {"class_type": "CauceExecutionProfile", "inputs": {"profile_name": "h3-5090-fl2va-640"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "8": {"class_type": "LoadImage", "inputs": {"image": "{{window.first_frame}}"}},
        "9": {"class_type": "LoadImage", "inputs": {"image": "{{window.last_frame}}"}},
        "10": {"class_type": "CauceH3FL2VA", "inputs": {
            "clip": ["5", 0], "vae": ["6", 0], "prompt": "{{window.prompt}}", "window": ["1", 0],
            "profile": ["2", 0], "first_frame": ["8", 0], "last_frame": ["9", 0],
        }},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["10", 0]}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "13": {"class_type": "BasicScheduler", "inputs": {"model": ["4", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": "{{window.seed}}"}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "CauceResolveParentLatent", "inputs": {"latent": ["15", 0], "window": ["1", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["6", 0]}},
        "19": {"class_type": "CauceAcceptDecodedWindow", "inputs": {"images": ["17", 0], "window": ["1", 0]}},
        "20": {"class_type": "CreateVideo", "inputs": {"images": ["19", 0], "fps": 24.0, "bit_depth": 8}},
        "21": {"class_type": "SaveVideo", "inputs": {"video": ["20", 0], "filename_prefix": "cauce/sequence/{{window.id}}", "format": "mp4", "codec": "auto"}},
        "22": {"class_type": "CauceRunReceipt", "inputs": {
            "artifact_id": "{{window.id}}", "window": ["1", 0], "profile": ["2", 0], "seed": "{{window.seed}}",
            "sampler": "res_multistep", "scheduler": "simple", "steps": 20, "cfg": 1.0, "parents_json": "[]",
        }},
        "23": {"class_type": "CauceSaveAVLatent", "inputs": {"latent": ["16", 0], "filename_prefix": "cauce/latents/{{window.id}}", "artifact_index": 1, "receipt": ["22", 0]}},
        "24": {"class_type": "CauceSaveReceipt", "inputs": {"receipt": ["22", 0], "relative_path": "cauce/receipts/{{window.id}}.json"}},
    }


def api_continuation():
    return {
        "1": {"class_type": "CauceGenerationWindow", "inputs": {
            "window_id": "{{window.id}}", "accepted_start_seconds": "{{window.accepted_start_seconds}}",
            "accepted_duration_seconds": "{{window.accepted_duration_seconds}}", "context_frames": "{{window.context_frames}}",
            "duplicate_prefix_frames": "{{window.duplicate_prefix_frames}}", "snap_mode": "ceil",
            "accept_mode": "{{window.accept_mode}}", "maximum_frames": 362,
        }},
        "2": {"class_type": "CauceExecutionProfile", "inputs": {"profile_name": "h3-5090-fl2va-640"}},
        "3": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "4": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": ["3", 0], "shift_video": 12.0, "shift_audio": 3.0}},
        "5": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type": "minimax", "device": "default"}},
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "8": {"class_type": "CauceH3FL2VA", "inputs": {
            "clip": ["5", 0], "vae": ["6", 0], "prompt": "{{window.prompt}}",
            "window": ["1", 0], "profile": ["2", 0], "first_frame": ["26", 0]
        }},
        "9": {"class_type": "CauceLoadAVLatent", "inputs": {"path_or_folder": "{{window.parent_latent}}", "artifact_index": 0}},
        "10": {"class_type": "CaucePrepareContinuation", "inputs": {
            "positive": ["8", 0], "target_latent": ["8", 1], "previous_latent": ["9", 0],
            "context_frames": "{{window.context_frames}}", "conditioning_mode": "mask_only"
        }},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["10", 0]}},
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "13": {"class_type": "BasicScheduler", "inputs": {"model": ["4", 0], "scheduler": "simple", "steps": 20, "denoise": 1.0}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": "{{window.seed}}"}},
        "15": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["14", 0], "guider": ["11", 0], "sampler": ["12", 0], "sigmas": ["13", 0], "latent_image": ["10", 1]}},
        "16": {"class_type": "CauceResolveParentLatent", "inputs": {"latent": ["15", 0], "window": ["1", 0]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["6", 0]}},
        "19": {"class_type": "CauceAcceptDecodedWindow", "inputs": {"images": ["17", 0], "window": ["1", 0]}},
        "20": {"class_type": "CreateVideo", "inputs": {"images": ["19", 0], "fps": 24.0, "bit_depth": 8}},
        "21": {"class_type": "SaveVideo", "inputs": {"video": ["20", 0], "filename_prefix": "cauce/sequence/{{window.id}}", "format": "mp4", "codec": "auto"}},
        "22": {"class_type": "CauceRunReceipt", "inputs": {
            "artifact_id": "{{window.id}}", "window": ["1", 0], "profile": ["2", 0],
            "seed": "{{window.seed}}", "sampler": "res_multistep", "scheduler": "simple",
            "steps": 20, "cfg": 1.0, "parents_json": "{{window.parents_json}}"
        }},
        "23": {"class_type": "CauceSaveAVLatent", "inputs": {"latent": ["16", 0], "filename_prefix": "cauce/latents/{{window.id}}", "artifact_index": 1, "receipt": ["22", 0]}},
        "24": {"class_type": "CauceSaveReceipt", "inputs": {"receipt": ["22", 0], "relative_path": "cauce/receipts/{{window.id}}.json"}},
        "25": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["6", 0]}},
        "26": {"class_type": "CauceSelectImageFrame", "inputs": {"images": ["25", 0], "frame_index": -1}}
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    visual = {
        "00_plate_sketch_handoff.json": build_plate(),
        "10_h3_fl2va_first_last.json": build_fl2va(),
        "20_h3_ref2va_motion_reference.json": build_ref2va(),
        "30_h3_timed_guide.json": build_timed_guide(),
        "40_h3_two_window_continuation.json": build_continuation(),
        "50_h3_latent_bridge.json": build_bridge(),
        "60_h3_confluence_seam_repair.json": build_confluence(),
    }
    for filename, data in visual.items():
        write_json(WORKFLOWS / filename, data)
    write_json(API / "h3_fl2va_window.template.json", api_fl2va())
    write_json(API / "h3_continuation_window.template.json", api_continuation())


if __name__ == "__main__":
    main()
