"""Native wrappers around ComfyUI's official MiniMax H3 nodes."""

from __future__ import annotations

import json

from ..cauce.h3 import (
    append_reference,
    empty_reference_set,
    execute_add_guide,
    execute_fl2va,
    execute_ref2va,
    frame_index_in_window,
    reference_tags,
)


CATEGORY = "CAUCE/H3"


def _assert_family(profile, expected):
    if profile.get("family") != expected:
        raise ValueError(
            f"profile {profile.get('name')!r} is for {profile.get('family')}, not {expected}"
        )


class CauceH3FL2VA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "window": ("CAUCE_WINDOW",),
                "profile": ("CAUCE_PROFILE",),
            },
            "optional": {"first_frame": ("IMAGE",), "last_frame": ("IMAGE",)},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CAUCE_WINDOW")
    RETURN_NAMES = ("positive", "latent", "window")
    FUNCTION = "compile"
    CATEGORY = CATEGORY
    DESCRIPTION = "Compile CAUCE media/time directly through the current official H3 FL2VA node."

    def compile(self, clip, vae, prompt, window, profile, first_frame=None, last_frame=None):
        _assert_family(profile, "FL2VA")
        positive, latent = execute_fl2va(
            clip=clip,
            vae=vae,
            prompt=prompt,
            width=int(profile["width"]),
            height=int(profile["height"]),
            length=int(window["shape"]["pixel_frames"]),
            first_frame=first_frame,
            last_frame=last_frame,
        )
        return positive, latent, window


class CauceH3ReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {"references": ("CAUCE_H3_REFS",)},
        }

    RETURN_TYPES = ("CAUCE_H3_REFS", "STRING")
    RETURN_NAMES = ("references", "tags")
    FUNCTION = "append"
    CATEGORY = CATEGORY

    def append(self, image, references=None):
        result = append_reference(references, kind="image", media=image)
        return result, reference_tags(result)


class CauceH3ReferenceVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"video_frames": ("IMAGE",)},
            "optional": {"audio": ("AUDIO",), "references": ("CAUCE_H3_REFS",)},
        }

    RETURN_TYPES = ("CAUCE_H3_REFS", "STRING")
    RETURN_NAMES = ("references", "tags")
    FUNCTION = "append"
    CATEGORY = CATEGORY

    def append(self, video_frames, audio=None, references=None):
        duration = float(video_frames.shape[0]) / 24.0
        result = append_reference(
            references,
            kind="video",
            media=video_frames,
            audio=audio,
            duration_seconds=duration,
        )
        return result, reference_tags(result)


class CauceH3ReferenceAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"audio": ("AUDIO",)},
            "optional": {"references": ("CAUCE_H3_REFS",)},
        }

    RETURN_TYPES = ("CAUCE_H3_REFS", "STRING")
    RETURN_NAMES = ("references", "tags")
    FUNCTION = "append"
    CATEGORY = CATEGORY

    def append(self, audio, references=None):
        waveform = audio.get("waveform")
        sample_rate = int(audio.get("sample_rate", 0))
        if waveform is None or sample_rate <= 0:
            raise ValueError("audio reference must contain waveform and sample_rate")
        duration = float(waveform.shape[-1]) / sample_rate
        result = append_reference(
            references, kind="audio", media=audio, duration_seconds=duration
        )
        return result, reference_tags(result)


class CauceH3Ref2VA:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "window": ("CAUCE_WINDOW",),
                "profile": ("CAUCE_PROFILE",),
                "ref_image_size": (["match", "max"], {"default": "match"}),
            },
            "optional": {"references": ("CAUCE_H3_REFS",)},
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "CAUCE_WINDOW", "STRING")
    RETURN_NAMES = ("positive", "latent", "window", "reference_tags")
    FUNCTION = "compile"
    CATEGORY = CATEGORY
    DESCRIPTION = "Compile an ordered opaque reference set through the official H3 Ref2VA node."

    def compile(
        self,
        clip,
        vae,
        audio_vae,
        prompt,
        window,
        profile,
        ref_image_size,
        references=None,
    ):
        _assert_family(profile, "Ref2VA")
        refs = references or empty_reference_set()
        positive, latent = execute_ref2va(
            refs,
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            prompt=prompt,
            width=int(profile["width"]),
            height=int(profile["height"]),
            length=int(window["shape"]["pixel_frames"]),
            ref_image_size=ref_image_size,
        )
        return positive, latent, window, reference_tags(refs)


class CauceH3TimedGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "window": ("CAUCE_WINDOW",),
                "master_seconds": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 99_999.0, "step": 0.001},
                ),
            },
            "optional": {
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("positive", "frame_idx", "guide_json")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "Resolve an absolute CAUCE time and call the official arbitrary H3 AddGuide node."

    def apply(
        self,
        positive,
        latent,
        window,
        master_seconds,
        vae=None,
        audio_vae=None,
        image=None,
        audio=None,
    ):
        frame_idx = frame_index_in_window(window, master_seconds)
        result = execute_add_guide(
            positive=positive,
            latent=latent,
            frame_idx=frame_idx,
            vae=vae,
            audio_vae=audio_vae,
            image=image,
            audio=audio,
        )
        manifest = {
            "schema": "cauce.h3-guide/1",
            "window_id": window["id"],
            "master_seconds": float(master_seconds),
            "frame_idx": frame_idx,
            "has_image": image is not None,
            "has_audio": audio is not None,
        }
        return result, frame_idx, json.dumps(manifest, ensure_ascii=False, indent=2)


NODE_CLASS_MAPPINGS = {
    "CauceH3FL2VA": CauceH3FL2VA,
    "CauceH3ReferenceImage": CauceH3ReferenceImage,
    "CauceH3ReferenceVideo": CauceH3ReferenceVideo,
    "CauceH3ReferenceAudio": CauceH3ReferenceAudio,
    "CauceH3Ref2VA": CauceH3Ref2VA,
    "CauceH3TimedGuide": CauceH3TimedGuide,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CauceH3FL2VA": "CAUCE · H3 FL2VA",
    "CauceH3ReferenceImage": "CAUCE · Add H3 Image Reference",
    "CauceH3ReferenceVideo": "CAUCE · Add H3 Video Reference",
    "CauceH3ReferenceAudio": "CAUCE · Add H3 Audio Reference",
    "CauceH3Ref2VA": "CAUCE · H3 Ref2VA",
    "CauceH3TimedGuide": "CAUCE · H3 Timed Guide",
}
